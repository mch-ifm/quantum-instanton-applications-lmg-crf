import gc
import warnings
import multiprocessing
import numpy as np
import pandas as pd
from scipy.sparse import (diags, kron, identity, coo_matrix, csc_matrix, block_diag)
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from pathlib import Path
from petsc4py import PETSc
from slepc4py import SLEPc

_BYTES_PER_NNZ = 24
_BASE_OVERHEAD_GB = 0.5
_KRYLOV_VECS_EXTRA = 10
_ILU_GROWTH_CAP = 4.0


def _estimate_worker_gb(nnz_L, dim, k, n_L_copies=1.5, nnz_ilu=0):
    """Conservative per-worker memory estimate in GB."""
    base   = n_L_copies * nnz_L * _BYTES_PER_NNZ
    krylov = (k + _KRYLOV_VECS_EXTRA) * dim * 16
    ilu    = nnz_ilu * _BYTES_PER_NNZ
    total  = base + krylov + ilu + _BASE_OVERHEAD_GB * (1024 ** 3)
    return total / (1024 ** 3)


def _estimate_ilu_nnz(nnz_L, avg_row_nnz, fill_level, growth_cap=_ILU_GROWTH_CAP):
    """Conservative nnz estimate for an ILU(fill_level) factorization."""
    if fill_level <= 0:
        return nnz_L
    g = min(max(avg_row_nnz, 1.0), growth_cap)
    return int(nnz_L * (g ** fill_level))


def choose_ilu_fill(nnz_L, dim, k, mem_per_worker_gb, max_fill=5, verbose=True):
    """
    Pick the largest ILU(fill) whose estimated memory fits in
    `mem_per_worker_gb`.  Returns (fill_level, ilu_usable).

    If `ilu_usable` is False, the caller should fall back to Jacobi
    (or another low-memory preconditioner), because even ILU(0) is
    estimated to exceed the budget.  A RuntimeWarning is emitted in
    that case.
    """
    base_gb = _estimate_worker_gb(nnz_L, dim, k, nnz_ilu=0)
    if base_gb >= mem_per_worker_gb:
        warnings.warn(
            f"[ILU] base memory estimate {base_gb:.2f} GB already exceeds "
            f"per-worker budget {mem_per_worker_gb:.2f} GB; ILU not usable.",
            RuntimeWarning,
        )
        return 0, False

    avg_row_nnz = nnz_L / float(dim)
    best = None
    for f in range(max_fill, -1, -1):
        nnz_ilu = _estimate_ilu_nnz(nnz_L, avg_row_nnz, f)
        est_gb = _estimate_worker_gb(nnz_L, dim, k, nnz_ilu=nnz_ilu)
        if est_gb <= mem_per_worker_gb:
            best = (f, est_gb)
            break

    if best is None:
        warnings.warn(
            f"[ILU] even ILU(0) estimate exceeds per-worker budget "
            f"{mem_per_worker_gb:.2f} GB; falling back to Jacobi.",
            RuntimeWarning,
        )
        return 0, False

    f, est_gb = best
    if verbose:
        print(f"[ILU] chose fill={f}, est. {est_gb:.2f} GB/worker "
              f"(budget {mem_per_worker_gb:.2f} GB)")
    return f, True


# =====================================================================
# CRF base operators
# =====================================================================

# def _Ap(j, m): return  np.sqrt(np.clip((j - m) * (j + m + 1), 0.0, None))
def _Am(j, m): return  np.sqrt(np.clip((j + m) * (j - m + 1), 0.0, None))
def _Bm(j, m): return -np.sqrt(np.clip((j + m) * (j + m - 1), 0.0, None))
def _Bz(j, m): return  np.sqrt(np.clip((j + m) * (j - m), 0.0, None))
def _Dm(j, m): return  np.sqrt(np.clip((j - m + 1) * (j - m + 2), 0.0, None))
def _Dz(j, m): return  np.sqrt(np.clip((j + m + 1) * (j - m + 1), 0.0, None))


def _coo_from_blocks(blocks, dim):
    rows = np.concatenate([b[0] for b in blocks])
    cols = np.concatenate([b[1] for b in blocks])
    vals = np.concatenate([b[2] for b in blocks])
    return coo_matrix((vals, (rows, cols)), shape=(dim, dim)).tocsc()


def build_crf_bases(n):
    """
    Precompute CRF base matrices for a system of size n.
    """
    n_half = n // 2
    dim = (n + 1) * (n + 2) * (n + 3) // 6
    block_offsets = np.array(
        [(4 * k**3 - k) // 3 for k in range(n_half + 1)], dtype=np.int64
    )

    # ----- 1. diagonal indices and <Sz> values -----
    diag_list, sz_list = [], []
    for k in range(n_half + 1):
        off = block_offsets[k]
        dk = 2 * k + 1
        for j in range(-k, k + 1):
            diag_list.append(off + (j + k) * dk + (j + k))
            sz_list.append(j)
    diag_indices = np.array(diag_list, dtype=np.int64)
    sz_diag_vals = np.array(sz_list, dtype=float)

    # ----- 2. diagonal parts m1, m1p, m1deph (vectorized) -----
    m1_diag = np.zeros(dim)
    m1p_diag = np.zeros(dim)
    m1deph_diag = np.zeros(dim)
    for k in range(n_half + 1):
        off = block_offsets[k]
        dk = 2 * k + 1
        jj, ii = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1), indexing='ij')
        idx = (off + (jj + k) * dk + (ii + k)).ravel()
        m1_diag[idx] = (-(n - ii - jj)).ravel()
        m1p_diag[idx] = (-(n + ii + jj)).ravel()
        if k == 0:
            m1deph_diag[idx] = -n_half
        else:
            m1deph_diag[idx] = (-(n_half - ii * jj * (n_half + 1) / (k * (k + 1)))).ravel()

    m1 = diags(m1_diag, 0, shape=(dim, dim), format='csc')
    m1p = diags(m1p_diag, 0, shape=(dim, dim), format='csc')
    m1deph = diags(m1deph_diag, 0, shape=(dim, dim), format='csc')

    # ----- 3. coupling matrices (vectorized) -----
    def _grid(k):
        return np.meshgrid(np.arange(-k, k + 1),
                           np.arange(-k, k + 1), indexing='ij')

    def build_m2():
        blk = []
        for k in range(n_half):
            o0, o1 = block_offsets[k], block_offsets[k + 1]
            d0, d1 = 2 * k + 1, 2 * (k + 1) + 1
            f = (n_half + k + 2) / ((k + 1) * (2 * k + 3))
            jj, ii = _grid(k)
            r = (o0 + (jj + k) * d0 + (ii + k)).ravel()
            c = (o1 + (jj + k) * d1 + (ii + k)).ravel()
            v = (f * _Bm(k + 1, -ii + 1) * _Bm(k + 1, -jj + 1)).ravel()
            blk.append((r, c, v))
        return _coo_from_blocks(blk, dim)

    def build_m3():
        blk = []
        for k in range(1, n_half + 1):
            o0 = block_offsets[k]
            d0 = 2 * k + 1
            f = (n_half + 1) / ((k + 1) * k)
            jj, ii = np.meshgrid(np.arange(-k + 1, k + 1),
                                 np.arange(-k + 1, k + 1), indexing='ij')
            r = (o0 + (jj + k) * d0 + (ii + k)).ravel()
            c = (o0 + (jj + k - 1) * d0 + (ii + k - 1)).ravel()
            v = (f * _Am(k, -(ii - 1)) * _Am(k, -(jj - 1))).ravel()
            blk.append((r, c, v))
        return _coo_from_blocks(blk, dim)

    def build_m4():
        blk = []
        for k in range(1, n_half + 1):
            o0, o1 = block_offsets[k], block_offsets[k - 1]
            d0, d1 = 2 * k + 1, 2 * (k - 1) + 1
            f = (n_half - k + 1) / ((2 * k - 1) * k)
            jj, ii = np.meshgrid(np.arange(-k + 2, k + 1),
                                 np.arange(-k + 2, k + 1), indexing='ij')
            r = (o0 + (jj + k) * d0 + (ii + k)).ravel()
            c = (o1 + (jj + k - 2) * d1 + (ii + k - 2)).ravel()
            v = (f * _Dm(k - 1, -(ii - 1)) * _Dm(k - 1, -(jj - 1))).ravel()
            blk.append((r, c, v))
        return _coo_from_blocks(blk, dim)

    def build_m2p():
        blk = []
        for k in range(n_half):
            o0, o1 = block_offsets[k], block_offsets[k + 1]
            d0, d1 = 2 * k + 1, 2 * (k + 1) + 1
            f = (n_half + k + 2) / ((k + 1) * (2 * k + 3))
            jj, ii = _grid(k)
            r = (o0 + (jj + k) * d0 + (ii + k)).ravel()
            c = (o1 + (jj + k + 2) * d1 + (ii + k + 2)).ravel()
            v = (f * _Bm(k + 1, ii + 1) * _Bm(k + 1, jj + 1)).ravel()
            blk.append((r, c, v))
        return _coo_from_blocks(blk, dim)

    def build_m3p():
        blk = []
        for k in range(1, n_half + 1):
            o0 = block_offsets[k]
            d0 = 2 * k + 1
            f = (n_half + 1) / ((k + 1) * k)
            jj, ii = np.meshgrid(np.arange(-k, k),
                                 np.arange(-k, k), indexing='ij')
            r = (o0 + (jj + k) * d0 + (ii + k)).ravel()
            c = (o0 + (jj + k + 1) * d0 + (ii + k + 1)).ravel()
            v = (f * _Am(k, ii + 1) * _Am(k, jj + 1)).ravel()
            blk.append((r, c, v))
        return _coo_from_blocks(blk, dim)

    def build_m4p():
        blk = []
        for k in range(1, n_half + 1):
            o0, o1 = block_offsets[k], block_offsets[k - 1]
            d0, d1 = 2 * k + 1, 2 * (k - 1) + 1
            f = (n_half - k + 1) / ((2 * k - 1) * k)
            jj, ii = np.meshgrid(np.arange(-k, k - 1),
                                 np.arange(-k, k - 1), indexing='ij')
            r = (o0 + (jj + k) * d0 + (ii + k)).ravel()
            c = (o1 + (jj + k) * d1 + (ii + k)).ravel()
            v = (f * _Dm(k - 1, ii + 1) * _Dm(k - 1, jj + 1)).ravel()
            blk.append((r, c, v))
        return _coo_from_blocks(blk, dim)

    def build_m2deph():
        blk = []
        for k in range(n_half):
            o0, o1 = block_offsets[k], block_offsets[k + 1]
            d0, d1 = 2 * k + 1, 2 * (k + 1) + 1
            f = (n_half + k + 2) / ((k + 1) * (2 * k + 3))
            jj, ii = _grid(k)
            r = (o0 + (jj + k) * d0 + (ii + k)).ravel()
            c = (o1 + (jj + k + 1) * d1 + (ii + k + 1)).ravel()
            v = (f * _Bz(k + 1, -ii) * _Bz(k + 1, -jj)).ravel()
            blk.append((r, c, v))
        return _coo_from_blocks(blk, dim)

    def build_m3deph():
        blk = []
        for k in range(1, n_half + 1):
            o0, o1 = block_offsets[k], block_offsets[k - 1]
            d0, d1 = 2 * k + 1, 2 * (k - 1) + 1
            f = (n_half - k + 1) / (k * (2 * k - 1))
            jj, ii = np.meshgrid(np.arange(-k + 1, k),
                                 np.arange(-k + 1, k), indexing='ij')
            r = (o0 + (jj + k) * d0 + (ii + k)).ravel()
            c = (o1 + (jj + k - 1) * d1 + (ii + k - 1)).ravel()
            v = (f * _Dz(k - 1, -ii) * _Dz(k - 1, -jj)).ravel()
            blk.append((r, c, v))
        return _coo_from_blocks(blk, dim)

    m2 = build_m2()
    m3 = build_m3()
    m4 = build_m4()
    m2p = build_m2p()
    m3p = build_m3p()
    m4p = build_m4p()
    m2deph = build_m2deph()
    m3deph = build_m3deph()

    L_decay = (m1 + m2 + m3 + m4) / 2.0
    L_pump = (m1p + m2p + m3p + m4p) / 2.0
    L_deph = m1deph + m2deph + m3deph
    # Free temporaries as soon as possible
    for m in (m2, m3, m4, m2p, m3p, m4p, m2deph, m3deph):
        del m
    gc.collect()

    # ----- 4. block-diagonal Hamiltonian, global dissipation, Sy vectors -----
    blocks_hamx, blocks_hamz, blocks_hamint, blocks_globdis = [], [], [], []
    sy_vec = np.zeros(dim, dtype=complex)
    sy2_vec = np.zeros(dim, dtype=complex)

    for k in range(n_half + 1):
        dk = 2 * k + 1
        off = block_offsets[k]
        if k == 0:
            Z = csc_matrix((1, 1), dtype=complex)
            blocks_hamx.append(Z)
            blocks_hamz.append(Z.copy())
            blocks_hamint.append(Z.copy())
            blocks_globdis.append(Z.copy())
            continue

        i_arr = np.arange(1, 2 * k + 1, dtype=float)
        tab_k = np.sqrt(2 * i_arr * (k + 1) - i_arr * (i_arr + 1))

        sx_k = diags([tab_k, tab_k], [1, -1], shape=(dk, dk), format='csc') / 2.0
        sy_k = diags([-1j * tab_k, 1j * tab_k],
                     [1, -1], shape=(dk, dk), format='csc') / 2.0
        sy2_k = sy_k.dot(sy_k)
        sp_k = sx_k + 1j * sy_k
        sm_k = sx_k - 1j * sy_k
        sz_k = diags(-np.arange(-k, k + 1, dtype=float), 0,
                     shape=(dk, dk), format='csc')

        sy_vec[off: off + dk * dk] = sy_k.T.toarray().ravel()
        sy2_vec[off: off + dk * dk] = sy2_k.T.toarray().ravel()

        Id_k = identity(dk, format='csc')

        lx_k = -1j * (kron(Id_k, sx_k, format='csc')
                      - kron(sx_k.T, Id_k, format='csc'))
        lz_k = -1j * (kron(Id_k, sz_k, format='csc')
                      - kron(sz_k.T, Id_k, format='csc'))

        sp_sm_k = sp_k.dot(sm_k) / float(n)
        lint_k = -1j * (kron(Id_k, sp_sm_k, format='csc')
                        - kron(sp_sm_k.T, Id_k, format='csc'))

        sm_conj = sm_k.conjugate()
        sm_dag_sm = sm_k.conjugate().T.dot(sm_k)
        ldis_k = (2 * kron(sm_conj, sm_k, format='csc')
                  - kron(Id_k, sm_dag_sm, format='csc')
                  - kron(sm_dag_sm.T, Id_k, format='csc')) / (2.0 * n)

        blocks_hamx.append(lx_k)
        blocks_hamz.append(lz_k)
        blocks_hamint.append(lint_k)
        blocks_globdis.append(ldis_k)

    L_hamx = block_diag(blocks_hamx, format='csc')
    L_hamz = block_diag(blocks_hamz, format='csc')
    L_hamint = block_diag(blocks_hamint, format='csc')
    L_globdis = block_diag(blocks_globdis, format='csc')
    del blocks_hamx, blocks_hamz, blocks_hamint, blocks_globdis
    gc.collect()

    return {
        'L_hamx': L_hamx, 'L_hamz': L_hamz,
        'L_hamint': L_hamint, 'L_globdis': L_globdis,
        'L_decay': L_decay, 'L_pump': L_pump, 'L_deph': L_deph,
        'diag_indices': diag_indices,
        'sz_diag_vals': sz_diag_vals,
        'sy_vec': sy_vec, 'sy2_vec': sy2_vec,
        'Id': identity(dim, format='csc'),
        'dim': dim,
    }


# =====================================================================
# Diagnostics
# =====================================================================

def diagnose_sparsity(L, label='L'):
    """Report nnz row-wise and column-wise.  Cheap even at dim=1e6."""
    L = L.tocsr() if not L.format == 'csr' else L
    row_nnz = np.diff(L.indptr)
    L_csc = L.tocsc()
    col_nnz = np.diff(L_csc.indptr)
    print(f"[{label}] dim={L.shape[0]}, nnz={L.nnz}")
    print(f"[{label}] nnz/row: min={row_nnz.min()}, "
          f"mean={row_nnz.mean():.2f}, max={row_nnz.max()}")
    print(f"[{label}] nnz/col: min={col_nnz.min()}, "
          f"mean={col_nnz.mean():.2f}, max={col_nnz.max()}")
    return dict(dim=L.shape[0], nnz=L.nnz,
                row_nnz=row_nnz, col_nnz=col_nnz)


def build_petsc_aij(L_csr):
    """Wrap a scipy CSR matrix as a PETSc AIJ matrix, one copy."""
    M = PETSc.Mat().createAIJ(
        size=L_csr.shape,
        csr=(L_csr.indptr.astype(PETSc.IntType),
             L_csr.indices.astype(PETSc.IntType),
             L_csr.data),
        comm=PETSc.COMM_WORLD,
    )
    M.assemble()
    return M


# =====================================================================
# Grid-point worker
# =====================================================================

def process_grid_point(omega, Gamma, matrices, solver_config):
    """
    Evaluates a specific (omega, Gamma) parameter point.

    solver_config keys (beyond n, k, sigma, hz, chi, gamma_p, kappa, J):
      method            : 'mumps' (default) | 'iterative' | 'no-shift'
      ksp_rtol, ksp_max_it, ksp_type
      pc_type           : 'ilu' (default) | 'jacobi' | 'none'
      ilu_fill          : optional explicit fill level (overrides auto)
      mem_per_worker_gb : RAM budget per worker; enables auto fill selection
      report_sparsity, report_memory
    """
    diag_indices = matrices['diag_indices']
    sz_diag_vals = matrices['sz_diag_vals']
    sy_vec       = matrices['sy_vec']
    sy2_vec      = matrices['sy2_vec']

    n     = solver_config['n']
    k     = solver_config['k']
    sigma = solver_config['sigma']
    method = solver_config.get('method', 'mumps')

    hz      = solver_config.get('hz', 0.0)
    chi     = solver_config.get('chi', 1.0)
    gamma_p = solver_config.get('gamma_p', 0.0)
    kappa   = solver_config.get('kappa', 0.01)
    J       = solver_config.get('J', 0.0)

    mem_per_worker_gb = solver_config.get('mem_per_worker_gb', None)

    # ---- assemble Liouvillian ----
    L = omega * matrices['L_hamx']
    if hz != 0.0:      L = L + hz      * matrices['L_hamz']
    if chi != 0.0:     L = L + chi     * matrices['L_globdis']
    if Gamma != 0.0:   L = L + Gamma   * matrices['L_decay']
    if gamma_p != 0.0: L = L + gamma_p * matrices['L_pump']
    if kappa != 0.0:   L = L + kappa   * matrices['L_deph']
    if J != 0.0:       L = L + J       * matrices['L_hamint']
    L.sum_duplicates()
    L.eliminate_zeros()

    if solver_config.get('report_sparsity', False):
        diagnose_sparsity(L, label=f"L(omega={omega:.3g},Gamma={Gamma:.3g})")

    nnz_L = int(L.nnz)
    dim   = int(L.shape[0])

    if solver_config.get('report_memory', False) and mem_per_worker_gb is not None:
        base_gb = _estimate_worker_gb(nnz_L, dim, k, nnz_ilu=0)
        print(f"[mem] dim={dim}, nnz={nnz_L}, "
              f"base estimate ≈ {base_gb:.2f} GB/worker "
              f"(budget {mem_per_worker_gb:.2f} GB)")

    L_csr = L.tocsr()
    del L
    gc.collect()

    L_petsc = build_petsc_aij(L_csr)
    del L_csr
    gc.collect()

    # ---- SLEPc EPS ----
    E = SLEPc.EPS().create(comm=PETSc.COMM_WORLD)
    E.setOperators(L_petsc)
    E.setProblemType(SLEPc.EPS.ProblemType.NHEP)
    E.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    E.setDimensions(k, PETSc.DECIDE)
    try:
        E.setTarget(sigma)
    except Exception:
        E.setTarget(np.real(sigma))

    st = E.getST()

    if method in ('iterative', 'mumps'):
        E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
        st.setType(SLEPc.ST.Type.SINVERT)
        try:
            st.setShift(sigma)
        except Exception:
            st.setShift(np.real(sigma))

        ksp = st.getKSP()
        pc  = ksp.getPC()

        if method == 'mumps':
            ksp.setType(PETSc.KSP.Type.PREONLY)
            pc.setType(PETSc.PC.Type.LU)
            pc.setFactorSolverType(PETSc.Mat.SolverType.MUMPS)

        else:
            ksp_type = solver_config.get('ksp_type', 'gmres')
            ksp.setType(PETSc.KSP.Type.GMRES if ksp_type == 'gmres' else PETSc.KSP.Type.BCGS)
            ksp.setTolerances(rtol=solver_config.get('ksp_rtol', 1e-8), max_it=solver_config.get('ksp_max_it', 2000))
            if ksp_type == 'gmres':
                ksp.setGMRESRestart(solver_config.get('ksp_gmres_restart', 30))

            pc_type = solver_config.get('pc_type', 'ilu')

            if pc_type == 'ilu':
                # --- Decide fill level ---
                if 'ilu_fill' in solver_config:
                    fill = solver_config['ilu_fill']
                    if mem_per_worker_gb is not None:
                        est_gb = _estimate_worker_gb(
                            nnz_L, dim, k,
                            nnz_ilu=_estimate_ilu_nnz(nnz_L, nnz_L / dim, fill),
                        )
                        if est_gb > mem_per_worker_gb:
                            warnings.warn(
                                f"[ILU] user-specified fill={fill} est. "
                                f"{est_gb:.2f} GB exceeds budget "
                                f"{mem_per_worker_gb:.2f} GB.",
                                RuntimeWarning,
                            )
                elif mem_per_worker_gb is not None:
                    fill, ilu_ok = choose_ilu_fill(nnz_L, dim, k,
                                                   mem_per_worker_gb)
                    if not ilu_ok:
                        # Fall back to Jacobi
                        pc_type = 'jacobi'
                else:
                    fill = 0  # no budget given: original behaviour

                if pc_type == 'ilu':
                    pc.setType(PETSc.PC.Type.ILU)
                    pc.setFactorLevels(fill)

            if pc_type == 'jacobi':
                pc.setType(PETSc.PC.Type.JACOBI)
            elif pc_type == 'none':
                pc.setType(PETSc.PC.Type.NONE)
            elif pc_type not in ('ilu', 'jacobi', 'none'):
                raise ValueError(f"Unknown pc_type {pc_type}")

    elif method == 'no-shift':
        E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
        st.setType(SLEPc.ST.Type.NONE)
    else:
        raise ValueError(f"Unknown method {method}")

    E.solve()

    nconv = E.getConverged()
    if nconv == 0:
        E.destroy(); L_petsc.destroy(); gc.collect()
        raise RuntimeError(
            f"SLEPc failed to converge at omega={omega}, Gamma={Gamma}")

    vals, vecs = [], []
    vr, vi = L_petsc.createVecs()
    for idx in range(min(nconv, k)):
        val = E.getEigenpair(idx, vr, vi)
        vals.append(val)
        vecs.append(vr.getArray() + 1j * vi.getArray())
    vals = np.asarray(vals)
    vecs = np.column_stack(vecs)

    # ---- steady state ----
    ss_idx = int(np.argmin(np.abs(vals)))
    rho_ss = vecs[:, ss_idx]
    rho_diag = rho_ss[diag_indices]
    trace = rho_diag.sum()
    rho_norm = rho_ss / trace
    rho_diag_norm = rho_diag / trace

    # ---- observables ----
    j_max = n / 2.0
    mz  = - (sz_diag_vals * rho_diag_norm).sum().real / j_max
    mz2 = ((sz_diag_vals ** 2) * rho_diag_norm).sum().real / (j_max ** 2)
    mz_fluct = mz2 - mz ** 2

    my  = - np.dot(sy_vec,  rho_norm).real / j_max
    my2 = np.dot(sy2_vec, rho_norm).real / (j_max ** 2)
    my_fluct = my2 - my ** 2

    # ---- gap ----
    decaying = np.delete(vals, ss_idx)
    gap = -np.max(decaying.real) if decaying.size else np.nan

    E.destroy()
    L_petsc.destroy()
    gc.collect()

    return omega, Gamma, gap, mz, mz_fluct, my, my_fluct


# ---- Custom Joblib Wrapper for Accurate Real-Time Completion Tracking -------
class ProgressParallel(Parallel):
    """Joblib Parallel override to ensure tqdm updates on job completion, not dispatch."""
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, iterable):
        with tqdm(disable=not self._use_tqdm, total=self._total, desc="Scanning Grid") as self._pbar:
            return super().__call__(iterable)

    def print_progress(self):
        if self._use_tqdm:
            self._pbar.n = self.n_completed_tasks
            self._pbar.refresh()


if __name__ == '__main__':
    # Configuration
    STEP_SIZE_O = 0.025
    STEP_SIZE_G_1 = 0.005
    STEP_SIZE_G_2 = 0.01
    MAX_MEMORY_GB = 50.0
    RECOMPUTE = False
    solver_k = 3

    n_list = [20,]
    kappas_list = np.round([0.0,], 10)

    available_cores = multiprocessing.cpu_count() // 2

    # Grid parameters setup
    hx_flat = np.round(np.concat([[STEP_SIZE_O/10], np.arange(STEP_SIZE_O, 0.52 + STEP_SIZE_O / 2, STEP_SIZE_O)]), 10)
    gamma_flat = np.round(np.concat([
        [STEP_SIZE_G_1/10],
        np.arange(STEP_SIZE_G_1, 0.05 - STEP_SIZE_G_1 / 2, STEP_SIZE_G_1),
        np.arange(0.05, 0.15 + STEP_SIZE_G_2 / 2, STEP_SIZE_G_2),
    ]), 10)

    for n in n_list:
        # Precompute shared base matrices
        shared_matrices = build_crf_bases(n)
        dim = shared_matrices['dim']
        
        # Resource allocation estimation
        mem_per_worker = 0.26336355 + 1.1770208 * (dim/1e5)**1.41401865  # for MUMPS + LU
        max_safe_jobs = int(MAX_MEMORY_GB // mem_per_worker)
        if max_safe_jobs < 1:
            break
        n_jobs = max(1, min(available_cores, max_safe_jobs))
        
        print(f"CRF System Size N = {n} (Liouvillian dimension: {dim})")
        print(f"Allocating {n_jobs} parallel workers (Estimated peak memory: {n_jobs * mem_per_worker:.2f} GB)\n")

        for kappa in kappas_list:
            print(f"Will compute {len(gamma_flat)} gamma by {len(hx_flat)} omega values")
            tasks = [(o, G) for o in hx_flat for G in gamma_flat]
            print(f"Total grid points: {len(tasks)}")
            
            if len(gamma_flat) == 1:
                signature = f"g={gamma_flat[0]:n}_hx={hx_flat[0]:n}-{hx_flat[-1]:n}_line"
            elif len(hx_flat) == 1:
                signature = f"g={gamma_flat[0]:n}-{gamma_flat[-1]:n}_hx={hx_flat[0]:n}_line"
            else:
                signature = f"g={gamma_flat[0]:n}-{gamma_flat[-1]:n}_hx={hx_flat[0]:n}-{hx_flat[-1]:n}_grid"
            
            solver_config = dict(
                n = n,
                k = solver_k,
                sigma = 1e-4 + 0j,
                hz = 0.0,
                chi = 1.0,
                gamma_p = 0.0,
                kappa = kappa,
                J = 0.0,
                mem_per_worker_gb = MAX_MEMORY_GB / n_jobs / 4,
                # report_sparsity = True,
                # report_memory = True,
                ksp_gmres_restart = 500,
            )
            
            output_path = Path.home() / f"results/kappa={kappa:n}_N={n:n}_{signature}_points={len(tasks):n}.csv"
            if output_path.exists() and not RECOMPUTE:
                print(f"Already computed: {output_path.stem}")
                continue
            
            # Parallel Execution
            raw_results = ProgressParallel(backend='loky', n_jobs=n_jobs, total=len(tasks))(
                delayed(process_grid_point)(o, G, shared_matrices, solver_config) for o, G in tasks
            )
            
            df_crf = pd.DataFrame(np.array(raw_results), columns=("hx", "gamma", "gap", "mz", "mzf", "my", "myf"))
            df_crf['action'] = -np.log(np.abs(df_crf.gap)) / n
            df_crf['kappa'] = kappa
            df_crf['N'] = n
            try:
                df_old
            except NameError:
                df_old = None
            df_joined = pd.concat([df_old, df_crf])
            df_joined.sort_values(by=["N", "kappa", "hx", "gamma"]).to_csv(output_path, index=False, float_format='%.10g')
            print(f"Results successfully written to {output_path.stem}\n")
            gc.collect()