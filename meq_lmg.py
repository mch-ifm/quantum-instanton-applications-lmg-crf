import multiprocessing
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator
from scipy.sparse import diags, kron, identity
from scipy.sparse.linalg import LinearOperator, eigs as sparse_eigs  # mind that meq_crf uses a much faster PETSc implementation
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from scikits.umfpack import splu as umfpack_splu
from pathlib import Path

APS_RCPARAMS_OUTERTICKS = {
    # LaTeX
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "text.latex.preamble": r"\usepackage{amsmath}",

    # Figure
    "figure.figsize": (4.5, 3.4),
    "figure.dpi": 150,
    "savefig.dpi": 300,

    # Text
    "font.size": 10,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,

    # Frame
    "axes.linewidth": 0.7,

    # Ticks
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,

    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,

    "xtick.minor.visible": True,
    "ytick.minor.visible": True,

    # Curves
    "lines.linewidth": 1.2,
}

APS_RCPARAMS = APS_RCPARAMS_OUTERTICKS | {
    # Ticks: inward, mirrored
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
}

mpl.rcParams.update(APS_RCPARAMS_OUTERTICKS)


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


# ---- Pure Structural Component Breakdown ------------------------------------
def build_liouvillian_bases(j, G=1.0):
    """Precomputes fully decoupled base operators in CSC format."""
    n = 2*j
    dim = int(n + 1)
    
    i_arr = np.arange(1, n+1, dtype=float)
    tab = np.sqrt(2*i_arr*(n/2 + 1) - i_arr*(i_arr + 1))
    
    sx = diags([tab, tab], [1, -1], shape=(dim, dim), format='csc')/2
    sy = diags([-1j*tab, 1j*tab], [1, -1], shape=(dim, dim), format='csc')/2
    sz = diags(-np.arange(-n/2, n/2+1, dtype=float), 0, shape=(dim, dim), format='csc')
    
    opm = (sx - 1j*sy).real
    Id = identity(dim, format='csc')
    
    # Decoupled parameter slices
    L_omega = -1j*(kron(Id, sz, format='csc') - kron(sz.T, Id, format='csc'))
    
    sx2 = sx.dot(sx)
    L_G = 1j*(kron(Id, sx2, format='csc') - kron(sx2.T, Id, format='csc'))/j * G
    
    octdo = opm.conjugate().T.dot(opm)
    L_gamma = (2*kron(opm.conjugate(), opm, format='csc')
               - kron(Id, octdo, format='csc')
               - kron(octdo.T, Id, format='csc'))/(2*j)
               
    return L_gamma, L_omega, L_G, sz, dim


# ---- Parallel Multi-Parameter Worker -----------------------------------------
def process_grid_point(gamma, Omega, matrices, solver_config):
    """Evaluates a specific (gamma, Omega) configuration."""
    L_gamma_even, L_omega_even, L_G_even, L_gamma_odd, L_omega_odd, L_G_odd = matrices['base_ops']
    Id_even, Id_odd, even_diag_idx = matrices['helpers']
    sz_diag = matrices['sz_diag']
    
    j = solver_config['j']
    k = solver_config['k']
    sigma = solver_config['sigma']
    include_odd = solver_config['include_odd']
    
    # 1. Even Subspace Processing
    L_even = (gamma * L_gamma_even) + (Omega * L_omega_even) + L_G_even
    A_even = L_even - (sigma * Id_even)
    A_even.sort_indices()
    
    lu_even = umfpack_splu(A_even)
    OP_even = LinearOperator(A_even.shape, matvec=lu_even.solve, dtype=np.complex128)
    
    vals_inv_even, vecs_even = sparse_eigs(OP_even, k=k, which='LM', return_eigenvectors=True)
    vals_even = 1.0 / vals_inv_even + sigma
    
    # Steady State Extraction
    ss_idx = np.argmin(np.abs(vals_even))
    rho_ss_unnorm = vecs_even[:, ss_idx]
    rho_diag = rho_ss_unnorm[even_diag_idx]
    trace = rho_diag.sum()
    rho_diag_norm = rho_diag / trace
    
    # Observables Calculation
    mz = (sz_diag * rho_diag_norm).sum().real / j
    mz2 = ((sz_diag**2) * rho_diag_norm).sum().real / (j**2)
    mz_fluct = mz2 - (mz**2)
    
    # Extract lowest even decaying eigenvalue
    vals_even_decaying = np.delete(vals_even, ss_idx)
    max_real_even = np.max(vals_even_decaying.real)
    
    # 2. Optional Odd Subspace Processing
    if include_odd:
        L_odd = (gamma * L_gamma_odd) + (Omega * L_omega_odd) + L_G_odd
        A_odd = L_odd - (sigma * Id_odd)
        A_odd.sort_indices()
        
        lu_odd = umfpack_splu(A_odd)
        OP_odd = LinearOperator(A_odd.shape, matvec=lu_odd.solve, dtype=np.complex128)
        
        vals_inv_odd = sparse_eigs(OP_odd, k=k, which='LM', return_eigenvectors=False)
        vals_odd = 1.0 / vals_inv_odd + sigma
        max_real_odd = np.max(vals_odd.real)
        
        gap = -max(max_real_even, max_real_odd)
    else:
        gap = -max_real_even
        max_real_odd = np.nan
        
    return gamma, Omega, gap, mz, mz_fluct, max_real_even, max_real_odd


if __name__ == '__main__':
    df_meq = pd.DataFrame([], columns=("gamma", "Omega", "gap", "mz", "mzf", "mre", "mro"))
    # Configuration
    G = 1.0
    STEP_SIZE = 0.05
    j_list = (10,)
    
    MAX_MEMORY_GB = 50.0
    include_odd = False
    WHOLE_DOMAIN = True

    output_folder = Path.home() / "results"
    
    # 1. Build Coordinate Tasks Grid with Physical Boundary Constraint
    gammas_flat = np.concat([[STEP_SIZE/10], np.arange(STEP_SIZE, 1.1 + STEP_SIZE/2, STEP_SIZE)])
    omegas_flat = np.concat([
        np.arange(0.00, 0.05 - STEP_SIZE/16, STEP_SIZE/8),
        np.arange(0.05, 0.10 - STEP_SIZE/ 8, STEP_SIZE/4),
        np.arange(0.10, 0.20 - STEP_SIZE/ 4, STEP_SIZE/2),
        np.arange(0.20, 2.10 + STEP_SIZE/ 2, STEP_SIZE),
    ])
    
    tasks = []
    for g in gammas_flat:
        for o in omegas_flat:
            if WHOLE_DOMAIN or (g**2 + (1.0 - o)**2) >= 1.0:
                tasks.append((g, o))
    
    for j in j_list:
        # 2. Resource Allocation Calculations
        mem_per_worker = 0.1 + (6.4e-8 * (j**3)) / 2
        available_cores = multiprocessing.cpu_count() // 2
        max_safe_jobs = int(MAX_MEMORY_GB // mem_per_worker)
        n_jobs = max(1, min(available_cores, max_safe_jobs))
        # n_jobs = 1
        
        print(f"Total points matching boundary constraints: {len(tasks)}")
        print(f"Allocating {n_jobs} parallel workers (Estimated peak memory: {n_jobs * mem_per_worker:.2f} GB)\n")
        
        # 3. Base Operator Precomputations
        L_gamma, L_omega, L_G, sz_op, dim = build_liouvillian_bases(j, G=G)
        
        # Parity Sector Slicing Matrix Indices
        all_idx = np.arange(dim**2)
        m = all_idx // dim
        m_prime = all_idx % dim
        even_mask = (m % 2) == (m_prime % 2)
        idx_even = all_idx[even_mask]
        idx_odd = all_idx[~even_mask]
        
        # Slice the static components down to their decoupled blocks once
        L_gamma_even = L_gamma[idx_even, :][:, idx_even]
        L_omega_even = L_omega[idx_even, :][:, idx_even]
        L_G_even = L_G[idx_even, :][:, idx_even]
        
        L_gamma_odd = L_gamma[idx_odd, :][:, idx_odd] if include_odd else None
        L_omega_odd = L_omega[idx_odd, :][:, idx_odd] if include_odd else None
        L_G_odd = L_G[idx_odd, :][:, idx_odd] if include_odd else None
        
        full_diag_idx = np.arange(dim) * (dim + 1)
        even_diag_idx = np.searchsorted(idx_even, full_diag_idx)
        
        shared_matrices = {
            'base_ops': (L_gamma_even, L_omega_even, L_G_even, L_gamma_odd, L_omega_odd, L_G_odd),
            'helpers': (identity(len(idx_even), format='csc'), 
                        identity(len(idx_odd),  format='csc') if include_odd else None, 
                        even_diag_idx),
            'sz_diag': sz_op.diagonal()
        }
        
        solver_config = {'j': j, 'k': 5, 'sigma': 1e-5 + 0j, 'include_odd': include_odd}
        
        # 4. Process Grid via ProgressParallel
        raw_results = ProgressParallel(backend='loky', n_jobs=n_jobs, total=len(tasks))(
            delayed(process_grid_point)(g, o, shared_matrices, solver_config) for g, o in tasks
        )
        
        df_new = pd.DataFrame(np.array(raw_results), columns=("gamma", "Omega", "gap", "mz", "mzf", "mre", "mro"))
        df_meq = pd.concat([df_meq, df_new])

        if len(omegas_flat) == 1 and omegas_flat[0] == 0.2:
            df_meq.sort_values(by="gamma").to_csv(
                output_folder / rf"ME Omega=0.2 j={j} step={STEP_SIZE:n} points={len(df_meq):n}.csv",
                columns=("gamma", "gap", "mz", "mzf", "mre", "mro"), index=False,
            )
        else:
            df_meq.sort_values(by=["Omega","gamma"]).to_csv(
                output_folder / (
                    rf"ME j={j:n} step={STEP_SIZE:n} points={len(df_meq):n} g={gammas_flat[0]:n}..{gammas_flat[-1]:n} "
                    rf"o={omegas_flat[0]:n}..{omegas_flat[-1]:n}.csv"
                ),
                columns=("Omega", "gamma", "gap", "mz", "mzf", "mre", "mro"), index=False,
            )

    # Plotting

    fig, axes = plt.subplots(2, 1, figsize=(4.5, 7), dpi=300)
    tt = np.linspace(0, np.pi, 361)

    for ax,x,y,z in (
        (axes[0], df_meq.gamma, df_meq.Omega, df_meq.mz),
        (axes[1], df_meq.Omega, df_meq.gamma, df_meq.mz),
    ):
        contour = ax.tricontourf(x, y, z, 256, cmap='viridis', rasterized=True, vmin=-1, vmax=0)
        fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04, ticks=MultipleLocator(1))
        
        ax.set_title(f"$m_z$, $j = {j}$", fontsize=12, pad=10)
        ax.plot(1, 1, 'w.', zorder=2)

    pp = np.linspace(0, np.pi/2, 181)
    ss = np.sin(2*pp)
    popt = [0.08215112166740672, 0.040058265944271296]  # fitted from instanton data, Husimi distribution
    rr = 1 + popt[0]*ss**2 + popt[1]*ss**4  # the fit function
    gg = rr * np.sin(pp)
    oo = 1 - rr * np.cos(pp)
    axes[0].plot(gg, oo, 'w--', lw=1)

    axes[0].xaxis.set_major_locator(MultipleLocator(0.5))
    axes[0].yaxis.set_major_locator(MultipleLocator(1))
    axes[0].xaxis.set_minor_locator(MultipleLocator(0.1))
    axes[0].yaxis.set_minor_locator(MultipleLocator(0.2))

    axes[1].yaxis.set_major_locator(MultipleLocator(0.5))
    axes[1].xaxis.set_major_locator(MultipleLocator(1))
    axes[1].yaxis.set_minor_locator(MultipleLocator(0.1))
    axes[1].xaxis.set_minor_locator(MultipleLocator(0.2))

    axes[0].set_xlim(0, 1.1)
    axes[1].set_xlim(0, 2.1)
    axes[0].set_ylim(0, 2.1)
    axes[1].set_ylim(0, 1.1)
    axes[0].plot([1,1], [0,1], 'w-', lw=1, zorder=2)
    axes[1].plot([0,1], [1,1], 'w-', lw=1, zorder=2)
    axes[0].plot(np.sin(tt), 1-np.cos(tt), 'w-', lw=1, zorder=2)
    axes[1].plot(1-np.cos(tt), np.sin(tt), 'w-', lw=1, zorder=2)

    axes[0].set_ylabel(r"$\omega/\chi$", fontsize=14)
    axes[1].set_xlabel(r"$\omega/\chi$", fontsize=14)
    axes[0].set_xlabel(r"$\Gamma/\chi$", fontsize=14)
    axes[1].set_ylabel(r"$\Gamma/\chi$", fontsize=14)

    axes[0].axhline(0.2, ls=':', color='w', lw=1)
    axes[1].axvline(0.2, ls=':', color='w', lw=1)

    fig.subplots_adjust(hspace=0.5)