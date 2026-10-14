import argparse
import json
import numpy as np
import errno
import warnings
import torch
from numba import njit, prange
from scipy.integrate import solve_bvp, trapezoid
from scipy.linalg import eig
from scipy.special import expit
from pathlib import Path
from filelock import FileLock

torch.set_default_device(torch.device('cpu'))
torch.set_default_dtype(torch.float64)


# ==========================================
# 1. PURE NUMPY MATH & CRITICAL POINTS
# ==========================================


def crit_point_eqs_num(gamma, Omega):
    """Pure numpy implementation of Sympy critical points."""
    if gamma**2 > 1.0:
        return []
    s = np.sqrt(1 - gamma**2)
    D = Omega**2 + 2*Omega + gamma**2
    E = Omega**2 + 2*Omega - gamma**2
    A = -Omega**2 + 2*Omega*gamma**2 - 2*Omega + gamma**2

    def R(sign):
        return (A + sign*s*E) / D

    def Nv(sign):
        return (Omega**2 * gamma**2 / 2 - Omega**2 - 2*Omega - gamma**4 / 2 -
                gamma**2 + sign*s*(Omega**2 + Omega*gamma**2 + 2*Omega + gamma**2))

    def Dv(sign):
        return gamma * (Omega**2 + 2*Omega*gamma**2 + 2*Omega + 3*gamma**2 - sign*s*E)

    pts = [(0.0, 0.0)]

    for sign_R, sign_N, sign_D in [(-1, 1, 1), (1, -1, -1)]:
        R_val = R(sign_R)
        if R_val >= 0:
            p_v = np.sqrt(2) * np.sqrt(R_val) * Nv(sign_N) / Dv(sign_D)
            p_w = (np.sqrt(2)/2) * np.sqrt(R_val)
            pts.extend([(-p_v, -p_w), (p_v, p_w)])
        else:
            pts.extend([(-np.nan, -np.nan), (np.nan, np.nan)])

    return pts


def jac2d_eigval_eqs_num(v, w, gamma, Omega):
    v = complex(v)
    w = complex(w)
    p1 = 1 + v**2 + w**2
    w2p = 1 + w**2
    r = np.sqrt(-(
        p1 * (Omega * w2p + v**2 * (Omega + 4)) *
        (w2p**2 * (Omega - 2) + v**4 * (Omega + 2) + 2 * v**2 * (Omega * w2p + 4))
    ))
    return [
        complex(-gamma - (4 * v * w * p1 + r) / p1**2),
        complex(-gamma - (4 * v * w * p1 - r) / p1**2),
    ]


def classify_pt(eigvals):
    ev = np.array(eigvals)
    e = np.sort(ev.real)
    if np.allclose(ev.imag, 0):
        if e[0] <= e[1] < 0:
            return 'na'
        elif e[0] < 0 < e[1]:
            return 's'
        elif 0 < e[0] <= e[1]:
            return 'nr'
        else:
            return 'd'
    else:
        if e[0] < 0:
            return 'sa'
        elif 0 < e[0]:
            return 'sr'
        else:
            return 'd'


def find_critical_points(gamma, Omega, four_d=False):
    """Identifies and sorts unique real critical points."""
    pts = np.array(crit_point_eqs_num(gamma, Omega))[[0,4,2]]
    assert np.allclose(pts.imag, 0)
    pts = pts.real
    assert classify_pt(jac2d_eigval_eqs_num(*pts[0], gamma, Omega)) == 'na'
    assert classify_pt(jac2d_eigval_eqs_num(*pts[1], gamma, Omega)) in {'sa','na'}
    assert classify_pt(jac2d_eigval_eqs_num(*pts[2], gamma, Omega)) == 's'
    if four_d:
        return np.pad(pts, ((0,0),(0,2)))
    return pts


@njit
def drift_4d_numba(v, w, pv, pw, gamma, Omega, backwards, distribution):
    v2 = v**2
    w2 = w**2
    vw = v*w
    denom = 1 + v2 + w2
    inv_denom = 1 / denom
    inv_denom2 = inv_denom**2

    c1 = 1 + v2 - w2
    c2 = 1 - v2 + w2
    w2_plus1 = 1 + w2

    if distribution == "Husimi" or distribution == "P":
        poly1 = v2**2 + w2_plus1**2 - 2*v2*(1 + 3*w2)
        poly_v = 1 - v2 + 3*w2
        poly_w = 1 + w2 - 3*v2

        p_diff = pw**2 - pv**2
        p_prod = pv*pw

        G_poly1 = 0.25*poly1
        G_vw_c2 = vw*c2
        G_half = 0.5

        G_vw_v_inv_4 = 4*vw*v*inv_denom
        G_v_c2_inv_2 = 2*v*c2*inv_denom

        # Cubic and fractional momentum distributions
        E3_cubic = p_diff*w*poly_w + 2*p_prod*v*poly_v
        E3_frac = 4*v*(2*pv*w*w2_plus1 - pw*v*(3*w2_plus1 + v2))*inv_denom2
        
        E4_cubic = poly_v*p_diff*v - 2*poly_w*p_prod*w
        E4_frac = 8*v2*(2*pw*vw + pv*c1)*inv_denom2

        if distribution == "Husimi":
            sgn = 1
            gc1 = c1
            gc2 = c2
            gamma_mod_pv = 0*v
            gamma_mod_pw = 0*w
        else: # P
            sgn = -1
            R2_sq = (v2 + w2)**2
            gc1 = v2 - w2 + R2_sq
            gc2 = -v2 + w2 + R2_sq
            
            # P-distribution subtracts a radial momentum correction in the dissipation
            gamma_mod_common = -gamma*(v2 + w2)*(pv**2 + pw**2)
            gamma_mod_pv = gamma_mod_common*v
            gamma_mod_pw = gamma_mod_common*w

        dv = sgn*G_poly1*pw + sgn*G_vw_c2*pv + (-v + vw*pw + 0.5*pv*gc1)*gamma - w*Omega - G_vw_v_inv_4
        dw = sgn*G_poly1*pv - sgn*G_vw_c2*pw + (-w + vw*pv + 0.5*pw*gc2)*gamma + v*Omega - G_v_c2_inv_2
        
        dpv = sgn*G_half*E3_cubic + E3_frac + (0.5*p_diff*v + pv - p_prod*w)*gamma - pw*Omega + 2*pw + gamma_mod_pv
        dpw = sgn*G_half*E4_cubic + G_half*E4_frac + (-0.5*p_diff*w + pw - p_prod*v)*gamma + pv*Omega + gamma_mod_pw

    elif distribution == "Wigner":
        v3 = v2*v
        v2_w2_m1 = v2 + w2 - 1
        
        dv = -4*v2*w*inv_denom - v*gamma + pw*vw*gamma + 0.25*pv*(denom**2 - 4*w2)*gamma - w*Omega
        dw = -2*v*c2*inv_denom - w*gamma + pv*vw*gamma + 0.25*pw*(denom**2 - 4*v2)*gamma + v*Omega
        
        dpv = -4*pw*v2*(3*w2_plus1 + v2)*inv_denom2 \
              - 0.5*pw**2*v*v2_w2_m1*gamma \
              - 0.5*pv**2*v*denom*gamma \
              + pv*(8*vw*w2_plus1*inv_denom2 + gamma) \
              - pw*(-2 + pv*w*gamma + Omega)
              
        dpw = 0.5*(-pw**2*w*denom*gamma \
                     + 2*pw*(8*v3*w*inv_denom2 + gamma - pv*v*gamma) \
                     + pv*(8*v2*c1*inv_denom2 - pv*w*v2_w2_m1*gamma + 2*Omega))

    else:
        raise ValueError("distribution must be 'Husimi', 'P', or 'Wigner'")

    if backwards:
        return -dv, -dw, -dpv, -dpw
    return dv, dw, dpv, dpw


@njit
def hamiltonian_num(v, w, pv, pw, gamma, Omega, distribution):
    v2 = v**2
    w2 = w**2
    denom = 1 + v2 + w2
    
    if distribution == "Husimi" or distribution == "P":
        # Shared base components for Term 1 (Kinetic / Nonlinear coupling)
        t1_a = -2*pv*v*w + pw*(-1 + v2 - w2)
        t1_b = pv*(v**4 - (1 + w2)**2) + 2*v*(4 + pw*w*denom)
        
        # Shared base components for Term 2 (Dissipative fluctuation)
        term2_base = 4*pv*v*(pw*w - 1) - 4*pw*w
        
        # Shared Term 3 (Coherent driving)
        term3 = 4*(pw*v - pv*w)*Omega
        
        if distribution == "Husimi":
            term1 = (t1_a*t1_b) / denom
            term2 = (term2_base + pv**2*(1 + v2 - w2) + pw**2*(1 - v2 + w2))*gamma
            
        else: # P
            # The P-distribution flips the sign of t1_a and shifts t1_b by 16v
            t1_b_P = t1_b - 16*v
            term1 = (-t1_a*t1_b_P) / denom
            
            # The constant 1 in the quadratic momentum terms becomes (v^2 + w^2)^2
            R2_sq = (v2 + w2)**2
            term2 = (term2_base + pv**2*(R2_sq + v2 - w2) + pw**2*(R2_sq - v2 + w2))*gamma
            
    elif distribution == "Wigner":
        c2 = 1 - v2 + w2
        term1 = 4*pv*pw*v*w*gamma + 0.5*pw**2*(denom**2 - 4*v2)*gamma + 0.5*pv**2*(denom**2 - 4*w2)*gamma
        term2 = 4*pw*(-2*v*c2 / denom - w*gamma + v*Omega) + 4*pv*(-4*v2*w / denom - v*gamma - w*Omega)
        term3 = 0.0 * v
        
    else:
        raise ValueError("distribution must be 'Husimi', 'P', or 'Wigner'")
        
    return 0.25*(term1 + term2 + term3)


@njit
def jacobian_4d_numba(v, w, pv, pw, gamma, Omega, distribution):
    v2, w2 = v**2, w**2
    v3, w3 = v2*v, w2*w
    v4, w4 = v2**2, w2**2
    v5, v6, v7 = v4*v, v4*v2, v4*v3
    
    vw = v*w
    v2w, vw2 = v2*w, v*w2
    pv2, pw2, pvpw = pv**2, pw**2, pv*pw
    
    denom = 1 + v2 + w2
    denom2, denom3 = denom**2, denom**3
    inv_denom2, inv_denom3 = 1 / denom2, 1 / denom3
    
    w2_plus1 = 1 + w2
    c1 = 1 + v2 - w2
    c2 = 1 - v2 + w2
    w_w3 = w + w3
    
    if distribution == "Husimi":
        J00 = -8*vw*w2_plus1*inv_denom2 - gamma + pv*(w - 3*v2w + w3 + v*gamma) + pw*(v3 - v*(1 + 3*w2) + w*gamma)
        
        t01_1 = pw*(-3*v6*w + v5*gamma + 2*v3*w2_plus1*gamma + v*w2_plus1**2*gamma)
        t01_2 = -pv*denom2*(v3 - v*(1 + 3*w2) + w*gamma)
        t01_3 = w2_plus1**2*(pw*w_w3 - Omega)
        t01_4 = -v4*(4 + 5*pw*w_w3 + Omega)
        t01_5 = -v2*(pw*w*w2_plus1**2 + 2*(2 + w2*(-2 + Omega) + Omega))
        J01 = (t01_1 + t01_2 + t01_3 + t01_4 + t01_5)*inv_denom2
        
        J02 = 0.5*(-2*v3*w + 2*v*w_w3 + gamma + v2*gamma - w2*gamma)
        J03 = 0.25*(v4 + w2_plus1**2 - 2*v2*(1 + 3*w2) + 4*vw*gamma)
        
        t10_1 = pw*(3*v6*w - v5*gamma - 2*v3*w2_plus1*gamma - v*w2_plus1**2*gamma)
        t10_2 = pv*denom2*(v3 - v*(1 + 3*w2) + w*gamma)
        t10_3 = -w2_plus1**2*(2 + pw*w_w3 - Omega)
        t10_4 = v4*(2 + 5*pw*w_w3 + Omega)
        t10_5 = v2*w2_plus1*(pw*w_w3 + 2*(4 + Omega))
        J10 = (t10_1 + t10_2 + t10_3 + t10_4 + t10_5)*inv_denom2
        
        J11 = -8*v3*w*inv_denom2 - gamma + pv*(w - 3*v2w + w3 + v*gamma) + pw*(v3 - v*(1 + 3*w2) + w*gamma)
        J12 = J03
        J13 = 0.5*(-1 + v2 - w2)*(2*vw - gamma)
        
        t20_1 = 8*pv*w*w2_plus1*(1 - 3*v2 + w2)*inv_denom3
        t20_2 = pvpw*(1 - 3*v2 + 3*w2)
        t20_3 = pv2*(3*vw - 0.5*gamma)
        t20_4_1 = -6*pw*v7*w - 18*pw*v5*w*w2_plus1
        t20_4_2 = -6*v*w2_plus1**2*(8 + pw*w_w3)
        t20_4_3 = -2*v3*w2_plus1*(-8 + 9*pw*w_w3)
        t20_4 = pw*(t20_4_1 + t20_4_2 + t20_4_3 + pw*denom3*gamma)*0.5*inv_denom3
        J20 = t20_1 + t20_2 + t20_3 + t20_4
        
        t21_1 = pv2*(-1 + 3*v2 - 3*w2)
        t21_2 = pw*(pw*(1 - 3*v2 + 3*w2) - 16*v2*w*(v2 - 3*w2_plus1)*inv_denom3)
        t21_3 = 4*pv*v*(3*pw*w + 4*(1 - w4 + v2*(1 + 3*w2))*inv_denom3)
        t21_4 = -2*pvpw*gamma
        J21 = 0.5*(t21_1 + t21_2 + t21_3 + t21_4)
        
        J22 = 8*v*w_w3*inv_denom2 + gamma - pv*(w - 3*v2w + w3 + v*gamma) + pw*(v - v3 + 3*vw2 - w*gamma)
        
        t23_3 = w2_plus1**2*(2 + pw*w_w3 - Omega)
        t23_4 = -v4*(2 + 5*pw*w_w3 + Omega)
        t23_5 = -v2*w2_plus1*(pw*w_w3 + 2*(4 + Omega))
        J23 = (t01_1 + t01_2 + t23_3 + t23_4 + t23_5)*inv_denom2
        
        J30 = J21
        
        t31_1 = 2*pv*(8*v2*w*(-3 - 3*v2 + w2) + pw*(-1 + 3*v2 - 3*w2)*denom3)
        t31_2 = -pv2*denom3*(6*vw - gamma)
        t31_3_1 = 6*pw*v7*w + 6*pw*vw*w2_plus1**3
        t31_3_2 = 2*v3*(8 - 24*w2 + 9*pw*w*w2_plus1**2)
        t31_3_3 = 2*v5*(8 + 9*pw*w_w3)
        t31_3 = pw*(t31_3_1 + t31_3_2 + t31_3_3 - pw*denom3*gamma)
        J31 = 0.5*(t31_1 + t31_2 + t31_3)*inv_denom3
        
        t32_0 = pv*(v3 - v*(1 + 3*w2) + w*gamma)
        t32_1 = pw*(3*v6*w - v5*gamma - 2*v3*w2_plus1*gamma - v*w2_plus1**2*gamma)
        t32_3 = w2_plus1**2*(-pw*w_w3 + Omega)
        t32_4 = v4*(4 + 5*pw*w_w3 + Omega)
        t32_5 = v2*(pw*w*w2_plus1**2 + 2*(2 + w2*(-2 + Omega) + Omega))
        J32 = t32_0 + (t32_1 + t32_3 + t32_4 + t32_5)*inv_denom2
        
        J33 = 8*v3*w*inv_denom2 + gamma - pv*(w - 3*v2w + w3 + v*gamma) + pw*(v - v3 + 3*vw2 - w*gamma)

    elif distribution == "P":
        J00 = -pv*w*(1 - 3*v2 + w2) - 8*vw*w2_plus1*inv_denom2 - gamma + pv*v*(1 + 2*v2 + 2*w2)*gamma + pw*(v - v3 + 3*vw2 + w*gamma)
        J01 = pv*v*(-1 + v2 - 3*w2) - 4*v2*c1*inv_denom2 + pv*w*(-1 + 2*v2 + 2*w2)*gamma - pw*(w - 3*v2w + w3 - v*gamma) - Omega
        J02 = vw*(-c2) + 0.5*(v4 - w2 + w4 + v2*(1 + 2*w2))*gamma
        J03 = 0.25*(-v4 - w2_plus1**2 + v2*(2 + 6*w2) + 4*vw*gamma)
        
        t10_1 = pw*(-3*v6*w + 2*v7*gamma + 6*v3*w2*w2_plus1*gamma + 3*v5*(1 + 2*w2)*gamma + v*(-1 + 3*w4 + 2*w**6)*gamma)
        t10_2 = -pv*denom2*(v3 - v*(1 + 3*w2) - w*gamma)
        t10_3 = v4*(2 - 5*pw*w_w3 + Omega)
        t10_4 = w2_plus1**2*(-2 + pw*w_w3 + Omega)
        t10_5 = -v2*w2_plus1*(pw*w_w3 - 2*(4 + Omega))
        J10 = (t10_1 + t10_2 + t10_3 + t10_4 + t10_5)*inv_denom2
        
        J11 = -8*v3*w*inv_denom2 - gamma - pv*(w - 3*v2w + w3 - v*gamma) + pw*(v - v3 + 3*vw2 + w*gamma + 2*v2w*gamma + 2*w3*gamma)
        J12 = J03
        J13 = vw*c2 + 0.5*(v4 + w2 + w4 + v2*(-1 + 2*w2))*gamma
        
        t20_1 = pvpw*(-1 + 3*v2 - 3*w2)
        t20_2 = 8*pv*w*w2_plus1*(1 - 3*v2 + w2)*inv_denom3
        t20_3 = -0.5*pv2*(6*vw + gamma + 6*v2*gamma + 2*w2*gamma)
        t20_4_1 = 6*pw*v7*w + 18*pw*v5*w*w2_plus1
        t20_4_2 = 6*v*w2_plus1**2*(-8 + pw*w_w3)
        t20_4_3 = 2*v3*w2_plus1*(8 + 9*pw*w_w3)
        t20_4_4 = -pw*denom3*(-1 + 6*v2 + 2*w2)*gamma
        t20_4 = pw*(t20_4_1 + t20_4_2 + t20_4_3 + t20_4_4)*0.5*inv_denom3
        J20 = t20_1 + t20_2 + t20_3 + t20_4
        
        t21_1 = 4*pv*v*(-3*pw*w + 4*(1 - w4 + v2*(1 + 3*w2))*inv_denom3)
        t21_2 = -2*pvpw*gamma
        t21_3 = pv2*(1 - 3*v2 + 3*w2 - 4*vw*gamma)
        t21_4 = pw*(-16*v2*w*(v2 - 3*w2_plus1)*inv_denom3 + pw*(-1 + 3*v2 - 3*w2 - 4*vw*gamma))
        J21 = 0.5*(t21_1 + t21_2 + t21_3 + t21_4)
        
        J22 = pw*v*(-1 + v2 - 3*w2) + 8*v*w_w3*inv_denom2 + gamma - pw*w*gamma + pv*(w - 3*v2w + w3 - v*(1 + 2*v2 + 2*w2)*gamma)
        
        t23_1 = pw*(3*v6*w - 2*v7*gamma - 6*v3*w2*w2_plus1*gamma - 3*v5*(1 + 2*w2)*gamma + v*(1 - 3*w4 - 2*w**6)*gamma)
        t23_2 = pv*denom2*(v3 - v*(1 + 3*w2) - w*gamma)
        t23_3 = v4*(-2 + 5*pw*w_w3 - Omega)
        t23_4 = -w2_plus1**2*(-2 + pw*w_w3 + Omega)
        t23_5 = v2*w2_plus1*(pw*w_w3 - 2*(4 + Omega))
        J23 = (t23_1 + t23_2 + t23_3 + t23_4 + t23_5)*inv_denom2
        
        J30 = J21
        
        t31_1 = 2*pv*(8*v2*w*(-3 - 3*v2 + w2)*inv_denom3 + pw*(1 - 3*v2 + 3*w2))
        t31_2 = pv2*(6*vw + gamma - 2*(v2 + 3*w2)*gamma)
        t31_3 = pw*(16*v3 + 16*v5 - 48*v3*w2 - 6*pw*vw*denom3 - pw*denom3*(1 + 2*v2 + 6*w2)*gamma)*inv_denom3
        J31 = 0.5*(t31_1 + t31_2 + t31_3)
        
        t32_0 = pv*(v - v3 + 3*vw2 + w*gamma - 2*v2w*gamma - 2*w3*gamma)
        t32_1 = -3*pw*v6*w + pw*w*w2_plus1**3 + v4*(4 - 5*pw*w_w3)
        t32_2 = -v2*(-4 + w*(4*w + pw*w2_plus1**2))
        t32_3 = -pw*v5*gamma - 2*pw*v3*w2_plus1*gamma - pw*v*w2_plus1**2*gamma + denom2*Omega
        J32 = t32_0 + (t32_1 + t32_2 + t32_3)*inv_denom2
        
        J33 = pw*v*(-1 + v2 - 3*w2) + 8*v3*w*inv_denom2 + gamma - pw*w*(1 + 2*v2 + 2*w2)*gamma + pv*(w - 3*v2w + w3 - v*gamma)

    elif distribution == "Wigner":
        v2_w2_m1 = v2 + w2 - 1

        J00 = pv*v3*gamma + (-1 + pw*w)*gamma + v*w2_plus1*(-8*w*inv_denom2 + pv*gamma)
        J01 = -4*v2*c1*inv_denom2 + pw*v*gamma + pv*w*v2_w2_m1*gamma - Omega
        J02 = 0.25*(denom2 - 4*w2)*gamma
        J03 = vw*gamma
        
        J10 = -2 + 4*v2*(3*w2_plus1 + v2)*inv_denom2 + pv*w*gamma + pw*v*v2_w2_m1*gamma + Omega
        J11 = -8*v3*w*inv_denom2 + (-1 + pv*v + pw*w*denom)*gamma
        J12 = vw*gamma
        J13 = 0.25*(denom2 - 4*v2)*gamma
        
        t20_1 = 8*w2_plus1*(pv*w*(1 - 3*v2 + w2) + pw*v*(v2 - 3*w2_plus1))*inv_denom3
        t20_2 = -0.5*(pw2*(-1 + 3*v2 + w2) + pv2*(1 + 3*v2 + w2))*gamma
        J20 = t20_1 + t20_2
        
        t21_1 = 8*v*(pv - pv*w4 + pv*v2*(1 + 3*w2) + pw*vw*(3 - v2 + 3*w2))*inv_denom3
        t21_2 = -(pvpw + (pv2 + pw2)*vw)*gamma
        J21 = t21_1 + t21_2
        
        J22 = 8*v*w_w3*inv_denom2 + gamma - pw*w*gamma - pv*v*denom*gamma
        J23 = -J10
        
        J30 = J21
        
        t31_1 = 8*pw*v3*(1 + v2 - 3*w2)*inv_denom3
        t31_2 = 8*pv*v2w*(-3 - 3*v2 + w2)*inv_denom3
        t31_3 = -0.5*pv2*(-1 + v2 + 3*w2)*gamma
        t31_4 = -0.5*pw2*(1 + v2 + 3*w2)*gamma
        J31 = t31_1 + t31_2 + t31_3 + t31_4
        
        J32 = -J01
        J33 = 8*v3*w*inv_denom2 + gamma - (pv*v + pw*w*denom)*gamma

    else:
        raise ValueError("distribution must be 'Husimi', 'P', or 'Wigner'")

    return np.array([[J00, J01, J02, J03],
                     [J10, J11, J12, J13],
                     [J20, J21, J22, J23],
                     [J30, J31, J32, J33]], dtype=np.float64)


def jacobian_4d(state, gamma, Omega, distribution):
    """
    Drop-in analytical replacement for the numerical jacobian implementation.
    """
    return jacobian_4d_numba(*state, gamma, Omega, distribution)


def classify_critical_points(crit4, gamma, Omega, distribution):
    """Calculates eigendecomposition and identifies sub-spaces."""
    info = []
    for i, p in enumerate(crit4):
        jac4d = jacobian_4d(p, gamma, Omega, distribution)
        vals, vecs = eig(jac4d)
        # Filter eigvals related to vectors with zero momentum
        fil = np.array([np.allclose(v[-2:], 0, atol=1e-6) for v in vecs.T])

        rel_vals = vals[fil]
        n_pos = np.sum(rel_vals.real > 1e-8)
        n_neg = np.sum(rel_vals.real < -1e-8)

        info.append({
            'idx': i,
            'pt': p,
            'eig': vals,
            'vecs': vecs,
            'n_pos': n_pos,
            'n_neg': n_neg,
            'sub_eig': rel_vals
        })
    return info


def get_ordered_eigvecs(node_info, unstable=True):
    """Extracts fast/slow sub-manifold vectors."""
    eigvals, eigvecs = node_info['eig'], node_info['vecs']
    valid_idx = np.where(eigvals.real > 1e-6)[0] if unstable else np.where(eigvals.real < -1e-6)[0]

    if np.allclose(eigvals[valid_idx].imag, 0):
        sorted_idx = valid_idx[np.argsort(-np.abs(eigvals[valid_idx]))]
        V_fast, V_slow = eigvecs[:, sorted_idx[0]].real, eigvecs[:, sorted_idx[1]].real
    else:
        V_fast, V_slow = eigvecs[:, valid_idx][:, 0].real, eigvecs[:, valid_idx][:, 0].imag

    return V_fast / np.linalg.norm(V_fast), V_slow / np.linalg.norm(V_slow)


# ==========================================
# 2. SHOOTING METHOD (DOPRI5 Integrator)
# ==========================================

# DOPRI5 Constants
A21 = 1/5
A31, A32 = 3/40, 9/40
A41, A42, A43 = 44/45, -56/15, 32/9
A51, A52, A53, A54 = 19372/6561, -25360/2187, 64448/6561, -212/729
A61, A62, A63, A64, A65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
B1, B3, B4, B5, B6 = 35/384, 500/1113, 125/192, -2187/6784, 11/84
E1, E3, E4, E5, E6, E7 = 71/57600, -71/16695, 71/1920, -17253/339200, 22/525, -1/40


@njit(inline='always')
def dopri5_step(y0, y1, y2, y3, h, k1_0, k1_1, k1_2, k1_3, gamma, Omega, backwards, atol, rtol, distribution):
    """Abstracted DOPRI5 integration step."""
    y_tmp_0 = y0 + h*A21*k1_0
    y_tmp_1 = y1 + h*A21*k1_1
    y_tmp_2 = y2 + h*A21*k1_2
    y_tmp_3 = y3 + h*A21*k1_3
    k2_0, k2_1, k2_2, k2_3 = drift_4d_numba(y_tmp_0, y_tmp_1, y_tmp_2, y_tmp_3, gamma, Omega, backwards, distribution)

    y_tmp_0 = y0 + h*(A31*k1_0 + A32*k2_0)
    y_tmp_1 = y1 + h*(A31*k1_1 + A32*k2_1)
    y_tmp_2 = y2 + h*(A31*k1_2 + A32*k2_2)
    y_tmp_3 = y3 + h*(A31*k1_3 + A32*k2_3)
    k3_0, k3_1, k3_2, k3_3 = drift_4d_numba(y_tmp_0, y_tmp_1, y_tmp_2, y_tmp_3, gamma, Omega, backwards, distribution)

    y_tmp_0 = y0 + h*(A41*k1_0 + A42*k2_0 + A43*k3_0)
    y_tmp_1 = y1 + h*(A41*k1_1 + A42*k2_1 + A43*k3_1)
    y_tmp_2 = y2 + h*(A41*k1_2 + A42*k2_2 + A43*k3_2)
    y_tmp_3 = y3 + h*(A41*k1_3 + A42*k2_3 + A43*k3_3)
    k4_0, k4_1, k4_2, k4_3 = drift_4d_numba(y_tmp_0, y_tmp_1, y_tmp_2, y_tmp_3, gamma, Omega, backwards, distribution)

    y_tmp_0 = y0 + h*(A51*k1_0 + A52*k2_0 + A53*k3_0 + A54*k4_0)
    y_tmp_1 = y1 + h*(A51*k1_1 + A52*k2_1 + A53*k3_1 + A54*k4_1)
    y_tmp_2 = y2 + h*(A51*k1_2 + A52*k2_2 + A53*k3_2 + A54*k4_2)
    y_tmp_3 = y3 + h*(A51*k1_3 + A52*k2_3 + A53*k3_3 + A54*k4_3)
    k5_0, k5_1, k5_2, k5_3 = drift_4d_numba(y_tmp_0, y_tmp_1, y_tmp_2, y_tmp_3, gamma, Omega, backwards, distribution)

    y_tmp_0 = y0 + h*(A61*k1_0 + A62*k2_0 + A63*k3_0 + A64*k4_0 + A65*k5_0)
    y_tmp_1 = y1 + h*(A61*k1_1 + A62*k2_1 + A63*k3_1 + A64*k4_1 + A65*k5_1)
    y_tmp_2 = y2 + h*(A61*k1_2 + A62*k2_2 + A63*k3_2 + A64*k4_2 + A65*k5_2)
    y_tmp_3 = y3 + h*(A61*k1_3 + A62*k2_3 + A63*k3_3 + A64*k4_3 + A65*k5_3)
    k6_0, k6_1, k6_2, k6_3 = drift_4d_numba(y_tmp_0, y_tmp_1, y_tmp_2, y_tmp_3, gamma, Omega, backwards, distribution)

    y_new_0 = y0 + h*(B1*k1_0 + B3*k3_0 + B4*k4_0 + B5*k5_0 + B6*k6_0)
    y_new_1 = y1 + h*(B1*k1_1 + B3*k3_1 + B4*k4_1 + B5*k5_1 + B6*k6_1)
    y_new_2 = y2 + h*(B1*k1_2 + B3*k3_2 + B4*k4_2 + B5*k5_2 + B6*k6_2)
    y_new_3 = y3 + h*(B1*k1_3 + B3*k3_3 + B4*k4_3 + B5*k5_3 + B6*k6_3)

    k7_0, k7_1, k7_2, k7_3 = drift_4d_numba(y_new_0, y_new_1, y_new_2, y_new_3, gamma, Omega, backwards, distribution)

    err_0 = h*(E1*k1_0 + E3*k3_0 + E4*k4_0 + E5*k5_0 + E6*k6_0 + E7*k7_0)
    err_1 = h*(E1*k1_1 + E3*k3_1 + E4*k4_1 + E5*k5_1 + E6*k6_1 + E7*k7_1)
    err_2 = h*(E1*k1_2 + E3*k3_2 + E4*k4_2 + E5*k5_2 + E6*k6_2 + E7*k7_2)
    err_3 = h*(E1*k1_3 + E3*k3_3 + E4*k4_3 + E5*k5_3 + E6*k6_3 + E7*k7_3)

    sc_0 = atol + rtol * max(abs(y0), abs(y_new_0))
    sc_1 = atol + rtol * max(abs(y1), abs(y_new_1))
    sc_2 = atol + rtol * max(abs(y2), abs(y_new_2))
    sc_3 = atol + rtol * max(abs(y3), abs(y_new_3))

    err_max = np.sqrt(((err_0/sc_0)**2 + (err_1/sc_1)**2 + (err_2/sc_2)**2 + (err_3/sc_3)**2) / 4.0)
    return y_new_0, y_new_1, y_new_2, y_new_3, k7_0, k7_1, k7_2, k7_3, err_max


@njit(parallel=True)
def integrate_batch_jit(y0s, gamma, Omega, tgt, t_max, h0, max_steps, check_every, r_escape, p_escape, backwards, atol, rtol, distribution):
    """Batch integrates trajectories and evaluates closest approach to target."""
    N = y0s.shape[0]
    best_dists_sq = np.full(N, np.inf)
    r_escape2, p_escape2 = r_escape**2, p_escape**2

    for i in prange(N):
        y0, y1, y2, y3 = y0s[i,0], y0s[i,1], y0s[i,2], y0s[i,3]
        t, h, step = 0.0, h0, 0
        active = True

        buf_idx = 0
        buffer_y = np.zeros((check_every, 4))
        k1_0, k1_1, k1_2, k1_3 = drift_4d_numba(y0, y1, y2, y3, gamma, Omega, backwards, distribution)

        while step < max_steps and active and t < t_max:
            if t + h > t_max: h = t_max - t

            yn_0, yn_1, yn_2, yn_3, k7_0, k7_1, k7_2, k7_3, err_max = dopri5_step(
                y0, y1, y2, y3, h, k1_0, k1_1, k1_2, k1_3, gamma, Omega, backwards, atol, rtol, distribution
            )
            scale_factor = 0.9 * (err_max ** -0.2) if err_max > 0.0 else 5.0

            if err_max <= 1.0:
                t += h
                y0, y1, y2, y3 = yn_0, yn_1, yn_2, yn_3
                k1_0, k1_1, k1_2, k1_3 = k7_0, k7_1, k7_2, k7_3

                buffer_y[buf_idx, 0] = y0
                buffer_y[buf_idx, 1] = y1
                buffer_y[buf_idx, 2] = y2
                buffer_y[buf_idx, 3] = y3
                buf_idx += 1

                if buf_idx == check_every:
                    d_sq = np.sum((buffer_y - tgt)**2, axis=1)
                    d_sq_min = d_sq.min()
                    d_sq_max = d_sq.max()
                    if d_sq_min < best_dists_sq[i]:
                        best_dists_sq[i] = d_sq_min
                    if d_sq_max > r_escape2 or np.sum(buffer_y[:,2:]**2, axis=1).max() > p_escape2:
                        active = False
                    buf_idx = 0

                h *= min(5.0, max(0.2, scale_factor))
            else:
                h *= min(0.9, max(0.1, scale_factor))
            step += 1

        for j in range(buf_idx):
            p = buffer_y[j]
            d_sq = (p[0]-tgt[0])**2 + (p[1]-tgt[1])**2 + (p[2]-tgt[2])**2 + (p[3]-tgt[3])**2
            if d_sq < best_dists_sq[i]: best_dists_sq[i] = d_sq

    dists = np.sqrt(best_dists_sq)
    b_idx = np.argmin(dists)

    return b_idx, dists[b_idx]


@njit
def integrate_single_path(y_init, gamma, Omega, t_max, h0, max_steps, backwards, atol, rtol, r_escape, p_escape, distribution):
    """Dense integrator for reconstructing the final single best path."""
    r_escape2, p_escape2 = r_escape**2, p_escape**2
    ys = np.zeros((max_steps, 4))
    ts = np.zeros(max_steps)
    y0, y1, y2, y3 = y_init[0], y_init[1], y_init[2], y_init[3]
    ys[0] = y_init
    count, t, h, step = 1, 0.0, h0, 0
    active = True
    k1_0, k1_1, k1_2, k1_3 = drift_4d_numba(y0, y1, y2, y3, gamma, Omega, backwards, distribution)

    while step < max_steps and active and t < t_max:
        if t + h > t_max: h = t_max - t
        yn_0, yn_1, yn_2, yn_3, k7_0, k7_1, k7_2, k7_3, err_max = dopri5_step(
            y0, y1, y2, y3, h, k1_0, k1_1, k1_2, k1_3, gamma, Omega, backwards, atol, rtol, distribution
        )
        scale_factor = 0.9 * (err_max ** -0.2) if err_max > 0.0 else 5.0
        if err_max <= 1.0:
            t += h
            y0, y1, y2, y3 = yn_0, yn_1, yn_2, yn_3
            k1_0, k1_1, k1_2, k1_3 = k7_0, k7_1, k7_2, k7_3
            if (step+1) % 10 == 0 and ((y0-y_init[0])**2 + (y1-y_init[1])**2 > r_escape2 or y2**2 + y3**2 > p_escape2):
                active = False
            if count < max_steps:
                ys[count, 0], ys[count, 1], ys[count, 2], ys[count, 3] = y0, y1, y2, y3
                ts[count] = t
                count += 1
            h *= min(5.0, max(0.2, scale_factor))
        else:
            h *= min(0.9, max(0.1, scale_factor))
        step += 1
    return ys[:count], ts[:count]


# ==========================================
# 3. BOUNDARY VALUE PROBLEM (BVP) METHOD
# ==========================================


def logitspace(n0, n1, num, rescale_to_0_1=False):
    """Spans limits with logarithmic spacing near bounds."""
    assert n0 <= 0 and n1 <= 0, "logitspace needs 10**n <= 1"
    ln10 = np.log(10.0)
    ya = n0 * ln10 - np.log1p(-10.0**n0)
    yb = np.log1p(-10.0**n1) - n1 * ln10
    y = expit(np.linspace(ya, yb, num))
    if rescale_to_0_1:
        y = (y - y[0]) / (y[-1] - y[0])
    return y


def get_manifold_vector(point_info, manifold='unstable', phase_angle=0.0):
    """
    Extracts local sub-manifold basis vectors. 
    If the manifold is 2D, phase_angle parameterizes the linear combination 
    of the basis vectors (or the real/imag components for spiral nodes).
    """
    eig = point_info['eig']
    vecs = point_info['vecs']
    vecs_4d = ~(np.isclose(vecs[2,:], 0) * np.isclose(vecs[3,:], 0))
    
    valid_idx = np.where((eig.real > 1e-10) * vecs_4d)[0] if manifold == 'unstable' else np.where((eig.real < -1e-10) * vecs_4d)[0]

    if len(valid_idx) == 0:
        return np.zeros(4)

    if len(valid_idx) >= 2:
        # We have a 2D manifold. Construct a basis.
        v1 = vecs[:, valid_idx[0]]
        v2 = vecs[:, valid_idx[1]]

        if np.iscomplexobj(v1) and np.isclose(eig[valid_idx[0]], np.conj(eig[valid_idx[1]])):
            # Spiral critical point: natural basis is real and imaginary parts
            b1 = np.real(v1)
            b2 = np.imag(v1)
        else:
            # Nodal critical point: two real eigenvectors
            b1 = np.real(v1)
            b2 = np.real(v2)

        # Mix them using the phase angle
        U = np.cos(phase_angle) * b1 + np.sin(phase_angle) * b2
    else:
        # 1D manifold fallback
        U = np.real(vecs[:, valid_idx[0]])

    return U / np.linalg.norm(U)


@njit
def bvp_fun_jac_numba(y, p, gamma, Omega, distribution):
    """
    Computes df/dy and df/dp analytically across the entire BVP mesh.
    """
    T = p[0]
    m = y.shape[1]
    df_dy = np.zeros((4, 4, m), dtype=np.float64)
    df_dp = np.zeros((4, 1, m), dtype=np.float64)
    
    for i in range(m):
        # The jacobian of T * drift with respect to y is T * jac_4d
        df_dy[:, :, i] = T * jacobian_4d_numba(y[0, i], y[1, i], y[2, i], y[3, i], gamma, Omega, distribution)
        
        # The derivative of T * drift with respect to the parameter T is just the drift
        dv, dw, dpv, dpw = drift_4d_numba(y[0, i], y[1, i], y[2, i], y[3, i], gamma, Omega, False, distribution)
        df_dp[0, 0, i] = dv
        df_dp[1, 0, i] = dw
        df_dp[2, 0, i] = dpv
        df_dp[3, 0, i] = dpw
        
    return df_dy, df_dp


def _subspace_basis(info, kind, tol=1e-10):
    """
    Real basis of the FULL stable or unstable eigenspace (handles complex pairs).
    Universally robust for 4D FW Hamiltonians: naturally handles 2D nodes, spirals, 
    attractors, and sources by sorting and extracting the 2 dominant eigenvalues.
    """
    eig = np.asarray(info['eig'])
    vecs = np.asarray(info['vecs'])
    
    # 4D FW Hamiltonians always have 2 stable and 2 unstable eigenvalues. 
    # Sort by real part to robustly select exactly the 2 we want.
    if kind == 'stable':
        idx = np.argsort(eig.real)[:2]       # Two most negative
    else:
        idx = np.argsort(eig.real)[::-1][:2]   # Two most positive
        
    v1, v2 = eig[idx[0]], eig[idx[1]]
    u1, u2 = vecs[:, idx[0]], vecs[:, idx[1]]
    
    # Check if they form a complex conjugate pair
    if abs(v1.imag) > tol and abs(v1 - np.conj(v2)) < 100 * tol:
        basis = [np.real(u1), np.imag(u1)]
    else:
        basis = [np.real(u1), np.real(u2)]
        
    return basis


def _compute_S_matrix(info, kind, tol=1e-10):
    """
    Computes the 2x2 matrix S relating momentum displacement to state displacement (p = S * dq)
    for the specified invariant subspace ('unstable' for origin A, 'stable' for target B).
    """
    basis = _subspace_basis(info, kind, tol=tol)
    assert len(basis) == 2, f"Expected 2D basis for {kind} subspace, got {len(basis)}"
    
    # Form 4x2 matrix V = [v1, v2]
    V = np.column_stack(basis)
    X = V[:2, :]  # Top 2x2 block (position components)
    P = V[2:, :]  # Bottom 2x2 block (momentum components)
    
    # Solve S @ X = P  =>  S = P @ X^-1
    S = np.linalg.solve(X.T, P.T).T
    
    # Enforce exact symmetry (Lagrangian manifold condition)
    return 0.5 * (S + S.T)


def bridge_to_critical_point(cut_pt, cp, J, kind, eps_target, num_pts=20):
    """
    Generates a bridge curve lying on the linear invariant manifold (stable or unstable).
    Natively handles spiral manifolds (complex eigen-pairs) and nodal manifolds (real).
    """
    vals, vecs = np.linalg.eig(J)
    
    if kind == 'unstable':
        idx = np.argsort(vals.real)[::-1][:2]
        sign = -1.0 # Trace backwards in time to approach the origin
    else:
        idx = np.argsort(vals.real)[:2]
        sign = 1.0  # Trace forwards in time to approach the target
        
    v1, v2 = vals[idx[0]], vals[idx[1]]
    vec1, vec2 = vecs[:, idx[0]], vecs[:, idx[1]]
    
    delta = cut_pt - cp
    is_spiral = abs(v1.imag) > 1e-10
    
    if is_spiral:
        lam, omega = v1.real, v1.imag
        u, v = vec1.real, vec1.imag
        V = np.column_stack((u, v))
        c, _, _, _ = np.linalg.lstsq(V, delta, rcond=None)
        A, B = c[0], c[1]
        lam_min = abs(lam)
    else:
        lam1, lam2 = v1.real, v2.real
        u1, u2 = vec1.real, vec2.real
        V = np.column_stack((u1, u2))
        c, _, _, _ = np.linalg.lstsq(V, delta, rcond=None)
        lam_min = min(abs(lam1), abs(lam2))
        
    def base_path(t):
        if is_spiral:
            # Analytical Log-Spiral
            return cp + np.exp(lam * t) * (
                (A * u + B * v) * np.cos(omega * t) + 
                (-A * v + B * u) * np.sin(omega * t)
            )
        else:
            # Analytical Exponential Node
            return cp + c[0] * np.exp(lam1 * t) * u1 + c[1] * np.exp(lam2 * t) * u2

    # Find the integration time 'T_end' that lands exactly at eps_target distance
    dist_0 = np.linalg.norm(delta)
    if lam_min > 1e-12:
        t_approx = np.log(eps_target / max(dist_0, 1e-12)) / lam_min
        t_search_max = 2.0 * abs(t_approx) + 10.0
    else:
        t_search_max = 1000.0

    t_eval = np.linspace(0, sign * t_search_max, 2000)
    dists = np.linalg.norm([base_path(t) - cp for t in t_eval], axis=1)
    cross_idx = np.where(dists < eps_target)[0]
    T_end = t_eval[cross_idx[0]] if len(cross_idx) > 0 else sign * t_search_max

    # Add a linear residual correction to ensure perfect C0 patching to the cut_pt
    residual = cut_pt - base_path(0)
    
    def corrected_path(t):
        tau_frac = t / T_end if abs(T_end) > 1e-12 else 0.0
        return base_path(t) + residual * (1.0 - tau_frac)

    # Discretize the resulting bridge
    if kind == 'unstable':
        tau = np.linspace(T_end, 0, num_pts) # From origin to cut_pt
    else:
        tau = np.linspace(0, T_end, num_pts) # From cut_pt to target
        
    return np.array([corrected_path(t) for t in tau]).T


def trim_and_interpolate_trajectory(traj, origin_2d, target_2d, drift4d_fn, params, eps1=1e-4, eps2=1e-4):
    """
    Trims the trajectory at both ends and mathematically bridges the gaps using the exact 
    linear stable/unstable manifolds.
    
    Assumes `params` is a tuple (e.g., `(gamma, Omega, distribution)`) that unpacks 
    correctly into `jacobian_4d_numba`.
    """
    origin = np.array([origin_2d[0], origin_2d[1], 0.0, 0.0])
    target = np.array([target_2d[0], target_2d[1], 0.0, 0.0])
    
    L_scale = np.linalg.norm(target - origin)
    trim_horizon = 0.05 * L_scale
    N = traj.shape[1]
    
    # 1. Trim at Origin (find first point that exceeds the trimming horizon)
    cut_idx_start = 0
    for i in range(N):
        if np.linalg.norm(traj[:, i] - origin) > trim_horizon:
            cut_idx_start = i
            break
            
    # 2. Trim at Target (find last well-aligned point just outside the horizon)
    cut_idx_end = N - 1
    best_metric = np.inf
    for i in range(N-1, -1, -1):
        pt = traj[:, i]
        dist_to_target = np.linalg.norm(target - pt)
        
        if dist_to_target > trim_horizon: 
            break
            
        v_drift = np.array(drift4d_fn(*pt, backwards=False, **params))
        v_norm = np.linalg.norm(v_drift)
        r_vec = target - pt
        r_norm = np.linalg.norm(r_vec)
        
        if v_norm > 1e-12 and r_norm > 1e-12:
            alignment = np.dot(v_drift / v_norm, r_vec / r_norm)
            metric = 1.0 - alignment
            if metric < best_metric:
                best_metric = metric
                cut_idx_end = i

    # Safeguard against crossover if trajectory is extremely short
    if cut_idx_start >= cut_idx_end:
        cut_idx_start = max(0, cut_idx_end - 2)

    trimmed_traj = traj[:, cut_idx_start:cut_idx_end+1]
    
    # 3. Bridge the Origin gap (Dynamically traces the spiral / node)
    cut_pt_start = trimmed_traj[:, 0]
    if np.linalg.norm(cut_pt_start - origin) > eps1:
        J_orig = jacobian_4d_numba(origin[0], origin[1], 0.0, 0.0, **params)
        bridge_orig = bridge_to_critical_point(cut_pt_start, origin, J_orig, 'unstable', eps1, num_pts=20)
        bridge_orig = bridge_orig[:, :-1] # Exclude final point to avoid overlap with trimmed_traj
    else:
        bridge_orig = np.empty((4, 0))

    # 4. Bridge the Target gap
    cut_pt_end = trimmed_traj[:, -1]
    if np.linalg.norm(target - cut_pt_end) > eps2:
        J_targ = jacobian_4d_numba(target[0], target[1], 0.0, 0.0, **params)
        bridge_targ = bridge_to_critical_point(cut_pt_end, target, J_targ, 'stable', eps2, num_pts=20)
        bridge_targ = bridge_targ[:, 1:] # Exclude first point to avoid overlap
    else:
        bridge_targ = np.empty((4, 0))
        
    # Stitch everything together seamlessly
    final_guess = np.hstack([bridge_orig, trimmed_traj, bridge_targ])

    # Calculate the magnitude of the drift (speed) at each point
    N_final = final_guess.shape[1]
    speeds = np.zeros(N_final)
    for i in range(N_final):
        v_drift = np.array(drift4d_fn(*final_guess[:, i], backwards=False, **params))
        speeds[i] = np.linalg.norm(v_drift)
        
    # Prevent division by zero (though we are eps away from critical points)
    inv_speeds = 1.0 / np.maximum(speeds, 1e-16)
    
    # Calculate cumulative path length 's'
    dp = np.diff(final_guess, axis=1, prepend=0.0)
    ds = np.linalg.norm(dp, axis=0)
    s_array = np.cumsum(ds)
    
    # Integrate dt = ds / v to get strictly increasing times
    from scipy.integrate import cumulative_trapezoid
    time_array = cumulative_trapezoid(inv_speeds, x=s_array, initial=0.0)
    
    return final_guess, time_array


def find_heteroclinic_bvp(
        infoA, infoB, gamma, Omega, nodes, max_nodes, tol,
        guess_tau=None, guess_y=None, guess_p=None,
        yA_target=None, yB_target=None,
        bc_mode='manifold_nl',              # 'manifold_nl' | 'manifold_S' | 'projection' | 'pinned'
        verbose=False, distribution=None,
        use_fun_jac=True, use_bc_jac=True,
        flipA=False, flipB=False, L_scale=1.0, eps=1e-4,
    ):
    """
    ODE: dy/dtau = T * f(y), tau in [0,1], p = [T] free. 4 states + 1 par = 5 BCs.

    bc_mode 'manifold_nl' : Linear manifold BCs scaled dynamically to enforce H=0 exactly.
                            Best for larger eps values. Universal for nodes/spirals.
    bc_mode 'manifold_S'  : Pure linear manifold (p = S * dq). Universal for nodes/spirals.
    bc_mode 'projection'  : (ya - A) perp stable eigenspace of A.
    bc_mode 'pinned'      : Fixed endpoints yA_target, yB_target.
    """
    assert distribution is not None
    ptA = np.asarray(infoA['pt'], float)
    ptB = np.asarray(infoB['pt'], float)
    vecAB = ptB - ptA
    AB_dist = np.linalg.norm(vecAB)

    if bc_mode in ('manifold_nl', 'manifold_S'):
        S_A = _compute_S_matrix(infoA, 'unstable')
        S_B = _compute_S_matrix(infoB, 'stable')
        is_nl = (bc_mode == 'manifold_nl')

    if yA_target is None or yB_target is None:
        if bc_mode in ('manifold_nl', 'manifold_S'):
            dir_2d = ptB[:2] - ptA[:2]
            norm_2d = np.hypot(dir_2d[0], dir_2d[1])
            dir_2d = dir_2d / norm_2d if norm_2d > 1e-12 else np.array([1.0, 0.0])
            
            dqA = eps * dir_2d
            pA = S_A @ dqA
            if yA_target is None:
                yA_target = np.array([ptA[0] + dqA[0], ptA[1] + dqA[1], pA[0], pA[1]])
                
            dqB = -eps * dir_2d
            pB = S_B @ dqB
            if yB_target is None:
                yB_target = np.array([ptB[0] + dqB[0], ptB[1] + dqB[1], pB[0], pB[1]])
        else:
            U_A_vector = -get_manifold_vector(infoA, 'unstable')
            U_B_vector =  get_manifold_vector(infoB, 'stable')
            if (U_A_vector.dot(vecAB) > 0) == flipA:
                U_A_vector = -U_A_vector
            if (U_B_vector.dot(-vecAB) > 0) == flipB:
                U_B_vector = -U_B_vector
            if yA_target is None:
                yA_target = ptA + eps * U_A_vector
            if yB_target is None:
                yB_target = ptB + eps * U_B_vector

    def fun(_, y, p):
        T = p[0]
        dv, dw, dpv, dpw = drift_4d_numba(y[0], y[1], y[2], y[3],
                                          gamma, Omega, False, distribution)
        return T * np.vstack((dv, dw, dpv, dpw))

    def fun_jac(x, y, p):
        return bvp_fun_jac_numba(y, p, gamma, Omega, distribution)

    # ------------------- BOUNDARY CONDITIONS SETUP -------------------
    if bc_mode in ('manifold_nl', 'manifold_S'):
        
        def get_alpha(q, p_dir):
            if not is_nl: 
                return 1.0
            
            # f(q) evaluated at p=0
            f_val = np.array(drift_4d_numba(q[0], q[1], 0.0, 0.0, gamma, Omega, False, distribution))[:2]
            
            # Extract D(q) from 4D Jacobian evaluated at p=0
            J4D = jacobian_4d_numba(q[0], q[1], 0.0, 0.0, gamma, Omega, distribution)
            D = J4D[0:2, 2:4] / 2.0
            
            den = np.dot(p_dir, D @ p_dir)
            if abs(den) < 1e-16: 
                return 1.0
            return -np.dot(p_dir, f_val) / den

        def bc(ya, yb, _):
            dA = ya[:2] - ptA[:2]
            dB = yb[:2] - ptB[:2]
            
            pA_dir = S_A @ dA
            pB_dir = S_B @ dB
            
            alpha_A = get_alpha(ya[:2], pA_dir)
            alpha_B = get_alpha(yb[:2], pB_dir)
            
            return np.array([
                np.hypot(dA[0], dA[1]) - eps,       # Distance eps from origin
                ya[2] - alpha_A * pA_dir[0],        # Nonlinear manifold pa_1
                ya[3] - alpha_A * pA_dir[1],        # Nonlinear manifold pa_2
                yb[2] - alpha_B * pB_dir[0],        # Nonlinear manifold pb_1
                yb[3] - alpha_B * pB_dir[1],        # Nonlinear manifold pb_2
            ])

        def bc_jac(ya, yb, p):
            dbc_dya = np.zeros((5, 4))
            dbc_dyb = np.zeros((5, 4))
            dbc_dp = np.zeros((5, 1))

            if is_nl:
                # Finite difference required due to complex alpha dependency
                eps_fd = 1e-8
                r0 = bc(ya, yb, p)
                for i in range(4):
                    ya_fd = ya.copy()
                    ya_fd[i] += eps_fd
                    dbc_dya[:, i] = (bc(ya_fd, yb, p) - r0) / eps_fd
                    
                    yb_fd = yb.copy()
                    yb_fd[i] += eps_fd
                    dbc_dyb[:, i] = (bc(ya, yb_fd, p) - r0) / eps_fd
            else:
                # Exact analytical Jacobian for the linear manifold
                dA = ya[:2] - ptA[:2]
                norm_dA = np.hypot(dA[0], dA[1])
                if norm_dA > 0:
                    dbc_dya[0, 0] = dA[0] / norm_dA
                    dbc_dya[0, 1] = dA[1] / norm_dA

                dbc_dya[1, 0] = -S_A[0, 0]
                dbc_dya[1, 1] = -S_A[0, 1]
                dbc_dya[1, 2] = 1.0

                dbc_dya[2, 0] = -S_A[1, 0]
                dbc_dya[2, 1] = -S_A[1, 1]
                dbc_dya[2, 3] = 1.0

                dbc_dyb[3, 0] = -S_B[0, 0]
                dbc_dyb[3, 1] = -S_B[0, 1]
                dbc_dyb[3, 2] = 1.0

                dbc_dyb[4, 0] = -S_B[1, 0]
                dbc_dyb[4, 1] = -S_B[1, 1]
                dbc_dyb[4, 3] = 1.0

            return dbc_dya, dbc_dyb, dbc_dp

    elif bc_mode == 'projection':
        S_A_subspace = _subspace_basis(infoA, 'stable')      # ya on unstable manifold of A
        U_B_subspace = _subspace_basis(infoB, 'unstable')    # yb on stable manifold of B
        assert len(S_A_subspace) == 2 and len(U_B_subspace) == 2, "expected 2D subspaces"

        def bc(ya, yb, _):
            return np.array(
                [np.dot(v, ya - ptA) for v in S_A_subspace]
                + [np.linalg.norm(ya - ptA) - eps]
                + [np.dot(v, yb - ptB) for v in U_B_subspace]
            )

        def bc_jac(ya, yb, p):
            dbc_dya = np.zeros((5, 4))
            dbc_dyb = np.zeros((5, 4))
            dbc_dp = np.zeros((5, 1))
            for r, v in enumerate(S_A_subspace):
                dbc_dya[r, :] = v
            d = ya - ptA
            n = np.linalg.norm(d)
            if n > 0:
                dbc_dya[2, :] = d / n
            for r, v in enumerate(U_B_subspace):
                dbc_dyb[3 + r, :] = v
            return dbc_dya, dbc_dyb, dbc_dp

    else:  # 'pinned'
        def bc(ya, yb, _):
            H_start = hamiltonian_num(ya[0], ya[1], ya[2], ya[3], gamma, Omega, distribution)
            return np.array([
                ya[0] - yA_target[0],
                ya[1] - yA_target[1],
                yb[0] - yB_target[0],
                yb[1] - yB_target[1],
                H_start,
            ])

        def bc_jac(ya, yb, p):
            dbc_dya = np.zeros((5, 4))
            dbc_dyb = np.zeros((5, 4))
            dbc_dp = np.zeros((5, 1))
            dbc_dya[0, 0] = 1.0
            dbc_dya[1, 1] = 1.0
            dbc_dyb[2, 0] = 1.0
            dbc_dyb[3, 1] = 1.0
            eps_fd = 1e-8
            H0 = hamiltonian_num(ya[0], ya[1], ya[2], ya[3], gamma, Omega, distribution)
            for i in range(4):
                ya_eps = ya.copy()
                ya_eps[i] += eps_fd
                H1 = hamiltonian_num(ya_eps[0], ya_eps[1], ya_eps[2], ya_eps[3],
                                     gamma, Omega, distribution)
                dbc_dya[4, i] = (H1 - H0) / eps_fd
            return dbc_dya, dbc_dyb, dbc_dp

    # ------------------- INITIAL GUESS GENERATION -------------------
    if guess_y is None:
        tau_guess = logitspace(np.log10(eps), np.log10(eps), nodes, rescale_to_0_1=True)
        t_curve = logitspace(np.log10(eps), np.log10(eps), nodes, rescale_to_0_1=False)
        L_A = L_B = AB_dist
        U_A_rel = (yA_target - ptA) / max(np.linalg.norm(yA_target - ptA), 1e-300)
        U_B_rel = (yB_target - ptB) / max(np.linalg.norm(yB_target - ptB), 1e-300)
        m_A, m_B = L_A * U_A_rel, L_B * U_B_rel

        h00 = 2*t_curve**3 - 3*t_curve**2 + 1
        h10 = t_curve**3 - 2*t_curve**2 + t_curve
        h01 = -2*t_curve**3 + 3*t_curve**2
        h11 = t_curve**3 - t_curve**2

        guess_y = (h00[np.newaxis, :] * ptA[:, np.newaxis] + h10[np.newaxis, :] * m_A[:, np.newaxis] +
                   h01[np.newaxis, :] * ptB[:, np.newaxis] + h11[np.newaxis, :] * m_B[:, np.newaxis])

        if bc_mode not in ('manifold_nl', 'manifold_S'):
            dir_pv, dir_pw = U_A_rel[2], U_A_rel[3]
            norm = np.hypot(dir_pv, dir_pw)
            if norm > 1e-12:
                dir_pv, dir_pw = dir_pv / norm, dir_pw / norm
            parabola = 4 * tau_guess * (1 - tau_guess)
            guess_y[2] += 0.01 * dir_pv * parabola
            guess_y[3] += 0.01 * dir_pw * parabola

        guess_y[2:] = L_scale * guess_y[2:]
        guess_tau = tau_guess
        guess_p = np.array([50.0]) if guess_p is None else guess_p

    res = solve_bvp(
        fun=fun, bc=bc, x=guess_tau, y=guess_y, p=guess_p,
        tol=tol, max_nodes=max_nodes,
        verbose=2 if verbose else 0,
        fun_jac=fun_jac if use_fun_jac else None,
        bc_jac=bc_jac if use_bc_jac else None,
    )

    if verbose:
        yb = res.y[:, -1]
        Hb = hamiltonian_num(yb[0], yb[1], yb[2], yb[3], gamma, Omega, distribution)
        print(f"T = {res.p[0]:.6g}, nodes = {res.x.size}, success = {res.success}")
        print(f"H(yb) = {Hb:.3e} (should be ~0; conserved by the ODE, not imposed)")
        print(f"|yb - B| = {np.linalg.norm(yb - ptB):.3e}")
        
    return res


def simplify_4d_path(x, y, M, alpha=0.5, beta=0.5, return_indices=False, min_res=200):
    """
    Simplifies a 4D path down to M points while preserving baseline density 
    profiles and geometric features.
    
    Parameters:
    -----------
    x : ndarray of shape (N,)
        The independent coordinate array.
    y : ndarray of shape (4, N)
        The 4-dimensional path array.
    M : int
        The target number of simplified points.
    alpha : float
        Weight for preserving original sampling density (0.0 to 1.0).
    beta : float
        Weight for preserving geometric features/curvature (0.0 to 1.0).
    return_indices : bool
        If True, returns the closest indices from the original array instead
        of interpolating new spatial positions.
    min_res : int
        minimal curve resolution relative to distance between start and end
        
    Returns:
    --------
    x_new, y_new (or indices) matching the requested density profile.
    """
    N = len(x)

    start_end_dist = np.linalg.norm(y[:,-1] - y[:,0])
    total_len = trapezoid(np.linalg.norm(np.diff(y, axis=-1, prepend=0.0), axis=0))
    min_res_points = round(min_res * total_len / start_end_dist)

    M = max(M, min_res_points)

    if M >= N:
        return x.copy(), y.copy()

    # 1. Compute original sampling density (1 / local spacing)
    dx = np.diff(x)
    spacing = np.zeros(N)
    spacing[1:-1] = (x[2:] - x[:-2]) / 2.0
    spacing[0] = dx[0]
    spacing[-1] = dx[-1]
    
    # Avoid division by zero on identical points
    spacing = np.maximum(spacing, 1e-12)
    density_orig = 1.0 / spacing
    
    # 2. Compute multi-dimensional geometric feature tracking (Curvature)
    # Calculate non-uniform second derivatives across all 4 channels
    dy_left = (y[:, 1:-1] - y[:, :-2]) / dx[:-1]
    dy_right = (y[:, 2:] - y[:, 1:-1]) / dx[1:]
    d2y_dx2 = (dy_right - dy_left) / spacing[1:-1]
    
    # Take L2 norm across the 4 independent channels
    geom_error = np.zeros(N)
    geom_error[1:-1] = np.linalg.norm(d2y_dx2, axis=0)
    geom_error[0] = geom_error[1]
    geom_error[-1] = geom_error[-2]
    
    # 3. Normalize profiles to balance their influence fairly
    norm_density = density_orig / np.max(density_orig)
    norm_geom = geom_error / np.max(geom_error) if np.max(geom_error) > 0 else geom_error
    
    # 4. Generate total importance weight map
    # A tiny floor constant (1e-5) ensures strictly monotonic behavior
    total_weight = alpha * norm_density + beta * norm_geom + 1e-5
    
    # 5. Integrate to build the Cumulative Distribution Function (CDF)
    weight_intervals = 0.5 * (total_weight[:-1] + total_weight[1:]) * dx
    cdf = np.zeros(N)
    cdf[1:] = np.cumsum(weight_intervals)
    cdf /= cdf[-1]  # Force normalization to [0.0, 1.0]
    
    # 6. Sample uniformly along the target CDF space
    target_cdf = np.linspace(0.0, 1.0, M)
    
    if return_indices:
        # Option A: Find the closest real indices from your original dataset
        # Best if you must preserve genuine experimental measurements exactly
        indices = np.searchsorted(cdf, target_cdf)
        indices = np.clip(indices, 0, N - 1)
        # Ensure exact boundaries
        indices[0], indices[-1] = 0, N - 1
        return indices
    else:
        # Option B: Re-interpolate coordinates (Higher quality smoothness)
        x_new = np.interp(target_cdf, cdf, x)
        y_new = np.zeros((4, M))
        for i in range(4):
            y_new[i, :] = np.interp(x_new, x, y[i, :])
        return x_new, y_new


# ==========================================
# 5. MAIN ORCHESTRATION & DB MANAGEMENT
# ==========================================


def evaluate_quality(trajectory):
    """Returns total action and quality metrics."""
    v, w, pv, pw = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], trajectory[:, 3]
    action = trapezoid(pv, v) + trapezoid(pw, w)
    return action


def output_file_paths(base_dir, run_params, do_not_write=False):
    base_path = Path(base_dir)
    meta_dir = base_path / 'metadata'
    bin_dir = base_path / 'trajectories'

    if not do_not_write:
        meta_dir.mkdir(parents=True, exist_ok=True)
        bin_dir.mkdir(parents=True, exist_ok=True)

    # Create a deterministic base filename from the defining index parameters
    g, o = run_params['gamma'], run_params['Omega']
    orig, tgt = run_params['origin_idx'], run_params['target_idx']
    base_name = f"g{g:.12f}_O{o:.12f}_o{orig}_t{tgt}"

    meta_path = meta_dir / f"{base_name}.json"
    lock_path = meta_path.with_suffix('.json.lock')
    npz_path = bin_dir / f"{base_name}.npz"

    return meta_path, lock_path, npz_path


def old_result_is_good_enough(base_dir, run_params, max_acceptable_quality=0):
    meta_path, _, npz_path = output_file_paths(base_dir, run_params)
    if meta_path.exists() and npz_path.exists():
        with meta_path.open('r') as f:
            existing_data = json.load(f)
        old_quality = existing_data.get('quality', float('inf'))
        if old_quality <= max_acceptable_quality:
            return True
    return False


def save_result_distributed(base_dir, run_params, action, quality, ys, ts, always_write=False, verbose=False):
    """
    Saves job parameters and quality metrics to a deterministic JSON file,
    updating only if the new quality is strictly better.
    """
    meta_path, lock_path, npz_path = output_file_paths(base_dir, run_params)

    new_data = {**run_params, 'action': action, 'quality': quality}

    with FileLock(str(lock_path)):
        if (not always_write) and meta_path.exists():
            with meta_path.open('r') as f:
                existing_data = json.load(f)

            # If the stored quality is better (lower), discard the new result
            old_quality = existing_data.get('quality', float('inf'))
            if quality >= old_quality:
                if verbose:
                    print(f"Not overwriting prior better quality, old: {old_quality:.3n}, new: {quality:.3n}")
                return
            else:
                if verbose:
                    print(f"Overwriting prior worse quality, old: {old_quality:.3n}, new: {quality:.3n}")
        
        if verbose:
            print(f"Will write to {meta_path}")

        # Write binary trajectory
        np.savez_compressed(npz_path, ys=ys, ts=ts)

        # Store the path as a standard string in the JSON
        new_data['npz_path'] = str(npz_path)

        # Write metadata
        with meta_path.open('w') as f:
            json.dump(new_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Instanton Path Finder")
    parser.add_argument('--gamma', type=float, default=0.9, help="Gamma parameter")
    parser.add_argument('--Omega', type=float, default=0.3, help="Omega parameter")
    parser.add_argument('--origin', type=int, default=0, help="Origin critical point index")
    parser.add_argument('--target', type=int, default=2, help="Target critical point index")
    parser.add_argument('--method', type=str, choices=['bvp', 'shooting', 'auto'], default='auto')
    parser.add_argument('--distribution', type=str, choices=["H", "Husimi", "W", "Wigner", "P"], default="Husimi")
    exclusive_args = parser.add_mutually_exclusive_group()
    exclusive_args.add_argument('--gamma-rel', type=float,
                                help="Gamma parameter, relative value for the narrow region, derived from --Omega, overrides --gamma")
    exclusive_args.add_argument('--Omega-rel', type=float,
                                help="Omega parameter, relative value for the narrow region, derived from --gamma, overrides --Omega")

    parser.add_argument('--spiral-factor', type=float, default=0.5, help="Ratio Im(L)/Re(L) to trigger shooting method")
    parser.add_argument('--quality-thresh', type=float, default=0.1, help="Discard thresholds for quality")
    parser.add_argument('--refinements', type=int, default=3, help="Number of refinement passes")
    parser.add_argument('--improve-fract', type=float, default=0.001, help="Min fraction improvement to accept refinement")
    parser.add_argument('--eps-shooting', type=float, default=0.001, help="Initial displacement for shooting method")
    parser.add_argument('--eps-bvp', type=float, default=0.001, help="Displacement from start/end for BVP")

    # Integrator Params
    parser.add_argument('--n-trajs', type=int, default=10_000, help="Shooting: Trajs per pass")
    parser.add_argument('--t-max', type=float, default=1_000.0, help="Shooting: Max integration time")

    parser.add_argument('--bvp-nodes', type=int, default=400, help="BVP: Initial nodes")
    parser.add_argument('--bvp-tol', type=float, default=1e-4, help="BVP: Tolerance")
    parser.add_argument('--flipA', action='store_true', help="BVP: Flip origin vector to point away from target")
    parser.add_argument('--flipB', action='store_true', help="BVP: Flip target vector to point away from origin")

    # Execution Params
    parser.add_argument('--max-save-pts', type=int, default=5_000,
                        help="Downsample the found instanton to max-save-pts if it's longer than max-save-pts * (1 + max-save-pts-grace)")
    parser.add_argument('--max-save-pts-grace', type=float, default=0.2, help="Grace value to avoid near-meaningless downsampling")
    parser.add_argument('--silent', action='store_true', help="Suppress progress information")
    parser.add_argument('--skip-existing', action='store_true', help="Quit execution early if a result already exists, no matter the quality")
    parser.add_argument('--L-scale', type=float, default=1.0, help="Scale of initial momentum guess")
    exclusive_args_write = parser.add_mutually_exclusive_group()
    exclusive_args_write.add_argument('--do-not-write', action='store_true', help="Do not write anything to disk")
    exclusive_args_write.add_argument('--always-write', action='store_true', help="Write results to disk even if lower quality")
    exclusive_args_write.add_argument('--skip-quality', type=float, help="Quickly quit if old result has quality <= this")

    args = parser.parse_args()

    BASE_DIR = f'./results_{args.distribution}'

    if args.distribution == "H":
        args.distribution = "Husimi"
    elif args.distribution == "W":
        args.distribution = "Wigner"

    # Resolving and validating the parameters
    if args.gamma_rel is not None:
        s = np.sqrt(1 - (1 - args.Omega)**2)
        args.gamma = s + (1 - s)*args.gamma_rel
        if not args.silent:
            print(f"Rescaled gamma to {args.gamma:n}")
    if args.Omega_rel is not None:
        s = np.sqrt(1 - args.gamma**2)
        args.Omega = (1 - s)*args.Omega_rel
        if not args.silent:
            print(f"Rescaled Omega to {args.Omega:n}")

    dg = (1 - args.gamma**2)**0.5
    if abs(args.gamma) > 1:
        if not args.silent:
            raise NotImplementedError("Only -1 <= gamma <= 1 region is implemented.")
        else:
            return errno.ENOSYS  # Function not implemented
    if abs(args.Omega) > 1 - dg:
        if not args.silent:
            raise NotImplementedError(f"For gamma = {args.gamma:n}, |Omega| can't be higher than {1 - dg:n}")
        else:
            return errno.ENOSYS  # Function not implemented
    # End of parameter resolution, except args.method

    run_params = {
        'gamma': args.gamma, 'Omega': args.Omega,
        'origin_idx': args.origin, 'target_idx': args.target,
    }
    out_json_path,_,_ = output_file_paths(base_dir=BASE_DIR, run_params=run_params, do_not_write=args.do_not_write)
    if args.skip_existing and out_json_path.exists():
        if not args.silent:
            print(f"Skipping, result exists: {out_json_path}")
        return
    if args.skip_quality is not None and args.skip_quality > 0:
        if old_result_is_good_enough(BASE_DIR, run_params, args.skip_quality):
            if not args.silent:
                print(f"Skipping, old result good enough: {out_json_path}")
            return

    crit4 = find_critical_points(args.gamma, args.Omega, four_d=True)
    info = classify_critical_points(crit4, args.gamma, args.Omega, args.distribution)
    start_info, end_info = info[args.origin], info[args.target]

    # 2. Automatic Method Selection
    if args.method == 'auto':
        real_parts, imag_parts = start_info['sub_eig'].real, start_info['sub_eig'].imag
        spiral_factors = np.abs(imag_parts) / (np.abs(real_parts) + 1e-15)
        args.method = 'shooting' if np.any(spiral_factors > args.spiral_factor) else 'bvp'
    run_params['method'] = args.method

    if not args.silent:
        print(f"Params: Gamma={args.gamma}, Omega={args.Omega}")
        print(f"Routing {info[args.origin]['pt'][:2]} -> {info[args.target]['pt'][:2]} via {args.method.upper()}")

    best_quality = np.inf
    best_traj = None
    best_time = None

    # 3. Execution & Refinements
    if args.method == 'shooting':
        V_fast, V_slow = get_ordered_eigvecs(start_info, unstable=True)
        theta_min, theta_max = -0.5*np.pi, 1.5*np.pi
        if not args.silent:
            print(f"Slow eigvec: {V_slow}")
            print(f"Fast eigvec: {V_fast}")

        for p_idx in range(args.refinements + 1):
            thetas = np.linspace(theta_min, theta_max, args.n_trajs, endpoint=False)
            inits = np.array([start_info['pt'] + args.eps_shooting*(np.cos(th)*V_slow + np.sin(th)*V_fast) for th in thetas])

            b_idx, quality = integrate_batch_jit(
                    inits, args.gamma, args.Omega, end_info['pt'], args.t_max,
                    1e-5, 200_000, 50, 5, 3,
                    False, 1e-10, 1e-8, args.distribution,
                )

            if quality < best_quality * (1.0 - args.improve_fract) or p_idx == 0:
                best_quality = quality
                best_traj, best_time = integrate_single_path(
                        inits[b_idx], args.gamma, args.Omega,
                        args.t_max, 1e-5, 200_000, False, 1e-12, 1e-9,
                        5, 3, args.distribution,
                    )
                dist_to_tgt = np.linalg.norm(best_traj - end_info['pt'], axis=1)
                mindist_arg = np.argmin(dist_to_tgt)
                best_traj, best_time = best_traj[:mindist_arg], best_time[:mindist_arg]
                if not args.silent:
                    print(f"Quality after shot {p_idx+1}: {quality:.12f},  action: {evaluate_quality(best_traj):.12f},  from theta = {thetas[b_idx]/np.pi:.15f} pi")

                # Setup next refinement pass narrowing search space
                l_th, r_th = thetas[(b_idx - 1) % len(thetas)], thetas[(b_idx + 1) % len(thetas)]
                theta_min, theta_max = min(l_th, r_th), max(l_th, r_th)
                if (theta_max - theta_min) > np.pi:
                    theta_min, theta_max = theta_max, theta_min + 2*np.pi
            else:
                if not args.silent:
                    print(f"Quality after shot {p_idx+1}: {quality:.12f} did not improve enough, aborting.")
                break
    
    if args.method in {'bvp', 'shooting'}:
        if args.method == 'bvp':
            g_tau, g_y, g_p = None, None, None
            best_quality = np.inf
            old_action = None
        elif args.method == 'shooting':
            interp_traj, interp_time = trim_and_interpolate_trajectory(
                best_traj.T, start_info['pt'][:2], end_info['pt'][:2], drift_4d_numba,
                {'gamma': args.gamma, 'Omega': args.Omega, 'distribution': args.distribution}, eps1=args.eps_bvp, eps2=args.eps_bvp,
            )
            g_tau, g_y, g_p = interp_time/interp_time[-1], interp_traj, interp_time[-1:]
            old_action = evaluate_quality(best_traj)

        max_refined_nodes = 1000
        cur_tol = args.bvp_tol

        for p_idx in range(args.refinements + 1):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    res = find_heteroclinic_bvp(
                            start_info, end_info, args.gamma, args.Omega,
                            args.bvp_nodes, 30_000, cur_tol,
                            g_tau, g_y, g_p, verbose=not args.silent,
                            distribution=args.distribution, flipA=args.flipA, flipB=args.flipB, L_scale=args.L_scale, eps=args.eps_bvp,
                        )
                    quality = res.rms_residuals.max()
            except RuntimeError:
                if not args.silent:
                    print("BVP raised RuntimeError, aborting")
                break

            action = evaluate_quality(res.y.T)
            if old_action is not None and action < 1e-4 and action < 1e-2*old_action:
                if not args.silent:
                    print("BVP collapsed to zero momentum, aborting")
                break
            old_action = action
            
            if quality < best_quality * (1.0 - args.improve_fract) or (args.method == 'bvp' and p_idx == 0):
                best_quality = quality
                best_traj, best_time = res.y.T, res.x
                g_p = res.p
                if len(res.x) > max_refined_nodes:
                    g_tau, g_y = simplify_4d_path(res.x, res.y, max_refined_nodes, alpha=0.5, beta=0.5)
                else:
                    g_tau, g_y = res.x, res.y
                cur_tol /= 10
                if not args.silent:
                    print(f"Quality after BVP {p_idx+1}: {quality:.12f},  action: {evaluate_quality(best_traj):.12f}\n")
            else:
                if not args.silent:
                    print("BVP Failed convergence.")
                break

    # 4. Storage Evaluation
    if best_quality < args.quality_thresh and best_traj is not None:
        action = evaluate_quality(best_traj)
        if action < 1e-16 or action > 3:
            if not args.silent:
                print(f"Action {action:.4f} outside of expected range, aborting.")
            return 1
        if len(best_time) > args.max_save_pts * (1 + args.max_save_pts_grace):
            best_time, best_traj = simplify_4d_path(best_time, best_traj.T, args.max_save_pts, alpha=0, beta=1)
            best_traj = best_traj.T
        if not args.do_not_write:
            save_result_distributed(
                base_dir=BASE_DIR,
                run_params=run_params,
                action=action,
                quality=best_quality,
                ys=best_traj,
                ts=best_time,
                always_write=args.always_write,
                verbose=not args.silent,
            )
        if not args.silent:
            print(f"Success! Action: {action:.4f}, Quality: {best_quality:.2e}.")
    else:
        if not args.silent:
            print(f"Discarding result. Target Quality not met ({best_quality:.2e} > {args.quality_thresh})")


if __name__ == "__main__":
    main()
