import argparse
import json
import numpy as np
import warnings
from numba import njit, prange
from scipy.integrate import solve_bvp, trapezoid, cumulative_trapezoid
from scipy.linalg import eig, null_space
from pathlib import Path
from filelock import FileLock


# ==========================================
# 1. PURE NUMPY MATH & CRITICAL POINTS
# ==========================================


@njit
def crit_point_eqs_num(gamma, h_x, kappa):
    gamma, h_x, kappa = np.array((gamma, h_x, kappa), dtype=np.complex128)
    pts_complex = np.array((
        (gamma*((1/3)*gamma + (2/3)*kappa + (1/3)*((-2*gamma - 4*kappa + 1)**2 - 3*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)/((27/2)*gamma**2 + 54*gamma*kappa + 54*kappa**2 + (1/2)*np.sqrt(-4*((-2*gamma - 4*kappa + 1)**2 - 3*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**3 + (27*gamma**2 + 108*gamma*kappa + 108*kappa**2 + 2*(-2*gamma - 4*kappa + 1)**3 - (-18*gamma - 36*kappa + 9)*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**2) + (-2*gamma - 4*kappa + 1)**3 - 1/2*(-18*gamma - 36*kappa + 9)*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**(1/3) + (1/3)*((27/2)*gamma**2 + 54*gamma*kappa + 54*kappa**2 + (1/2)*np.sqrt(-4*((-2*gamma - 4*kappa + 1)**2 - 3*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**3 + (27*gamma**2 + 108*gamma*kappa + 108*kappa**2 + 2*(-2*gamma - 4*kappa + 1)**3 - (-18*gamma - 36*kappa + 9)*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**2) + (-2*gamma - 4*kappa + 1)**3 - 1/2*(-18*gamma - 36*kappa + 9)*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**(1/3) + 1/3)*((2/3)*gamma + (4/3)*kappa - 1/3*((-2*gamma - 4*kappa + 1)**2 - 3*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)/((27/2)*gamma**2 + 54*gamma*kappa + 54*kappa**2 + (1/2)*np.sqrt(-4*((-2*gamma - 4*kappa + 1)**2 - 3*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**3 + (27*gamma**2 + 108*gamma*kappa + 108*kappa**2 + 2*(-2*gamma - 4*kappa + 1)**3 - (-18*gamma - 36*kappa + 9)*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**2) + (-2*gamma - 4*kappa + 1)**3 - 1/2*(-18*gamma - 36*kappa + 9)*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**(1/3) - 1/3*((27/2)*gamma**2 + 54*gamma*kappa + 54*kappa**2 + (1/2)*np.sqrt(-4*((-2*gamma - 4*kappa + 1)**2 - 3*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**3 + (27*gamma**2 + 108*gamma*kappa + 108*kappa**2 + 2*(-2*gamma - 4*kappa + 1)**3 - (-18*gamma - 36*kappa + 9)*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**2) + (-2*gamma - 4*kappa + 1)**3 - 1/2*(-18*gamma - 36*kappa + 9)*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**(1/3) + 2/3)/(h_x*(gamma + 2*kappa)),
        (2/3)*gamma + (4/3)*kappa - 1/3*((-2*gamma - 4*kappa + 1)**2 - 3*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)/((27/2)*gamma**2 + 54*gamma*kappa + 54*kappa**2 + (1/2)*np.sqrt(-4*((-2*gamma - 4*kappa + 1)**2 - 3*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**3 + (27*gamma**2 + 108*gamma*kappa + 108*kappa**2 + 2*(-2*gamma - 4*kappa + 1)**3 - (-18*gamma - 36*kappa + 9)*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**2) + (-2*gamma - 4*kappa + 1)**3 - 1/2*(-18*gamma - 36*kappa + 9)*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**(1/3) - 1/3*((27/2)*gamma**2 + 54*gamma*kappa + 54*kappa**2 + (1/2)*np.sqrt(-4*((-2*gamma - 4*kappa + 1)**2 - 3*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**3 + (27*gamma**2 + 108*gamma*kappa + 108*kappa**2 + 2*(-2*gamma - 4*kappa + 1)**3 - (-18*gamma - 36*kappa + 9)*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**2) + (-2*gamma - 4*kappa + 1)**3 - 1/2*(-18*gamma - 36*kappa + 9)*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**(1/3) - 1/3),
        (gamma*((1/3)*gamma + (2/3)*kappa + (1/3)*((-2*gamma - 4*kappa + 1)**2 - 3*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)/((-1/2 - 1/2*np.sqrt(3)*1j)*((27/2)*gamma**2 + 54*gamma*kappa + 54*kappa**2 + (1/2)*np.sqrt(-4*((-2*gamma - 4*kappa + 1)**2 - 3*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**3 + (27*gamma**2 + 108*gamma*kappa + 108*kappa**2 + 2*(-2*gamma - 4*kappa + 1)**3 - (-18*gamma - 36*kappa + 9)*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**2) + (-2*gamma - 4*kappa + 1)**3 - 1/2*(-18*gamma - 36*kappa + 9)*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**(1/3)) + (1/3)*(-1/2 - 1/2*np.sqrt(3)*1j)*((27/2)*gamma**2 + 54*gamma*kappa + 54*kappa**2 + (1/2)*np.sqrt(-4*((-2*gamma - 4*kappa + 1)**2 - 3*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**3 + (27*gamma**2 + 108*gamma*kappa + 108*kappa**2 + 2*(-2*gamma - 4*kappa + 1)**3 - (-18*gamma - 36*kappa + 9)*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**2) + (-2*gamma - 4*kappa + 1)**3 - 1/2*(-18*gamma - 36*kappa + 9)*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**(1/3) + 1/3)*((2/3)*gamma + (4/3)*kappa - 1/3*((-2*gamma - 4*kappa + 1)**2 - 3*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)/((-1/2 - 1/2*np.sqrt(3)*1j)*((27/2)*gamma**2 + 54*gamma*kappa + 54*kappa**2 + (1/2)*np.sqrt(-4*((-2*gamma - 4*kappa + 1)**2 - 3*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**3 + (27*gamma**2 + 108*gamma*kappa + 108*kappa**2 + 2*(-2*gamma - 4*kappa + 1)**3 - (-18*gamma - 36*kappa + 9)*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**2) + (-2*gamma - 4*kappa + 1)**3 - 1/2*(-18*gamma - 36*kappa + 9)*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**(1/3)) - 1/3*(-1/2 - 1/2*np.sqrt(3)*1j)*((27/2)*gamma**2 + 54*gamma*kappa + 54*kappa**2 + (1/2)*np.sqrt(-4*((-2*gamma - 4*kappa + 1)**2 - 3*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**3 + (27*gamma**2 + 108*gamma*kappa + 108*kappa**2 + 2*(-2*gamma - 4*kappa + 1)**3 - (-18*gamma - 36*kappa + 9)*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**2) + (-2*gamma - 4*kappa + 1)**3 - 1/2*(-18*gamma - 36*kappa + 9)*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**(1/3) + 2/3)/(h_x*(gamma + 2*kappa)),
        (2/3)*gamma + (4/3)*kappa - 1/3*((-2*gamma - 4*kappa + 1)**2 - 3*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)/((-1/2 - 1/2*np.sqrt(3)*1j)*((27/2)*gamma**2 + 54*gamma*kappa + 54*kappa**2 + (1/2)*np.sqrt(-4*((-2*gamma - 4*kappa + 1)**2 - 3*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**3 + (27*gamma**2 + 108*gamma*kappa + 108*kappa**2 + 2*(-2*gamma - 4*kappa + 1)**3 - (-18*gamma - 36*kappa + 9)*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**2) + (-2*gamma - 4*kappa + 1)**3 - 1/2*(-18*gamma - 36*kappa + 9)*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**(1/3)) - 1/3*(-1/2 - 1/2*np.sqrt(3)*1j)*((27/2)*gamma**2 + 54*gamma*kappa + 54*kappa**2 + (1/2)*np.sqrt(-4*((-2*gamma - 4*kappa + 1)**2 - 3*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**3 + (27*gamma**2 + 108*gamma*kappa + 108*kappa**2 + 2*(-2*gamma - 4*kappa + 1)**3 - (-18*gamma - 36*kappa + 9)*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**2) + (-2*gamma - 4*kappa + 1)**3 - 1/2*(-18*gamma - 36*kappa + 9)*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**(1/3) - 1/3),
        (gamma*((1/3)*gamma + (2/3)*kappa + (1/3)*((-2*gamma - 4*kappa + 1)**2 - 3*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)/((-1/2 + (1/2)*np.sqrt(3)*1j)*((27/2)*gamma**2 + 54*gamma*kappa + 54*kappa**2 + (1/2)*np.sqrt(-4*((-2*gamma - 4*kappa + 1)**2 - 3*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**3 + (27*gamma**2 + 108*gamma*kappa + 108*kappa**2 + 2*(-2*gamma - 4*kappa + 1)**3 - (-18*gamma - 36*kappa + 9)*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**2) + (-2*gamma - 4*kappa + 1)**3 - 1/2*(-18*gamma - 36*kappa + 9)*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**(1/3)) + (1/3)*(-1/2 + (1/2)*np.sqrt(3)*1j)*((27/2)*gamma**2 + 54*gamma*kappa + 54*kappa**2 + (1/2)*np.sqrt(-4*((-2*gamma - 4*kappa + 1)**2 - 3*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**3 + (27*gamma**2 + 108*gamma*kappa + 108*kappa**2 + 2*(-2*gamma - 4*kappa + 1)**3 - (-18*gamma - 36*kappa + 9)*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**2) + (-2*gamma - 4*kappa + 1)**3 - 1/2*(-18*gamma - 36*kappa + 9)*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**(1/3) + 1/3)*((2/3)*gamma + (4/3)*kappa - 1/3*((-2*gamma - 4*kappa + 1)**2 - 3*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)/((-1/2 + (1/2)*np.sqrt(3)*1j)*((27/2)*gamma**2 + 54*gamma*kappa + 54*kappa**2 + (1/2)*np.sqrt(-4*((-2*gamma - 4*kappa + 1)**2 - 3*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**3 + (27*gamma**2 + 108*gamma*kappa + 108*kappa**2 + 2*(-2*gamma - 4*kappa + 1)**3 - (-18*gamma - 36*kappa + 9)*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**2) + (-2*gamma - 4*kappa + 1)**3 - 1/2*(-18*gamma - 36*kappa + 9)*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**(1/3)) - 1/3*(-1/2 + (1/2)*np.sqrt(3)*1j)*((27/2)*gamma**2 + 54*gamma*kappa + 54*kappa**2 + (1/2)*np.sqrt(-4*((-2*gamma - 4*kappa + 1)**2 - 3*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**3 + (27*gamma**2 + 108*gamma*kappa + 108*kappa**2 + 2*(-2*gamma - 4*kappa + 1)**3 - (-18*gamma - 36*kappa + 9)*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**2) + (-2*gamma - 4*kappa + 1)**3 - 1/2*(-18*gamma - 36*kappa + 9)*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**(1/3) + 2/3)/(h_x*(gamma + 2*kappa)),
        (2/3)*gamma + (4/3)*kappa - 1/3*((-2*gamma - 4*kappa + 1)**2 - 3*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)/((-1/2 + (1/2)*np.sqrt(3)*1j)*((27/2)*gamma**2 + 54*gamma*kappa + 54*kappa**2 + (1/2)*np.sqrt(-4*((-2*gamma - 4*kappa + 1)**2 - 3*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**3 + (27*gamma**2 + 108*gamma*kappa + 108*kappa**2 + 2*(-2*gamma - 4*kappa + 1)**3 - (-18*gamma - 36*kappa + 9)*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**2) + (-2*gamma - 4*kappa + 1)**3 - 1/2*(-18*gamma - 36*kappa + 9)*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**(1/3)) - 1/3*(-1/2 + (1/2)*np.sqrt(3)*1j)*((27/2)*gamma**2 + 54*gamma*kappa + 54*kappa**2 + (1/2)*np.sqrt(-4*((-2*gamma - 4*kappa + 1)**2 - 3*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**3 + (27*gamma**2 + 108*gamma*kappa + 108*kappa**2 + 2*(-2*gamma - 4*kappa + 1)**3 - (-18*gamma - 36*kappa + 9)*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**2) + (-2*gamma - 4*kappa + 1)**3 - 1/2*(-18*gamma - 36*kappa + 9)*(gamma**3 + 4*gamma**2*kappa - 2*gamma**2 + 2*gamma*h_x**2 + 4*gamma*kappa**2 - 4*gamma*kappa + 4*h_x**2*kappa)/gamma)**(1/3) - 1/3),
    ))
    near_zero_imag = np.abs(pts_complex.imag) < 1e-10
    return np.where(near_zero_imag, pts_complex.real, np.nan)


def jac2d_eigval_eqs_num(m_y, m_z, gamma, h_x, kappa):
    m_y, m_z, gamma, h_x, kappa = np.array((m_y, m_z, gamma, h_x, kappa), dtype=np.complex128)
    p0 = -3/4*gamma - 1/2*kappa + (1/4)*m_z
    p1 = (1/4)*np.sqrt(gamma**2 - 4*gamma*kappa + 2*gamma*m_z - 16*h_x**2 + 24*h_x*m_y + 4*kappa**2 - 4*kappa*m_z - 8*m_y**2 + m_z**2)
    return (p0 - p1, p0 + p1)


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


def find_critical_points(gamma, h_x, kappa, four_d=False):
    """Identifies and sorts unique real critical points."""
    pts = np.array(crit_point_eqs_num(gamma, h_x, kappa))
    if np.any(np.isnan(pts)):
        return None
    assert classify_pt(jac2d_eigval_eqs_num(*pts[0], gamma, h_x, kappa)) in {'na', 'sa'}
    assert classify_pt(jac2d_eigval_eqs_num(*pts[1], gamma, h_x, kappa)) == 's'
    assert classify_pt(jac2d_eigval_eqs_num(*pts[2], gamma, h_x, kappa)) in {'na', 'sa'}
    if four_d:
        return np.pad(pts, ((0,0),(0,2)))
    return pts


@njit
def hamiltonian_num(m_y, m_z, pi_y, pi_z, gamma, h_x, kappa, distribution):
    if distribution == "G":
        return -1/2*gamma*(m_y*pi_y + 2*pi_z*(m_z + 1)) + h_x*(m_y*pi_z - m_z*pi_y) - kappa*m_y*pi_y + (1/2)*m_y**2*pi_z*(pi_z*(m_z + 1) - 1) + (1/2)*m_y*pi_y*pi_z*(m_y**2 - (m_z + 1)**2) + (1/2)*m_z*pi_y*(m_y + pi_y*(-m_y**2 + m_z + 1))
    elif distribution == "N":
        return (-1/2*gamma*(m_y*pi_y*np.exp(2*pi_z) - (1 - np.exp(2*pi_z))*(m_z + 1)) - kappa*pi_y*(m_y*np.exp(2*pi_z) - pi_y*(m_z + 1)) + (1/4)*(-2*h_x*(m_y*pi_y**2 + m_y*(1 - np.exp(2*pi_z)) + 2*m_z*pi_y) + m_y**2*pi_y**2 + m_y**2*(1 - np.exp(2*pi_z)) + 2*m_y*m_z*pi_y)*np.exp(2*pi_z))*np.exp(-2*pi_z)
    elif distribution == "SW":
        return (1/2)*gamma*(m_y*pi_y*(2*pi_z - 1) + pi_y**2 + 2*pi_z*(m_z + 1)*(pi_z - 1)) + h_x*(m_y*pi_z - m_z*pi_y) - kappa*pi_y*(m_y - pi_y) + (1/2)*m_y**2*pi_z*(pi_z - 1) - 1/2*m_y*m_z*pi_y*(2*pi_z - 1) + (1/2)*m_z**2*pi_y**2
    else:
        raise ValueError("distribution must be 'G', 'N', or 'SW'")


@njit
def drift_4d_numba(m_y, m_z, pi_y, pi_z, gamma, h_x, kappa, backwards, distribution):
    if distribution == "G":
        q0,q1,p0,p1 = (
            -1/2*gamma*m_y - h_x*m_z - kappa*m_y + (1/2)*m_y**3*pi_z - m_y**2*m_z*pi_y + (1/2)*m_y*m_z - 1/2*m_y*pi_z*(m_z + 1)**2 + m_z**2*pi_y + m_z*pi_y,
            -gamma*m_z - gamma + h_x*m_y + (1/2)*m_y**3*pi_y + m_y**2*m_z*pi_z + m_y**2*pi_z - 1/2*m_y**2 - 1/2*m_y*pi_y*(m_z + 1)**2,
            (1/2)*gamma*pi_y - h_x*pi_z + kappa*pi_y - m_y**2*pi_y*pi_z - m_y*pi_z*(pi_z*(m_z + 1) - 1) + (1/2)*m_z*pi_y*(2*m_y*pi_y - 1) - 1/2*pi_y*pi_z*(m_y**2 - (m_z + 1)**2),
            gamma*pi_z + h_x*pi_y - 1/2*m_y**2*pi_z**2 + m_y*pi_y*pi_z*(m_z + 1) - 1/2*m_z*pi_y**2 - 1/2*pi_y*(m_y + pi_y*(-m_y**2 + m_z + 1))
        )
    elif distribution == "N":
        q0,q1,p0,p1 = (
            -1/2*gamma*m_y - h_x*m_y*pi_y - h_x*m_z - kappa*m_y + 2*kappa*m_z*pi_y*np.exp(-2*pi_z) + 2*kappa*pi_y*np.exp(-2*pi_z) + (1/2)*m_y**2*pi_y + (1/2)*m_y*m_z,
            (-gamma*m_z - gamma + h_x*m_y*np.exp(4*pi_z) - 2*kappa*m_z*pi_y**2 - 2*kappa*pi_y**2 - 1/2*m_y**2*np.exp(4*pi_z))*np.exp(-2*pi_z),
            (1/2)*gamma*pi_y + (1/2)*h_x*pi_y**2 - 1/2*h_x*np.exp(2*pi_z) + (1/2)*h_x + kappa*pi_y - 1/2*m_y*pi_y**2 + (1/2)*m_y*np.exp(2*pi_z) - 1/2*m_y - 1/2*m_z*pi_y,
            (1/2)*(-gamma - 2*kappa*pi_y**2 + (gamma + 2*h_x*pi_y - m_y*pi_y)*np.exp(2*pi_z))*np.exp(-2*pi_z),
        )
    elif distribution == "SW":
        q0,q1,p0,p1 = (
            gamma*m_y*pi_z - 1/2*gamma*m_y + gamma*pi_y - h_x*m_z - kappa*m_y + 2*kappa*pi_y - m_y*m_z*pi_z + (1/2)*m_y*m_z + m_z**2*pi_y,
            gamma*m_y*pi_y + 2*gamma*m_z*pi_z - gamma*m_z + 2*gamma*pi_z - gamma + h_x*m_y + m_y**2*pi_z - 1/2*m_y**2 - m_y*m_z*pi_y,
            -1/2*gamma*pi_y*(2*pi_z - 1) - h_x*pi_z + kappa*pi_y - m_y*pi_z*(pi_z - 1) + (1/2)*m_z*pi_y*(2*pi_z - 1),
            -gamma*pi_z*(pi_z - 1) + h_x*pi_y + (1/2)*m_y*pi_y*(2*pi_z - 1) - m_z*pi_y**2,
        )
    else:
        raise ValueError("distribution must be 'G', 'N', or 'SW'")
    
    if backwards:
        return -q0,-q1,-p0,-p1
    
    return q0,q1,p0,p1


@njit
def jacobian_4d_numba(m_y, m_z, pi_y, pi_z, gamma, h_x, kappa, distribution):
    J = np.empty((4, 4), dtype=np.float64)
    if distribution == "G":
        J[0, 0] = -1/2*gamma - kappa + (3/2)*m_y**2*pi_z - 2*m_y*m_z*pi_y + (1/2)*m_z - 1/2*pi_z*(m_z + 1)**2
        J[0, 1] = -h_x - m_y**2*pi_y - m_y*pi_z*(m_z + 1) + (1/2)*m_y + 2*m_z*pi_y + pi_y
        J[0, 2] = m_z*(-m_y**2 + m_z + 1)
        J[0, 3] = (1/2)*m_y*(m_y**2 - (m_z + 1)**2)
        J[1, 0] = h_x + (3/2)*m_y**2*pi_y + 2*m_y*m_z*pi_z + 2*m_y*pi_z - m_y - 1/2*pi_y*(m_z + 1)**2
        J[1, 1] = -gamma + m_y**2*pi_z - m_y*pi_y*(m_z + 1)
        J[1, 2] = (1/2)*m_y*(m_y**2 - (m_z + 1)**2)
        J[1, 3] = m_y**2*(m_z + 1)
        J[2, 0] = -3*m_y*pi_y*pi_z + m_z*pi_y**2 - pi_z*(pi_z*(m_z + 1) - 1)
        J[2, 1] = m_y*pi_y**2 - m_y*pi_z**2 + m_z*pi_y*pi_z + pi_y*pi_z - 1/2*pi_y
        J[2, 2] = (1/2)*gamma + kappa - 3/2*m_y**2*pi_z + 2*m_y*m_z*pi_y + (1/2)*m_z**2*pi_z + m_z*pi_z - 1/2*m_z + (1/2)*pi_z
        J[2, 3] = -h_x - m_y**2*pi_y - m_y*pi_z*(m_z + 1) - m_y*(pi_z*(m_z + 1) - 1) - 1/2*pi_y*(m_y**2 - (m_z + 1)**2)
        J[3, 0] = m_y*pi_y**2 - m_y*pi_z**2 + m_z*pi_y*pi_z + pi_y*pi_z - 1/2*pi_y
        J[3, 1] = pi_y*(m_y*pi_z - pi_y)
        J[3, 2] = h_x + m_y*pi_z*(m_z + 1) - 1/2*m_y - m_z*pi_y - pi_y*(-m_y**2 + m_z + 1)
        J[3, 3] = gamma - m_y**2*pi_z + m_y*pi_y*(m_z + 1)
    elif distribution == "N":
        J[0, 0] = -1/2*gamma - h_x*pi_y - kappa + m_y*pi_y + (1/2)*m_z
        J[0, 1] = -h_x + 2*kappa*pi_y*np.exp(-2*pi_z) + (1/2)*m_y
        J[0, 2] = (1/2)*(4*kappa*m_z + 4*kappa + m_y*(-2*h_x + m_y)*np.exp(2*pi_z))*np.exp(-2*pi_z)
        J[0, 3] = 4*kappa*pi_y*(-m_z - 1)*np.exp(-2*pi_z)
        J[1, 0] = (h_x - m_y)*np.exp(2*pi_z)
        J[1, 1] = (-gamma - 2*kappa*pi_y**2)*np.exp(-2*pi_z)
        J[1, 2] = 4*kappa*pi_y*(-m_z - 1)*np.exp(-2*pi_z)
        J[1, 3] = (2*gamma*m_z + 2*gamma + 2*h_x*m_y*np.exp(4*pi_z) + 4*kappa*m_z*pi_y**2 + 4*kappa*pi_y**2 - m_y**2*np.exp(4*pi_z))*np.exp(-2*pi_z)
        J[2, 0] = -1/2*pi_y**2 + (1/2)*np.exp(2*pi_z) - 1/2
        J[2, 1] = -1/2*pi_y
        J[2, 2] = (1/2)*gamma + h_x*pi_y + kappa - m_y*pi_y - 1/2*m_z
        J[2, 3] = (-h_x + m_y)*np.exp(2*pi_z)
        J[3, 0] = -1/2*pi_y
        J[3, 1] = 0.0
        J[3, 2] = h_x - 2*kappa*pi_y*np.exp(-2*pi_z) - 1/2*m_y
        J[3, 3] = (gamma + 2*kappa*pi_y**2)*np.exp(-2*pi_z)
    elif distribution == "SW":
        J[0, 0] = gamma*pi_z - 1/2*gamma - kappa - m_z*pi_z + (1/2)*m_z
        J[0, 1] = -h_x - m_y*pi_z + (1/2)*m_y + 2*m_z*pi_y
        J[0, 2] = gamma + 2*kappa + m_z**2
        J[0, 3] = m_y*(gamma - m_z)
        J[1, 0] = gamma*pi_y + h_x + 2*m_y*pi_z - m_y - m_z*pi_y
        J[1, 1] = 2*gamma*pi_z - gamma - m_y*pi_y
        J[1, 2] = m_y*(gamma - m_z)
        J[1, 3] = 2*gamma*m_z + 2*gamma + m_y**2
        J[2, 0] = pi_z*(1 - pi_z)
        J[2, 1] = pi_y*(pi_z - 1/2)
        J[2, 2] = -gamma*pi_z + (1/2)*gamma + kappa + m_z*pi_z - 1/2*m_z
        J[2, 3] = -gamma*pi_y - h_x - 2*m_y*pi_z + m_y + m_z*pi_y
        J[3, 0] = pi_y*(pi_z - 1/2)
        J[3, 1] = -pi_y**2
        J[3, 2] = h_x + m_y*pi_z - 1/2*m_y - 2*m_z*pi_y
        J[3, 3] = -2*gamma*pi_z + gamma + m_y*pi_y
    else:
        raise ValueError("distribution must be 'G', 'N', or 'SW'")
    return J


def jacobian_4d(state, gamma, h_x, kappa, distribution):
    """
    Drop-in analytical replacement for the numerical jacobian implementation.
    """
    return jacobian_4d_numba(*state, gamma, h_x, kappa, distribution)


def classify_critical_points(crit4, gamma, h_x, kappa, distribution):
    """Calculates eigendecomposition and identifies sub-spaces."""
    info = []
    for i, p in enumerate(crit4):
        jac4d = jacobian_4d(p, gamma, h_x, kappa, distribution)
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
def dopri5_step(y0, y1, y2, y3, h, k1_0, k1_1, k1_2, k1_3, gamma, h_x, kappa, backwards, atol, rtol, distribution):
    """Abstracted DOPRI5 integration step."""
    y_tmp_0 = y0 + h*A21*k1_0
    y_tmp_1 = y1 + h*A21*k1_1
    y_tmp_2 = y2 + h*A21*k1_2
    y_tmp_3 = y3 + h*A21*k1_3
    k2_0, k2_1, k2_2, k2_3 = drift_4d_numba(y_tmp_0, y_tmp_1, y_tmp_2, y_tmp_3, gamma, h_x, kappa, backwards, distribution)

    y_tmp_0 = y0 + h*(A31*k1_0 + A32*k2_0)
    y_tmp_1 = y1 + h*(A31*k1_1 + A32*k2_1)
    y_tmp_2 = y2 + h*(A31*k1_2 + A32*k2_2)
    y_tmp_3 = y3 + h*(A31*k1_3 + A32*k2_3)
    k3_0, k3_1, k3_2, k3_3 = drift_4d_numba(y_tmp_0, y_tmp_1, y_tmp_2, y_tmp_3, gamma, h_x, kappa, backwards, distribution)

    y_tmp_0 = y0 + h*(A41*k1_0 + A42*k2_0 + A43*k3_0)
    y_tmp_1 = y1 + h*(A41*k1_1 + A42*k2_1 + A43*k3_1)
    y_tmp_2 = y2 + h*(A41*k1_2 + A42*k2_2 + A43*k3_2)
    y_tmp_3 = y3 + h*(A41*k1_3 + A42*k2_3 + A43*k3_3)
    k4_0, k4_1, k4_2, k4_3 = drift_4d_numba(y_tmp_0, y_tmp_1, y_tmp_2, y_tmp_3, gamma, h_x, kappa, backwards, distribution)

    y_tmp_0 = y0 + h*(A51*k1_0 + A52*k2_0 + A53*k3_0 + A54*k4_0)
    y_tmp_1 = y1 + h*(A51*k1_1 + A52*k2_1 + A53*k3_1 + A54*k4_1)
    y_tmp_2 = y2 + h*(A51*k1_2 + A52*k2_2 + A53*k3_2 + A54*k4_2)
    y_tmp_3 = y3 + h*(A51*k1_3 + A52*k2_3 + A53*k3_3 + A54*k4_3)
    k5_0, k5_1, k5_2, k5_3 = drift_4d_numba(y_tmp_0, y_tmp_1, y_tmp_2, y_tmp_3, gamma, h_x, kappa, backwards, distribution)

    y_tmp_0 = y0 + h*(A61*k1_0 + A62*k2_0 + A63*k3_0 + A64*k4_0 + A65*k5_0)
    y_tmp_1 = y1 + h*(A61*k1_1 + A62*k2_1 + A63*k3_1 + A64*k4_1 + A65*k5_1)
    y_tmp_2 = y2 + h*(A61*k1_2 + A62*k2_2 + A63*k3_2 + A64*k4_2 + A65*k5_2)
    y_tmp_3 = y3 + h*(A61*k1_3 + A62*k2_3 + A63*k3_3 + A64*k4_3 + A65*k5_3)
    k6_0, k6_1, k6_2, k6_3 = drift_4d_numba(y_tmp_0, y_tmp_1, y_tmp_2, y_tmp_3, gamma, h_x, kappa, backwards, distribution)

    y_new_0 = y0 + h*(B1*k1_0 + B3*k3_0 + B4*k4_0 + B5*k5_0 + B6*k6_0)
    y_new_1 = y1 + h*(B1*k1_1 + B3*k3_1 + B4*k4_1 + B5*k5_1 + B6*k6_1)
    y_new_2 = y2 + h*(B1*k1_2 + B3*k3_2 + B4*k4_2 + B5*k5_2 + B6*k6_2)
    y_new_3 = y3 + h*(B1*k1_3 + B3*k3_3 + B4*k4_3 + B5*k5_3 + B6*k6_3)

    k7_0, k7_1, k7_2, k7_3 = drift_4d_numba(y_new_0, y_new_1, y_new_2, y_new_3, gamma, h_x, kappa, backwards, distribution)

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
def integrate_batch_jit(y0s, gamma, h_x, kappa, tgt, t_max, h0, max_steps, check_every, r_escape, p_escape, backwards, atol, rtol, distribution):
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
        k1_0, k1_1, k1_2, k1_3 = drift_4d_numba(y0, y1, y2, y3, gamma, h_x, kappa, backwards, distribution)

        while step < max_steps and active and t < t_max:
            if t + h > t_max: h = t_max - t

            yn_0, yn_1, yn_2, yn_3, k7_0, k7_1, k7_2, k7_3, err_max = dopri5_step(
                y0, y1, y2, y3, h, k1_0, k1_1, k1_2, k1_3, gamma, h_x, kappa, backwards, atol, rtol, distribution
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
def integrate_single_path(y_init, gamma, h_x, kappa, t_max, h0, max_steps, backwards, atol, rtol, r_escape, p_escape, distribution):
    """Dense integrator for reconstructing the final single best path."""
    r_escape2, p_escape2 = r_escape**2, p_escape**2
    ys = np.zeros((max_steps, 4))
    ts = np.zeros(max_steps)
    y0, y1, y2, y3 = y_init[0], y_init[1], y_init[2], y_init[3]
    ys[0] = y_init
    count, t, h, step = 1, 0.0, h0, 0
    active = True
    k1_0, k1_1, k1_2, k1_3 = drift_4d_numba(y0, y1, y2, y3, gamma, h_x, kappa, backwards, distribution)

    while step < max_steps and active and t < t_max:
        if t + h > t_max: h = t_max - t
        yn_0, yn_1, yn_2, yn_3, k7_0, k7_1, k7_2, k7_3, err_max = dopri5_step(
            y0, y1, y2, y3, h, k1_0, k1_1, k1_2, k1_3, gamma, h_x, kappa, backwards, atol, rtol, distribution
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
def bvp_fun_jac_numba(y, p, gamma, h_x, kappa, distribution):
    """
    Computes df/dy and df/dp analytically across the entire BVP mesh.
    """
    T = p[0]
    m = y.shape[1]
    df_dy = np.zeros((4, 4, m), dtype=np.float64)
    df_dp = np.zeros((4, 1, m), dtype=np.float64)
    
    for i in range(m):
        # The jacobian of T * drift with respect to y is T * jac_4d
        df_dy[:, :, i] = T * jacobian_4d_numba(y[0, i], y[1, i], y[2, i], y[3, i], gamma, h_x, kappa, distribution)
        
        # The derivative of T * drift with respect to the parameter T is just the drift
        dv, dw, dpv, dpw = drift_4d_numba(y[0, i], y[1, i], y[2, i], y[3, i], gamma, h_x, kappa, False, distribution)
        df_dp[0, 0, i] = dv
        df_dp[1, 0, i] = dw
        df_dp[2, 0, i] = dpv
        df_dp[3, 0, i] = dpw
        
    return df_dy, df_dp


def eigenspace_basis(info, kind, representation='basis', tol=1e-10):
    eig = np.asarray(info['eig'])
    vecs = np.asarray(info['vecs'])

    if kind == 'stable':
        idx = np.argsort(eig.real)[:2]
    elif kind == 'unstable':
        idx = np.argsort(eig.real)[::-1][:2]
    else:
        raise ValueError("kind must be 'stable' or 'unstable'")

    lam1, lam2 = eig[idx]
    vec1, vec2 = vecs[:, idx[0]], vecs[:, idx[1]]

    # Convert a complex-conjugate pair into a real basis.
    if abs(lam1.imag) > tol and abs(lam1 - np.conj(lam2)) < 100 * tol:
        basis = np.column_stack((vec1.real, vec1.imag))
    else:
        basis = np.column_stack((vec1.real, vec2.real))

    if representation == 'basis':
        result = basis
    elif representation == 'orthonormal':
        result = np.linalg.qr(basis, mode='reduced')[0]
    elif representation == 'complement':
        result = null_space(basis.T)
    else:
        raise ValueError(
            "representation must be 'basis', 'orthonormal', or 'complement'"
        )

    return [result[:, i] for i in range(result.shape[1])]


def _compute_S_matrix(info, kind, p_matrix=False, tol=1e-10):
    basis = eigenspace_basis(info, kind, tol=tol)
    assert len(basis) == 2

    V = np.column_stack(basis)
    X = V[:2, :]
    P = V[2:, :]

    if p_matrix:
        if np.linalg.matrix_rank(P, tol=tol) < 2:
            raise RuntimeError("Singular P matrix in BVP")
        S = np.linalg.solve(P.T, X.T).T     # dq = S_p dp
    else:
        if np.linalg.matrix_rank(X, tol=tol) < 2:
            raise RuntimeError("Singular X matrix in BVP")
        S = np.linalg.solve(X.T, P.T).T     # dp = S dq

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
    
    Assumes `params` is a tuple (e.g., `(gamma, h_x, kappa, distribution)`) that unpacks 
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
    dp = np.pad(np.diff(final_guess, axis=1), ((0,0),(1,0)))
    ds = np.linalg.norm(dp, axis=0)
    s_array = cumulative_trapezoid(ds, initial=0)
    
    # Integrate dt = ds / v to get strictly increasing times
    time_array = cumulative_trapezoid(inv_speeds, x=s_array, initial=0.0)
    
    return final_guess, time_array


def make_4d_spline_f(source, target, n_samples=10000, zero_tol=1e-9, flip_s=False, flip_t=False, fast_origin=False):
    """
    Build a smooth 4D trajectory from `source` to `target`.

    Both `source` and `target` are dicts with keys:
        'pt'   : (4,) point
        'eig'  : (4,) eigenvalues
        'vecs' : (4, 4) eigenvectors in columns

    Returns
    -------
    traj : callable
        traj(s) where s is a relative arc-length value in [0, 1].
        s=0 -> source, s=1 -> target.
    """
    S = np.asarray(source['pt'], dtype=float)
    T = np.asarray(target['pt'], dtype=float)
    chord = T - S
    L = np.linalg.norm(chord)

    if L < 1e-12:
        return lambda s: np.tile(S, (np.size(s), 1)) if np.ndim(s) else S.copy()

    def real_vecs(info):
        return np.real(info['vecs']), np.real(info['eig'])

    # ---------------- source eigenvector ----------------
    # slow unstable manifold: eigval > 0, v[2:] not near zero, slowest = smallest positive
    vecs_s, eig_s = real_vecs(source)
    mask_s = (eig_s > 0) & (np.linalg.norm(vecs_s[2:, :], axis=0) > zero_tol)
    idx_s = np.where(mask_s)[0]
    if idx_s.size == 0:
        raise ValueError("No suitable source eigenvector found.")

    if fast_origin:
        i_s = idx_s[np.argmax(eig_s[idx_s])]
    else:
        i_s = idx_s[np.argmin(eig_s[idx_s])]
    v_s = vecs_s[:, i_s].copy()

    # orient so increasing parameter moves source -> target
    if bool(np.dot(v_s, chord) < 0) ^ bool(flip_s):
        v_s = -v_s
    v_s /= np.linalg.norm(v_s)

    # ---------------- target eigenvector ----------------
    # stable manifold: eigval < 0, v[2:] not near zero
    vecs_t, eig_t = real_vecs(target)
    mask_t = (eig_t < 0) & (np.linalg.norm(vecs_t[2:, :], axis=0) > zero_tol)
    idx_t = np.where(mask_t)[0]
    if idx_t.size == 0:
        raise ValueError("No suitable target eigenvector found.")

    # if multiple, pick slowest stable = closest to zero
    i_t = idx_t[np.argmax(eig_t[idx_t])]
    v_t = vecs_t[:, i_t].copy()

    # orient so increasing parameter moves source -> target
    if bool(np.dot(v_t, chord) < 0) ^ bool(flip_t):
        v_t = -v_t
    v_t /= np.linalg.norm(v_t)

    # ---------------- cubic Hermite tangents ----------------
    # Scale by chord length so that after arc-length reparametrization
    # the endpoint derivatives are unit vectors along v_s and v_t.
    m0 = L * v_s
    m1 = L * v_t

    def hermite(u):
        u = np.asarray(u, dtype=float)
        u2 = u * u
        u3 = u2 * u

        h00 = 2 * u3 - 3 * u2 + 1
        h10 = u3 - 2 * u2 + u
        h01 = -2 * u3 + 3 * u2
        h11 = u3 - u2

        if u.ndim == 0:
            return h00 * S + h10 * m0 + h01 * T + h11 * m1
        return (
              h00[:, None] * S
            + h10[:, None] * m0
            + h01[:, None] * T
            + h11[:, None] * m1
        )

    # ---------------- arc-length reparametrization ----------------
    u_grid = np.linspace(0.0, 1.0, n_samples)
    pts = hermite(u_grid)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]

    if total < 1e-12:
        return lambda s: np.tile(S, (np.size(s), 1)) if np.ndim(s) else S.copy()

    s_grid = cum / total

    # remove any duplicate arc-length grid points just in case
    keep = np.concatenate([[True], np.diff(s_grid) > 0])
    s_grid = s_grid[keep]
    u_grid = u_grid[keep]

    def traj(s):
        s_arr = np.asarray(s, dtype=float)
        s_clip = np.clip(s_arr, 0.0, 1.0)
        u = np.interp(s_clip, s_grid, u_grid)
        return hermite(u)

    return traj


def make_timed_4d_spline(source, target, epsilon, gamma, h_x, kappa, distribution, n_samples=2000, flip_s=False, flip_t=False, fast_origin=False):
    """
    Build a trimmed 4D trajectory between two saddles, estimate the
    traversal time, and return a normalized cumulative time array.

    Parameters
    ----------
    source, target : dict
        Same dicts used by `make_traj` (contain 'pt', 'eig', 'vecs').
    epsilon : float
        Absolute 4D distance to trim from each end.
    gamma, h_x, kappa, distribution :
        Passed through to `drift_4d_numba`.
    n_samples : int
        Number of samples along the trimmed trajectory.

    Returns
    -------
    traj_4xN : ndarray, shape (4, N)
        Points along the curve. First column is ~epsilon away from source,
        last column is ~epsilon away from target (measured in 4D arc length).
    time_estimate : float
        Approximate total traversal time (last entry of the cumulative
        time integral).
    t_norm : ndarray, shape (N,)
        Cumulative normalized time, from exactly 0 at the first point to
        exactly 1 at the last point.
    """
    # ---- 1. Full Hermite curve, reparametrized by arc length ----
    curve_f = make_4d_spline_f(source, target, n_samples=n_samples, flip_s=flip_s, flip_t=flip_t, fast_origin=fast_origin)
    s = np.linspace(0.0, 1.0, n_samples)
    pts = curve_f(s)                                    # (N, 4)

    # ---- 2. 4D arc length along the full curve ----
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]

    # ---- 3. Trim `epsilon` off each end (in 4D arc length) ----
    s_lo = float(np.interp(epsilon, cum, s))
    s_hi = float(np.interp(total - epsilon, cum, s))
    if s_hi <= s_lo:            # degenerate case; fall back to middle half
        s_lo, s_hi = 0.25, 0.75

    s_trim = np.linspace(s_lo, s_hi, n_samples)
    pts_trim = curve_f(s_trim)                          # (N, 4)

    # ---- 4. Arc length along the trimmed curve ----
    seg_trim = np.linalg.norm(np.diff(pts_trim, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_trim)])

    # ---- 5. Evaluate drift and integrate time along the curve ----
    speeds = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        y = pts_trim[i]
        v = drift_4d_numba(y[0], y[1], y[2], y[3], gamma, h_x, kappa, False, distribution)
        speeds[i] = np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3])

    inv_speed = 1.0 / np.maximum(speeds, 1e-12)

    # cumulative_trapezoid returns array of length N-1; prepend 0
    cum_time = np.concatenate([[0.0], cumulative_trapezoid(inv_speed, arc)])
    time_estimate = float(cum_time[-1])

    # ---- 6. Normalized cumulative time (exactly 0 -> exactly 1) ----
    if time_estimate > 0:
        t_norm = cum_time / time_estimate
    else:
        t_norm = np.linspace(0.0, 1.0, n_samples)
    t_norm[0] = 0.0
    t_norm[-1] = 1.0

    return pts_trim.T, np.array([time_estimate,]), t_norm


def find_heteroclinic_bvp(
        infoA, infoB, gamma, h_x, kappa, nodes, max_nodes, tol,
        guess_tau=None, guess_y=None, guess_p=None,
        yA_target=None, yB_target=None,
        bc_mode='manifold_nl',              # 'manifold_nl' | 'manifold_P' | 'projection' | 'pinned'
        verbose=False, distribution=None,
        use_fun_jac=True, use_bc_jac=True,
        flipA=False, flipB=False, fastOrigin=False, L_scale=1.0, eps=1e-4,
    ):
    """
    ODE: dy/dtau = T * f(y), tau in [0,1], p = [T] free. 4 states + 1 par = 5 BCs.

    bc_mode 'manifold_nl' : Linear manifold BCs scaled dynamically to enforce H=0 exactly.
                            Best for larger eps values. Universal for nodes/spirals.
    bc_mode 'projection'  : (ya - A) perp stable eigenspace of A.
    bc_mode 'pinned'      : Fixed endpoints yA_target, yB_target.
    """
    print(f"BC MODE: {bc_mode}")
    assert distribution is not None
    ptA = np.asarray(infoA['pt'], float)
    ptB = np.asarray(infoB['pt'], float)
    vecAB = ptB - ptA

    if bc_mode == 'manifold_nl':
        S_A = _compute_S_matrix(infoA, 'unstable')
    elif bc_mode == 'manifold_P':
        S_Ap = _compute_S_matrix(infoA, 'unstable', p_matrix=True)

    S_B = _compute_S_matrix(infoB, 'stable')

    if (bc_mode == 'projection' or bc_mode is None) and (yA_target is None or yB_target is None):
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
        dv, dw, dpv, dpw = drift_4d_numba(y[0], y[1], y[2], y[3], gamma, h_x, kappa, False, distribution)
        return T * np.vstack((dv, dw, dpv, dpw))

    def fun_jac(x, y, p):
        return bvp_fun_jac_numba(y, p, gamma, h_x, kappa, distribution)
    
    def get_alpha(q, p_dir):
        # f(q) evaluated at p=0
        f_val = np.array(drift_4d_numba(q[0], q[1], 0.0, 0.0, gamma, h_x, kappa, False, distribution))[:2]
        
        # Extract D(q) from 4D Jacobian evaluated at p=0
        J4D = jacobian_4d_numba(q[0], q[1], 0.0, 0.0, gamma, h_x, kappa, distribution)
        D = J4D[0:2, 2:4] / 2.0
        
        den = np.dot(p_dir, D @ p_dir)
        if abs(den) < 1e-16: 
            return 1.0
        return -np.dot(p_dir, f_val) / den

    def get_alpha_p(q_star, p, q_dir, alpha0=1.0):
        """
        Find alpha such that
            H(q_star + alpha*q_dir, p) = 0
        using Newton iterations.

        dH/dalpha = -q_dir · p_dot
        from Hamilton's equations.
        """
        alpha = alpha0

        for _ in range(6):
            q = q_star + alpha * q_dir

            H = hamiltonian_num(q[0], q[1], p[0], p[1], gamma, h_x, kappa, distribution)

            dp = np.asarray(
                drift_4d_numba(
                    q[0], q[1], p[0], p[1],
                    gamma, h_x, kappa, False, distribution
                )
            )[2:]

            dH = -np.dot(q_dir, dp)

            if abs(dH) < 1e-14:
                break

            delta = H / dH
            alpha -= delta

            if abs(delta) < 1e-10 * max(1.0, abs(alpha)):
                break

        return alpha

    # ------------------- BOUNDARY CONDITIONS SETUP -------------------
    if bc_mode == 'manifold_nl':
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

            return dbc_dya, dbc_dyb, dbc_dp

    elif bc_mode == 'manifold_P':
        def bc(ya, yb, _):
            dA_p = ya[2:] - ptA[2:]
            dB_q = yb[:2] - ptB[:2]

            qA_dir = S_Ap @ dA_p
            pB_dir = S_B @ dB_q

            alpha_A = get_alpha_p(ptA[:2], ya[2:], qA_dir)
            alpha_B = get_alpha(yb[:2], pB_dir)

            return np.array([
                np.linalg.norm(dA_p) - eps,       # epsilon in p-space
                ya[0] - (ptA[0] + alpha_A * qA_dir[0]),
                ya[1] - (ptA[1] + alpha_A * qA_dir[1]),
                yb[2] - alpha_B * pB_dir[0],
                yb[3] - alpha_B * pB_dir[1],
            ])
        
        def bc_jac(ya, yb, p):
            dbc_dya = np.zeros((5, 4))
            dbc_dyb = np.zeros((5, 4))
            dbc_dp = np.zeros((5, 1))

            r0 = bc(ya, yb, p)

            for i in range(4):
                h = 1e-7 * max(1.0, abs(ya[i]))

                ya_p = ya.copy()
                ya_m = ya.copy()
                ya_p[i] += h
                ya_m[i] -= h

                dbc_dya[:, i] = (
                    bc(ya_p, yb, p) -
                    bc(ya_m, yb, p)
                ) / (2.0 * h)

            for i in range(4):
                h = 1e-7 * max(1.0, abs(yb[i]))

                yb_p = yb.copy()
                yb_m = yb.copy()
                yb_p[i] += h
                yb_m[i] -= h

                dbc_dyb[:, i] = (
                    bc(ya, yb_p, p) -
                    bc(ya, yb_m, p)
                ) / (2.0 * h)

            return dbc_dya, dbc_dyb, dbc_dp

    elif bc_mode == 'projection':
        S_A_subspace = eigenspace_basis(infoA, 'stable', 'complement')      # ya on unstable manifold of A
        U_B_subspace = eigenspace_basis(infoB, 'unstable', 'complement')    # yb on stable manifold of B
        assert len(S_A_subspace) == 2 and len(U_B_subspace) == 2, "expected 2D subspaces"

        def bc(ya, yb, _):
            d_ptA = ya - ptA
            d_ptB = yb - ptB
            return np.array(
                  [np.dot(v, d_ptA) for v in S_A_subspace]
                + [np.linalg.norm(d_ptA) - eps]
                + [np.dot(v, d_ptB) for v in U_B_subspace]
            )

        def bc_jac(ya, yb, p):
            raise NotImplementedError

    else:  # 'pinned'
        def bc(ya, yb, _):
            H_start = hamiltonian_num(ya[0], ya[1], ya[2], ya[3], gamma, h_x, kappa, distribution)
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
            H0 = hamiltonian_num(ya[0], ya[1], ya[2], ya[3], gamma, h_x, kappa, distribution)
            for i in range(4):
                ya_eps = ya.copy()
                ya_eps[i] += eps_fd
                H1 = hamiltonian_num(ya_eps[0], ya_eps[1], ya_eps[2], ya_eps[3],
                                     gamma, h_x, kappa, distribution)
                dbc_dya[4, i] = (H1 - H0) / eps_fd
            return dbc_dya, dbc_dyb, dbc_dp

    # ------------------- INITIAL GUESS GENERATION -------------------
    if guess_y is None:
        guess_y, guess_p, guess_tau = make_timed_4d_spline(infoA, infoB, eps, gamma, h_x, kappa, distribution, nodes, flipA, flipB, fastOrigin)

    res = solve_bvp(
        fun=fun, bc=bc, x=guess_tau, y=guess_y, p=guess_p,
        tol=tol, max_nodes=max_nodes,
        verbose=2 if verbose else 0,
        fun_jac=fun_jac if use_fun_jac else None,
        bc_jac=bc_jac if use_bc_jac else None,
    )

    if verbose:
        yb = res.y[:, -1]
        Hb = hamiltonian_num(yb[0], yb[1], yb[2], yb[3], gamma, h_x, kappa, distribution)
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
    g, hx, k = run_params['gamma'], run_params['h_x'], run_params['kappa']
    orig, tgt = run_params['origin_idx'], run_params['target_idx']
    base_name = f"g{g:.9f}_hx{hx:.9f}_k{k:.9f}_o{orig}_t{tgt}"

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
    parser.add_argument('--gamma', type=float, default=0.01, help="Gamma parameter")
    parser.add_argument('--h-x', type=float, default=0.3, help="h_x parameter")
    parser.add_argument('--kappa', type=float, default=0.0, help="kappa parameter")
    parser.add_argument('--origin', type=int, default=0, choices=[0, 2], help="Origin critical point index")
    # parser.add_argument('--target', type=int, default=1, help="Target critical point index")  # if there is more than one saddle
    parser.add_argument('--method', type=str, choices=['bvp', 'shooting', 'auto'], default='auto')
    parser.add_argument('--distribution', type=str, choices=["G", "N", "SW"], default="N")

    parser.add_argument('--spiral-factor', type=float, default=0.5, help="Ratio Im(L)/Re(L) to trigger shooting method")
    parser.add_argument('--quality-thresh', type=float, default=0.1, help="Discard thresholds for quality")
    parser.add_argument('--refinements', type=int, default=3, help="Number of refinement passes")
    parser.add_argument('--improve-fract', type=float, default=0.001, help="Min fraction improvement to accept refinement")
    parser.add_argument('--eps-shooting', type=float, default=0.001, help="Initial displacement for shooting method")
    parser.add_argument('--eps-bvp', type=float, default=0.001, help="Displacement from start/end for BVP")

    # Integrator Params
    parser.add_argument('--n-trajs', type=int, default=10_000, help="Shooting: Trajs per pass")
    parser.add_argument('--t-max', type=float, default=1_000.0, help="Shooting: Max integration time")

    parser.add_argument('--p-guess', type=float, default=50.0, help="BVP time guess")
    parser.add_argument('--bvp-nodes', type=int, default=400, help="BVP: Initial nodes")
    parser.add_argument('--bvp-tol', type=float, default=1e-4, help="BVP: Tolerance")
    parser.add_argument('--flipA', action='store_true', help="BVP: Flip origin vector to point away from target")
    parser.add_argument('--flipB', action='store_true', help="BVP: Flip target vector to point away from origin")
    parser.add_argument('--fast-origin', action='store_true', help="BVP: start from the fast manifold (slow manifold is default)")

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
    args.target = 1  # the only saddle in the system

    BASE_DIR = f'./results_CRF_{args.distribution}'

    # End of parameter resolution, except args.method

    run_params = {
        'gamma': args.gamma, 'h_x': args.h_x, 'kappa': args.kappa,
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

    crit4 = find_critical_points(args.gamma, args.h_x, args.kappa, four_d=True)
    if crit4 is None:
        if not args.silent:
            print(f"Skipping, no multistability found: {out_json_path}")
        return
    info = classify_critical_points(crit4, args.gamma, args.h_x, args.kappa, args.distribution)
    start_info, end_info = info[args.origin], info[args.target]

    # 2. Automatic Method Selection
    if args.method == 'auto':
        real_parts, imag_parts = start_info['sub_eig'].real, start_info['sub_eig'].imag
        spiral_factors = np.abs(imag_parts) / (np.abs(real_parts) + 1e-15)
        args.method = 'shooting' if np.any(spiral_factors > args.spiral_factor) else 'bvp'
    run_params['method'] = args.method

    if not args.silent:
        print(f"Params: Gamma={args.gamma:.9n}, h_x={args.h_x:.9n}, kappa={args.kappa:.9n}")
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
                    inits, args.gamma, args.h_x, args.kappa, end_info['pt'], args.t_max,
                    1e-5, 200_000, 50, 5, 3,
                    False, 1e-10, 1e-8, args.distribution,
                )

            if quality < best_quality * (1.0 - args.improve_fract) or p_idx == 0:
                best_quality = quality
                best_traj, best_time = integrate_single_path(
                        inits[b_idx], args.gamma, args.h_x, args.kappa,
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
            g_tau, g_y, g_p = None, None, np.array([args.p_guess,])
            best_quality = np.inf
            old_action = None
        elif args.method == 'shooting':
            interp_traj, interp_time = trim_and_interpolate_trajectory(
                best_traj.T, start_info['pt'][:2], end_info['pt'][:2], drift_4d_numba,
                {'gamma': args.gamma, 'h_x': args.h_x, 'kappa': args.kappa, 'distribution': args.distribution}, eps1=args.eps_bvp, eps2=args.eps_bvp,
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
                            start_info, end_info, args.gamma, args.h_x, args.kappa, args.bvp_nodes,
                            30_000, cur_tol, g_tau, g_y, g_p, verbose=not args.silent,
                            distribution=args.distribution, flipA=args.flipA, flipB=args.flipB,
                            fastOrigin=args.fast_origin, L_scale=args.L_scale, eps=args.eps_bvp,
                            bc_mode='manifold_P' if args.distribution=="G" else 'manifold_nl',
                        )
                    quality = res.rms_residuals.max()
            except RuntimeError as e:
                if not args.silent:
                    print("BVP raised RuntimeError, aborting")
                    raise
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
            print(f"Success! Action: {action:.9n}, Quality: {best_quality:.2e}.")
    else:
        if not args.silent:
            print(f"Discarding result. Target Quality not met ({best_quality:.2e} > {args.quality_thresh})")


if __name__ == "__main__":
    main()
