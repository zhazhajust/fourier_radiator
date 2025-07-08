from scipy.constants import alpha as alpha_fs, hbar, c, e, epsilon_0, m_e
import numpy as np

def norm_val(val):
    """
    Normalize the spectrum values.
    """
    lambda_u = 1.0
    J_in_um = 2e6 * np.pi * hbar * c
    val = alpha_fs / (4 * np.pi**2) * val
    lambda0_um = lambda_u / 1e-6
    val *= J_in_um / lambda0_um
    return val
