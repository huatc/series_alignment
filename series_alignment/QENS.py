import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis
import os
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import linregress
import warnings
from tqdm import tqdm  # optional progress bar
warnings.filterwarnings('ignore')
from numpy.fft import fft, fftfreq, fftshift
import random
from scipy.special import sinc


def simulate_QENS(Q_VALUES, TOPO, TRAJ, positions, MAX_LAG_FRACTION, T0_STRIDE):
    DT_PS = None
    HBAR_MEV_PS = 0.6582119514  # ħ in meV·ps

    # -----------------------------
    # LOAD TRAJECTORY
    # -----------------------------
    u = mda.Universe(TOPO, TRAJ)

    T, N, _ = positions.shape
    dt = DT_PS if DT_PS is not None else u.trajectory.dt 
    max_lag = int(T * MAX_LAG_FRACTION)

    # Storage
    Fqt_dict = {}
    Sqw_dict = {}
    energy_dict = {}

    # -----------------------------
    # LOOP OVER Q VALUES
    # -----------------------------
    for Q in Q_VALUES:
        Fqt = np.zeros(max_lag)
        counts = np.zeros(max_lag)

        for t0 in range(0, T - max_lag, T0_STRIDE):
            r0 = positions[t0]

            for lag in range(max_lag):
                dr_vec = positions[t0 + lag] - r0     # (N, 3)
                dr = np.linalg.norm(dr_vec, axis=1)   # (N,)

                Fqt[lag] += np.mean(sinc(Q * dr / np.pi))
                counts[lag] += 1

        # Average over time origins
        Fqt /= counts

        # Normalize
        Fqt /= Fqt[0]

        # Fourier transform
        Sqw = np.real(fft(Fqt))
        omega = fftfreq(len(Fqt), dt)

        Sqw = fftshift(Sqw)
        omega = fftshift(omega)

        energy_meV = HBAR_MEV_PS * omega

        # Store
        Fqt_dict[Q] = Fqt
        Sqw_dict[Q] = Sqw
        energy_dict[Q] = energy_meV
        return Fqt_dict, Sqw_dict, energy_dict
    

# def convolve_with_resolution(Sqw_sim, Sqw_res, omega):
#     """
#     Convolve simulated S(Q,ω) with experimental resolution.
#     Assumes same ω grid.
#     """
#     # Time domain
#     Fqt_sim = np.real(np.fft.ifft(np.fft.ifftshift(Sqw_sim)))
#     R_t     = np.real(np.fft.ifft(np.fft.ifftshift(Sqw_res)))

#     # Apply resolution
#     Fqt_conv = Fqt_sim * R_t

#     # Back to frequency domain
#     Sqw_conv = np.real(np.fft.fftshift(np.fft.fft(Fqt_conv)))

#     # Normalize intensity
#     Sqw_conv *= np.trapezoid(Sqw_sim, omega) / np.trapezoid(Sqw_conv, omega)

#     return Sqw_conv

# import numpy as np


def convolve_with_resolution(
    Sqw_sim,
    Sqw_res,
    omega,
    elastic_weight
):
    """
    Convolve simulated S(Q,ω) + elastic delta contribution
    with experimental resolution.

    Parameters
    ----------
    Sqw_sim : array
        Simulated dynamic spectrum (no elastic line).
    Sqw_res : array
        Experimental resolution spectrum.
    omega : array
        Frequency grid (uniform spacing).
    elastic_weight : float
        A0(Q) elastic fraction (0 <= A0 <= 1).
    """

    Sqw_sim /= np.trapezoid(Sqw_sim, omega)

    # -------------------------------------------------
    # 1. Construct delta function on discrete grid
    # -------------------------------------------------

    domega = omega[1] - omega[0]

    delta = np.zeros_like(omega)
    zero_index = np.argmin(np.abs(omega))
    delta[zero_index] = 1.0 / domega   # ensures ∫δ dω = 1

    # -------------------------------------------------
    # 2. Build full spectrum (elastic + quasielastic)
    # -------------------------------------------------

    Sqw_total = (
        elastic_weight * delta
        + (1.0 - elastic_weight) * Sqw_sim
    )

    # -------------------------------------------------
    # 3. Transform to time domain
    # -------------------------------------------------

    Fqt_total = np.real(np.fft.ifft(np.fft.ifftshift(Sqw_total)))
    R_t       = np.real(np.fft.ifft(np.fft.ifftshift(Sqw_res)))

    # -------------------------------------------------
    # 4. Apply resolution (multiplication in time)
    # -------------------------------------------------

    Fqt_conv = Fqt_total * R_t

    # -------------------------------------------------
    # 5. Back to frequency domain
    # -------------------------------------------------

    Sqw_conv = np.real(np.fft.fftshift(np.fft.fft(Fqt_conv)))

    # -------------------------------------------------
    # 6. Normalize total intensity
    # -------------------------------------------------

    Sqw_conv *= np.trapezoid(Sqw_total, omega) / np.trapezoid(Sqw_conv, omega)

    return Sqw_conv




def truncate_simulated_to_experimental(
    E_exp: np.ndarray,
    E_sim: np.ndarray,
    I_sim: np.ndarray
):
    """
    Truncate simulated 1D QENS data to the experimental energy range.

    Parameters
    ----------
    E_exp : (N_exp,) array
        Experimental energy axis
    E_sim : (N_sim,) array
        Simulated energy axis
    I_sim : (N_sim,) array
        Simulated intensity

    Returns
    -------
    E_sim_trunc : array
        Truncated simulated energy axis
    I_sim_trunc : array
        Truncated simulated intensity
    """

    E_min, E_max = E_exp.min(), E_exp.max()
    mask = (E_sim >= E_min) & (E_sim <= E_max)

    return E_sim[mask], I_sim[mask]

def interpolate_simulated_to_experimental(
    E_exp: np.ndarray,
    E_sim: np.ndarray,
    I_sim: np.ndarray
):
    """
    Interpolate simulated 1D QENS data onto the experimental energy axis.

    Parameters
    ----------
    E_exp : (N_exp,) array
        Experimental energy axis
    E_sim : (N_sim,) array
        Simulated energy axis
    I_sim : (N_sim,) array
        Simulated intensity

    Returns
    -------
    I_sim_interp : (N_exp,) array
        Simulated intensity evaluated at experimental energies
    """

    return np.interp(E_exp, E_sim, I_sim)

import numpy as np
import re

def extract_tagged_dat(filename):
    """
    Extract data blocks from a .dat file where each block starts with '#'.

    Returns
    -------
    data : dict
        Keys are header strings (without '#'),
        values are NumPy arrays of the extracted numeric data.
    """

    data = {}
    current_key = None
    current_block = []

    def flush_block():
        """Save the current block into the dictionary."""
        if current_key is None or not current_block:
            return

        # Convert to numpy array
        arr = np.array(current_block, dtype=float)
        data[current_key] = arr

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # New header
            if line.startswith("#"):
                flush_block()

                # Clean header text
                current_key = line.lstrip("#").strip()
                current_block = []

            else:
                # Extract all numeric values from the line
                numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
                if numbers:
                    current_block.append([float(x) for x in numbers])

        # Flush last block
        flush_block()

    return data


def truncate_simulated_to_experimental(
    E_exp: np.ndarray,
    E_sim: np.ndarray,
    I_sim: np.ndarray
):
    """
    Truncate simulated 1D QENS data to the experimental energy range.

    Parameters
    ----------
    E_exp : (N_exp,) array
        Experimental energy axis
    E_sim : (N_sim,) array
        Simulated energy axis
    I_sim : (N_sim,) array
        Simulated intensity

    Returns
    -------
    E_sim_trunc : array
        Truncated simulated energy axis
    I_sim_trunc : array
        Truncated simulated intensity
    """

    E_min, E_max = E_exp.min(), E_exp.max()
    mask = (E_sim >= E_min) & (E_sim <= E_max)

    return E_sim[mask], I_sim[mask]

def interpolate_simulated_to_experimental(
    E_exp: np.ndarray,
    E_sim: np.ndarray,
    I_sim: np.ndarray
):
    """
    Interpolate simulated 1D QENS data onto the experimental energy axis.

    Parameters
    ----------
    E_exp : (N_exp,) array
        Experimental energy axis
    E_sim : (N_sim,) array
        Simulated energy axis
    I_sim : (N_sim,) array
        Simulated intensity

    Returns
    -------
    I_sim_interp : (N_exp,) array
        Simulated intensity evaluated at experimental energies
    """

    return np.interp(E_exp, E_sim, I_sim)

