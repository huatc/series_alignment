import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from scipy import integrate
import sys
sys.path.append('../series_alignment')
from scipy.interpolate import CubicSpline
import time
import series_alignment

from apdist.distances import AmplitudePhaseDistance as numpy_apdist 
from apdist.utils import plot_warping 
from apdist.geometry import SquareRootSlopeFramework as SRSF

# Load experimental and simulated SAXS data
path = '../Data/SAXS/Kinetics_Sq/'
filenames = sorted(os.listdir(path))

exp_saxs_data = []
for i in range(0, 10):
    #data = read_DAT_file(path + filenames[i])
    data = np.load(path + filenames[i])
    exp_saxs_data.append(data)

file_path = '../Data/Simulations/Simulation_File_8/Optimization_Results/Sample_0/Density_0.005_U_0_46.95_r0_2.25_n_1.0_m_20.89/scattering_data_mcdfm_sq/'
filenames = sorted(os.listdir(file_path))

sim_saxs_data = []
for i in range(len(filenames)):
    saxs_data = np.load(file_path + filenames[i])
    sim_saxs_data.append(saxs_data)

print(f'Number of experimental SAXS curves: {len(exp_saxs_data)}')  
print(f'Number of simulated SAXS curves: {len(sim_saxs_data)}')

def process_saxs_data(x, y):
    x_lim = [0.004, 0.03]
    y_lim = [7e-2, 5e0]

    mask = (x >= x_lim[0]) & (x <= x_lim[1]) & (y >= y_lim[0]) & (y <= y_lim[1])
    x_processed = np.log10(x[mask])
    y_processed = np.log10(y[mask])

    x_unique, unique_indices = np.unique(x_processed, return_index=True)
    y_unique = y_processed[unique_indices]

    return x_unique, y_unique

def shape_distance_saxs(expt_x, expt_y, sim_x, sim_y, plot=False):
    expt_x, expt_y = process_saxs_data(expt_x, expt_y)
    sim_x, sim_y = process_saxs_data(sim_x, sim_y)

    x_dense = np.linspace(expt_x.min(), expt_x.max(), 100)
    x_dense_scaled = (x_dense-min(x_dense))/(max(x_dense)-min(x_dense))
    try:
        spline_expt = CubicSpline(expt_x, expt_y)
        y_dense_expt = spline_expt(x_dense)
    except Exception as e:
        print(f"Error in spline interpolation: {e}")
        for i in range(len(expt_x) - 1):
            if expt_x[i] >= expt_x[i + 1]:
                print(f"Sequence breaks at index {i}: {expt_x[i]} -> {expt_x[i+1]}")
        import pdb; pdb.set_trace()

    spline_expt = CubicSpline(expt_x, expt_y)
    y_dense_expt = spline_expt(x_dense)

    spline_sim = CubicSpline(sim_x, sim_y)
    y_dense_sim = spline_sim(x_dense)

    optim_kwargs = {"optim":"DP", "grid_dim":10}
    amplitude, phase = numpy_apdist(x_dense_scaled, y_dense_expt, y_dense_sim, **optim_kwargs)
    if plot:
        srsf = SRSF(x_dense_scaled)
        q_ref = srsf.to_srsf(y_dense_expt)
        q_query = srsf.to_srsf(y_dense_sim)
        gamma = srsf.get_gamma(q_ref, q_query, **optim_kwargs)
        f_query_gamma = srsf.warp_f_gamma(y_dense_sim, gamma)
        plot_warping(x_dense_scaled, y_dense_expt, y_dense_sim, f_query_gamma, gamma)
        plt.savefig('../Figures/saxs_shape_distance.png', dpi=600, bbox_inches="tight")
        plt.close()
    
    return (amplitude+phase).item()    

def create_distance_matrix(exp_data, sim_data):
    score_matrix = np.zeros((len(exp_data), len(sim_data)))

    plot_i = np.random.randint(len(exp_data))
    plot_j = np.random.randint(len(sim_data))
    print(f"Plotting alignment for exp {plot_i} and sim {plot_j}")

    for i, exp_curve in enumerate(exp_data):
        for j, sim_curve in enumerate(sim_data):
            start = time.time()
            dist = shape_distance_saxs(
                exp_curve[:, 0], exp_curve[:, 1],
                sim_curve[:, 0], sim_curve[:, 1],
                plot=(i == plot_i and j == plot_j)
            )
            score_matrix[i, j] = dist
            end = time.time()
            print(f"Distance between exp {i} and sim {j}: {dist:.4f} (Time: {end - start:.4f}s)")

    return score_matrix

def plot_alignment_with_cost(cost_matrix, idx_cols_min, idx_cols_lin, save_path=None):
    """
    Plot cost matrix heatmap with alignment path overlaid.
    """

    idx_cols_min = np.array(idx_cols_min)
    idx_cols_lin = np.array(idx_cols_lin)
    n_exp, n_sim = cost_matrix.shape
    exp_indices = np.arange(n_exp)

    plt.figure(figsize=(15, 3))
    plt.imshow(cost_matrix, aspect='auto', origin='lower')
    plt.colorbar(label="Distance", aspect=7)
    plt.plot(idx_cols_min, exp_indices, marker='o', color='red', label='AP Distance')


    plt.plot(idx_cols_lin, exp_indices, marker='o', color='orange', label='Peaks Distance')

    plt.xlabel("Simulated Index")
    plt.ylabel("Exp. Index")
    #plt.title("Alignment Path on Cost Matrix")
    plt.tight_layout()
    #plt.legend(fontsize=18)
    plt.legend(loc='center', bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize=14)

    if save_path is None:
        plt.savefig('../Figures/saxs_shape_distance_matching.png', dpi=600, bbox_inches="tight")
    plt.show()

score_matrix_ap = create_distance_matrix(exp_saxs_data, sim_saxs_data)
idx_cols_min, total_dist_min = series_alignment.align_monotone_min(score_matrix_ap)

score_matrix_peaks = series_alignment.create_distance_matrix(exp_saxs_data, sim_saxs_data,'peak_position')
idx_cols_peaks, _ = series_alignment.align_monotone_min(score_matrix_peaks)

plot_alignment_with_cost(score_matrix_ap, idx_cols_min, idx_cols_peaks)