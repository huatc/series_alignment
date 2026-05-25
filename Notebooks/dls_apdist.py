import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
sys.path.append('../series_alignment')
from sklearn.preprocessing import MinMaxScaler
import DLS
import series_alignment 
import time
from apdist.torch import TorchAmplitudePhaseDistance as torch_apdist 
from apdist.distances import AmplitudePhaseDistance as numpy_apdist 

import torch 
from scipy.interpolate import CubicSpline

path = '../Data/DLS/Kinetics_data/'
filenames = sorted(os.listdir(path))

df = pd.read_excel('../Data/DLS/Assembly_kinetics_data.xlsx').values[:,5:].T
data = df.astype('float')
#df = np.flip(df)
sizes = data[1:71,0].reshape(-1,1)
intensity = data[71:,:]
n_samples = 19
exp_DLS_data = []
for i in range(0,n_samples):
    sample_int = intensity[:,i*3:i*3 + 3]
    avg = np.mean(sample_int, axis=1).reshape(-1,1)
    scaler = MinMaxScaler()
    avg = scaler.fit(avg).transform(avg)
    exp_DLS_data.append(np.hstack((sizes, avg)))
    if i == 0:
        avg_data_silica = avg.reshape(-1,1)
    else:
        avg_data_silica = np.hstack((avg_data_silica, avg.reshape(-1,1)))

file_path = '../Data/DLS/Simulated_DLS_10/'
filenames = sorted(os.listdir(file_path))
for i in range(len(filenames)):
    if len(filenames[i]) == 6:
        new_name = filenames[i][0:5] + '0' + filenames[i][-1]
        os.rename(file_path + filenames[i], file_path + '/' + new_name)

file_path = '../Data/DLS/Simulated_DLS_10/'
filenames = sorted(os.listdir(file_path))
sim_DLS_data = []
for i in range(0, len(filenames)):
    DLS_data = pd.read_csv(file_path + filenames[i], skiprows=1, delimiter= ' ') #this code loads the data incorrectly so modify the correct column titles
    radius_of_gyration = DLS_data['Cluster Size'].values*2.0*25*2*1e-9 #meters 
    radius_of_gyration = radius_of_gyration[radius_of_gyration != 0]
    cluster_size = np.round(DLS_data['Cluster Identifier'].values)
    cluster_size = cluster_size[cluster_size != 0]
    hydrodynamic_radius = radius_of_gyration/0.77 #assuming spherical clusters 

    clusters = [
        {"N": int(N), "Rg": float(Rg), "Rh": float(Rh)}
        for N, Rg, Rh in zip(cluster_size, radius_of_gyration, hydrodynamic_radius)
    ]
    # ---- example usage ----
    # Instrument settings
    lam = 632.8e-9        # HeNe laser (m)
    n_medium = 1.333      # water
    theta = np.deg2rad(173)  # backscatter example
    q = (4*np.pi*n_medium/lam) * np.sin(theta/2)
    kB = 1.380649e-23  # J/K
    T = 298.15            # K
    eta = 0.00089         # Pa*s

    tau = np.logspace(-6, 0, 100)  # correlation times (s) matching your experiment
    g2_sim, g1_sim, Ds, wts = DLS.build_g2(clusters, tau, q, T, eta, beta=0.85)

    # Now invert like the instrument would
    #Dgrid, Rh_grid, w_intensity = invert_g2_to_distribution(g2_sim, tau, q, T, eta, beta=0.85)
    Dgrid, wD, info = DLS.contin_like_invert(g2_sim, tau, q, T=T, eta=eta)
    Rh_grid = info["Rh_grid"]*1e9 
    Rh_grid = np.flip(Rh_grid)
    scaler = MinMaxScaler()
    wD = scaler.fit(wD.reshape(-1,1)).transform(wD.reshape(-1,1))
    wD = np.flip(wD)
    dls_data = np.hstack((Rh_grid.reshape(-1,1), wD.reshape(-1,1)))
    sim_DLS_data.append(dls_data)
    # Dgrid/Rh_grid with weights w_intensity is your synthetic DLS "size distribution"
    # Plot Rh_grid vs w_intensity and compare peak/width to experimental inversion.


def shape_distance_dls_torch(expt_x, expt_y, sim_x, sim_y):
    x_log10 = np.log10(expt_x)
    x_dense = np.linspace(1.0, 4.0, 100)
    x_dense_scaled = (x_dense-min(x_dense))/(max(x_dense)-min(x_dense))

    optim_kwargs = {"n_iters":100, 
                "n_basis":20, 
                "n_layers":20,
                "n_restarts":128,
                "domain_type":"linear",
                "basis_type":"palais",
                "lr":1e-1,
                "n_domain":x_dense.shape[0],
                "eps":1e-2
            }
    
    spline_expt = CubicSpline(x_log10, expt_y)
    y_dense_expt = spline_expt(x_dense)

    spline_sim = CubicSpline(np.log10(sim_x), sim_y)
    y_dense_sim = spline_sim(x_dense)

    amplitude, phase, _ = torch_apdist(torch.from_numpy(x_dense_scaled),
                                       torch.from_numpy(y_dense_expt),
                                       torch.from_numpy(y_dense_sim),
                                       **optim_kwargs
                                    )
    
    return (amplitude+phase).item()

def shape_distance_dls_numpy(expt_x, expt_y, sim_x, sim_y):
    x_log10 = np.log10(expt_x)
    x_dense = np.linspace(1.0, 4.0, 100)
    x_dense_scaled = (x_dense-min(x_dense))/(max(x_dense)-min(x_dense))

    spline_expt = CubicSpline(x_log10, expt_y)
    y_dense_expt = spline_expt(x_dense)

    spline_sim = CubicSpline(np.log10(sim_x), sim_y)
    y_dense_sim = spline_sim(x_dense)

    optim_kwargs = {"optim":"DP", "grid_dim":10}
    amplitude, phase = numpy_apdist(x_dense_scaled, y_dense_expt, y_dense_sim, **optim_kwargs)
    
    return (amplitude+phase).item()    

def create_distance_matrix(exp_data, sim_data):
    score_matrix = np.zeros((len(exp_data), len(sim_data)))

    for i, exp_curve in enumerate(exp_data):
        for j, sim_curve in enumerate(sim_data):
            start = time.time()
            dist = shape_distance_dls_numpy(
                exp_curve[:, 0], exp_curve[:, 1],
                sim_curve[:, 0], sim_curve[:, 1]
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
    plt.plot(idx_cols_lin, exp_indices, marker='o', color='white', label='Peaks Distance')
    plt.legend()
    plt.xlabel("Simulated Index")
    plt.ylabel("Exp. Index")
    #plt.title("Alignment Path on Cost Matrix")
    plt.tight_layout()
    #plt.legend(fontsize=18)
    #plt.legend(loc='center', bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize=14)

    if save_path is not None:
        plt.savefig('../Figures/dls_shape_distance_matching.png', dpi=600, bbox_inches="tight")
    plt.show()

score_matrix_ap = create_distance_matrix(exp_DLS_data, sim_DLS_data)
idx_cols_min_ap, _ = series_alignment.align_monotone_min(score_matrix_ap)

score_matrix_peaks = series_alignment.create_distance_matrix(exp_DLS_data, sim_DLS_data,'peak_position')
idx_cols_peaks, _ = series_alignment.align_monotone_min(score_matrix_peaks)

plot_alignment_with_cost(score_matrix_ap, idx_cols_min_ap, idx_cols_peaks, save_path=True)