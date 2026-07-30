# %% [markdown]
# # Bayesian optimization of a coarse-grained potential against SAXS *and* DLS series
#
# Each candidate potential is simulated once with HOOMD, then a *series* of
# simulated curves is extracted from the trajectory (one per selected frame) for
# both SAXS and DLS.  Each simulated series is matched against the corresponding
# experimental kinetic series using the amplitude-phase shape distance plus the
# monotone alignment in `series_alignment`, and the aligned total cost is the
# objective that Bayesian optimization minimizes.
#
# The HOOMD simulation only runs on Linux/GPU.  Everything downstream of the
# trajectory (clustering, scattering, distances, alignment) imports without
# HOOMD so it can be developed and checked on any platform:
#
#     python Bridge_20_Hyak_250803.py --check     # scoring self-test, no HOOMD
#     python Bridge_20_Hyak_250803.py             # full optimization (Linux/GPU)

# %%
import os
import sys
import glob
import time

import numpy as np
import pandas as pd
import matplotlib

# Every figure here is written straight to disk, and the optimization runs on a
# headless cluster node, so avoid a GUI backend unless we are inside a notebook.
if 'ipykernel' not in sys.modules:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt                      # noqa: E402
from scipy.interpolate import CubicSpline            # noqa: E402
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.preprocessing import MinMaxScaler

# --- Repository-relative paths (work regardless of the current directory) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)          # .../series_alignment

for _p in (os.path.join(_ROOT, 'series_alignment'),
           os.path.join(_ROOT, os.pardir, 'MC-DFM'),
           os.path.join(_ROOT, os.pardir, os.pardir, 'MC-DFM')):
    _p = os.path.abspath(_p)
    if _p not in sys.path:
        sys.path.append(_p)

import DLS                              # noqa: E402  (series_alignment/DLS.py)
import series_alignment                 # noqa: E402
from apdist.distances import AmplitudePhaseDistance   # noqa: E402

# --- Optional heavy dependencies -------------------------------------------
# HOOMD 2.9.4 and the MC-DFM scattering simulator are Linux/GPU only; gsd and
# botorch may also be absent on an analysis-only machine.  Import them lazily
# so the scoring code below stays importable everywhere.
try:
    import hoomd
    import hoomd.md
    _HAS_HOOMD = True
except ImportError:                     # pragma: no cover
    _HAS_HOOMD = False

try:
    import gsd.hoomd
    _HAS_GSD = True
except ImportError:                     # pragma: no cover
    _HAS_GSD = False

try:
    from Scattering_Simulator import pairwise_method
    _HAS_MCDFM = True
except ImportError:                     # pragma: no cover
    _HAS_MCDFM = False


# %%
# ============================== CONFIGURATION ==============================

# Number of trajectory frames turned into simulated curves per candidate.
# `series_alignment.align_monotone_min` requires at least as many simulated
# curves as experimental ones, so this must exceed N_EXP_DLS (19) below.
# NOTE ON COST: each SAXS frame is a full Monte-Carlo scattering calculation
# (SAXS_MC_SAMPLES below), so wall time per BO iteration scales linearly with
# this number.  40 frames is ~8x the cost of the old 5-frame average.
N_SERIES_FRAMES = 40

# Objective weights for the two modalities (each already normalized per curve).
SAXS_WEIGHT = 0.5
DLS_WEIGHT = 0.5

# Experimental series sizes
N_EXP_SAXS = 10
N_EXP_DLS = 19

# --- SAXS analysis settings ---
SAXS_Q_SIM = np.geomspace(0.8, 20, 2000)   # simulation q-grid (sim units)
SAXS_Q_RESCALE = 260.0                     # sim q -> physical 1/Angstrom
SAXS_HIST_BINS = 10000
SAXS_MC_SAMPLES = 10_000_000
SAXS_SPHERE_R = 130.0                      # form-factor sphere radius (Angstrom)
SAXS_NORM_RANGE = (0.02, 0.03)             # q-window used to normalize P(q)
SAXS_Q_LIM = (0.004, 0.03)                 # comparison window (1/Angstrom)
SAXS_S_LIM = (7e-2, 5e0)                   # keep physically sensible S(q)
BUILDING_BLOCK_D = 1.0                     # sphere diameter (sim units)
BUILDING_BLOCK_SPACING = 0.03

# --- DLS analysis settings (instrument constants match Scripts/dls_apdist.py) ---
DLS_LAMBDA = 632.8e-9        # HeNe laser (m)
DLS_N_MEDIUM = 1.333         # water
DLS_THETA = np.deg2rad(173)  # backscatter detection
DLS_T = 298.15               # K
DLS_ETA = 0.00089            # Pa*s
DLS_BETA = 0.85
DLS_TAU = np.logspace(-6, 0, 100)
DLS_LOG_SIZE_LIM = (1.0, 4.0)   # log10(Rh / nm) comparison window

# One simulation length unit -> metres.  Matches the conversion used in
# Scripts/dls_apdist.py (Rg_sim * 2.0 * 25 * 2 * 1e-9).
SIM_LENGTH_TO_M = 2.0 * 25 * 2 * 1e-9
RG_TO_RH = 1.0 / 0.77           # Rh from Rg, assuming compact/spherical clusters

# Bond cutoff for cluster analysis, as a multiple of the potential minimum r0.
CLUSTER_CUTOFF_FACTOR = 1.25
MIN_CLUSTER_SIZE = 2

# Shared amplitude-phase distance options (dynamic-programming warp search).
AP_KWARGS = {"optim": "DP", "grid_dim": 10}

N_DENSE = 100          # points on the common grid used for the AP distance
FAILED_SCORE = 999.0   # sentinel returned when a candidate cannot be evaluated


# %% [markdown]
# ## Simulation (HOOMD, Linux/GPU only)

# %%
def modified_LJ(r, rmin, rmax, U_0, n, m, r0):
    """Generalized (n, m) Lennard-Jones potential and force."""
    U = U_0 / (n - m) * (m * (r0 / r) ** n - n * (r0 / r) ** m)
    F = U_0 * m * n * ((r0 / r) ** n - (r0 / r) ** m) / ((n - m) * r)
    return U, F


def generate_positions(N, L, min_dist, rng=None):
    """Rejection-sample N non-overlapping positions in a cubic box of edge L.

    Uses an incremental cKDTree query rather than an all-pairs Python loop so
    that N = 5000 initializes in seconds instead of minutes.
    """
    rng = np.random.default_rng() if rng is None else rng
    positions = np.empty((N, 3))
    count = 0
    attempts = 0
    max_attempts = N * 1000
    batch = max(1, N // 10)

    while count < N and attempts < max_attempts:
        candidates = rng.uniform(-L / 2, L / 2, size=(batch, 3))
        for pos in candidates:
            attempts += 1
            if count == 0:
                positions[count] = pos
                count += 1
                continue
            d = positions[:count] - pos
            if np.min(np.einsum('ij,ij->i', d, d)) >= min_dist ** 2:
                positions[count] = pos
                count += 1
                if count == N:
                    break
    if count < N:
        raise RuntimeError(
            f"Failed to generate non-overlapping configuration ({count}/{N}).")
    return positions


def simulation(density, U_0, r0, n, m, save_dir,
               N=5000, dt=0.001, steps=15_000_000, kT=1.0, gsd_period=50_000):
    """Run Langevin dynamics of N attractive spheres and dump a GSD trajectory."""
    if not _HAS_HOOMD:
        raise RuntimeError(
            "HOOMD is not available: the simulation only runs on Linux/GPU. "
            "Use --check to exercise the scoring pipeline without it.")

    hoomd.context.initialize("--mode=gpu")
    os.makedirs(save_dir, exist_ok=True)

    rmin = 0.75 * r0
    rmax = 5
    width = 1000                       # points used to tabulate the pair potential

    # === Box size from density ===
    L = (N / density) ** (1.0 / 3.0)

    positions = generate_positions(N, L, min_dist=1.1)

    # === Initial snapshot ===
    snapshot = hoomd.data.make_snapshot(N=N,
                                        box=hoomd.data.boxdim(L=L),
                                        particle_types=['A'])
    for i in range(N):
        snapshot.particles.position[i] = positions[i]
        snapshot.particles.diameter[i] = 1.0

    hoomd.init.read_snapshot(snapshot)

    # === Sanity-check the tabulated potential ===
    r_vals = np.linspace(rmin, rmax, width)
    U, F = modified_LJ(r_vals, rmin, rmax, U_0, n, m, r0)
    assert np.all(np.isfinite(U)), "U not finite over [rmin, rmax]"
    assert np.all(np.isfinite(F)), "F not finite over [rmin, rmax]"

    plt.figure(figsize=(6, 4))
    plt.plot(r_vals, U, label=f'LJ-nm: n={n}, m={m}, r0={r0}, U_0={U_0}')
    plt.xlabel("r")
    plt.ylabel("Potential Energy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "potential_plot.png"),
                dpi=600, bbox_inches="tight")
    plt.close()

    # === Langevin dynamics ===
    group_all = hoomd.group.all()
    hoomd.md.integrate.mode_standard(dt=dt)
    langevin = hoomd.md.integrate.langevin(group=group_all, kT=kT, seed=42)
    langevin.set_gamma('A', gamma=1.0)

    nl = hoomd.md.nlist.cell()
    table = hoomd.md.pair.table(width=width, nlist=nl)
    table.pair_coeff.set('A', 'A', rmin=rmin, rmax=rmax, func=modified_LJ,
                         coeff=dict(U_0=U_0, n=n, m=m, r0=r0))

    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    filename = os.path.join(save_dir, f"DNA_assembly_{timestamp}.gsd")
    hoomd.dump.gsd(filename=filename, period=gsd_period, group=group_all,
                   overwrite=True)

    hoomd.analyze.log(filename=os.path.join(save_dir, 'potential_energy.csv'),
                      quantities=['potential_energy'], period=5000,
                      overwrite=True)

    print(f"Running {steps} steps with {N} spheres at density {density:.5f}")
    hoomd.run(steps)
    print("Simulation complete.")

    df = pd.read_csv(os.path.join(save_dir, "potential_energy.csv"),
                     delimiter='\t').values
    plt.figure(figsize=(6, 4))
    plt.plot(df[6:, 0], df[6:, 1], label='Potential Energy')
    plt.xlabel("Time")
    plt.ylabel("Potential Energy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "potential_energy_plot.png"),
                dpi=600, bbox_inches="tight")
    plt.close()

    return filename


# %% [markdown]
# ## Trajectory reading and frame selection

# %%
def quaternion_to_euler(quat, degrees=True, order='xyz'):
    """Convert HOOMD quaternions [qw, qx, qy, qz] to Euler angles.

    scipy expects [qx, qy, qz, qw], so the components are reordered here.
    """
    quat = np.atleast_2d(np.asarray(quat, dtype=float))
    scipy_quat = np.column_stack((quat[:, 1], quat[:, 2], quat[:, 3], quat[:, 0]))
    return R.from_quat(scipy_quat).as_euler(order, degrees=degrees)


def find_gsd_file(save_dir):
    """Locate the GSD trajectory in save_dir (robust to other files present)."""
    matches = sorted(glob.glob(os.path.join(save_dir, '*.gsd')))
    if not matches:
        raise FileNotFoundError(f"No .gsd trajectory found in {save_dir}")
    return matches[-1]


def read_trajectory(filename):
    """Read a GSD trajectory.

    Returns
    -------
    positions : list of (N, 3) arrays, one per frame
    lattice_coordinates : list of (N, 6) arrays (position + Euler angles)
    box_L : float, cubic box edge length
    """
    if not _HAS_GSD:
        raise RuntimeError("The gsd package is required to read trajectories.")

    positions, lattice_coordinates = [], []
    box_L = None

    with gsd.hoomd.open(name=filename, mode='r') as traj:
        for frame in traj:
            pos = np.asarray(frame.particles.position, dtype=float).copy()
            positions.append(pos)

            if box_L is None:
                box_L = float(frame.configuration.box[0])

            quat = getattr(frame.particles, 'orientation', None)
            if quat is None:
                angles = np.zeros((len(pos), 3))
            else:
                angles = quaternion_to_euler(np.asarray(quat, dtype=float))
            lattice_coordinates.append(np.hstack((pos, angles)))

    return positions, lattice_coordinates, box_L


def select_frame_indices(n_frames, n_series):
    """Pick n_series frame indices spread evenly across a trajectory.

    The simulated series must be time-ordered for the monotone alignment to be
    meaningful, and must be at least as long as the experimental series.
    """
    if n_frames == 0:
        raise ValueError("Trajectory contains no frames.")
    idx = np.unique(np.linspace(0, n_frames - 1, min(n_series, n_frames)).astype(int))
    return idx.tolist()


# %% [markdown]
# ## Simulated SAXS series

# %%
def grid_points_in_sphere(D, spacing):
    """Regular 3D grid of points filling a sphere of diameter D."""
    radius = D / 2.0
    coords = np.arange(-radius, radius + spacing, spacing)
    X, Y, Z = np.meshgrid(coords, coords, coords)
    grid = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    return grid[np.sum(grid ** 2, axis=1) <= radius ** 2]


def sphere(q, r):
    """Form factor of a monodisperse sphere of radius r."""
    return 3 * (np.sin(q * r) - q * r * np.cos(q * r)) ** 2 / (q * r) ** 6


def convert_data(data, model):
    """Resample `model` onto the q values of `data` by nearest-neighbour lookup."""
    model_x, model_y = model[:, 0], model[:, 1]
    loc = np.abs(model_x[None, :] - data[:, 0][:, None]).argmin(axis=1)
    return np.column_stack((model_x[loc], model_y[loc]))


def normalize_scattering_curves(q, I1, I2, q_min, q_max):
    """Scale I2 onto I1 using their mean ratio over [q_min, q_max]."""
    mask = (q >= q_min) & (q <= q_max)
    if not np.any(mask):
        raise ValueError("No data points found within the specified q-range.")
    return I2 * (np.mean(I1[mask]) / np.mean(I2[mask]))


def calculate_structure_factor(form_factor, intensity, q_min, q_max, plot=False):
    """Divide out the sphere form factor to obtain S(q) = I(q) / P(q)."""
    resampled = convert_data(form_factor, intensity)
    normalized_P = normalize_scattering_curves(
        resampled[:, 0], resampled[:, 1], form_factor[:, 1], q_min, q_max)

    if plot:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(form_factor[:, 0], normalized_P, color='blue', label='P(q)')
        ax.scatter(resampled[:, 0], resampled[:, 1], color='red', label='I(q)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('Intensity (arb. unit)')
        ax.set_xlabel('q ($\\AA^{-1}$)')
        ax.legend()
        plt.close(fig)

    structure_factor = resampled[:, 1] / normalized_P
    return np.column_stack((resampled[:, 0], structure_factor))


def convert_to_SAXS_series(lattice_coordinates, frame_indices, save_dir,
                           n_samples=SAXS_MC_SAMPLES, plot=True):
    """Compute one simulated structure factor per selected trajectory frame.

    Returns a list of (n_q, 2) arrays [q, S(q)], ordered by increasing frame
    index (i.e. increasing simulation time).
    """
    if not _HAS_MCDFM:
        raise RuntimeError("The MC-DFM Scattering_Simulator package is required.")

    out_dir = os.path.join(save_dir, 'scattering_data_series')
    os.makedirs(out_dir, exist_ok=True)

    points = grid_points_in_sphere(BUILDING_BLOCK_D, BUILDING_BLOCK_SPACING)
    points = np.hstack((points, np.ones((len(points), 1))))

    q_sim = SAXS_Q_SIM
    q_phys = q_sim / SAXS_Q_RESCALE

    q_sphere = np.geomspace(0.003, 0.08, 500)
    monodisperse_sphere = np.column_stack((q_sphere, sphere(q_sphere, SAXS_SPHERE_R)))

    series = []
    for k in frame_indices:
        coords = lattice_coordinates[k]

        simulator = pairwise_method.scattering_simulator(n_samples)
        simulator.sample_building_block(points)
        simulator.sample_lattice_coordinates(coords)
        simulator.calculate_structure_coordinates()
        I_q = simulator.simulate_multiple_scattering_curves_lattice_coords(
            points, coords, SAXS_HIST_BINS, q_sim, save=False).cpu().numpy()
        I_q = np.mean(I_q, axis=1)

        S_q = calculate_structure_factor(
            monodisperse_sphere, np.column_stack((q_phys, I_q)),
            *SAXS_NORM_RANGE, plot=False)

        np.save(os.path.join(out_dir, f'structure_factor_frame_{k:05d}.npy'), S_q)
        np.save(os.path.join(out_dir, f'intensity_frame_{k:05d}.npy'),
                np.column_stack((q_phys, I_q)))
        series.append(S_q)

    if plot and series:
        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = plt.get_cmap('viridis')
        for i, S_q in enumerate(series):
            ax.plot(S_q[:, 0], S_q[:, 1], color=cmap(i / max(1, len(series) - 1)),
                    linewidth=1.5)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('q ($\\AA^{-1}$)')
        ax.set_ylabel('S(q)')
        ax.set_title('Simulated SAXS series (dark = early, light = late)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'saxs_series.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

    return series


# %% [markdown]
# ## Simulated DLS series (cluster analysis -> g2 -> size distribution)

# %%
def _bfs_unwrap(pos, members, neighbors, box_L):
    """Unwrap one cluster across periodic boundaries by walking its bond graph.

    Positions are reconstructed relative to a seed particle using minimum-image
    displacements, so Rg is correct even for clusters that span the box.
    """
    member_set = set(int(m) for m in members)
    seed = int(members[0])
    coords = {seed: pos[seed].copy()}
    stack = [seed]

    while stack:
        cur = stack.pop()
        for nb in neighbors[cur]:
            if nb in coords or nb not in member_set:
                continue
            d = pos[nb] - coords[cur]
            d -= box_L * np.round(d / box_L)
            coords[nb] = coords[cur] + d
            stack.append(nb)

    # Disconnected stragglers cannot occur for a connected component, but fall
    # back to the wrapped coordinate rather than raising.
    return np.array([coords.get(int(m), pos[int(m)]) for m in members])


def cluster_analysis(positions, box_L, cutoff, min_size=MIN_CLUSTER_SIZE):
    """Group particles into clusters and measure their size and radius of gyration.

    Neighbours are found with a periodic KD-tree; clusters are the connected
    components of the resulting bond graph.

    Returns
    -------
    list of dict with keys 'N' (particles), 'Rg' and 'Rh' (metres), matching the
    input format expected by DLS.build_g2.
    """
    pos = np.mod(np.asarray(positions, dtype=float) + box_L / 2.0, box_L)
    n = len(pos)

    tree = cKDTree(pos, boxsize=box_L)
    pairs = tree.query_pairs(cutoff, output_type='ndarray')

    if len(pairs) == 0:
        labels = np.arange(n)
        n_labels = n
    else:
        adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                         shape=(n, n))
        n_labels, labels = connected_components(adj, directed=False)

    neighbors = [[] for _ in range(n)]
    for a, b in pairs:
        neighbors[a].append(b)
        neighbors[b].append(a)

    order = np.argsort(labels, kind='stable')
    boundaries = np.searchsorted(labels[order], np.arange(n_labels + 1))

    clusters = []
    for lab in range(n_labels):
        members = order[boundaries[lab]:boundaries[lab + 1]]
        if len(members) < min_size:
            continue
        unwrapped = _bfs_unwrap(pos, members, neighbors, box_L)
        com = unwrapped.mean(axis=0)
        rg_sim = np.sqrt(np.mean(np.sum((unwrapped - com) ** 2, axis=1)))
        rg_m = rg_sim * SIM_LENGTH_TO_M
        clusters.append({'N': int(len(members)),
                         'Rg': float(rg_m),
                         'Rh': float(rg_m * RG_TO_RH)})
    return clusters


def clusters_from_ovito_file(path):
    """Read an OVITO cluster-analysis export into DLS.build_g2 input format.

    The exported header line begins with '#', which shifts every column label
    by one when parsed.  Columns are therefore read *by position*:
    0 = cluster identifier, 1 = cluster size, 2 = radius of gyration.
    (Scripts/dls_apdist.py relies on the same shift via the mislabelled names
    'Cluster Identifier' -> size and 'Cluster Size' -> Rg.)
    """
    values = pd.read_csv(path, skiprows=1, delimiter=' ').values
    cluster_size = np.round(values[:, 1].astype(float))
    rg_sim = values[:, 2].astype(float)

    keep = (cluster_size > 0) & (rg_sim > 0)
    cluster_size, rg_sim = cluster_size[keep], rg_sim[keep]

    rg_m = rg_sim * SIM_LENGTH_TO_M
    return [{'N': int(N), 'Rg': float(Rg), 'Rh': float(Rg * RG_TO_RH)}
            for N, Rg in zip(cluster_size, rg_m)]


def dls_distribution_from_clusters(clusters):
    """Turn a cluster list into a normalized DLS size distribution [Rh_nm, weight].

    Builds the field autocorrelation function the instrument would measure, then
    inverts it with the CONTIN-like regularized solver so the simulated
    distribution is subject to the same broadening as the experiment.
    """
    q = (4 * np.pi * DLS_N_MEDIUM / DLS_LAMBDA) * np.sin(DLS_THETA / 2)

    g2, _, _, _ = DLS.build_g2(clusters, DLS_TAU, q, DLS_T, DLS_ETA, beta=DLS_BETA)
    _, wD, info = DLS.contin_like_invert(g2, DLS_TAU, q, T=DLS_T, eta=DLS_ETA)

    Rh_nm = np.asarray(info["Rh_grid"], dtype=float) * 1e9
    wD = np.asarray(wD, dtype=float)

    # CubicSpline needs strictly increasing x; Rh decreases with D.
    order = np.argsort(Rh_nm)
    Rh_nm, wD = Rh_nm[order], wD[order]

    wD = MinMaxScaler().fit_transform(wD.reshape(-1, 1)).ravel()
    return np.column_stack((Rh_nm, wD))


def convert_to_DLS_series(positions, frame_indices, box_L, r0, save_dir, plot=True):
    """Compute one simulated DLS size distribution per selected trajectory frame."""
    out_dir = os.path.join(save_dir, 'dls_data_series')
    os.makedirs(out_dir, exist_ok=True)

    cutoff = CLUSTER_CUTOFF_FACTOR * r0
    series, cluster_counts = [], []

    for k in frame_indices:
        clusters = cluster_analysis(positions[k], box_L, cutoff)
        if not clusters:
            raise ValueError(
                f"Frame {k}: no clusters of size >= {MIN_CLUSTER_SIZE} found "
                f"(cutoff {cutoff:.3f}); cannot build a DLS curve.")
        dist = dls_distribution_from_clusters(clusters)
        np.save(os.path.join(out_dir, f'dls_frame_{k:05d}.npy'), dist)
        series.append(dist)
        cluster_counts.append(len(clusters))

    pd.DataFrame({'frame': frame_indices, 'n_clusters': cluster_counts}).to_csv(
        os.path.join(out_dir, 'cluster_counts.csv'), index=False)

    if plot and series:
        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = plt.get_cmap('viridis')
        for i, dist in enumerate(series):
            ax.plot(dist[:, 0], dist[:, 1],
                    color=cmap(i / max(1, len(series) - 1)), linewidth=1.5)
        ax.set_xscale('log')
        ax.set_xlabel('$R_h$ (nm)')
        ax.set_ylabel('Scaled intensity')
        ax.set_title('Simulated DLS series (dark = early, light = late)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'dls_series.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

    return series


# %% [markdown]
# ## Experimental series

# %%
def load_experimental_saxs(n_curves=N_EXP_SAXS, root=_ROOT, include_no_bridge=False):
    """Load the experimental SAXS kinetic series (already structure factors).

    The `no_bridge` file is a reference measurement rather than a kinetic time
    point and sorts first, so it is excluded by default.
    """
    path = os.path.join(root, 'Data', 'SAXS', 'Kinetics_Sq')
    names = sorted(os.listdir(path))
    if not include_no_bridge:
        names = [f for f in names if 'no_bridge' not in f]
    names = names[:n_curves]
    return [np.load(os.path.join(path, f)) for f in names]


def load_experimental_dls(n_samples=N_EXP_DLS, root=_ROOT, n_replicates=3):
    """Load the experimental DLS kinetic series, averaging replicates per sample."""
    path = os.path.join(root, 'Data', 'DLS', 'Assembly_kinetics_data.xlsx')
    data = pd.read_excel(path).values[:, 5:].T.astype(float)

    sizes = data[1:71, 0].reshape(-1, 1)
    intensity = data[71:, :]

    series = []
    for i in range(n_samples):
        block = intensity[:, i * n_replicates:(i + 1) * n_replicates]
        avg = np.mean(block, axis=1).reshape(-1, 1)
        avg = MinMaxScaler().fit_transform(avg)
        series.append(np.hstack((sizes, avg)))
    return series


# %% [markdown]
# ## Shape distances and series alignment

# %%
def _process_saxs_curve(x, y, q_lim=SAXS_Q_LIM, s_lim=SAXS_S_LIM):
    """Crop a SAXS curve to the comparison window and move to log-log space."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = ((x >= q_lim[0]) & (x <= q_lim[1]) &
            (y >= s_lim[0]) & (y <= s_lim[1]) & np.isfinite(y))
    if mask.sum() < 4:
        raise ValueError("Fewer than 4 usable points inside the SAXS window.")

    x_log, y_log = np.log10(x[mask]), np.log10(y[mask])
    x_unique, idx = np.unique(x_log, return_index=True)
    return x_unique, y_log[idx]


def shape_distance_saxs(exp_curve, sim_curve, n_dense=N_DENSE):
    """Amplitude-phase shape distance between two SAXS curves (log-log space)."""
    ex, ey = _process_saxs_curve(exp_curve[:, 0], exp_curve[:, 1])
    sx, sy = _process_saxs_curve(sim_curve[:, 0], sim_curve[:, 1])

    # Compare only where both curves have data, so neither spline extrapolates.
    lo, hi = max(ex.min(), sx.min()), min(ex.max(), sx.max())
    if not hi > lo:
        raise ValueError("Experimental and simulated q-ranges do not overlap.")

    x_dense = np.linspace(lo, hi, n_dense)
    x_scaled = (x_dense - x_dense.min()) / (x_dense.max() - x_dense.min())

    y_exp = CubicSpline(ex, ey)(x_dense)
    y_sim = CubicSpline(sx, sy)(x_dense)

    amplitude, phase = AmplitudePhaseDistance(x_scaled, y_exp, y_sim, **AP_KWARGS)
    return float(amplitude + phase)


def shape_distance_dls(exp_curve, sim_curve, log_size_lim=DLS_LOG_SIZE_LIM,
                       n_dense=N_DENSE):
    """Amplitude-phase shape distance between two DLS size distributions.

    Both curves are splined onto a common log10(size) grid, matching the
    treatment in Scripts/dls_apdist.py.
    """
    x_dense = np.linspace(log_size_lim[0], log_size_lim[1], n_dense)
    x_scaled = (x_dense - x_dense.min()) / (x_dense.max() - x_dense.min())

    def _spline(curve):
        x = np.log10(np.asarray(curve[:, 0], dtype=float))
        y = np.asarray(curve[:, 1], dtype=float)
        order = np.argsort(x)
        x, y = x[order], y[order]
        keep = np.concatenate(([True], np.diff(x) > 0))
        return CubicSpline(x[keep], y[keep])(x_dense)

    y_exp = _spline(exp_curve)
    y_sim = _spline(sim_curve)

    amplitude, phase = AmplitudePhaseDistance(x_scaled, y_exp, y_sim, **AP_KWARGS)
    return float(amplitude + phase)


def build_cost_matrix(exp_series, sim_series, distance_fn, label='', verbose=True):
    """Cost matrix of shape (n_exp, n_sim) of pairwise shape distances."""
    cost = np.full((len(exp_series), len(sim_series)), np.nan)
    start = time.time()

    for i, exp_curve in enumerate(exp_series):
        for j, sim_curve in enumerate(sim_series):
            cost[i, j] = distance_fn(exp_curve, sim_curve)

    if verbose:
        print(f"  {label} cost matrix {cost.shape} in {time.time() - start:.1f}s "
              f"(min {np.nanmin(cost):.4f}, max {np.nanmax(cost):.4f})")
    return cost


def plot_alignment_with_cost(cost_matrix, idx_ap, idx_reference, save_path,
                            title=None):
    """Heatmap of the cost matrix with the alignment paths overlaid."""
    exp_indices = np.arange(cost_matrix.shape[0])

    plt.figure(figsize=(15, 3))
    plt.imshow(cost_matrix, aspect='auto', origin='lower')
    plt.colorbar(label="Distance", aspect=7)
    plt.plot(np.asarray(idx_ap), exp_indices, marker='o', color='red',
             label='Monotone alignment')
    if idx_reference is not None:
        plt.plot(np.asarray(idx_reference), exp_indices, marker='o', color='white',
                 label='Linear alignment')
    plt.xlabel("Simulated frame index")
    plt.ylabel("Exp. index")
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


def score_series(exp_series, sim_series, distance_fn, save_dir, label):
    """Align a simulated series to an experimental series and return its cost.

    Returns
    -------
    score : float
        Total aligned cost divided by the number of experimental curves, so
        SAXS and DLS scores are on a comparable per-curve scale.
    idx_cols : list[int]
        Simulated frame matched to each experimental curve.
    cost : np.ndarray
    """
    if len(sim_series) < len(exp_series):
        raise ValueError(
            f"{label}: need at least as many simulated curves as experimental "
            f"({len(sim_series)} < {len(exp_series)}); raise N_SERIES_FRAMES.")

    cost = build_cost_matrix(exp_series, sim_series, distance_fn, label=label)
    idx_cols, total_cost = series_alignment.align_monotone_min(cost)
    idx_linear, _ = series_alignment.align_monotone_linear(cost)

    score = total_cost / len(exp_series)

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f'{label}_cost_matrix.npy'), cost)
    pd.DataFrame({'exp_index': np.arange(len(exp_series)),
                  'sim_index': idx_cols,
                  'distance': [cost[i, j] for i, j in enumerate(idx_cols)]}
                 ).to_csv(os.path.join(save_dir, f'{label}_alignment.csv'),
                          index=False)
    plot_alignment_with_cost(
        cost, idx_cols, idx_linear,
        os.path.join(save_dir, f'{label}_alignment.png'),
        title=f'{label.upper()} series alignment (score {score:.4f})')

    print(f"  {label}: aligned cost/curve = {score:.4f}, path = {idx_cols}")
    return score, idx_cols, cost


# %% [markdown]
# ## Objective

# %%
def objective(Experiment_Name, iteration, Sample, param_names, *simulation_inputs,
              exp_saxs=None, exp_dls=None):
    """Simulate one candidate potential and score it against both data series."""
    # Accept a single tensor/array/list as well as separate floats.
    if len(simulation_inputs) == 1 and isinstance(
            simulation_inputs[0], (np.ndarray, list, tuple)):
        simulation_inputs = simulation_inputs[0]
    elif len(simulation_inputs) == 1 and hasattr(simulation_inputs[0], 'detach'):
        simulation_inputs = simulation_inputs[0].detach().cpu().numpy()

    if hasattr(simulation_inputs, 'detach'):
        simulation_inputs = simulation_inputs.detach().cpu().numpy()
    simulation_inputs = np.asarray(simulation_inputs, dtype=float).flatten()

    if len(simulation_inputs) != len(param_names):
        raise ValueError(
            f"Expected {len(param_names)} parameters but got {len(simulation_inputs)}.")

    params = dict(zip(param_names, simulation_inputs))

    sample_dir = os.path.join(Experiment_Name, 'Optimization_Results',
                              f'Sample_{Sample}')
    os.makedirs(sample_dir, exist_ok=True)
    pd.DataFrame([simulation_inputs], columns=param_names).to_csv(
        os.path.join(sample_dir, 'input_params.csv'), index=False)

    param_strs = [f"{name}_{np.round(val, 5)}" for name, val in params.items()]
    save_dir = os.path.join(sample_dir, "_".join(param_strs))
    os.makedirs(save_dir, exist_ok=True)

    if exp_saxs is None:
        exp_saxs = load_experimental_saxs()
    if exp_dls is None:
        exp_dls = load_experimental_dls()

    try:
        gsd_file = simulation(*simulation_inputs, save_dir)
        if gsd_file is None or not os.path.exists(gsd_file):
            gsd_file = find_gsd_file(save_dir)

        positions, lattice_coordinates, box_L = read_trajectory(gsd_file)
        frame_indices = select_frame_indices(len(positions), N_SERIES_FRAMES)
        print(f"Scoring {len(frame_indices)} frames out of {len(positions)}")

        print('Converting simulation to a SAXS series')
        sim_saxs = convert_to_SAXS_series(lattice_coordinates, frame_indices, save_dir)

        print('Converting simulation to a DLS series')
        sim_dls = convert_to_DLS_series(positions, frame_indices, box_L,
                                       params['r0'], save_dir)

        print('Aligning against the experimental series')
        saxs_score, _, _ = score_series(exp_saxs, sim_saxs, shape_distance_saxs,
                                        save_dir, 'saxs')
        dls_score, _, _ = score_series(exp_dls, sim_dls, shape_distance_dls,
                                       save_dir, 'dls')

        score = SAXS_WEIGHT * saxs_score + DLS_WEIGHT * dls_score

        pd.DataFrame([{'saxs_score': saxs_score, 'dls_score': dls_score,
                       'saxs_weight': SAXS_WEIGHT, 'dls_weight': DLS_WEIGHT,
                       'score': score}]).to_csv(
            os.path.join(save_dir, 'scores.csv'), index=False)
        print('Score: ', score)
        pd.DataFrame([score]).to_csv(
            os.path.join(sample_dir, f'Score_{score}.csv'), index=False)
    except Exception as e:
        print(f'Evaluation failed: {type(e).__name__}: {e}')
        score = FAILED_SCORE

    return score


# %% [markdown]
# ## Bayesian optimization loop

# %%
def scale_to_range(x_scaled, mins, maxs):
    """Map inputs from the unit cube onto their physical ranges."""
    return mins + x_scaled * (maxs - mins)


def run_optimization(Experiment_Name='20_Bridge_250803', n_iters=100,
                     param_names=('Density', 'U_0', 'r0', 'n', 'm'),
                     target_mins=(0.00005, 0.1, 2.15, 1, 1),
                     target_maxs=(0.005, 150, 2.25, 30, 30)):
    """Minimize the combined SAXS + DLS series-alignment cost with BoTorch."""
    import torch
    from botorch.models import SingleTaskGP
    from botorch.acquisition import qLogExpectedImprovement
    from botorch.optim.optimize import optimize_acqf
    from gpytorch.kernels import ScaleKernel, RBFKernel
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.constraints import GreaterThan
    from gpytorch.mlls import ExactMarginalLogLikelihood

    dtype = torch.float64
    param_names = list(param_names)
    target_mins = torch.tensor(target_mins, dtype=dtype)
    target_maxs = torch.tensor(target_maxs, dtype=dtype)
    bounds = torch.tensor([[0.0] * len(param_names), [1.0] * len(param_names)],
                          dtype=dtype)

    results_dir = os.path.join(Experiment_Name, 'Optimization_Results')
    os.makedirs(results_dir, exist_ok=True)

    # Load the experimental series once and reuse for every candidate.
    exp_saxs = load_experimental_saxs()
    exp_dls = load_experimental_dls()
    print(f"Experimental series: {len(exp_saxs)} SAXS curves, "
          f"{len(exp_dls)} DLS curves")
    if N_SERIES_FRAMES < max(len(exp_saxs), len(exp_dls)):
        raise ValueError(
            f"N_SERIES_FRAMES ({N_SERIES_FRAMES}) must be >= the longest "
            f"experimental series ({max(len(exp_saxs), len(exp_dls))}).")

    def fit_gpytorch_model_with_adam(mll, lr=0.01, steps=100):
        mll.train()
        mll.model.train()
        optimizer = torch.optim.Adam(mll.parameters(), lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            output = mll.model(mll.model.train_inputs[0])
            loss = -mll(output, mll.model.train_targets)
            loss.backward()
            optimizer.step()
        mll.eval()
        mll.model.eval()

    # === Initial design ===
    train_x = torch.rand(1, len(param_names), dtype=dtype)
    y_values = []
    for i, x in enumerate(train_x):
        x_physical = scale_to_range(x, target_mins, target_maxs)
        print('Inputs: ', x_physical)
        y_values.append(objective(Experiment_Name, 0, i, param_names,
                                  *x_physical, exp_saxs=exp_saxs, exp_dls=exp_dls))
    train_y = torch.tensor(y_values, dtype=dtype).unsqueeze(-1)

    best_score_lst = [float(train_y.min())]

    # === BO loop ===
    for i in range(1, n_iters):
        neg_train_y = -train_y

        kernel = ScaleKernel(
            RBFKernel(ard_num_dims=len(param_names),
                      lengthscale_constraint=GreaterThan(1e-3)),
            outputscale_constraint=GreaterThan(1e-3))
        likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-4))

        model = SingleTaskGP(train_x, neg_train_y, covar_module=kernel,
                            likelihood=likelihood)
        model.covar_module.outputscale = torch.tensor(1.0, dtype=dtype)
        model.covar_module.base_kernel.lengthscale = torch.tensor(0.2, dtype=dtype)
        model.likelihood.noise = torch.tensor(1e-2, dtype=dtype)

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model_with_adam(mll)

        acq_func = qLogExpectedImprovement(model=model, best_f=neg_train_y.max())
        new_x, _ = optimize_acqf(acq_func, bounds=bounds, q=1,
                                 num_restarts=5, raw_samples=20)

        new_x_physical = scale_to_range(new_x, target_mins, target_maxs)
        y_values = []
        for x_physical in new_x_physical:
            print('Inputs: ', x_physical)
            y_values.append([objective(Experiment_Name, 0, i, param_names,
                                       *x_physical, exp_saxs=exp_saxs,
                                       exp_dls=exp_dls)])
        new_y = torch.tensor(y_values, dtype=dtype)

        train_x = torch.cat([train_x, new_x], dim=0)
        train_y = torch.cat([train_y, new_y], dim=0)

        best_idx = int(train_y.argmin())
        best_x_physical = scale_to_range(train_x[best_idx], target_mins, target_maxs)

        print(f"Iter {i + 1}:")
        print(f"  Physical input:  {new_x_physical.squeeze().tolist()}")
        print(f"  Objective value: {new_y.item():.6f}")
        print(f"  Best so far:     {train_y[best_idx].item():.6f} "
              f"at {best_x_physical.numpy()} (sample {best_idx})")

        best_score_lst.append(train_y[best_idx].item())

        pd.DataFrame(
            np.hstack([scale_to_range(train_x, target_mins, target_maxs).numpy(),
                       train_y.numpy()]),
            columns=param_names + ['score']).to_csv(
            os.path.join(results_dir, 'all_evaluations.csv'), index=False)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(np.arange(len(best_score_lst)), best_score_lst, linewidth=3)
        ax.scatter(np.arange(len(best_score_lst)), best_score_lst, linewidth=0.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Best aligned cost')
        plt.savefig(os.path.join(results_dir, 'score_plot.png'),
                    dpi=600, bbox_inches="tight")
        plt.close(fig)

    return train_x, train_y


# %% [markdown]
# ## Self-check of the scoring pipeline (no HOOMD required)
#
# Scores the *existing* simulated series in `Data/` against the experimental
# series, exercising exactly the distance + alignment code the objective uses.

# %%
def check_scoring(root=_ROOT, out_dir=None, n_sim=25):
    """Verify the scoring pipeline against already-generated simulated data."""
    out_dir = out_dir or os.path.join(root, 'Figures', 'series_check')
    os.makedirs(out_dir, exist_ok=True)

    exp_saxs = load_experimental_saxs()
    exp_dls = load_experimental_dls()
    print(f"Experimental: {len(exp_saxs)} SAXS, {len(exp_dls)} DLS curves")

    # --- Simulated SAXS series from a previous optimization run ---
    saxs_dir = os.path.join(
        root, 'Data', 'Simulations', 'Simulation_File_8', 'Optimization_Results',
        'Sample_0', 'Density_0.005_U_0_46.95_r0_2.25_n_1.0_m_20.89',
        'scattering_data_mcdfm_sq')
    saxs_files = sorted(glob.glob(os.path.join(saxs_dir, '*.npy')))[:n_sim]
    sim_saxs = [np.load(f) for f in saxs_files]
    print(f"Simulated SAXS series: {len(sim_saxs)} curves")
    saxs_score, saxs_path, _ = score_series(
        exp_saxs, sim_saxs, shape_distance_saxs, out_dir, 'saxs')

    # --- Simulated DLS series from the OVITO cluster exports ---
    dls_dir = os.path.join(root, 'Data', 'DLS', 'Simulated_DLS_10')
    dls_files = sorted(glob.glob(os.path.join(dls_dir, 'cluster_size_*')))[:n_sim]
    sim_dls = [dls_distribution_from_clusters(clusters_from_ovito_file(f))
               for f in dls_files]
    print(f"Simulated DLS series: {len(sim_dls)} curves")
    dls_score, dls_path, _ = score_series(
        exp_dls, sim_dls, shape_distance_dls, out_dir, 'dls')

    combined = SAXS_WEIGHT * saxs_score + DLS_WEIGHT * dls_score
    print(f"\nCombined objective = {SAXS_WEIGHT} * {saxs_score:.4f} + "
          f"{DLS_WEIGHT} * {dls_score:.4f} = {combined:.4f}")
    print(f"Plots and cost matrices written to {out_dir}")
    return combined


# %%
if __name__ == '__main__':
    if '--check' in sys.argv:
        check_scoring()
    else:
        run_optimization()
