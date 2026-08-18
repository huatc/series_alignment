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
import re
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
from matplotlib.colors import Normalize               # noqa: E402
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
N_SERIES_FRAMES = 75
# Frames at each end of the trajectory that are always taken consecutively.
# Assembly changes fastest just after the quench and the structure settles at
# the end, so both ends are sampled densely and the middle is filled evenly.
N_EDGE_FRAMES = 20

# Objective weights for the two modalities (each already normalized per curve).
#SAXS_WEIGHT = 1.0
#DLS_WEIGHT = 0.0

#SAXS_WEIGHT = 30
#DLS_WEIGHT = 5

SAXS_WEIGHT = 1
DLS_WEIGHT = 0

# Experimental series sizes.  35 SAXS files exist (10 min each, so the full run
# reaches 350 min) but assembly stops changing measurably after ~90 min, so only
# the first 9 are scored.  That ends both series at 90 min, which means every
# time point in the grid carries a DLS curve and none is scored on SAXS alone.
N_EXP_SAXS = 9
N_EXP_DLS = 18

# --- Experimental acquisition times (minutes) ---
# Each measurement is stamped at the END of its acquisition window: a SAXS frame
# takes 10 min so the first is complete at t = 10, and a DLS frame takes 5 min so
# the first is complete at t = 5.  The grids therefore coincide every 10 min.
#
# A simulated frame is compared against whichever experimental curves were
# recorded at the same time, and both modalities share one alignment: a single
# trajectory cannot sit at two points in its own evolution at once, which is what
# independent per-modality alignment implied.
SAXS_START_MIN = 10.0        # first SAXS frame complete at t = 10 min
SAXS_INTERVAL_MIN = 10.0     # -> 10, 20, ... 100 min for the 10 curves used
DLS_START_MIN = 5.0          # first DLS frame complete at t = 5 min
DLS_INTERVAL_MIN = 5.0       # -> 5, 10, ... 90 min for the 18 curves
TIME_MATCH_TOL_MIN = 1e-6    # tolerance when pairing SAXS and DLS times

# How the two acquisition grids are combined:
#   'intersection' -> only times where BOTH were measured (t = 10, 20 ... 90),
#                     giving 9 points each carrying one SAXS and one DLS curve.
#                     The 5-minute DLS measurements in between are not scored.
#   'union'        -> every time either was measured (18 points here); the
#                     intermediate DLS curves contribute on their own.
TIME_GRID_MODE = 'intersection'

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

# One simulation length unit -> metres.  Soft Matter 2025, 21, 9398 (Fig. 8):
# "one reduced unit is equal to the diameter of the silica nanoparticle (25 nm)".
# The SAXS path already assumes this (q_phys = q_sim / 260 puts 1 unit at 260 A),
# so the two modalities now share one length scale.  The old value here,
# 2.0 * 25 * 2 * 1e-9 = 100 nm, carried two spurious factors of 2 and made the
# DLS analysis read the same trajectory at 4x the SAXS scale.
SIM_LENGTH_TO_M = 25e-9
RG_TO_RH = 1.0 / 0.77           # Rh from Rg, assuming compact/spherical clusters

# The Zetasizer reports an intensity-weighted hydrodynamic DIAMETER, while
# Stokes-Einstein yields a radius, so the simulated distribution is doubled to
# put both axes in the same units.  Affects scoring and plots alike.
DLS_SIM_AS_DIAMETER = True

# Bond cutoff for cluster analysis.  The paper's OVITO analysis used an absolute
# cutoff of 3.5 reduced units; set CLUSTER_CUTOFF_ABS to None to fall back to
# CLUSTER_CUTOFF_FACTOR * r0 (the previous behaviour, ~2.9 at r0 = 2.33, which
# is tighter and splits single aggregates into several counted clusters).
CLUSTER_CUTOFF_ABS = 3.5
CLUSTER_CUTOFF_FACTOR = 1.25
MIN_CLUSTER_SIZE = 2
# Radius of gyration of a single solid sphere, sqrt(3/5)*R, in simulation units.
# No cluster can be more compact than one building block, so this is the floor
# for Rg.  Without it a fully collapsed cluster gives Rg = 0, and build_g2's
# Stokes-Einstein step (a pure-Python division) raises ZeroDivisionError.
MONOMER_RG = float(np.sqrt(3.0 / 5.0) * (BUILDING_BLOCK_D / 2.0))

# Shared amplitude-phase distance options (dynamic-programming warp search).
AP_KWARGS = {"optim": "DP", "grid_dim": 10}

# SAXS distance metric: 'log_mse' (mean squared difference of log10 S(q)) or
# 'apdist' (elastic amplitude-phase).  DLS always uses the amplitude-phase
# distance.  Note the two are on very different numeric scales, so SAXS_WEIGHT
# needs revisiting whenever this changes.
SAXS_METRIC = 'log_mse'
# Subtract the mean log residual before squaring, which lets the simulated curve
# float by one overall scale factor so log_mse grades shape rather than level.
SAXS_LOG_MSE_MATCH_OFFSET = False

N_DENSE = 100          # points on the common grid used for the AP distance
FAILED_SCORE = 999.0   # sentinel returned when a candidate cannot be evaluated

# --- Bayesian-optimization settings ---
MIN_EXPONENT_GAP = 1.0   # smallest |n - m| that is numerically safe
MIN_GP_POINTS = 3        # successful evaluations needed before fitting a GP


# %% [markdown]
# ## Simulation (HOOMD, Linux/GPU only)

# %%

F_CAP = 1.0e4          # F*dt^2 = 0.01 sigma per step at dt = 1e-3

def modified_LJ(r, rmin, rmax, U_0, n, m, r0):
    """Generalized (n, m) Lennard-Jones potential and force."""
    U = U_0 / (n - m) * (m * (r0 / r) ** n - n * (r0 / r) ** m)
    F = U_0 * m * n * ((r0 / r) ** n - (r0 / r) ** m) / ((n - m) * r)
    # Cap the repulsive core so a steep wall cannot eject a particle in one step
    return np.clip(U, -F_CAP, F_CAP), np.clip(F, -F_CAP, F_CAP)


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
               N=1000, dt=0.001, steps=500_000, kT=1.0, gsd_period=1_000):
    """Run Langevin dynamics of N attractive spheres and dump a GSD trajectory."""
    if not _HAS_HOOMD:
        raise RuntimeError(
            "HOOMD is not available: the simulation only runs on Linux/GPU. "
            "Use --check to exercise the scoring pipeline without it.")

    #hoomd.context.initialize("--mode=gpu")
    hoomd.context.initialize(os.environ.get('HOOMD_MODE', '--mode=cpu'))
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


def select_frame_indices(n_frames, n_series, n_edge=N_EDGE_FRAMES):
    """Pick n_series frames: the first n_edge, the last n_edge, and an even
    spread of the remainder in between.

    Nucleation happens in the first few frames and the structure stops evolving
    in the last few, so sampling both ends consecutively captures the fast
    early kinetics and the converged state, while the evenly spaced middle
    covers the slow growth regime.

    The result is sorted and duplicate-free, so the simulated series stays
    time-ordered for the monotone alignment. Exactly n_series frames are
    returned whenever the trajectory is long enough.
    """
    if n_frames <= 0:
        raise ValueError("Trajectory contains no frames.")

    n_series = int(min(n_series, n_frames))
    if n_frames <= n_series:
        return list(range(n_frames))

    # Never let the two ends overlap or crowd out the middle entirely.
    n_edge = int(min(n_edge, n_frames // 2, n_series // 2))
    if n_edge <= 0:
        return np.unique(
            np.linspace(0, n_frames - 1, n_series).astype(int)).tolist()

    head = list(range(n_edge))
    tail = list(range(n_frames - n_edge, n_frames))
    idx = set(head) | set(tail)

    n_mid = n_series - len(idx)
    lo, hi = head[-1] + 1, tail[0] - 1
    if n_mid > 0 and hi >= lo:
        idx |= {int(v) for v in np.linspace(lo, hi, n_mid).round().astype(int)}

    # Rounding can collapse neighbouring middle frames; top up from whatever is
    # left so the caller still gets n_series curves.
    if len(idx) < n_series:
        spare = [f for f in range(n_frames) if f not in idx]
        if spare:
            take = min(n_series - len(idx), len(spare))
            idx |= {spare[int(round(p))]
                    for p in np.linspace(0, len(spare) - 1, take)}

    return sorted(idx)


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
        I_q = simulator.simulate_scattering_curve_fast_lattice(
            points, coords, SAXS_HIST_BINS, q_sim, save=False).cpu().numpy()
        #I_q = np.mean(I_q, axis=1)

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
        # Particles that collapse into the soft core can coincide exactly; floor
        # Rg at the single-sphere value so Rh stays physical and non-zero.
        rg_sim = max(float(rg_sim), MONOMER_RG)
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
    if DLS_SIM_AS_DIAMETER:
        # Stokes-Einstein gives a hydrodynamic RADIUS, but the Zetasizer's
        # intensity-weighted distribution is in hydrodynamic DIAMETER (see the
        # paper's Methods), so both axes must be diameters before they are
        # compared.  Applied here so scoring and plotting stay consistent.
        Rh_nm = 2.0 * Rh_nm
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

    cutoff = (CLUSTER_CUTOFF_ABS if CLUSTER_CUTOFF_ABS is not None
              else CLUSTER_CUTOFF_FACTOR * r0)
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
def _load_curve(filepath):
    """Load a 2-column [x, y] .npy curve, tolerating object/pickled arrays."""
    try:
        arr = np.load(filepath)
    except ValueError:
        # dtype=object file (e.g. np.save on a ragged list) needs pickle to read
        arr = np.load(filepath, allow_pickle=True)

    # Unwrap 0-d object arrays from np.save(path, <python object>)
    if arr.dtype == object:
        if arr.shape == ():
            arr = arr.item()
        arr = np.asarray(arr)

    try:
        arr = np.asarray(arr, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"{os.path.basename(filepath)}: not a numeric array ({exc}). "
            "Looks ragged/object -- re-save it as a plain (N, 2) float array."
        ) from exc

    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(
            f"{os.path.basename(filepath)}: expected an (N, 2) [q, I] array, "
            f"got shape {arr.shape}"
        )
    return arr[:, :2]

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
    return [_load_curve(os.path.join(path, f)) for f in names]


def _numeric_key(filename):
    """Sort key from the first integer in a filename.

    Plain sorted() puts data_10 immediately after data_1, which would scramble
    the kinetic order and silently misalign every curve in time.
    """
    match = re.search(r'(\d+)', os.path.basename(filename))
    return int(match.group(1)) if match else -1


def load_experimental_dls(n_samples=N_EXP_DLS, root=_ROOT, n_replicates=3):
    """Load the experimental DLS kinetic series as [size_nm, scaled_intensity].

    Prefers the per-time-point arrays in Data/DLS/Kinetics_data, which are the
    replicate-averaged, MinMax-scaled curves already extracted from the
    spreadsheet. Falls back to re-deriving them from Assembly_kinetics_data.xlsx
    if that folder is missing.
    """
    path = os.path.join(root, 'Data', 'DLS', 'Kinetics_data')
    if os.path.isdir(path):
        names = sorted((f for f in os.listdir(path) if f.endswith('.npy')),
                       key=_numeric_key)
        if len(names) < n_samples:
            raise ValueError(
                f"{path} holds {len(names)} curves but {n_samples} were "
                f"requested; lower N_EXP_DLS.")
        return [_load_curve(os.path.join(path, f)) for f in names[:n_samples]]

    xlsx = os.path.join(root, 'Data', 'DLS', 'Assembly_kinetics_data.xlsx')
    data = pd.read_excel(xlsx).values[:, 5:].T.astype(float)

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


def _prepare_saxs_pair(exp_curve, sim_curve, n_dense=N_DENSE):
    """Spline a SAXS pair onto the shared log-q grid the distance is scored on.

    Split out from shape_distance_saxs so the diagnostic plots can draw exactly
    the curves the metric saw, rather than an independently processed version.
    """
    ex, ey = _process_saxs_curve(exp_curve[:, 0], exp_curve[:, 1])
    sx, sy = _process_saxs_curve(sim_curve[:, 0], sim_curve[:, 1])

    # Compare only where both curves have data, so neither spline extrapolates.
    lo, hi = max(ex.min(), sx.min()), min(ex.max(), sx.max())
    if not hi > lo:
        raise ValueError("Experimental and simulated q-ranges do not overlap.")

    x_dense = np.linspace(lo, hi, n_dense)
    return x_dense, CubicSpline(ex, ey)(x_dense), CubicSpline(sx, sy)(x_dense)


def shape_distance_saxs(exp_curve, sim_curve, n_dense=N_DENSE, metric=None):
    """Distance between two SAXS curves on the shared log-q grid.

    'log_mse' : mean squared difference of log10 S(q). A plain point-wise
        comparison -- it penalises a peak that is in the wrong place, which the
        elastic distance partly forgives by warping the q axis to line peaks up.
    'apdist'  : amplitude-phase (elastic) distance, the previous default.

    See SAXS_METRIC.
    """
    metric = SAXS_METRIC if metric is None else metric
    x_dense, y_exp, y_sim = _prepare_saxs_pair(exp_curve, sim_curve, n_dense)

    if metric == 'log_mse':
        # _prepare_saxs_pair already returns log10 S(q), so this is the MSE
        # in log space; no further transform is needed.
        residual = y_exp - y_sim
        if SAXS_LOG_MSE_MATCH_OFFSET:
            # Remove a constant log offset, i.e. allow the simulated curve to be
            # rescaled by one factor, so the metric grades shape not scale.
            residual = residual - residual.mean()
        return float(np.mean(residual ** 2))

    if metric == 'apdist':
        x_scaled = (x_dense - x_dense.min()) / (x_dense.max() - x_dense.min())
        amplitude, phase = AmplitudePhaseDistance(
            x_scaled, y_exp, y_sim, **AP_KWARGS)
        return float(amplitude + phase)

    raise ValueError(f"SAXS metric must be 'log_mse' or 'apdist', got {metric!r}")


def _prepare_dls_pair(exp_curve, sim_curve, log_size_lim=DLS_LOG_SIZE_LIM,
                      n_dense=N_DENSE):
    """Spline a DLS pair onto the common log10(size) grid used for scoring.

    Matches the treatment in Scripts/dls_apdist.py.
    """
    x_dense = np.linspace(log_size_lim[0], log_size_lim[1], n_dense)

    def _spline(curve):
        x = np.log10(np.asarray(curve[:, 0], dtype=float))
        y = np.asarray(curve[:, 1], dtype=float)
        order = np.argsort(x)
        x, y = x[order], y[order]
        keep = np.concatenate(([True], np.diff(x) > 0))
        return CubicSpline(x[keep], y[keep])(x_dense)

    return x_dense, _spline(exp_curve), _spline(sim_curve)


def shape_distance_dls(exp_curve, sim_curve, log_size_lim=DLS_LOG_SIZE_LIM,
                       n_dense=N_DENSE):
    """Amplitude-phase shape distance between two DLS size distributions."""
    x_dense, y_exp, y_sim = _prepare_dls_pair(
        exp_curve, sim_curve, log_size_lim, n_dense)
    x_scaled = (x_dense - x_dense.min()) / (x_dense.max() - x_dense.min())

    amplitude, phase = AmplitudePhaseDistance(x_scaled, y_exp, y_sim, **AP_KWARGS)
    return float(amplitude + phase)


# Acquisition times (minutes) for the experimental series, used to colour the
# overlay figures.  Leave as None to colour by curve index instead.
EXP_TIMES_SAXS = None
EXP_TIMES_DLS = None

# Kept only so older saved figures/labels still resolve; the radius -> diameter
# conversion now happens in dls_distribution_from_clusters via
# DLS_SIM_AS_DIAMETER, which affects scoring as well as plotting.
SIM_RH_TO_DIAMETER = False

# Panels drawn in the overlay figures: SAXS as 3x3 and DLS as 6x3.  Scoring and
# alignment always use every experimental curve; this only trims the figure.
OVERLAY_MAX_PANELS = {'saxs': 9, 'dls': 18}
OVERLAY_NCOLS = 3

# Font sizes for the overlay figures.  Sized for a figure that will be shrunk
# into a paper column, so the axis and colourbar text stays readable.
OVERLAY_FONTS = {
    'axis_label': 24,     # the shared "q (1/A)" / "Intensity" labels
    'cbar_label': 22,     # "Experimental Data Time (Mins)"
    'cbar_ticks': 20,     # the numbers up the colourbar
    'panel_label': 14,    # the "(exp, sim)" tag in each panel
    'legend': 16,
    'tick_label': 18,     # axis tick numbers (cost-matrix heatmap)
    'title': 20,
}


def _overlay_pair(exp_curve, sim_curve, label):
    """Return (exp_x, exp_y, sim_x, sim_y) in native units for plotting.

    The experimental curve keeps its raw sampling so the scatter shows real
    measured points; both are cropped to the same window the distance uses.
    """
    if label == 'saxs':
        ex, ey = _process_saxs_curve(exp_curve[:, 0], exp_curve[:, 1])
        sx, sy = _process_saxs_curve(sim_curve[:, 0], sim_curve[:, 1])
        return 10 ** ex, 10 ** ey, 10 ** sx, 10 ** sy

    lo, hi = DLS_LOG_SIZE_LIM
    out = []
    for curve, is_sim in ((exp_curve, False), (sim_curve, True)):
        x = np.asarray(curve[:, 0], dtype=float)
        y = np.asarray(curve[:, 1], dtype=float)
        # No conversion here: dls_distribution_from_clusters has already put the
        # simulated curve on the same (diameter) axis as the experiment.
        keep = (x > 0)
        x, y = x[keep], y[keep]
        keep = (np.log10(x) >= lo) & (np.log10(x) <= hi)
        out += [x[keep], y[keep]]
    return tuple(out)


def plot_alignment_overlay(exp_series, sim_series, idx_cols, save_path, label,
                           exp_times=None, ncols=OVERLAY_NCOLS, cmap='jet',
                           annotate=True, max_panels=None,
                           time_label='Experimental Data Time (Mins)',
                           panel_size=(3.1, 2.6), fonts=None):
    """Publication-style overlay of each experimental curve on its match.

    One tightly packed panel per experimental curve: the measured data as
    scatter points coloured by acquisition time, and the simulated frame the
    monotone alignment chose as a black line. A shared colourbar maps colour to
    time, and each panel is annotated with its (experimental, simulated) pair.

    exp_times : sequence of float, optional
        Acquisition time per experimental curve. When omitted the panels are
        coloured by curve index and the colourbar is relabelled accordingly.
    max_panels : int, optional
        Draw only the first max_panels experimental curves, so the grid comes
        out a chosen shape (9 for a 3x3 SAXS figure, 18 for a 6x3 DLS one).
        Scoring and alignment are unaffected -- this trims the figure only.
    fonts : dict, optional
        Font sizes, merged over OVERLAY_FONTS. Keys: 'axis_label',
        'cbar_label', 'cbar_ticks', 'panel_label', 'legend'.
    """
    fs = dict(OVERLAY_FONTS)
    if fonts:
        fs.update(fonts)
    n_total = len(exp_series)
    n = int(min(n_total, max_panels)) if max_panels else n_total
    ncols = int(min(ncols, n))
    nrows = int(np.ceil(n / ncols))

    if exp_times is None:
        times = np.arange(n_total, dtype=float)
        time_label = 'Experimental curve index'
    else:
        times = np.asarray(exp_times, dtype=float)
        if len(times) != n_total:
            raise ValueError(
                f"exp_times has {len(times)} entries but there are "
                f"{n_total} curves.")
    times = times[:n]        # colour spans the panels actually shown

    norm = Normalize(vmin=float(times.min()), vmax=float(times.max()))
    colormap = plt.get_cmap(cmap)

    fig, axes = plt.subplots(
        nrows, ncols, squeeze=False, sharex=True, sharey=True,
        figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
        gridspec_kw={'wspace': 0, 'hspace': 0})

    sim_handle = exp_handle = None
    for k in range(nrows * ncols):
        ax = axes[k // ncols][k % ncols]
        if k >= n:
            ax.axis('off')
            continue

        j = int(idx_cols[k])
        colour = colormap(norm(times[k]))
        try:
            ex, ey, sx, sy = _overlay_pair(exp_series[k], sim_series[j], label)
        except Exception as exc:
            ax.text(0.5, 0.5, f'failed:\n{exc}', ha='center', va='center',
                    transform=ax.transAxes, fontsize=7, color='crimson')
            continue

        exp_handle = ax.scatter(ex, ey, s=12, color=colour, zorder=2)
        sim_handle, = ax.plot(sx, sy, color='k', lw=2.2, zorder=3)

        ax.set_xscale('log')
        if label == 'saxs':
            ax.set_yscale('log')
        if annotate:
            ax.text(0.03, 0.94, f'({k + 1}, {j})', transform=ax.transAxes,
                    fontsize=fs['panel_label'], va='top', ha='left')
        ax.tick_params(direction='in', which='both', labelbottom=False,
                       labelleft=False, top=True, right=True)

    if label == 'saxs':
        xlabel, ylabel = r'q ($\mathrm{\AA}^{-1}$)', 'Intensity (arb. unit)'
    else:
        size = 'Hydrodynamic Diameter' if DLS_SIM_AS_DIAMETER else 'Hydrodynamic Radius'
        xlabel, ylabel = f'{size} (nm)', 'Intensity (arb. unit)'
    fig.supxlabel(xlabel, fontsize=fs['axis_label'])
    fig.supylabel(ylabel, fontsize=fs['axis_label'])

    handles = [h for h in (sim_handle, exp_handle) if h is not None]
    if handles:
        fig.legend(handles, ['Simulated data', 'Experimental data'][:len(handles)],
                   loc='upper center', bbox_to_anchor=(0.5, 1.0),
                   ncol=2, fontsize=fs['legend'], frameon=True)

    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.04, pad=0.02)
    cbar.set_label(time_label, fontsize=fs['cbar_label'])
    cbar.ax.tick_params(labelsize=fs['cbar_ticks'])

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


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


def plot_alignment_with_cost(cost_matrix, idx_ap, save_path, title=None,
                             fonts=None, figsize=(15, 4.5)):
    """Heatmap of the cost matrix with the alignment path overlaid.

    Font sizes come from OVERLAY_FONTS so this matches the overlay figures;
    pass `fonts` to override individual keys. The default figure is taller than
    the original 15x3 so the enlarged axis and colourbar text has room.
    """
    fs = dict(OVERLAY_FONTS)
    if fonts:
        fs.update(fonts)

    exp_indices = np.arange(cost_matrix.shape[0])

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cost_matrix, aspect='auto', origin='lower')

    cbar = fig.colorbar(im, ax=ax, aspect=7)
    cbar.set_label("Distance", fontsize=fs['cbar_label'])
    cbar.ax.tick_params(labelsize=fs['cbar_ticks'])

    ax.plot(np.asarray(idx_ap), exp_indices, marker='o', color='red',
            label='Monotone alignment')
    ax.set_xlabel("Simulated frame index", fontsize=fs['axis_label'])
    ax.set_ylabel("Exp. index", fontsize=fs['axis_label'])
    ax.tick_params(labelsize=fs['tick_label'])
    if title:
        ax.set_title(title, fontsize=fs['title'])
    ax.legend(fontsize=fs['legend'])

    fig.tight_layout()
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


# Per-modality curve preparation and axis labels, used by the diagnostic plots.
CURVE_AXES = {
    'saxs': (_prepare_saxs_pair,
             r'$\log_{10}\,q\ (\mathrm{\AA}^{-1})$',
             r'$\log_{10}\,S(q)$'),
    'dls': (_prepare_dls_pair,
            r'$\log_{10}\,R_h\ (\mathrm{nm})$',
            'Normalized intensity'),
}


def plot_aligned_curves(exp_series, sim_series, idx_cols, cost, save_path, label,
                        score=None, ncols=5):
    """Overlay every experimental curve on the simulated frame it matched.

    One panel per experimental curve, drawn in exactly the space the
    amplitude-phase distance saw them (splined onto the shared dense grid), so
    the panels explain the numbers in the cost matrix rather than approximating
    them. Panel titles carry the pairing and its distance.
    """
    prepare, xlabel, ylabel = CURVE_AXES[label]
    n = len(exp_series)
    ncols = int(min(ncols, n))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.9 * nrows),
                             squeeze=False)
    for k in range(nrows * ncols):
        ax = axes[k // ncols][k % ncols]
        if k >= n:
            ax.axis('off')
            continue

        j = int(idx_cols[k])
        try:
            x, y_exp, y_sim = prepare(exp_series[k], sim_series[j])
        except Exception as exc:                       # keep one bad pair local
            ax.text(0.5, 0.5, f'failed:\n{exc}', ha='center', va='center',
                    transform=ax.transAxes, fontsize=7, color='crimson')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'exp {k}  <-  sim {j}', fontsize=9)
            continue

        ax.plot(x, y_exp, color='k', lw=2.0, label='experiment')
        ax.plot(x, y_sim, color='crimson', lw=1.6, ls='--', label='simulation')
        ax.set_title(f'exp {k}  <-  sim {j}   d = {cost[k, j]:.3f}', fontsize=9)
        ax.tick_params(labelsize=7)
        if k % ncols == 0:
            ax.set_ylabel(ylabel, fontsize=8)
        if k // ncols == nrows - 1:
            ax.set_xlabel(xlabel, fontsize=8)
        if k == 0:
            ax.legend(fontsize=7, frameon=False)

    title = f'{label.upper()} aligned pairs'
    if score is not None:
        title += f'   (aligned cost/curve = {score:.4f})'
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


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

    score = total_cost / len(exp_series)

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f'{label}_cost_matrix.npy'), cost)
    pd.DataFrame({'exp_index': np.arange(len(exp_series)),
                  'sim_index': idx_cols,
                  'distance': [cost[i, j] for i, j in enumerate(idx_cols)]}
                 ).to_csv(os.path.join(save_dir, f'{label}_alignment.csv'),
                          index=False)
    plot_alignment_with_cost(
        cost, idx_cols,
        os.path.join(save_dir, f'{label}_alignment.png'),
        title=f'{label.upper()} series alignment (score {score:.4f})')
    plot_aligned_curves(
        exp_series, sim_series, idx_cols, cost,
        os.path.join(save_dir, f'{label}_aligned_curves.png'),
        label, score=score)
    plot_alignment_overlay(
        exp_series, sim_series, idx_cols,
        os.path.join(save_dir, f'{label}_overlay.png'), label,
        exp_times=EXP_TIMES_SAXS if label == 'saxs' else EXP_TIMES_DLS,
        max_panels=OVERLAY_MAX_PANELS.get(label))

    print(f"  {label}: aligned cost/curve = {score:.4f}, path = {idx_cols}")
    return score, idx_cols, cost


# %% [markdown]
# ## Joint (time-matched) scoring
#
# SAXS and DLS were measured on the same sample at known times, so a simulated
# frame must be compared with the experimental curves recorded at the *same*
# time and both modalities must share one alignment.

# %%
def experimental_times(n_curves, start, interval):
    """Acquisition time of each experimental curve, in minutes."""
    return start + interval * np.arange(int(n_curves), dtype=float)


def build_time_grid(n_saxs, n_dls,
                    saxs_start=SAXS_START_MIN, saxs_interval=SAXS_INTERVAL_MIN,
                    dls_start=DLS_START_MIN, dls_interval=DLS_INTERVAL_MIN,
                    tol=TIME_MATCH_TOL_MIN, mode=None):
    """Time points at which the simulation is compared with experiment.

    Returns a list of (time_min, saxs_index or None, dls_index or None).

    With SAXS on 10-minute and DLS on 5-minute intervals, 'intersection' keeps
    the 10-minute marks where both were measured -- one SAXS and one DLS curve
    per point -- while 'union' also keeps the intervening DLS-only points.
    See TIME_GRID_MODE.
    """
    mode = TIME_GRID_MODE if mode is None else mode
    if mode not in ('intersection', 'union'):
        raise ValueError(f"mode must be 'intersection' or 'union', got {mode!r}")

    t_saxs = experimental_times(n_saxs, saxs_start, saxs_interval)
    t_dls = experimental_times(n_dls, dls_start, dls_interval)

    grid = []
    for t in np.unique(np.concatenate([t_saxs, t_dls])):
        i = np.flatnonzero(np.abs(t_saxs - t) <= tol)
        j = np.flatnonzero(np.abs(t_dls - t) <= tol)
        i = int(i[0]) if i.size else None
        j = int(j[0]) if j.size else None
        if mode == 'intersection' and (i is None or j is None):
            continue
        grid.append((float(t), i, j))

    if not grid:
        raise ValueError(
            "No usable time points. Check SAXS_START_MIN / SAXS_INTERVAL_MIN "
            "against DLS_START_MIN / DLS_INTERVAL_MIN -- under 'intersection' "
            "the two grids must actually coincide somewhere.")
    return grid


def build_joint_cost_matrix(exp_saxs, exp_dls, sim_saxs, sim_dls, time_grid,
                            saxs_weight=SAXS_WEIGHT, dls_weight=DLS_WEIGHT,
                            normalize_rows=True, verbose=True):
    """Cost of matching each experimental *time point* to each simulated frame.

    Both modalities are scored against the same simulated frame, so the two
    simulated series must be parallel -- element k of each must come from the
    same trajectory frame.

    Rows are normalized by the weight actually present, so a time point carrying
    only DLS is not systematically cheaper than one carrying both and therefore
    does not bias the alignment path.

    Returns (cost, saxs_part, dls_part) where the parts hold the raw per-modality
    distances (NaN where that modality was not measured at that time).
    """
    n_sim = len(sim_saxs)
    if len(sim_dls) != n_sim:
        raise ValueError(
            f"Simulated series must be parallel: {n_sim} SAXS frames vs "
            f"{len(sim_dls)} DLS frames.")

    n_t = len(time_grid)
    cost = np.zeros((n_t, n_sim))
    saxs_part = np.full((n_t, n_sim), np.nan)
    dls_part = np.full((n_t, n_sim), np.nan)
    start = time.time()

    for r, (t, i_saxs, i_dls) in enumerate(time_grid):
        row = np.zeros(n_sim)
        w_total = 0.0
        if i_saxs is not None:
            d = np.array([shape_distance_saxs(exp_saxs[i_saxs], s)
                          for s in sim_saxs])
            saxs_part[r] = d
            row += saxs_weight * d
            w_total += saxs_weight
        if i_dls is not None:
            d = np.array([shape_distance_dls(exp_dls[i_dls], s)
                          for s in sim_dls])
            dls_part[r] = d
            row += dls_weight * d
            w_total += dls_weight
        if w_total == 0.0:
            raise ValueError(f"No experimental data at t = {t} min.")
        cost[r] = row / w_total if normalize_rows else row

    if verbose:
        both = sum(1 for _, i, j in time_grid if i is not None and j is not None)
        print(f"  joint cost matrix {cost.shape} in {time.time() - start:.1f}s "
              f"({both}/{n_t} time points carry both modalities, "
              f"min {cost.min():.4f}, max {cost.max():.4f})")
    return cost, saxs_part, dls_part


def score_series_joint(exp_saxs, exp_dls, sim_saxs, sim_dls, save_dir,
                       time_grid=None, frame_indices=None):
    """Align both modalities to the simulation on a shared experimental clock.

    Returns
    -------
    score : float
        Aligned cost per experimental time point.
    idx_cols : list[int]
        Simulated frame matched to each time point.
    info : dict
        Cost matrices, the time grid and the per-modality score breakdown.
    """
    if time_grid is None:
        time_grid = build_time_grid(len(exp_saxs), len(exp_dls))

    if len(sim_saxs) < len(time_grid):
        raise ValueError(
            f"Need at least as many simulated frames as experimental time "
            f"points ({len(sim_saxs)} < {len(time_grid)}); raise "
            f"N_SERIES_FRAMES or shorten the experimental series.")

    cost, saxs_part, dls_part = build_joint_cost_matrix(
        exp_saxs, exp_dls, sim_saxs, sim_dls, time_grid)
    idx_cols, total_cost = series_alignment.align_monotone_min(cost)
    score = total_cost / len(time_grid)

    # Per-modality means along the shared path, for reporting only.
    chosen = [(r, int(j)) for r, j in enumerate(idx_cols)]
    saxs_vals = [saxs_part[r, j] for r, j in chosen if not np.isnan(saxs_part[r, j])]
    dls_vals = [dls_part[r, j] for r, j in chosen if not np.isnan(dls_part[r, j])]
    saxs_mean = float(np.mean(saxs_vals)) if saxs_vals else float('nan')
    dls_mean = float(np.mean(dls_vals)) if dls_vals else float('nan')

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'joint_cost_matrix.npy'), cost)

    rows = []
    for r, (t, i_saxs, i_dls) in enumerate(time_grid):
        j = int(idx_cols[r])
        rows.append({
            'time_min': t,
            'saxs_index': i_saxs, 'dls_index': i_dls,
            'sim_frame_index': j,
            'sim_frame': None if frame_indices is None else int(frame_indices[j]),
            'saxs_distance': saxs_part[r, j],
            'dls_distance': dls_part[r, j],
            'combined': cost[r, j]})
    pd.DataFrame(rows).to_csv(
        os.path.join(save_dir, 'joint_alignment.csv'), index=False)

    plot_alignment_with_cost(
        cost, idx_cols, os.path.join(save_dir, 'joint_alignment.png'),
        title=f'Time-matched SAXS + DLS alignment (score {score:.4f})')

    # Per-modality figures, both driven by the one shared path.
    for label, part, exp_series, sim_series in (
            ('saxs', saxs_part, exp_saxs, sim_saxs),
            ('dls', dls_part, exp_dls, sim_dls)):
        key = 1 if label == 'saxs' else 2
        pairs = [(g[key], int(idx_cols[r]), g[0])
                 for r, g in enumerate(time_grid) if g[key] is not None]
        if not pairs:
            continue
        sub_exp = [exp_series[i] for i, _, _ in pairs]
        sub_idx = [j for _, j, _ in pairs]
        sub_times = [t for _, _, t in pairs]
        sub_cost = np.array(
            [[part[r, jj] for jj in range(cost.shape[1])]
             for r, g in enumerate(time_grid) if g[key] is not None])

        plot_aligned_curves(
            sub_exp, sim_series, sub_idx, sub_cost,
            os.path.join(save_dir, f'{label}_aligned_curves.png'),
            label, score=saxs_mean if label == 'saxs' else dls_mean)
        plot_alignment_overlay(
            sub_exp, sim_series, sub_idx,
            os.path.join(save_dir, f'{label}_overlay.png'), label,
            exp_times=sub_times, max_panels=OVERLAY_MAX_PANELS.get(label))

    print(f"  joint: cost/time-point = {score:.4f} "
          f"(SAXS {saxs_mean:.4f}, DLS {dls_mean:.4f})")
    print(f"  path = {[int(j) for j in idx_cols]}")
    return score, idx_cols, {
        'cost': cost, 'saxs_part': saxs_part, 'dls_part': dls_part,
        'time_grid': time_grid, 'saxs_mean': saxs_mean, 'dls_mean': dls_mean}


# %% [markdown]
# ## Objective

# %%

    
def bo_params_to_physical(values, param_names):
    """Map the optimizer's parameters onto simulation()'s (density, U_0, r0, n, m).

    The (n, m) potential is symmetric under swapping n and m, and singular at
    n == m.  Searching both independently therefore wastes half the domain on
    mirror-image duplicates and puts a 0/0 singularity in the middle of it.
    When the optimizer searches 'm' and 'gap' instead, n = m + gap reaches every
    distinct potential exactly once and the singularity becomes unreachable.

    Both parametrizations are accepted so older runs remain reproducible.
    """
    p = dict(zip(list(param_names), np.asarray(values, dtype=float).flatten()))
    if 'gap' in p:
        m = p['m']
        n = m + p['gap']
    else:
        n, m = p['n'], p['m']
    return p['Density'], p['U_0'], p['r0'], float(n), float(m)


def objective(Experiment_Name, iteration, Sample, param_names, *simulation_inputs,
              exp_saxs=None, exp_dls=None):
    """Simulate one candidate potential and score it against both data series.

    Experiment_Name is the directory everything for this evaluation is written
    under. run_optimization passes the per-run folder from make_run_dir, so
    repeated runs never overwrite each other.
    """
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

    searched = dict(zip(list(param_names), simulation_inputs))
    density, U_0, r0, n, m = bo_params_to_physical(simulation_inputs, param_names)

    # Unreachable under the (m, gap) parametrization; kept so runs that still
    # search n and m independently stay safe.
    if abs(n - m) < MIN_EXPONENT_GAP:
        print(f"Rejecting |n-m| = {abs(n - m):.3f}: too close to the 0/0 "
              f"singularity in U_0/(n-m)")
        return FAILED_SCORE

    sample_dir = os.path.join(Experiment_Name, 'Optimization_Results',
                              f'Sample_{Sample}')
    os.makedirs(sample_dir, exist_ok=True)
    record = dict(searched)
    record.update({'n_derived': n, 'm_derived': m})
    pd.DataFrame([record]).to_csv(
        os.path.join(sample_dir, 'input_params.csv'), index=False)

    param_strs = [f"{name}_{np.round(val, 5)}" for name, val in searched.items()]
    save_dir = os.path.join(sample_dir, "_".join(param_strs))
    os.makedirs(save_dir, exist_ok=True)

    if exp_saxs is None:
        exp_saxs = load_experimental_saxs()
    if exp_dls is None:
        exp_dls = load_experimental_dls()

    try:
        gsd_file = simulation(density, U_0, r0, n, m, save_dir)
        if gsd_file is None or not os.path.exists(gsd_file):
            gsd_file = find_gsd_file(save_dir)

        positions, lattice_coordinates, box_L = read_trajectory(gsd_file)
        frame_indices = select_frame_indices(len(positions), N_SERIES_FRAMES)
        print(f"Scoring {len(frame_indices)} frames out of {len(positions)}")

        print('Converting simulation to a SAXS series')
        sim_saxs = convert_to_SAXS_series(lattice_coordinates, frame_indices, save_dir)

        print('Converting simulation to a DLS series')
        sim_dls = convert_to_DLS_series(positions, frame_indices, box_L,
                                       r0, save_dir)

        print('Aligning against the experimental series (shared clock)')
        score, idx_cols, info = score_series_joint(
            exp_saxs, exp_dls, sim_saxs, sim_dls, save_dir,
            frame_indices=frame_indices)

        pd.DataFrame([{'saxs_score': info['saxs_mean'],
                       'dls_score': info['dls_mean'],
                       'saxs_weight': SAXS_WEIGHT, 'dls_weight': DLS_WEIGHT,
                       'n_time_points': len(info['time_grid']),
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


def make_run_dir(experiment_name):
    """Create a fresh timestamped directory under experiment_name.

    Every invocation gets its own folder, so re-running never overwrites the
    trajectories, scores or plots of a previous run. A counter is appended if
    two runs start within the same second.
    """
    stamp = time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(experiment_name, f'run_{stamp}')
    suffix = 0
    while os.path.exists(run_dir):
        suffix += 1
        run_dir = os.path.join(experiment_name, f'run_{stamp}_{suffix}')
    os.makedirs(run_dir)
    return run_dir


def run_optimization(Experiment_Name='20_Bridge_250803', n_iters=20, n_init=3,
                     param_names=('Density', 'U_0', 'r0', 'm', 'gap'),
                     target_mins=(0.00005, 0.1, 2.20, 1, 2),
                     target_maxs=(0.005, 300, 2.30, 30, 30)):
    """Minimize the combined SAXS + DLS series-alignment cost with BoTorch.

    The exponents are searched as 'm' and 'gap', with n = m + gap; see
    bo_params_to_physical for why. gap >= 2 keeps the potential away from the
    n == m singularity, and n therefore spans [3, 30] as before.

    Failed evaluations are recorded but excluded from the GP training set: the
    FAILED_SCORE sentinel is ~300x larger than a real score, so fitting on it
    destroys the length scales and flattens the acquisition surface. Until
    MIN_GP_POINTS successes exist the loop samples at random instead.
    """
    import torch
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Standardize
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

    # Each run writes into its own timestamped folder; nothing is overwritten.
    run_dir = make_run_dir(Experiment_Name)
    results_dir = os.path.join(run_dir, 'Optimization_Results')
    os.makedirs(results_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

    pd.DataFrame([{
        'started': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_iters': n_iters, 'n_init': n_init,
        'param_names': ' '.join(param_names),
        'target_mins': ' '.join(str(v) for v in target_mins.tolist()),
        'target_maxs': ' '.join(str(v) for v in target_maxs.tolist()),
        'N_SERIES_FRAMES': N_SERIES_FRAMES, 'N_EDGE_FRAMES': N_EDGE_FRAMES,
        'saxs_weight': SAXS_WEIGHT, 'dls_weight': DLS_WEIGHT,
        'hoomd_mode': os.environ.get('HOOMD_MODE', '--mode=cpu'),
    }]).to_csv(os.path.join(run_dir, 'run_config.csv'), index=False)

    # Load the experimental series once and reuse for every candidate.
    exp_saxs = load_experimental_saxs()
    exp_dls = load_experimental_dls()
    print(f"Experimental series: {len(exp_saxs)} SAXS curves, "
          f"{len(exp_dls)} DLS curves")
    time_grid = build_time_grid(len(exp_saxs), len(exp_dls))
    both = sum(1 for _, i, j in time_grid if i is not None and j is not None)
    print(f"Time grid: {len(time_grid)} points from {time_grid[0][0]:.0f} to "
          f"{time_grid[-1][0]:.0f} min ({both} carry SAXS + DLS)")
    if N_SERIES_FRAMES < len(time_grid):
        raise ValueError(
            f"N_SERIES_FRAMES ({N_SERIES_FRAMES}) must be >= the number of "
            f"experimental time points ({len(time_grid)}).")

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
    n_init = max(1, min(int(n_init), n_iters))
    train_x = torch.rand(n_init, len(param_names), dtype=dtype)
    y_values = []
    for i, x in enumerate(train_x):
        x_physical = scale_to_range(x, target_mins, target_maxs)
        print(f'Initial design {i + 1}/{n_init}: {x_physical.tolist()}')
        y_values.append(objective(run_dir, 0, i, param_names,
                                  *x_physical, exp_saxs=exp_saxs, exp_dls=exp_dls))
    train_y = torch.tensor(y_values, dtype=dtype).unsqueeze(-1)

    best_score_lst = [float(train_y.min())]

    # === BO loop ===
    for i in range(n_init, n_iters):
        # Fit on successful evaluations only; sentinels would dominate the fit.
        ok = train_y.squeeze(-1) != FAILED_SCORE
        n_ok = int(ok.sum())
        can_fit = n_ok >= MIN_GP_POINTS and float(train_y[ok].std()) > 1e-9

        if not can_fit:
            reason = (f"only {n_ok} successful evaluation(s)" if n_ok < MIN_GP_POINTS
                      else "successful scores are all identical")
            print(f"  {reason}; sampling at random instead of fitting a GP")
            new_x = torch.rand(1, len(param_names), dtype=dtype)
        else:
            fit_x = train_x[ok]
            fit_y = -train_y[ok]          # BoTorch maximizes

            kernel = ScaleKernel(
                RBFKernel(ard_num_dims=len(param_names),
                          lengthscale_constraint=GreaterThan(1e-3)),
                outputscale_constraint=GreaterThan(1e-3))
            likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-4))

            # Standardize keeps the hyperparameter initialization below on a
            # sensible scale and silences BoTorch's InputDataWarning.
            model = SingleTaskGP(fit_x, fit_y, covar_module=kernel,
                                likelihood=likelihood,
                                outcome_transform=Standardize(m=1))
            model.covar_module.outputscale = torch.tensor(1.0, dtype=dtype)
            model.covar_module.base_kernel.lengthscale = torch.tensor(0.2, dtype=dtype)
            model.likelihood.noise = torch.tensor(1e-2, dtype=dtype)

            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model_with_adam(mll)

            print(f"  fitting GP on {n_ok}/{len(train_y)} successful evaluations")
            acq_func = qLogExpectedImprovement(model=model, best_f=fit_y.max())
            new_x, _ = optimize_acqf(acq_func, bounds=bounds, q=1,
                                     num_restarts=5, raw_samples=20)

        new_x_physical = scale_to_range(new_x, target_mins, target_maxs)
        y_values = []
        for x_physical in new_x_physical:
            print('Inputs: ', x_physical)
            y_values.append([objective(run_dir, 0, i, param_names,
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

        evals = pd.DataFrame(
            np.hstack([scale_to_range(train_x, target_mins, target_maxs).numpy(),
                       train_y.numpy()]),
            columns=param_names + ['score'])
        if 'gap' in evals.columns:
            evals['n_derived'] = evals['m'] + evals['gap']
        evals['status'] = np.where(evals['score'] == FAILED_SCORE, 'failed', 'ok')
        evals.to_csv(os.path.join(results_dir, 'all_evaluations.csv'), index=False)

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

    # --- Simulated DLS series from the OVITO cluster exports ---
    dls_dir = os.path.join(root, 'Data', 'DLS', 'Simulated_DLS_10')
    dls_files = sorted(glob.glob(os.path.join(dls_dir, 'cluster_size_*')))[:n_sim]
    sim_dls = [dls_distribution_from_clusters(clusters_from_ovito_file(f))
               for f in dls_files]
    print(f"Simulated DLS series: {len(sim_dls)} curves")

    n = min(len(sim_saxs), len(sim_dls))
    sim_saxs, sim_dls = sim_saxs[:n], sim_dls[:n]

    grid = build_time_grid(len(exp_saxs), len(exp_dls))
    both = sum(1 for _, i, j in grid if i is not None and j is not None)
    print(f"\nTime grid: {len(grid)} points from "
          f"{grid[0][0]:.0f} to {grid[-1][0]:.0f} min "
          f"({both} carry SAXS + DLS, {len(grid) - both} DLS only)")
    print("NOTE: these two stored series come from different runs, so the joint "
          "score is only a structural check, not a physical result.")

    score, path, info = score_series_joint(
        exp_saxs, exp_dls, sim_saxs, sim_dls, out_dir, time_grid=grid)

    print(f"\nJoint objective = {score:.4f}")
    print(f"Plots and cost matrices written to {out_dir}")
    return score


# %%
if __name__ == '__main__':
    if '--check' in sys.argv:
        check_scoring()
    else:
        run_optimization()
