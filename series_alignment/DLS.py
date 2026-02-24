import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.optimize import nnls
from scipy.optimize import lsq_linear
from scipy.signal import savgol_filter



def read_DAT_file(name):
    with open(name) as pdbfile:
        q = []
        I = []
        dI = []
        start = 10000
        for i,line in enumerate(pdbfile):
            if 'q(A-1)' in line:
                start = i
            if i > start: 
                splitted_line = [line[0:20], line[23:50], line[50:]]
                q.append(splitted_line[0])
                I.append(splitted_line[1])
                dI.append(splitted_line[2])
        q = np.array([float(i) for i in q])
        I = np.array([float(i) for i in I])
        dI = np.array([float(i) for i in dI])
        data = np.hstack((q.reshape(-1,1), I.reshape(-1,1), dI.reshape(-1,1)))
    return data

def gaus(x,a,mu,sigma):
    return a*np.exp(-(x-mu)**2/(2*sigma**2))

def calculate_max_wv(sizes, spec):
    peak_pos = find_peaks(spec, height=0.75)[0][0]
    lower = 8
    upper = 8
    scaler = MinMaxScaler()
    coords = spec[int(peak_pos-lower):int(peak_pos+upper)].reshape(-1,1)
    coords = scaler.fit(coords).transform(coords)*100
    sizes = np.log10(sizes)
    new_wv = sizes[int(peak_pos-lower):int(peak_pos+upper)]
    
    angles = new_wv - np.median(new_wv)
    data = coords
    angles = np.array(angles).flatten()
    data = np.array(data).flatten()

    n = len(data)  ## <---
    mean = np.sum(data*angles)/n
    sigma = np.sqrt(np.sum(data*(angles-mean)**2)/n)

    popt,pcov = curve_fit(gaus,angles,data)#,p0=[0.18,mean,sigma])  ## <--- leave out the first estimation of the parameters
    xx = np.linspace(np.min(angles), np.max(angles), 100)  ## <--- calculate against a continuous variable

    normal_fit = gaus(xx,*popt)
    xx_wv = xx + np.median(new_wv)
    #fig, ax = plt.subplots()
    plt.plot(xx_wv, normal_fit,'r',label='Fit')  ## <--- plot against the contious variable
    plt.scatter(new_wv, data)
    peak_position = xx_wv[np.argmax(normal_fit)]
    peak_position = 10**peak_position
    plt.title('Peak Position: ' + str(peak_position)[0:3])
    return peak_position

def extract_positions_orientations(filename):
    """
    Extract positions and orientations from each frame of a GSD file.

    Parameters
    ----------
    filename : str
        Path to the GSD trajectory file.

    Returns
    -------
    positions : list of np.ndarray
        List of arrays of shape (N, 3) with particle positions per frame.
    orientations : list of np.ndarray or None
        List of arrays of shape (N, 4) with quaternions per frame,
        or None if not present in the frame.
    """
    traj = hoomd.open(name=filename, mode='r')

    positions = []
    orientations = []

    for frame in traj:
        positions.append(frame.particles.position.copy())

        if hasattr(frame.particles, 'orientation'):
            orientations.append(frame.particles.orientation.copy())
        else:
            orientations.append(None)

    return positions, orientations


def quaternion_to_euler(quat, degrees=True, order='xyz'):
    """
    Convert a quaternion (HOOMD format: [qw, qx, qy, qz]) to Euler angles.

    Parameters:
    - quat: array-like, quaternion [qw, qx, qy, qz]
    - degrees: bool, return angles in degrees if True (default), radians if False
    - order: str, axes sequence for Euler angles ('xyz', 'zyx', etc.)

    Returns:
    - tuple of 3 Euler angles (angle_x, angle_y, angle_z)
    """
    # Convert to scipy format (qx, qy, qz, qw)
    #scipy_quat = [quat[1], quat[2], quat[3], quat[0]]
    scipy_quat = quat
    r = R.from_quat(scipy_quat)
    angles = r.as_euler(order, degrees=degrees)
    return angles

def grid_points_in_sphere(D, spacing):
    """
    Generate a regular 3D grid of points spaced by 'spacing' that fit inside a sphere.

    Parameters
    ----------
    D : float
        Diameter of the sphere.
    spacing : float
        Distance between adjacent grid points.

    Returns
    -------
    points : np.ndarray of shape (M, 3)
        Grid points inside the sphere.
    """
    radius = D / 2.0
    r2 = radius ** 2

    # Create a 3D grid
    coords = np.arange(-radius, radius + spacing, spacing)
    X, Y, Z = np.meshgrid(coords, coords, coords)
    grid = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    # Keep only the points inside the sphere
    mask = np.sum(grid**2, axis=1) <= r2
    points = grid[mask]

    return points

def read_DAT_file(name):
    with open(name) as pdbfile:
        q = []
        I = []
        dI = []
        start = 10000
        for i,line in enumerate(pdbfile):
            if 'q(A-1)' in line:
                start = i
            if i > start: 
                splitted_line = [line[0:20], line[23:50], line[50:]]
                q.append(splitted_line[0])
                I.append(splitted_line[1])
                dI.append(splitted_line[2])
        q = np.array([float(i) for i in q])
        I = np.array([float(i) for i in I])
        dI = np.array([float(i) for i in dI])
        data = np.hstack((q.reshape(-1,1), I.reshape(-1,1), dI.reshape(-1,1)))
    return data


def sphere_guinier_P(q, Rg):
    # simple Guinier-like factor for aggregates/spheres at small q
    return np.exp(-(q*Rg)**2 / 3.0)

def stokes_einstein_Rh(D, T, eta):
    kB = 1.380649e-23  # J/K
    return kB*T/(6*np.pi*eta*D)

def build_g2(clusters, tau, q, T, eta, beta=0.9, n_norm=True):
    """
    clusters: list of dicts with keys {'N', 'Rg', 'D'} or {'N','Rg','Rh'}
    If 'Rh' provided, compute D; else use 'D'.
    """
    kB = 1.380649e-23  # J/K
    Ds, weights = [], []
    for cl in clusters:
        D = cl.get('D', kB*T/(6*np.pi*eta*cl['Rh']))
        Rg = cl['Rg']
        N  = cl['N']
        # intensity weight ~ N^2 * P(q)
        w  = (N**2) * sphere_guinier_P(q, Rg)
        Ds.append(D); weights.append(w)

    Ds = np.array(Ds); weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    g1 = np.sum(weights[:,None] * np.exp(- (Ds[:,None]) * (q**2) * tau[None,:]), axis=0)
    g2 = 1.0 + beta * (g1**2)
    return g2, g1, Ds, weights

def invert_g2_to_distribution(g2, tau, q, T, eta, beta=0.9, n_bins=60, D_bounds=(1e-14, 1e-9)):
    """
    NNLS inversion on g1 (since g2 ~ 1 + beta*g1^2 => take sqrt piece).
    We fit g1 = A * w, where A_{ij} = exp(-D_j q^2 tau_i), w_j >= 0.
    """
    kB = 1.380649e-23  # J/K
    # estimate g1 from g2 (keep sign positive)
    g1 = np.sqrt(np.maximum(g2-1.0, 0.0)/beta)

    logD = np.linspace(np.log(D_bounds[0]), np.log(D_bounds[1]), n_bins)
    Dgrid = np.exp(logD)

    A = np.exp(- np.outer(tau, Dgrid) * q**2)  # shape (len(tau), n_bins)
    w, _ = nnls(A, g1)                          # nonnegative weights
    w /= (w.sum() + 1e-16)

    # Convert to intensity-weighted Rh distribution
    Rh_grid = stokes_einstein_Rh(Dgrid, T, eta)
    return Dgrid, Rh_grid, w


def _second_diff_matrix(n):
    L = np.zeros((n-2, n))
    for i in range(n-2):
        L[i, i]   = 1.0
        L[i, i+1] = -2.0
        L[i, i+2] = 1.0
    return L

def _lcurve_corner(res_norms, reg_norms, alphas):
    """Pick alpha at maximum curvature of L-curve in log-log space."""
    x = np.log(res_norms)
    y = np.log(reg_norms)
    # discrete curvature via second derivative on arc-length parametrization
    dx = np.gradient(x);  dy = np.gradient(y)
    ddx = np.gradient(dx); ddy = np.gradient(dy)
    kappa = np.abs(dx*ddy - dy*ddx) / np.power(dx*dx + dy*dy, 1.5)
    i = int(np.nanargmax(kappa))
    return alphas[i], i

def contin_like_invert(
    g2, tau, q,
    beta=None,                      # set to float if you already know it
    T=None, eta=None,               # optional: to return Rh grid
    D_bounds=(1e-14, 1e-9),         # m^2/s
    n_bins=60,
    smooth_g2=True, sg_window=11, sg_poly=3,
    alphas=np.logspace(-5, 0, 15)   # grid of regularization strengths. This parameter controls the polydispersity of sample. The alpha used in Malvern GP software is proprietary so choose any good values here. 
):
    """
    Returns:
      Dgrid [m^2/s], weights w (sum=1), dict(info) with g2_inf, beta_used, alpha, Rh_grid (if T,eta)
    """
    kB = 1.380649e-23  # J/K
    g2 = np.asarray(g2, float).reshape(-1)
    tau = np.asarray(tau, float).reshape(-1)
    assert g2.size == tau.size and g2.size > 5, "g2 and tau must be same length (>5)."
    assert np.all(np.isfinite(g2)) and np.all(np.isfinite(tau)) and np.isfinite(q) and q>0

    # --- Baseline and beta ---
    k_tail = max(3, int(0.2*len(g2)))
    g2_inf = float(np.nanmean(g2[-k_tail:]))
    if beta is None:
        amp = float(np.nanmax(g2 - g2_inf))
        beta = float(np.clip(amp, 1e-3, 0.99))  # conservative
    else:
        beta = float(np.clip(beta, 1e-6, 0.999))

    # --- Pre-smooth in log-tau space (very light) ---
    excess = np.clip(g2 - g2_inf, 0.0, None)
    if smooth_g2 and len(excess) >= sg_window:
        excess = savgol_filter(excess, sg_window, sg_poly, mode="interp")
        excess = np.clip(excess, 0.0, None)

    # --- Siegert -> g1 ---
    g1 = np.sqrt(excess / beta)
    # Guard against residual non-finites
    m = np.isfinite(g1) & np.isfinite(tau)
    tau = tau[m]; g1 = g1[m]

    # --- Kernel on log-D grid ---
    logD = np.linspace(np.log(D_bounds[0]), np.log(D_bounds[1]), n_bins)
    Dgrid = np.exp(logD)
    A = np.exp(- np.outer(tau, Dgrid) * (q**2))  # shape (M, n_bins)

    # Column normalization (stabilizes fit + alpha selection)
    col_norms = np.linalg.norm(A, axis=0)
    nz = col_norms > 1e-14
    A = A[:, nz] / col_norms[nz]
    Dgrid = Dgrid[nz]
    n = A.shape[1]

    # Regularizer (2nd derivative in log-D)
    L = _second_diff_matrix(n)

    # --- Choose alphas to sweep ---
    if alphas is None:
        alphas = np.logspace(-3, 2, 25)  # adjust if needed

    res_norms, reg_norms, sols = [], [], []

    # --- Sweep alpha and solve nonnegative Tikhonov with bounds ---
    for a in alphas:
        A_aug = np.vstack([A, np.sqrt(a)*L])
        b_aug = np.concatenate([g1, np.zeros(L.shape[0])])
        res = lsq_linear(A_aug, b_aug, bounds=(0, np.inf), lsmr_tol='auto')
        w = np.maximum(res.x, 0.0)
        # normalize to unit area (intensity distribution)
        if w.sum() > 0: w = w / w.sum()
        sols.append(w)
        # residual and smoothness norms on original (un-augmented) terms:
        r = A @ w - g1
        s = L @ w
        res_norms.append(np.linalg.norm(r))
        reg_norms.append(np.linalg.norm(s) + 1e-18)

    res_norms = np.array(res_norms); reg_norms = np.array(reg_norms)

    # --- L-curve corner selection ---
    alpha_star, idx = _lcurve_corner(res_norms, reg_norms, alphas)
    w = sols[idx]

    info = {
        "g2_inf": g2_inf,
        "beta_used": beta,
        "alpha": float(alpha_star),
        "residual_norm": float(res_norms[idx]),
        "smoothness_norm": float(reg_norms[idx]),
        "logD": np.log(Dgrid)
    }

    # Optional: Rh grid
    kB = 1.380649e-23  # J/K
    if (T is not None) and (eta is not None):
        Rh = kB*T / (6*np.pi*eta*Dgrid)
        info["Rh_grid"] = Rh

    return Dgrid, w, info


