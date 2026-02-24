import numpy as np
from scipy.interpolate import interp1d

def compare_curves(
    exp_data,
    sim_data,
    metric,
    q_range=None,
):
    """
    Compare experimental and simulated data with multiple intensity columns.
    Simulated data is truncated and interpolated onto the experimental q-grid.

    Parameters:
        exp_data: np.ndarray, shape (N_exp, 1+K)
        sim_data: np.ndarray, shape (N_sim, 1+K)
        q_range: tuple (q_min, q_max), optional
        metric: str or callable
            Supported strings:
                "mse", "rmse", "mae", "log_mse", "relative_mse"
            Callable:
                f(I_exp, I_sim) -> float

    Returns:
        score: float
        q_ref: np.ndarray
        I_exp: np.ndarray, shape (N_used, K)
        I_sim: np.ndarray, shape (N_used, K)
    """

    # Extract q and intensities
    q_exp = exp_data[:, 0]
    I_exp = exp_data[:, 1:]

    q_sim = sim_data[:, 0]
    I_sim = sim_data[:, 1:]

    # Determine common q-range
    q_min = max(q_exp.min(), q_sim.min())
    q_max = min(q_exp.max(), q_sim.max())

    if q_range is not None:
        q_min = max(q_min, q_range[0])
        q_max = min(q_max, q_range[1])

    # Truncate experimental data
    mask_exp = (q_exp >= q_min) & (q_exp <= q_max)
    q_ref = q_exp[mask_exp]
    I_exp = I_exp[mask_exp]

    # Truncate simulated data
    mask_sim = (q_sim >= q_min) & (q_sim <= q_max)
    q_sim_crop = q_sim[mask_sim]
    I_sim_crop = I_sim[mask_sim]

    if len(q_sim_crop) < 2:
        raise ValueError("Not enough simulated points for interpolation")

    # Interpolate simulated intensities onto experimental q-grid
    interp_func = interp1d(
        q_sim_crop,
        I_sim_crop,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    I_sim_resampled = interp_func(q_ref)

    # Numerical safety
    eps = 1e-10
    I_exp = np.clip(I_exp, eps, None)
    I_sim_resampled = np.clip(I_sim_resampled, eps, None)

    # -----------------------
    # Distance metrics
    # -----------------------
    if callable(metric):
        score = metric(I_exp, I_sim_resampled)
    elif metric== "peak_position":
        q_AP = np.linspace(q_ref[0], q_ref[-1], len(q_ref))
        peak_exp = q_AP[np.argmax(I_exp)]
        peak_sim = q_AP[np.argmax(I_sim_resampled)]
        score = np.abs(peak_exp - peak_sim)
    elif metric == "mse":
        score = np.mean((I_exp - I_sim_resampled) ** 2)

    elif metric == "rmse":
        score = np.sqrt(np.mean((I_exp - I_sim_resampled) ** 2))

    elif metric == "mae":
        score = np.mean(np.abs(I_exp - I_sim_resampled))

    elif metric == "log_mse":
        score = np.mean((np.log10(I_exp) - np.log10(I_sim_resampled)) ** 2)

    elif metric == "relative_mse":
        score = np.mean(((I_exp - I_sim_resampled) / I_exp) ** 2)

    else:
        raise ValueError(f"Unknown metric: {metric}")

    return score, q_ref, I_exp, I_sim_resampled


def create_distance_matrix(exp_saxs_data, sim_saxs_data, metric, q_range=None):
    """
    Create distance matrix comparing multiple experimental and simulated datasets.

    Parameters:
        exp_saxs_data: list of np.ndarray, each (N, 1+K)
        sim_saxs_data: list of np.ndarray, each (N, 1+K)
        q_range: tuple (q_min, q_max), optional

    Returns:
        score_matrix: np.ndarray, shape (n_exp, n_sim)
    """

    score_matrix = np.zeros((len(exp_saxs_data), len(sim_saxs_data)))

    for i, exp_data in enumerate(exp_saxs_data):
        for j, sim_data in enumerate(sim_saxs_data):
            score, _, _, _ = compare_curves(
                exp_data, sim_data, metric, q_range=q_range)
            score_matrix[i, j] = score

    return score_matrix


def align_monotone_min(cost: np.ndarray):
    """
    Align experimental curves (rows) to simulated curves (columns) with
    strictly increasing column indices, minimizing total cost.

    Parameters
    ----------
    cost : (N, M) numpy.ndarray
        cost[i, j] = distance between experimental curve i and simulated curve j.
        Must have M >= N.

    Returns
    -------
    idx_cols : list[int]
        For each experimental row i (0..N-1), the chosen simulated column j.
    total_cost : float
        Sum of costs cost[i, idx_cols[i]].
    """
    N, M = cost.shape
    if M < N:
        raise ValueError("Requires at least as many simulated columns as experimental rows (M >= N).")

    # Mask out column positions that would make completion impossible.
    # Row i cannot pick columns < i, and cannot pick columns > M - (N - i)
    # (so there remain enough columns for remaining rows).
    valid = np.full((N, M), False, dtype=bool)
    for i in range(N):
        j_min = i
        j_max = M - (N - i)
        valid[i, j_min:j_max+1] = True

    INF = np.finfo(float).max / 10.0
    C = np.where(valid, cost, INF)

    # DP: dp[i, j] = min total cost aligning rows 0..i with row i -> col j
    dp = np.full((N, M), INF)
    back = np.full((N, M), -1, dtype=int)

    # Initialize first row: can pick any valid j
    dp[0, :] = C[0, :]
    back[0, :] = -1

    # Fill DP with prefix minima for O(N*M)
    for i in range(1, N):
        # prefix minima of previous row: pm[j] = min_{k < j} dp[i-1, k]
        pm = np.minimum.accumulate(dp[i-1, :])
        # For each column j, best previous is any k < j
        best_prev = pm.copy()
        # First valid j for this row
        j_start = i
        j_end = M - (N - i)
        # Compute dp only where valid
        for j in range(j_start, j_end+1):
            prev_cost = best_prev[j-1]  # min over k < j
            if prev_cost < INF:
                dp[i, j] = C[i, j] + prev_cost
                # Find the actual k that achieved pm[j-1]
                # (optional backtrack search within small window)
                # To avoid O(M^2), we can store argmins separately; for simplicity, do a small scan:
                k = np.argmin(dp[i-1, :j])
                back[i, j] = k

    # Choose the best ending column for last row
    j_last = int(np.argmin(dp[N-1, :]))
    total_cost = float(dp[N-1, j_last])

    # Backtrack columns
    idx_cols = [0] * N
    j = j_last
    for i in range(N-1, -1, -1):
        idx_cols[i] = j
        j = back[i, j]

    return idx_cols, total_cost


# ----- Optional: a simple greedy variant (fast, not always optimal) -----
def align_monotone_greedy(cost: np.ndarray):
    """
    Greedy monotone alignment: for each row in order, pick the cheapest
    column >= previous+1 that still leaves enough columns for remaining rows.
    Fast but not guaranteed globally optimal.
    """
    N, M = cost.shape
    if M < N:
        raise ValueError("Requires M >= N")

    assignment = []
    prev_j = -1
    for i in range(N):
        j_min = max(prev_j + 1, i)
        j_max = M - (N - i)
        j_range = slice(j_min, j_max + 1)
        j_rel = int(np.argmin(cost[i, j_range]))
        j = j_min + j_rel
        assignment.append(j)
        prev_j = j

    total = float(sum(cost[i, j] for i, j in enumerate(assignment)))
    return assignment, total
