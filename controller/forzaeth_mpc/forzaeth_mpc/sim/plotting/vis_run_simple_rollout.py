import math
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_closed_loop_summary(
    refline_xy: np.ndarray,
    left_dist: np.ndarray,
    right_dist: np.ndarray,
    # (K, 10) plant states: [s, n, theta, vx, vy, delta, yaw_rate, accel, derDelta, jerk]
    x_hist: np.ndarray,
    u_hist: np.ndarray,              # (K, 2)  inputs: [jerk, steer_rate]
    *,
    # indices in x_hist
    idx_s: int = 0,
    idx_n: int = 1,
    idx_vx: int = 3,
    idx_delta: int = 5,
    idx_yaw_rate: int = 6,
    idx_accel: int = 7,
    # indices in u_hist
    ui_jerk: int = 0,
    ui_steer_rate: int = 1,
    titles: Optional[dict] = None,
) -> plt.Figure:
    """
    Left:   Track (centerline + boundaries) and ACTUAL plant positions (from x_hist[:, [s,n]]).
    Right:  3x2 time plots of actual inputs & states (jerk, steer_rate, delta, accel, v_x, yaw_rate).
    """

    assert refline_xy.ndim == 2 and refline_xy.shape[1] == 2
    assert left_dist.shape[0] == refline_xy.shape[0]
    assert right_dist.shape[0] == refline_xy.shape[0]
    assert x_hist.ndim == 2 and x_hist.shape[1] >= 8
    assert u_hist.ndim == 2 and u_hist.shape[1] >= 2

    def _compute_unit_tangents(centerline_xy: np.ndarray) -> np.ndarray:
        diffs = np.vstack((centerline_xy[1:] - centerline_xy[:-1],
                           centerline_xy[0] - centerline_xy[-1]))
        return diffs / (np.linalg.norm(diffs, axis=1, keepdims=True) + 1e-12)

    def _compute_unit_left_normals(tangents: np.ndarray) -> np.ndarray:
        return np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)

    def _arc_length(centerline_xy: np.ndarray) -> np.ndarray:
        seg = np.linalg.norm(np.diff(centerline_xy, axis=0, append=centerline_xy[:1]), axis=1)
        s = np.cumsum(seg)
        s = np.insert(s, 0, 0.0)
        # s has length N+1; last point equals track length
        return s

    def _wrap_s(s: np.ndarray, L: float) -> np.ndarray:
        return np.mod(s, L)

    def _frenet_to_xy_batch(centerline_xy: np.ndarray,
                            s_ref: np.ndarray,
                            sn_batch: np.ndarray) -> np.ndarray:
        """
        Convert a sequence of (s,n) to XY along a closed track.
        s_ref: (N+1,) cumulative arc-length with s_ref[-1] = track length
        """
        N = centerline_xy.shape[0]
        L = s_ref[-1]
        s = _wrap_s(sn_batch[:, 0], L)
        n = sn_batch[:, 1]

        # For each s, find segment i so that s in [s_ref[i], s_ref[i+1])
        idx = np.searchsorted(s_ref, s, side='right') - 1
        idx = np.clip(idx, 0, N - 1)
        idx_next = (idx + 1) % N

        s_i = s_ref[idx]
        ds_i = s_ref[idx_next] - s_ref[idx]
        # wrap across the end
        ds_i = np.where(ds_i <= 0.0, ds_i + L, ds_i)

        alpha = np.where(ds_i > 1e-12, (s - s_i) / ds_i, 0.0)
        p_i = centerline_xy[idx]
        p_j = centerline_xy[idx_next]
        p = p_i + (p_j - p_i) * alpha[:, None]

        tang = _compute_unit_tangents(centerline_xy)
        left_n = _compute_unit_left_normals(tang)
        n_vec = left_n[idx]

        xy = p + n[:, None] * n_vec
        return xy

    # --- Track geometry ---
    tang = _compute_unit_tangents(refline_xy)
    left_n = _compute_unit_left_normals(tang)
    left_b = refline_xy + left_n * left_dist[:, None]
    right_b = refline_xy - left_n * right_dist[:, None]
    s_ref = _arc_length(refline_xy)  # (N+1,), s_ref[-1] = total length

    # --- Actual plant positions (convert Frenet to XY) ---
    sn_hist = x_hist[:, [idx_s, idx_n]]  # (K, 2)
    xy_hist = _frenet_to_xy_batch(refline_xy, s_ref, sn_hist)

    # --- Figure layout ---
    fig = plt.figure(figsize=(14, 7))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(nrows=3, ncols=4, figure=fig, wspace=0.35, hspace=0.5)

    # Left big axis: track + actual path
    ax_left = fig.add_subplot(gs[:, :2])
    ax_left.plot(refline_xy[:, 0], refline_xy[:, 1], lw=2, label='Centerline')
    ax_left.plot(left_b[:, 0], left_b[:, 1], label='Left boundary')
    ax_left.plot(right_b[:, 0], right_b[:, 1], label='Right boundary')
    # Direction arrow
    ax_left.arrow(refline_xy[0, 0], refline_xy[0, 1],
                  tang[0, 0], tang[0, 1], head_width=0.5, head_length=0.5)
    # Actual plant trajectory
    ax_left.plot(xy_hist[:, 0], xy_hist[:, 1], 'o-', ms=3, label='Actual (plant)', zorder=5)

    ax_left.set_title((titles or {}).get('left_title', 'Track & Actual Plant Trajectory'))
    ax_left.set_xlabel('X (m)')
    ax_left.set_ylabel('Y (m)')
    ax_left.axis('equal')
    ax_left.grid(True)
    ax_left.legend(loc='best')

    # Right: 3x2 panels
    axes = [
        fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3]),
        fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[1, 3]),
        fig.add_subplot(gs[2, 2]), fig.add_subplot(gs[2, 3]),
    ]

    K = x_hist.shape[0]
    t = np.arange(K)

    # (1) Input: jerk
    axes[0].plot(t, u_hist[:, ui_jerk])
    axes[0].set_title((titles or {}).get('jerk', 'Input: jerk'))
    axes[0].set_xlabel('step')
    axes[0].set_ylabel('jerk')
    axes[0].grid(True)

    # (2) Input: steering rate
    axes[1].plot(t, u_hist[:, ui_steer_rate])
    axes[1].set_title((titles or {}).get('steer_rate', 'Input: steering rate'))
    axes[1].set_xlabel('step')
    axes[1].set_ylabel('dδ/dt')
    axes[1].grid(True)

    # (3) State: steering (delta)
    axes[2].plot(t, x_hist[:, idx_delta])
    axes[2].set_title((titles or {}).get('delta', 'Steering δ'))
    axes[2].set_xlabel('step')
    axes[2].set_ylabel('rad')
    axes[2].grid(True)

    # (4) State: acceleration
    axes[3].plot(t, x_hist[:, idx_accel])
    axes[3].set_title((titles or {}).get('accel', 'Acceleration'))
    axes[3].set_xlabel('step')
    axes[3].set_ylabel('m/s²')
    axes[3].grid(True)

    # (5) State: velocity v_x
    axes[4].plot(t, x_hist[:, idx_vx])
    axes[4].set_title((titles or {}).get('vx', 'Velocity vₓ'))
    axes[4].set_xlabel('step')
    axes[4].set_ylabel('m/s')
    axes[4].grid(True)

    # (6) State: yaw rate
    axes[5].plot(t, x_hist[:, idx_yaw_rate])
    axes[5].set_title((titles or {}).get('yaw_rate', 'Angular velocity (yaw rate)'))
    axes[5].set_xlabel('step')
    axes[5].set_ylabel('rad/s')
    axes[5].grid(True)

    fig.suptitle((titles or {}).get('suptitle', 'Closed-loop Summary (Actual Plant)'))
    fig.tight_layout()
    return fig


def plot_state_prediction_vs_actual(
    x_pred1_hist: np.ndarray,          # (K, nx)  solver 1-step predictions per iteration
    x_hist: np.ndarray,                # (K, M)   plant history (M >= nx); first nx are the MPC model states
    *,
    state_names: Optional[Sequence[str]] = None,  # e.g., ["s","n","theta","vx","vy","delta","yaw_rate","accel"]
    suptitle: str = "1-step Prediction vs Actual (per iteration)",
) -> plt.Figure:
    """
    For each modeled state (0..nx-1), plot over iterations:
      - predicted x_{k+1|k} from the solver (stage-1 state at iteration k)
      - actual plant state after applying u_k and integrating one step
      - absolute error on a secondary y-axis (thin line)

    Assumes you collected:
      x_pred1_hist[k] = solver.get(1, "x") BEFORE integrating the plant at iteration k
      x_hist[k]       = plant state AFTER integrating once at iteration k
    """
    assert x_pred1_hist.ndim == 2 and x_hist.ndim == 2, "Inputs must be 2D arrays"
    K, nx = x_pred1_hist.shape
    assert x_hist.shape[0] == K, "Pred and actual must have same number of samples"
    assert x_hist.shape[1] >= nx, "Plant history must contain at least nx modeled states"

    x_act = x_hist[:, :nx]                 # (K, nx)
    err = x_act - x_pred1_hist           # (K, nx)
    t = np.arange(K)

    # layout: 3 columns, ceil(nx/3) rows
    ncols = 3
    nrows = math.ceil(nx / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 3.8 * nrows), squeeze=False)
    axes = axes.reshape(nrows, ncols)

    if state_names is None or len(state_names) < nx:
        # sensible defaults if not provided / incomplete
        default_names = [f"x[{i}]" for i in range(nx)]
        if state_names is None:
            state_names = default_names
        else:
            # pad
            state_names = list(state_names) + default_names[len(state_names):nx]

    for i in range(nx):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        ax2 = ax.twinx()

        ax.plot(t, x_pred1_hist[:, i], label="pred", lw=2)
        ax.plot(t, x_act[:, i], label="actual", lw=1.5, linestyle="--")
        ax.set_title(f"{state_names[i]}")
        ax.set_xlabel("iteration")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.4)

        # error on secondary axis
        ax2.plot(t, np.abs(err[:, i]), linewidth=1.0, alpha=0.6)
        ax2.set_ylabel("|error|", rotation=270, labelpad=12)
        ax2.tick_params(axis='y', labelsize=8)

        # small legend per row, first subplot only to avoid clutter
        if i % ncols == 0:
            ax.legend(loc="upper right", fontsize=9)

    # hide any unused subplots
    total_ax = nrows * ncols
    for j in range(nx, total_ax):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")

    fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def plot_spaghetti_predictions_vs_actual(
    x_pred_seq_hist: np.ndarray,   # (K, H+1, nx): predictions per iteration
    x_hist: np.ndarray,            # (K, M): actual plant states (first nx are modeled states)
    *,
    state_names: Optional[Sequence[str]] = None,
    plot_every: int = 1,           # draw every 'plot_every' prediction to reduce clutter
    suptitle: str = "Spaghetti: MPC Predicted States vs Actual",
) -> plt.Figure:
    """
    For each state i, draw:
      - All predicted sequences from each iteration k: x_pred_seq_hist[k, :, i] at times t = k..k+H
      - The actual realized state over iterations: x_hist[:, i]

    Time base:
      - Actual: t = 0..K-1
      - Prediction k: plotted at t = k + [0..H]
    """

    assert x_pred_seq_hist.ndim == 3, "x_pred_seq_hist must be (K, H+1, nx)"
    assert x_hist.ndim == 2, "x_hist must be (K, M)"
    K, Hp1, nx = x_pred_seq_hist.shape
    H = Hp1 - 1
    assert x_hist.shape[0] == K, "x_hist and x_pred_seq_hist must share K"

    x_act = x_hist[:, :nx]  # (K, nx)
    t_act = np.arange(K)

    if state_names is None or len(state_names) < nx:
        defaults = [f"x[{i}]" for i in range(nx)]
        state_names = defaults if state_names is None else list(state_names) + defaults[len(state_names):nx]

    # layout: 3 columns
    ncols = 3
    nrows = math.ceil(nx / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 3.8 * nrows), squeeze=False)
    axes = axes.reshape(nrows, ncols)

    # Choose y-limits per state from combined data for readability
    for i in range(nx):
        r, c = divmod(i, ncols)
        ax = axes[r, c]

        # Spaghetti: all predictions
        for k in range(0, K, plot_every):
            t_pred = k + np.arange(H + 1)
            y_pred = x_pred_seq_hist[k, :, i]
            ax.plot(t_pred, y_pred, linewidth=0.8, alpha=0.35)

        # Actual
        ax.plot(t_act, x_act[:, i], linewidth=2.0, label="actual", zorder=5)

        ax.set_title(state_names[i])
        ax.set_xlabel("iteration (step)")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.4)

        # Legend on first subplot only (to reduce clutter)
        if i == 0:
            ax.legend(loc="best", fontsize=9)

    # Hide unused axes
    total_ax = nrows * ncols
    for j in range(nx, total_ax):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")

    fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def plot_velocity_vs_progress(
    waypoints: np.ndarray,   # columns: [x, y, vx_ref, kappa, s, d_left, d_right]
    x_hist: np.ndarray,      # plant/model states per tick; first nx correspond to MPC states
    *,
    idx_s: int = 0,          # index of s in x_hist
    idx_vx: int = 3,         # index of v_x (longitudinal) in x_hist
    title: Optional[str] = "Target vs Actual velocity along track",
) -> plt.Figure:
    """
    Plots v_target(s) from 'waypoints' vs actual v_x measured in 'x_hist', both vs progress s.

    Assumptions:
      - 'waypoints[:, 4]' stores cumulative arc-length s along a CLOSED loop (0..L).
      - 'waypoints[:, 2]' stores the target speed profile v_ref(s) [m/s].
      - 'x_hist' rows store states; 'idx_s' is Frenet s, 'idx_vx' is longitudinal speed v_x.
      - For closed tracks, s is wrapped to [0, L).

    Returns:
      matplotlib Figure.
    """

    assert waypoints.ndim == 2 and waypoints.shape[1] >= 5, "waypoints must have at least 5 columns"
    assert x_hist.ndim == 2 and x_hist.shape[1] > max(idx_s, idx_vx), "x_hist shape/index mismatch"

    # --- Extract from inputs ---
    s_wp = waypoints[:, 4].astype(float)   # (N,)
    vref = waypoints[:, 2].astype(float)   # (N,)
    s_act = x_hist[:, idx_s].astype(float)  # (K,)
    v_act = x_hist[:, idx_vx].astype(float)  # (K,)

    # --- Ensure a clean closed-loop parameterization for interpolation ---
    # Track length L from waypoints' last s (assumes increasing with a wrap at the end)
    # If s_wp is not strictly increasing (due to wrap), make an increasing copy.
    s_sorted = s_wp.copy()
    # If the track tool already gives 0..L increasing, this is fine; otherwise detect wrap:
    # Create a strictly increasing version by unwrapping:
    jumps = np.where(np.diff(s_sorted) < 0)[0]
    if jumps.size > 0:
        j = jumps[0]
        s_sorted = np.concatenate([s_sorted[j + 1:] - s_sorted[j + 1],
                                   s_sorted[:j + 1] + (s_sorted[-1] - s_sorted[j + 1])])
        vref = np.concatenate([vref[j + 1:], vref[:j + 1]])

    L = s_sorted[-1]
    if L <= 0:
        raise ValueError("Non-positive track length inferred from waypoints s column.")

    # To interpolate periodically across the wrap, append one extra point at s=L with vref=same as s=0
    s_ext = np.concatenate([s_sorted, [s_sorted[0] + L]])
    v_ext = np.concatenate([vref, [vref[0]]])

    # --- Wrap actual s into [0, L) and interpolate v_ref at those s ---
    s_act_wrapped = np.mod(s_act, L)
    # Sort for nicer plotting (optional)
    order = np.argsort(s_act_wrapped)
    s_plot = s_act_wrapped[order]
    v_act_plot = v_act[order]

    v_ref_at_s = np.interp(s_plot, s_ext, v_ext)

    # --- Build figure ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(s_ext, v_ext, alpha=0.25, label="v_ref (profile, over 1 lap)")  # light baseline
    ax.plot(s_plot, v_ref_at_s, lw=2, label="v_ref @ actual s")
    ax.plot(s_plot, v_act_plot, lw=1.8, linestyle="--", label="v_actual (v_x)")

    ax.set_xlabel("Progress s (m)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="best")

    # Optional: show second x-axis in laps
    def s_to_lap(x): return x / L
    secax = ax.secondary_xaxis('top', functions=(s_to_lap, lambda lap: lap * L))
    secax.set_xlabel("Laps")

    fig.tight_layout()
    return fig
