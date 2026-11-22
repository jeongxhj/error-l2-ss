from __future__ import annotations

from typing import Optional

import numpy as np


def safe_mean(x: np.ndarray) -> float:
    """NaN-safe mean; returns NaN if empty."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    return float(np.nanmean(x))


def unwrap_s(s_seq: np.ndarray, track_len: float) -> np.ndarray:
    """
    Unwrap arc-length sequence across the 0/track_len boundary.
    Preserves the first element and accumulates smallest jumps.
    """
    s_seq = np.asarray(s_seq, dtype=float)
    if s_seq.size == 0:
        return s_seq
    unwrapped = np.zeros_like(s_seq)
    unwrapped[0] = s_seq[0]
    half = 0.5 * track_len
    for k in range(1, len(s_seq)):
        ds = s_seq[k] - s_seq[k - 1]
        if ds < -half:
            ds += track_len
        elif ds > half:
            ds -= track_len
        unwrapped[k] = unwrapped[k - 1] + ds
    return unwrapped


def lap_time(x_hist: np.ndarray, dt: float, track_len: float, idx_s: int = 0) -> float:
    """
    Time to complete one full lap from the initial s, using "unwrap_s".
    Returns NaN if the lap is not completed within the provided history.
    """
    if x_hist.size == 0 or track_len <= 0.0:
        return float("nan")
    s_unw = unwrap_s(x_hist[:, idx_s], track_len)
    target = s_unw[0] + track_len
    # First index where we reached (or exceeded) one lap
    hit = np.nonzero(s_unw >= target)[0]
    if hit.size == 0:
        return float("nan")
    k = int(hit[0])
    return k * float(dt)


def completion_ratio(x_hist: np.ndarray, track_len: float, idx_s: int = 0) -> float:
    """
    Fraction of a lap completed (>=1.0 means at least one full lap).
    Useful when a simulation is interrupted early.
    """
    if x_hist.size == 0 or track_len <= 0.0:
        return 0.0
    s_unw = unwrap_s(x_hist[:, idx_s], track_len)
    return float((s_unw[-1] - s_unw[0]) / track_len)


def mean_abs_track_error(x_hist: np.ndarray, idx_n: int = 1) -> float:
    """Mean absolute lateral error |n|."""
    if x_hist.size == 0:
        return float("nan")
    return float(np.mean(np.abs(x_hist[:, idx_n])))


def rms_track_error(x_hist: np.ndarray, idx_n: int = 1) -> float:
    """RMS lateral error."""
    if x_hist.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(x_hist[:, idx_n]))))


def mean_abs_speed_error(
    waypoints: np.ndarray,
    x_hist: np.ndarray,
    track_len: float,
    *,
    idx_s: int = 0,
    idx_vx: int = 3,
    wp_col_s: int = 4,
    wp_col_vx: int = 2
) -> float:
    """
    Mean |vx - vx_target(s)| using nearest-neighbor matching in s.
    """
    if x_hist.size == 0 or waypoints.size == 0:
        return float("nan")

    s_seq = np.mod(x_hist[:, idx_s], track_len) if track_len > 0 else x_hist[:, idx_s]
    wp_s = np.asarray(waypoints[:, wp_col_s], dtype=float)
    wp_v = np.asarray(waypoints[:, wp_col_vx], dtype=float)

    # nearest neighbor along s
    # (vectorized argmin over a (T x Nwp) distance matrix)
    idxs = np.abs(wp_s[None, :] - s_seq[:, None]).argmin(axis=1)
    v_targets = wp_v[idxs]
    return float(np.mean(np.abs(x_hist[:, idx_vx] - v_targets)))


def mean_cost(cost_hist: np.ndarray) -> float:
    """Mean solver cost ignoring NaNs."""
    return safe_mean(cost_hist)


def success_rate(status_hist: np.ndarray, success_codes: tuple = (0, 2)) -> float:
    """
    Fraction of solver calls that ended with a 'success' status.
    By default, 0=success and 2=max_iter_reached-but-acceptable are considered OK.
    """
    if status_hist.size == 0:
        return float("nan")
    status_hist = np.asarray(status_hist)
    ok = np.isin(status_hist, success_codes)
    return float(np.mean(ok))


def lap_boundaries(x_hist: np.ndarray, track_len: float, idx_s: int = 0) -> list:
    """Step indices where each lap is completed, using unwrapped arc length.

    Returns [k_1, k_2, ...] with k_i the first step at or past lap i.
    """
    if x_hist.size == 0 or track_len <= 0.0:
        return []
    s_unw = unwrap_s(x_hist[:, idx_s], track_len)
    bounds, lap = [], 1
    while True:
        hit = np.nonzero(s_unw >= s_unw[0] + lap * track_len)[0]
        if hit.size == 0:
            return bounds
        bounds.append(int(hit[0]))
        lap += 1


def per_lap_times(x_hist: np.ndarray, dt: float, track_len: float, idx_s: int = 0) -> np.ndarray:
    """Individual lap times in seconds (not cumulative)."""
    bounds = lap_boundaries(x_hist, track_len, idx_s)
    return np.diff([0] + bounds) * float(dt)
