#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse

from forzaeth_mpc.sim.tracks import load_track
from forzaeth_mpc.sim.waypoints import WP


@dataclass
class PlotHorizonCovConfig:
    # uncertainty visualization
    nsig: float = 2.0                 # number of sigmas for bands/ellipses (e.g. 2 = ~95%)
    ellipse_stride: int = 4           # draw one ellipse every N nodes
    ellipse_alpha: float = 0.20
    band_alpha: float = 0.20

    # layout
    figsize: Tuple[float, float] = (14.5, 8.0)

    # cosmetic
    grid_alpha: float = 0.30


def plot_horizon_with_cov(
    X_pred: np.ndarray,
    Pdiag_pred: Optional[np.ndarray] = None,
    *,
    ref_xy: np.ndarray,          # (M,2) raceline / centerline [x,y] (closed track)
    track_left: np.ndarray,      # (M,) left distances from ref_xy
    track_right: np.ndarray,     # (M,) right distances from ref_xy
    idx_map: Optional[Dict[str, int]] = None,
    idx_s: Union[int, str] = "pos_on_center_line_s",
    idx_n: Union[int, str] = "min_dist_to_center_line_n",
    series: Tuple[str, str, str, str, str] = (
        "velocity_v_x",
        "velocity_v_y",
        "yaw_rate",
        "steering_angle_delta",
        "accel",
    ),
    titles: Optional[dict] = None,
    cfg: PlotHorizonCovConfig = PlotHorizonCovConfig(),
):
    """
    Left: 5 time-series plots (mean + ±nsig*sqrt(var)) for selected states.
    Right: track (ref + boundaries) + predicted (s,n)->(x,y) trajectory mean
           + covariance ellipses derived from (var_s, var_n) and local tangent direction.

    Notes:
    - Uses *diagonal* covariance only (Pdiag_pred). Track ellipses assume independence of s and n.
    - s is arc-length progress along ref_xy. n is lateral deviation w.r.t. ref_xy.
    """

    # ----------------- helpers -----------------
    def _get_idx(x):
        if isinstance(x, (int, np.integer)):
            return int(x)
        if idx_map is None:
            raise ValueError("idx_map must be provided when using string indices.")
        return int(idx_map[x])

    def _compute_unit_tangents(centerline_xy: np.ndarray) -> np.ndarray:
        d = np.diff(centerline_xy, axis=0, append=centerline_xy[:1])
        nrm = np.linalg.norm(d, axis=1, keepdims=True)
        nrm = np.where(nrm < 1e-12, 1.0, nrm)
        return d / nrm  # (M,2)

    def _compute_unit_left_normals(tangents: np.ndarray) -> np.ndarray:
        return np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)  # (M,2)

    def _arc_length(centerline_xy: np.ndarray) -> np.ndarray:
        seg = np.linalg.norm(np.diff(centerline_xy, axis=0, append=centerline_xy[:1]), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])  # (M+1,)
        return s

    def _wrap_s(s: np.ndarray, L: float) -> np.ndarray:
        return np.mod(s, L)

    def _frenet_to_xy_and_tangent_batch(
        centerline_xy: np.ndarray,
        s_ref: np.ndarray,
        sn_batch: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert (s,n) to XY along a closed track defined by centerline_xy and cumulative arc s_ref.
        Returns:
          xy: (K,2)
          tang_unit: (K,2) unit tangent at the segment used for each sample
          idx_seg: (K,) segment indices used
        """
        M = centerline_xy.shape[0]
        L = float(s_ref[-1])

        s = _wrap_s(sn_batch[:, 0], L)
        n = sn_batch[:, 1]

        # segment index so that s in [s_ref[i], s_ref[i+1})
        idx = np.searchsorted(s_ref, s, side="right") - 1
        idx = np.clip(idx, 0, M - 1)
        idx_next = (idx + 1) % M

        s_i = s_ref[idx]
        ds_i = s_ref[idx_next] - s_ref[idx]
        ds_i = np.where(ds_i <= 0.0, ds_i + L, ds_i)  # wrap at end

        alpha = np.where(ds_i > 1e-12, (s - s_i) / ds_i, 0.0)

        p_i = centerline_xy[idx]
        p_j = centerline_xy[idx_next]
        p = p_i + (p_j - p_i) * alpha[:, None]

        tang = _compute_unit_tangents(centerline_xy)
        left_n = _compute_unit_left_normals(tang)

        n_vec = left_n[idx]
        xy = p + n[:, None] * n_vec

        tang_unit = tang[idx]
        return xy, tang_unit, idx

    # ----------------- checks -----------------
    X_pred = np.asarray(X_pred)
    if X_pred.ndim != 2:
        raise ValueError("X_pred must be 2D array (K, nx).")

    K = X_pred.shape[0]

    if Pdiag_pred is not None:
        Pdiag_pred = np.asarray(Pdiag_pred)
        if Pdiag_pred.shape != X_pred.shape:
            raise ValueError(f"Pdiag_pred must have same shape as X_pred: {X_pred.shape}")

    ref_xy = np.asarray(ref_xy)
    if ref_xy.ndim != 2 or ref_xy.shape[1] != 2:
        raise ValueError("ref_xy must be (M,2).")

    track_left = np.asarray(track_left).reshape(-1)
    track_right = np.asarray(track_right).reshape(-1)
    if track_left.shape[0] != ref_xy.shape[0] or track_right.shape[0] != ref_xy.shape[0]:
        raise ValueError("track_left/track_right length must match ref_xy length.")

    # ----------------- geometry (track + boundaries) -----------------
    s_ref = _arc_length(ref_xy)  # (M+1,)
    tang_ref = _compute_unit_tangents(ref_xy)
    left_n_ref = _compute_unit_left_normals(tang_ref)

    left_b = ref_xy + left_n_ref * track_left[:, None]
    right_b = ref_xy - left_n_ref * track_right[:, None]

    is_s = _get_idx(idx_s)
    is_n = _get_idx(idx_n)

    sn_pred = X_pred[:, [is_s, is_n]]
    xy_pred, tang_pred, _ = _frenet_to_xy_and_tangent_batch(ref_xy, s_ref, sn_pred)

    # ----------------- layout: 5x2 (left=5 series, right=map spanning all rows) -----------------
    fig = plt.figure(figsize=cfg.figsize)
    gs = fig.add_gridspec(
        5, 2,
        width_ratios=[1.15, 1.0],
        wspace=0.25,
        hspace=0.35
    )

    axs_ts = [fig.add_subplot(gs[i, 0]) for i in range(5)]
    ax_map = fig.add_subplot(gs[:, 1])

    # Show an arrow
    k0 = 0
    x0, y0 = xy_pred[k0]

    itheta = _get_idx("orientation_theta")
    theta0 = X_pred[k0, itheta]

    t0 = tang_pred[k0]
    arrow_len = 1.0
    dx, dy = arrow_len * t0[0], arrow_len * t0[1]
    # dx = arrow_len * np.cos(theta0)
    # dy = arrow_len * np.sin(theta0)

    ax_map.plot([x0], [y0], marker="o", markersize=6, label="Start")
    ax_map.arrow(
        x0, y0, dx, dy,
        length_includes_head=True,
        head_width=0.35,   # tune
        head_length=0.45,  # tune
        linewidth=2.0,
    )

    # ----------------- right: map -----------------
    ax_map.plot(ref_xy[:, 0], ref_xy[:, 1], lw=2, label="Raceline")
    ax_map.plot(left_b[:, 0], left_b[:, 1], ls="--", label="Left limit")
    ax_map.plot(right_b[:, 0], right_b[:, 1], ls="--", label="Right limit")
    ax_map.plot(xy_pred[:, 0], xy_pred[:, 1], ls=":", lw=2, label="Predicted traj (mean)")

    # covariance ellipses on track
    if Pdiag_pred is not None:
        var_s = np.maximum(Pdiag_pred[:, is_s], 0.0)
        var_n = np.maximum(Pdiag_pred[:, is_n], 0.0)

        for k in range(0, K, max(1, int(cfg.ellipse_stride))):
            if not np.isfinite(var_s[k]) or not np.isfinite(var_n[k]):
                continue

            sig_s = np.sqrt(var_s[k])
            sig_n = np.sqrt(var_n[k])

            # degenerate -> skip
            if sig_s < 1e-12 and sig_n < 1e-12:
                continue

            # ellipse aligned with local tangent direction:
            # width along tangent ~ 2*nsig*sig_s, height along normal ~ 2*nsig*sig_n
            tvec = tang_pred[k]
            ang = np.degrees(np.arctan2(tvec[1], tvec[0]))

            # Elipse tweaks for visualization
            tweak_width = 1.0
            tweak_height = 1.0

            e = Ellipse(
                xy=(xy_pred[k, 0], xy_pred[k, 1]),
                width=2.0 * tweak_width * cfg.nsig * sig_s,
                height=2.0 * tweak_height * cfg.nsig * sig_n,
                angle=ang,
                fill=True,
                alpha=cfg.ellipse_alpha,
                linewidth=0.0,
            )
            ax_map.add_patch(e)

    ax_map.set_title((titles or {}).get("map_title", "Track + Prediction (mean + covariance)"))
    ax_map.set_xlabel("x [m]")
    ax_map.set_ylabel("y [m]")
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.grid(True, alpha=cfg.grid_alpha)
    ax_map.legend(loc="best")

    # ----------------- left: 5 time-series (mean + bands) -----------------
    t = np.arange(K)

    for i, name in enumerate(series):
        j = _get_idx(name)
        y = X_pred[:, j]

        axs_ts[i].plot(t, y, lw=2, label="mean")

        if Pdiag_pred is not None:
            v = np.maximum(Pdiag_pred[:, j], 0.0)
            sig = np.sqrt(v)
            lo = y - cfg.nsig * sig
            hi = y + cfg.nsig * sig

            # handle NaNs gracefully (matplotlib breaks fills across NaNs)
            axs_ts[i].fill_between(t, lo, hi, alpha=cfg.band_alpha, label=f"±{cfg.nsig:.1f}σ")

        axs_ts[i].set_ylabel(name)
        axs_ts[i].grid(True, alpha=cfg.grid_alpha)
        if i == 0:
            axs_ts[i].legend(loc="best")

    axs_ts[-1].set_xlabel("horizon node")

    fig.suptitle((titles or {}).get("suptitle", "Single Horizon Prediction (Mean + Diagonal Covariance)"))
    fig.tight_layout()
    return fig, (axs_ts, ax_map)


def build_arrays_for_step(hp: pd.DataFrame, hc: pd.DataFrame, step: int):
    # Filter
    hp_s = hp[(hp["step"] == step)].copy()
    hc_s = hc[(hc["step"] == step)].copy()
    if hp_s.empty:
        raise RuntimeError(f"No rows found in horizon_pred.csv for step={step}")
    if hc_s.empty:
        raise RuntimeError(f"No rows found in horizon_covdig.csv for step={step}")

    # Sort by node
    hp_s = hp_s.sort_values("node")
    hc_s = hc_s.sort_values("node")

    # Define a compact internal state vector order.
    # You can expand this if you want to include more states.
    cols_mean = [
        "x_pos_on_center_line_s",
        "x_min_dist_to_center_line_n",
        "x_orientation_theta",
        "x_velocity_v_x",
        "x_velocity_v_y",
        "x_yaw_rate",
        "x_steering_angle_delta",
        "x_accel",
    ]

    cols_var = [
        "Pdiag_pos_on_center_line_s",
        "Pdiag_min_dist_to_center_line_n",
        "Pdiag_orientation_theta",
        "Pdiag_velocity_v_x",
        "Pdiag_velocity_v_y",
        "Pdiag_yaw_rate",
        "Pdiag_steering_angle_delta",
        "Pdiag_accel",
    ]

    for c in cols_mean:
        if c not in hp_s.columns:
            raise RuntimeError(f"Missing column {c} in horizon_pred.csv")
    for c in cols_var:
        if c not in hc_s.columns:
            raise RuntimeError(f"Missing column {c} in horizon_covdig.csv")

    X = hp_s[cols_mean].to_numpy(dtype=float)
    Pdiag = hc_s[cols_var].to_numpy(dtype=float)

    # Map short names to indices (used by plotter)
    idx_map: Dict[str, int] = {
        "pos_on_center_line_s": 0,
        "min_dist_to_center_line_n": 1,
        "orientation_theta": 2,
        "velocity_v_x": 3,
        "velocity_v_y": 4,
        "yaw_rate": 5,
        "steering_angle_delta": 6,
        "accel": 7,
    }

    return X, Pdiag, idx_map


def main():

    step = 2300
    ellipse_stride = 4
    base_dir = "/home/gil/catkin_ws/src/race_stack/controller/lbmpc/data/offline_viz/"
    csv_dir = base_dir + "csv"
    fig_dir = base_dir + "figs"

    hp_path = os.path.join(csv_dir, "horizon_pred.csv")
    hc_path = os.path.join(csv_dir, "horizon_covdiag.csv")

    # Read csvs to get prediction mean and covariance
    hp = pd.read_csv(hp_path)
    hc = pd.read_csv(hc_path)

    X, Pdiag, idx_map = build_arrays_for_step(hp, hc, step=step)

    # === Waypoints & configs ===
    waypoints, _ = load_track("GLC_pit_rbring1")
    refline_xy = waypoints[:, [WP.X, WP.Y]]
    left_dist = waypoints[:, WP.D_LEFT]
    right_dist = waypoints[:, WP.D_RIGHT]

    cfg = PlotHorizonCovConfig(
        nsig=1.0,
        ellipse_stride=ellipse_stride,
    )

    titles = {
        "suptitle": f"Prediction Horizon @ step={step}",
        "map_title": "Track + predicted trajectory (mean + covariance ellipses)",
    }

    fig, _ = plot_horizon_with_cov(
        X, Pdiag,
        ref_xy=refline_xy,
        track_left=left_dist,
        track_right=right_dist,
        idx_map=idx_map,
        idx_s="pos_on_center_line_s",
        idx_n="min_dist_to_center_line_n",
        series=("velocity_v_x", "velocity_v_y", "yaw_rate", "steering_angle_delta", "accel"),
        titles=titles,
        cfg=cfg,
    )

    out_path = os.path.join(fig_dir, f"horizon_step_{step:06d}.png")
    fig.savefig(out_path, dpi=180)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
