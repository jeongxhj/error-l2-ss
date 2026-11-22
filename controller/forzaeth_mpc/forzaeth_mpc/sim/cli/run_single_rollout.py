#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAIN TEST MPC — single closed-loop run using Scenario + RolloutRunner

- Loads scenario via Scenario (waypoints + configs)
- Builds STMPC with your real constructor
- Splits μ: track/plant via set_sim_mu; MPC μ via p-vector each step
- Integrates plant by RK4 over mpc._dynamics_of_car(0.0, x), with control injected into x[8], x[9]
- Captures full-horizon predictions each step via RolloutRunner (for spaghetti/state-1 plots)
"""

from __future__ import annotations

import argparse
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from forzaeth_mpc.utils.indices import StateIndex
from forzaeth_mpc.sim.waypoints import WP

from ..core.controllers import ControllerMPC
from ..core.plant import Plant
from ..core.rollout import RolloutRunner
from ..tracks import REFLINES
from ..core.scenario import Scenario
from ..metrics.metrics import (mean_abs_speed_error, mean_abs_track_error,
                               rms_track_error, unwrap_s)
from ..plotting.vis_run_simple_rollout import (
    plot_closed_loop_summary, plot_spaghetti_predictions_vs_actual,
    plot_state_prediction_vs_actual, plot_velocity_vs_progress)


def print_per_lap_summary(
    x_hist: np.ndarray,
    waypoints: np.ndarray,
    dt: float,
    track_len: float,
    *,
    idx_s: int = 0,
    idx_n: int = 1,
    idx_vx: int = 3,
    wp_col_s: int = 4,
    wp_col_vx: int = 2,
    log_fn=print,
) -> None:
    """
    Find completed laps in x_hist and print lap time + error metrics per lap.
    A 'lap' is whenever unwrapped s increases by +track_len.
    """
    if x_hist.size == 0 or track_len <= 0:
        log_fn("No data to summarize.")
        return

    s_unw = unwrap_s(x_hist[:, idx_s], track_len)
    s0 = float(s_unw[0])

    # Identify [start_idx, end_idx] slices for each completed lap
    lap_slices = []
    lap_start = 0
    next_target = s0 + track_len
    for k, s in enumerate(s_unw):
        if s >= next_target:
            lap_slices.append((lap_start, k))   # inclusive start, inclusive end
            lap_start = k                        # next lap starts at this sample
            next_target += track_len

    if not lap_slices:
        log_fn("No complete laps detected.")
        return

    # Header
    log_fn("\nPer-lap summary (completed laps):")
    log_fn(f"{'Lap':>3} | {'Steps':>5} | {'Lap time [s]':>12} | "
           f"{'mean|n| [m]':>11} | {'rms n [m]':>10} | {'mean|vx-v*| [m/s]':>16}")

    # Metrics for each lap (use existing metrics functions)
    for i, (i0, i1) in enumerate(lap_slices, start=1):
        # Use slice i0..i1 (inclusive). Steps in slice = (i1 - i0)
        xlap = x_hist[i0:i1 + 1]

        steps = (i1 - i0)
        lap_time_s = steps * float(dt)  # sample-aligned lap time

        mean_n = mean_abs_track_error(xlap, idx_n=idx_n)
        rms_n = rms_track_error(xlap, idx_n=idx_n)
        mean_v_err = mean_abs_speed_error(
            waypoints=waypoints,
            x_hist=xlap,
            track_len=track_len,
            idx_s=idx_s,
            idx_vx=idx_vx,
            wp_col_s=wp_col_s,
            wp_col_vx=wp_col_vx,
        )

        log_fn(f"{i:>3d} | {steps:>5d} | {lap_time_s:>12.3f} | "
               f"{mean_n:>11.3f} | {rms_n:>10.3f} | {mean_v_err:>16.3f}")


def main():
    ap = argparse.ArgumentParser(description="Single MPC rollout with RolloutRunner.")
    ap.add_argument("--track", type=str, default="THETRACK")
    ap.add_argument("--racecar_version", type=str, default="NUC5")
    ap.add_argument("--floor", type=str, default="dubi")
    ap.add_argument("--refline", type=str, default="iqp", choices=sorted(REFLINES))

    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=600, help="Number of control steps to simulate (default: 1).")
    ap.add_argument("--integrator", type=str, default="rk4", choices=["rk4", "euler"])

    ap.add_argument("--mu_track", type=float, default=1.00, help="Plant/track μ via set_sim_mu")
    ap.add_argument("--mu_mpc", type=float, default=1.00, help="MPC μ inside p-vector")

    ap.add_argument("--s0", type=float, default=20.0)
    ap.add_argument("--n0", type=float, default=0.0)
    ap.add_argument("--psi0", type=float, default=0.0)
    ap.add_argument("--vx0", type=float, default=3.0)

    # visualization flags
    ap.add_argument("--vis_spaghetti_predictions_vs_actual", action="store_true")
    ap.add_argument("--vis_state_prediction_vs_actual", action="store_true")
    ap.add_argument(
        "--no_vis_closed_loop_summary",
        dest="vis_closed_loop_summary",
        action="store_false",
        help="Disable visualization of the closed-loop summary."
    )
    ap.add_argument("--vis_target_vel_actual", action="store_true")

    # possibility to add noise to the plant
    ap.add_argument(
        "--noise_std", type=float, nargs="+", default=None,
        help="Gaussian noise σ on plant states. Pass 1 value (same for vx,vy,yaw_rate) "
        "or 3 values (σ_vx σ_vy σ_r). Variance = σ²."
    )
    ap.add_argument("--seed", type=int, default=None, help="Random seed for noise.")
    args = ap.parse_args()

    # -------------------- scenario + MPC build -------------------- #
    scn = Scenario(track=args.track,
                   racecar_version=args.racecar_version,
                   floor=args.floor,
                   refline=args.refline)

    mpc = scn.make_mpc(s0=args.s0, n0=args.n0, psi0=args.psi0, dt=args.dt)
    waypoints = scn.waypoints
    track_len = scn.track_len

    # initial plant state (full)
    x0_full = scn.make_x0_full(mpc, s0=args.s0, n0=args.n0, psi0=args.psi0, vx0=args.vx0)

    # MPC weights from config
    stmpc_config = scn.stmpc_config
    qdict: Dict[str, float] = dict(
        q_adv=stmpc_config.qadv if getattr(stmpc_config, "adv_maximization", False) else 0.0,
        q_v=stmpc_config.qv, q_n=stmpc_config.qn, q_alpha=stmpc_config.qalpha,
        q_jerk=stmpc_config.qjerk, q_ddelta=stmpc_config.qddelta,
        terminal_multiplier=stmpc_config.terminal_multiplier
    )

    # ----------------- controller + plant + runner ---------------- #
    ctrl = ControllerMPC(mpc, track_len)
    plant = Plant(dyn_fun=mpc._dynamics_of_car, dt=args.dt, integrator=args.integrator)
    runner = RolloutRunner(controller=ctrl, plant=plant, track_len=track_len)

    # -------------------------- μ split --------------------------- #
    # Track/plant μ once:
    try:
        mpc.set_sim_mu(float(args.mu_track))
    except Exception:
        pass
    # MPC μ is applied inside the runner each step via p-vector.

    if args.seed is not None:
        np.random.seed(int(args.seed))
    # Optional: enable plant noise only if --noise_std was provided
    if args.noise_std is not None:
        sigmas = np.asarray(args.noise_std, dtype=float)
        if sigmas.size == 1:
            sigma_vx = sigma_vy = sigma_r = float(sigmas[0])
        elif sigmas.size == 3:
            sigma_vx, sigma_vy, sigma_r = map(float, sigmas)
        else:
            raise ValueError("--noise_std expects 1 or 3 values (σ or σ_vx σ_vy σ_r).")

        # Variances only on [vx, vy, yaw_rate] => indices [3, 4, 5]
        noise_vars = np.zeros_like(x0_full, dtype=float)
        # index 5 is the steering angle under STMPC's ordering, not the yaw rate --
        # same mislabelling as noise_robustness_analysis had.
        noise_vars[StateIndex.VELOCITY_V_X] = sigma_vx ** 2
        noise_vars[StateIndex.VELOCITY_V_Y] = sigma_vy ** 2
        noise_vars[StateIndex.YAW_RATE] = sigma_r ** 2

        # Apply to plant (uses your Plant.update_noise_vars)
        plant.update_noise_vars(noise_vars)

    # --------------------------- run once ------------------------- #
    X, U, C, S, X_pred_seq = runner.run(
        x0_full=x0_full,
        waypoints=waypoints,
        iterations=args.steps,
        qdict=qdict,
        mpc_mu=float(args.mu_mpc),
        set_sim_mu_fn=None,
        capture_predictions=True,      # <-- important for spaghetti/state-1 plots
        log_fn=tqdm.write,
    )

    print_per_lap_summary(
        x_hist=X,
        waypoints=waypoints,
        dt=args.dt,
        track_len=track_len,
        log_fn=tqdm.write,     # pretty prints under tqdm
    )

    # ---------------------------- plots --------------------------- #
    # 1) Spaghetti predictions vs actual
    if args.vis_spaghetti_predictions_vs_actual and X_pred_seq is not None:
        state_names = ["s", "n", "theta", "v_x", "v_y", "delta", "yaw_rate", "accel"]
        plot_spaghetti_predictions_vs_actual(
            x_pred_seq_hist=X_pred_seq,             # (K, H+1, nx)
            x_hist=X,                                # (K, x_full)
            state_names=state_names[:X_pred_seq.shape[2]],
            plot_every=1
        )
        plt.show()

    # 2) State prediction (stage 1) vs actual
    if args.vis_state_prediction_vs_actual and X_pred_seq is not None:
        nx = mpc.model.n_x
        state_names = ["s", "n", "theta", "v_x", "v_y", "delta", "yaw_rate", "accel"]
        plot_state_prediction_vs_actual(
            x_pred1_hist=X_pred_seq[:, 1, :],       # (K, nx)
            x_hist=X,                                # (K, x_full)
            state_names=state_names[:nx]
        )
        plt.show()

    # 3) Closed-loop summary plot (track + XY path etc.)
    if args.vis_closed_loop_summary:
        refline_xy = waypoints[:, [WP.X, WP.Y]]
        left_dist = waypoints[:, WP.D_LEFT]
        right_dist = waypoints[:, WP.D_RIGHT]
        plot_closed_loop_summary(
            refline_xy=refline_xy,
            left_dist=left_dist,
            right_dist=right_dist,
            x_hist=X,   # (K, x_full)
            u_hist=U
        )
        plt.show()

    # 4) Target vs actual speed along s
    if args.vis_target_vel_actual:
        plot_velocity_vs_progress(
            waypoints=waypoints,   # (N,7)
            x_hist=X,              # (K, x_full)
            idx_s=0,
            idx_vx=3,
            title="Target vs Actual speed along s"
        )
        plt.show()


if __name__ == "__main__":
    main()
