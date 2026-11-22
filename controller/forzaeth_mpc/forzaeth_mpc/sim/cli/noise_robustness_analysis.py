#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROBUSTNESS ANALYSIS — Plant noise sweep (vx, vy, yaw_rate)

- Noise is injected ONLY in the plant integration step via Plant.noise_vars
- Sweeps scalar std deviations σ over a user-provided list
- Uses the same Scenario/MPC/Rollout/metrics structure as the μ-sensitivity script
- Plots results in a similar multi-metric style

Assumptions about state indexing (consistent with your stack):
  s = x[0], n = x[1], psi = x[2], vx = x[3], vy = x[4], yaw_rate = x[5]
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from ...utils.indices import StateIndex
from ..core.controllers import ControllerMPC
from ..core.plant import Plant
from ..core.rollout import RolloutRunner
from ..tracks import REFLINES
from ..core.scenario import Scenario
from ..metrics.metrics import (completion_ratio, lap_time,
                               mean_abs_speed_error, mean_abs_track_error,
                               mean_cost, rms_track_error, success_rate)


def _make_noise_vars(state_dim: int, sigma: float) -> np.ndarray:
    """
    Build a variance vector with noise only on [vx, vy, yaw_rate] = indices [3, 4, 5].
    Variance = sigma^2 (same sigma applied to the three dynamic states).
    """
    noise_vars = np.zeros(state_dim, dtype=float)
    if state_dim < 6:
        raise ValueError(
            f"Expected at least 6 states to set vx, vy, yaw_rate noise; got {state_dim}."
        )
    var = float(sigma) ** 2
    # These were literal 3/4/5 with a "# yaw_rate" comment on the last one, but index 5
    # is the steering angle under STMPC's ordering -- it has been injecting steering
    # noise and calling it yaw rate.
    noise_vars[StateIndex.VELOCITY_V_X] = var
    noise_vars[StateIndex.VELOCITY_V_Y] = var
    noise_vars[StateIndex.YAW_RATE] = var
    return noise_vars


def _plot_noise_robustness(
    sigmas: np.ndarray,
    lap_times: np.ndarray,
    mean_abs_lat_dev: np.ndarray,
    mean_abs_vel_err: np.ndarray,
    mean_cost_arr: np.ndarray,
):
    """
    Simple, clean plotting with multiple aligned panels,
    mirroring the style/intent of the μ-sensitivity plots.
    """

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    ax = axes.ravel()

    # 1) Lap time
    ax[0].plot(sigmas, lap_times, marker='o')
    ax[0].set_title("Lap time vs noise σ")
    ax[0].set_xlabel("σ (vx, vy, yaw_rate)")
    ax[0].set_ylabel("Lap time [s]")
    ax[0].grid(True, alpha=0.3)

    # 2) Mean |n| (lateral deviation)
    ax[1].plot(sigmas, mean_abs_lat_dev, marker='o')
    ax[1].set_title("Mean |n| vs noise σ")
    ax[1].set_xlabel("σ (vx, vy, yaw_rate)")
    ax[1].set_ylabel("Mean |n| [m]")
    ax[1].grid(True, alpha=0.3)

    # 3) Mean |vx - v*|
    ax[2].plot(sigmas, mean_abs_vel_err, marker='o')
    ax[2].set_title("Mean |vx - v*| vs noise σ")
    ax[2].set_xlabel("σ (vx, vy, yaw_rate)")
    ax[2].set_ylabel("Mean speed error [m/s]")
    ax[2].grid(True, alpha=0.3)

    # 4) Mean cost
    ax[3].plot(sigmas, mean_cost_arr, marker='o')
    ax[3].set_title("Mean cost vs noise σ")
    ax[3].set_xlabel("σ (vx, vy, yaw_rate)")
    ax[3].set_ylabel("Mean stage cost")
    ax[3].grid(True, alpha=0.3)

    fig.suptitle("MPC Robustness to Plant Integration Noise", fontsize=14)


def main():
    ap = argparse.ArgumentParser(description="Robustness analysis: plant noise sweep (vx, vy, yaw_rate).")
    ap.add_argument("--track", type=str, default="THETRACK")
    ap.add_argument("--racecar_version", type=str, default="NUC5")
    ap.add_argument("--floor", type=str, default="dubi")
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--integrator", type=str, default="rk4")
    ap.add_argument("--refline", type=str, default="iqp", choices=sorted(REFLINES))

    # Plant/track μ (kept constant here, unlike the μ sweep script)
    ap.add_argument("--mu_track", type=float, default=1.00, help="Plant/track μ via set_sim_mu")

    # Noise sweep: scalar std (σ) applied equally to vx, vy, yaw_rate (variance = σ²)
    ap.add_argument("--noise_std_list", type=float, nargs="+",
                    default=[0.0, 0.01, 0.02, 0.03, 0.05],
                    help="Std devs (σ) for vx, vy, yaw_rate. Variance = σ².")

    # Initial conditions
    ap.add_argument("--vx0", type=float, default=3.0)
    ap.add_argument("--s0", type=float, default=20.0)
    ap.add_argument("--n0", type=float, default=0.0)
    ap.add_argument("--psi0", type=float, default=0.0)

    # Reproducibility
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    # Seed numpy RNG (affects Gaussian noise draws)
    np.random.seed(int(args.seed))

    # ---------------- Scenario: load everything ---------------- #
    scn = Scenario(track=args.track,
                   racecar_version=args.racecar_version,
                   floor=args.floor,
                   refline=args.refline)

    # Build MPC
    mpc = scn.make_mpc(s0=args.s0, n0=args.n0, psi0=args.psi0, dt=args.dt)
    track_len = scn.track_len
    waypoints = scn.waypoints

    # Initial full state
    x0_full = scn.make_x0_full(mpc, s0=args.s0, n0=args.n0, psi0=args.psi0, vx0=args.vx0)
    state_dim = int(x0_full.shape[0])

    # --- MPC weights from config (same as your cost analysis script)
    qdict: Dict[str, float] = dict(
        q_adv=mpc.stmpc_config.qadv if getattr(mpc.stmpc_config, "adv_maximization", False) else 0.0,
        q_v=mpc.stmpc_config.qv, q_n=mpc.stmpc_config.qn, q_alpha=mpc.stmpc_config.qalpha,
        q_jerk=mpc.stmpc_config.qjerk, q_ddelta=mpc.stmpc_config.qddelta,
        terminal_multiplier=mpc.stmpc_config.terminal_multiplier
    )

    # --- Controller + Plant + Rollout
    ctrl = ControllerMPC(mpc, track_len)
    plant = Plant(dyn_fun=mpc._dynamics_of_car, dt=args.dt, integrator=args.integrator)
    runner = RolloutRunner(controller=ctrl, plant=plant, track_len=track_len)

    # Set PLANT/track μ once
    mpc.set_sim_mu(float(args.mu_track))

    # ------------------ Noise sweep ------------------ #
    rows: List[tuple] = []

    for sigma in args.noise_std_list:
        # Update plant noise variances: only vx, vy, yaw_rate
        noise_vars = _make_noise_vars(state_dim, sigma=float(sigma))
        plant.update_noise_vars(noise_vars if sigma > 0.0 else None)

        # Warm start between runs
        try:
            mpc.apply_warm_start(
                pose_frenet=np.array([args.s0, args.n0, args.psi0, args.vx0], dtype=float)
            )
        except Exception:
            tqdm.write("Failed to apply warm start!")
            pass

        X, U, C, S, _ = runner.run(
            x0_full=x0_full, waypoints=waypoints, iterations=args.steps,
            qdict=qdict, mpc_mu=float(args.mu_track),  # MPC μ same as track μ here
            set_sim_mu_fn=None,
            capture_predictions=False,
            log_fn=tqdm.write,
        )

        lt = lap_time(X, args.dt, track_len, idx_s=0)
        comp = completion_ratio(X, track_len, idx_s=0)
        man = mean_abs_track_error(X, idx_n=1)
        rms = rms_track_error(X, idx_n=1)
        mve = mean_abs_speed_error(waypoints, X, track_len, idx_s=0, idx_vx=3)
        mc = mean_cost(C)
        sr = success_rate(S)

        rows.append((float(sigma), lt, comp, man, rms, mve, mc, sr))

        tqdm.write(
            f"[σ={sigma:.3f} | μ_track={args.mu_track:.3f}] "
            f"lap={lt:.3f}s, comp={comp:.3f}, |n|={man:.3f}, rms(n)={rms:.3f}, "
            f"|vx-v*|={mve:.3f}, mean_cost={mc:.3f}, solver_ok={sr:.2%}"
        )

    # ------------------ Plots (same style/intent) ------------------ #
    rows_arr = np.asarray(rows, dtype=object)
    sigma_arr = rows_arr[:, 0].astype(float)   # noise σ values you swept
    lap_arr = rows_arr[:, 1].astype(float)
    man_arr = rows_arr[:, 3].astype(float)
    mve_arr = rows_arr[:, 5].astype(float)
    mc_arr = rows_arr[:, 6].astype(float)

    _plot_noise_robustness(
        sigmas=sigma_arr,
        lap_times=lap_arr,
        mean_abs_lat_dev=man_arr,
        mean_abs_vel_err=mve_arr,
        mean_cost_arr=mc_arr,
    )

    plt.show()


if __name__ == "__main__":
    main()
