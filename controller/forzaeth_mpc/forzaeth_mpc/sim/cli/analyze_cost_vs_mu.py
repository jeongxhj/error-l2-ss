#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COST ANALYSIS — μ split + metrics with Scenario

- Track/plant μ set once via mpc.set_sim_mu(mu_track)
- MPC μ swept via 'p' vector (mu_mpc per horizon stage)
- Plant integrates x_{k+1} = RK4( mpc._dynamics_of_car(0.0, x_k) )
  with u written in x[8], x[9] before each step.
"""

from __future__ import annotations

import argparse
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from ..core.controllers import ControllerMPC
from ..core.plant import Plant
from ..core.rollout import RolloutRunner
from ..tracks import REFLINES
from ..core.scenario import Scenario
from ..metrics.metrics import (completion_ratio, lap_time,
                               mean_abs_speed_error, mean_abs_track_error,
                               mean_cost, rms_track_error, success_rate)
from ..plotting.vis_analyse_cost_vs_mu import plot_mu_sensitivity_violin


def main():
    ap = argparse.ArgumentParser(description="Cost analysis: μ sweep + metrics.")
    ap.add_argument("--track", type=str, default="THETRACK")
    ap.add_argument("--racecar_version", type=str, default="NUC5")
    ap.add_argument("--floor", type=str, default="dubi")
    ap.add_argument("--dt", type=float, default=0.02)
    # ~377 steps is one lap at dt=0.02; 400 finishes with 6% margin, which the
    # lower-mu cases eat into
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--integrator", type=str, default="rk4")
    ap.add_argument("--refline", type=str, default="iqp", choices=sorted(REFLINES))

    # Split μ explicitly:
    ap.add_argument("--mu_track", type=float, default=1.00, help="Plant/track μ via set_sim_mu")
    ap.add_argument("--mu_mpc_list", type=float, nargs="+",
                    default=[0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15],
                    help="Values used inside MPC p-vector")

    ap.add_argument("--vx0", type=float, default=3.0)
    ap.add_argument("--s0", type=float, default=20.0)
    ap.add_argument("--n0", type=float, default=0.0)
    ap.add_argument("--psi0", type=float, default=0.0)
    args = ap.parse_args()

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

    # --- MPC weights from config (what you were using)
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

    # Set PLANT/track μ once (decoupled from μ_mpc sweep)
    mpc.set_sim_mu(float(args.mu_track))

    # --- μ_mpc sweep (MPC internal μ differs from plant μ)
    rows = []
    for mu_mpc in args.mu_mpc_list:
        # Warm start between runs
        try:
            mpc.apply_warm_start(pose_frenet=np.array([args.s0, args.n0, args.psi0, args.vx0], dtype=float))
        except Exception:
            print(f"Failed to apply warm start!")
            pass

        X, U, C, S, _ = runner.run(
            x0_full=x0_full, waypoints=waypoints, iterations=args.steps,
            qdict=qdict, mpc_mu=float(mu_mpc),
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

        rows.append((mu_mpc, lt, comp, man, rms, mve, mc, sr))

        print(f"[μ_mpc={mu_mpc:.3f} | μ_track={args.mu_track:.3f}] "
              f"lap={lt:.3f}s, comp={comp:.3f}, |n|={man:.3f}, rms(n)={rms:.3f}, "
              f"|vx-v*|={mve:.3f}, mean_cost={mc:.3f}, solver_ok={sr:.2%}")

    # ------------------ plots like the previous script ------------------ #
    rows_arr = np.asarray(rows, dtype=object)
    mu_arr = rows_arr[:, 0].astype(float)   # MPC μ values you swept
    lap_arr = rows_arr[:, 1].astype(float)
    man_arr = rows_arr[:, 3].astype(float)
    mve_arr = rows_arr[:, 5].astype(float)
    mc_arr = rows_arr[:, 6].astype(float)

    # Δμ = μ_track − μ_mpc (label expects 'track − MPC nominal')
    delta_mu = float(args.mu_track) - mu_arr

    plot_mu_sensitivity_violin(
        delta_mu=delta_mu,
        lap_times=lap_arr,
        mean_abs_lat_dev=man_arr,
        mean_abs_vel_err=mve_arr,
        mean_cost=mc_arr,
    )

    plt.show()


if __name__ == "__main__":
    main()
