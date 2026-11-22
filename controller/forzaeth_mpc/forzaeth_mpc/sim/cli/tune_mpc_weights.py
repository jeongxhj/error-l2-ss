#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OPTUNA AUTOTUNING — MPC cost weights

- Uses Scenario + ControllerMPC + Plant + RolloutRunner
- Splits μ: plant/track via mpc.set_sim_mu(mu_track), MPC μ via p-vector each step
- Plant integrates x_{k+1} with RK4 over mpc._dynamics_of_car(0.0, x_k),
  after injecting the control into x[8], x[9] (handled inside RolloutRunner->Plant).

Objective:
    J = w_lt * lap_time + w_cost * mean_cost
Where:
    lap_time: time to complete one lap (NaN -> heavy penalty)
    mean_cost: mean ACADOS objective over rollout (NaN -> heavy penalty)
"""

from __future__ import annotations

import argparse
import math
from typing import Dict

import numpy as np
import optuna
from tqdm.auto import tqdm

from ..core.controllers import ControllerMPC
from ..core.plant import Plant
from ..core.rollout import RolloutRunner
# Core stack
from ..tracks import REFLINES
from ..core.scenario import Scenario
# Metrics
from ..metrics.metrics import (lap_time, mean_abs_speed_error,
                               mean_abs_track_error)
from ..metrics.metrics import mean_cost as metric_mean_cost
from ..plotting.vis_tune_mpc_weights import plot_parallel_params_vs_metrics

PARAM_SPECS = {
    "q_adv": dict(low=1.0, high=50.0, log=True),  # This actual limits are for simulating MPCC
    "q_v": dict(low=0.5, high=0.6, log=True),
    "q_n": dict(low=10.0, high=50.0, log=True),
    "q_alpha": dict(low=1e-6, high=1e-1, log=True),
    "q_jerk": dict(low=1e-6, high=1e-1, log=True),
    "q_ddelta": dict(low=1e-6, high=1e-1, log=True),
    "terminal_multiplier": dict(low=1e-2, high=5.0, log=True),
}
PARAM_ORDER = list(PARAM_SPECS.keys())

# ----------------------------
# Search space (Optuna params)
# ----------------------------


def suggest_weights(trial: optuna.Trial) -> Dict[str, float]:
    """
    Define/search your MPC weighting hyperparameters here.
    Names must match what ControllerMPC.set_online_params expects in 'qdict'.
    """
    return {
        name: trial.suggest_float(name, spec["low"], spec["high"], log=spec["log"])
        for name, spec in PARAM_SPECS.items()
    }


# ----------------------------
# Objective wrapper
# ----------------------------
class MPCOptunaObjective:
    def __init__(
        self,
        scn: Scenario,
        dt: float,
        steps: int,
        integrator: str,
        mu_track: float,
        mu_mpc: float,
        w_lt: float,
        w_cost: float,
        w_abs_n: float,
        w_abs_v: float,
        seed: int = 42,
    ):
        # Store scenario + constants
        self.scn = scn
        self.dt = float(dt)
        self.steps = int(steps)
        self.integrator = integrator
        self.mu_track = float(mu_track)
        self.mu_mpc = float(mu_mpc)
        self.w_lt = float(w_lt)
        self.w_cost = float(w_cost)
        self.w_abs_n = float(w_abs_n)
        self.w_abs_v = float(w_abs_v)

        # Build MPC + wrappers once; reuse per-trial with proper resets
        # Initial MPC built around scenario's init pose (vx comes from make_mpc)
        self.mpc = scn.make_mpc(
            s0=float(scn.init_pose_frenet[0]),
            n0=float(scn.init_pose_frenet[1]),
            psi0=float(scn.init_pose_frenet[2]),
            dt=self.dt,
        )
        self.track_len = scn.track_len
        self.waypoints = scn.waypoints

        self.ctrl = ControllerMPC(self.mpc, self.track_len)
        self.plant = Plant(dyn_fun=self.mpc._dynamics_of_car, dt=self.dt, integrator=self.integrator)
        self.runner = RolloutRunner(controller=self.ctrl, plant=self.plant, track_len=self.track_len)

        # Set PLANT μ once (decoupled from μ_mpc)
        try:
            self.mpc.set_sim_mu(self.mu_track)
        except Exception:
            pass

        # Seed any stochastic parts if needed (not used here, but harmless)
        np.random.default_rng(seed)

    def _reset_mpc_state(self) -> np.ndarray:
        """
        Bring MPC + plant state back to scenario's initial pose between trials.
        Returns the full plant state x0_full used by the plant/runner.
        """
        # Warm start + clear horizon guesses
        try:
            self.mpc.mpc_sd = np.zeros((self.mpc.stmpc_config.N + 1, 2))
            self.mpc.mpc_init_params()
        except Exception:
            pass

        pose = self.scn.init_pose_frenet
        try:
            self.mpc.apply_warm_start(pose_frenet=pose.copy())
        except Exception:
            pass

        x0_full = self.scn.make_x0_full(
            self.mpc, s0=pose[0], n0=pose[1], psi0=pose[2], vx0=float(self.scn.init_vx0)
        )

        # Also seed acados x(0) in case your solver uses it
        try:
            self.mpc.acados_solver.set(0, "x", x0_full[: self.mpc.model.n_x])
        except Exception:
            pass

        return x0_full

    def __call__(self, trial: optuna.Trial) -> float:
        # 1) Propose a set of weights for this trial
        qdict = suggest_weights(trial)

        # 2) Reset MPC and plant state to identical initial conditions
        x0_full = self._reset_mpc_state()

        # 3) Run a closed-loop rollout (no prediction capture needed here)
        X, U, C, S, _ = self.runner.run(
            x0_full=x0_full,
            waypoints=self.waypoints,
            iterations=self.steps,
            qdict=qdict,
            mpc_mu=self.mu_mpc,
            set_sim_mu_fn=None,
            capture_predictions=False,
            log_fn=tqdm.write,
        )

        # 4) Compute metrics
        lt = lap_time(X, self.dt, self.track_len, idx_s=0)
        mc = metric_mean_cost(C)
        man = mean_abs_track_error(X, idx_n=1)
        mve = mean_abs_speed_error(self.waypoints, X, self.track_len, idx_s=0, idx_vx=3)

        # 5) Penalties for failed runs
        if math.isnan(lt):
            # A flat penalty makes every non-finishing trial tie, so a study where most
            # trials fail looks like it converged when it has learned nothing. Say so.
            print(f"[tune] trial {trial.number}: no lap completed in {self.steps} steps "
                  f"(dt={self.dt}) -- penalised. Raise --steps if most trials do this.")
            lt = 1e6
        if math.isnan(mc):
            mc = 1e6
        if math.isnan(man):
            man = 1e6
        if math.isnan(mve):
            mve = 1e6

        # 6) Scalarized objective
        J = self.w_lt * float(lt) + self.w_cost * float(mc) + self.w_abs_n * float(man) + self.w_abs_v * float(mve)

        # 7) Log metrics for convenience
        trial.set_user_attr("lap_time", float(lt))
        trial.set_user_attr("mean_cost", float(mc))
        trial.set_user_attr("mean_abs_n", float(man))
        trial.set_user_attr("mean_abs_v_err", float(mve))
        trial.set_user_attr("objective_J", float(J))

        # Optional pruning based on lap time
        trial.report(float(lt), step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return float(J)


# ----------------------------
# CLI
# ----------------------------
def main():
    p = argparse.ArgumentParser(description="Optuna autotuning for MPC cost weights (modular).")
    # Scenario / build
    p.add_argument("--track", type=str, default="THETRACK")
    p.add_argument("--racecar_version", type=str, default="NUC5")
    p.add_argument("--floor", type=str, default="dubi")
    p.add_argument("--refline", type=str, default="iqp", choices=sorted(REFLINES))
    p.add_argument("--dt", type=float, default=0.02)
    # a lap on THETRACK is ~377 steps at dt=0.02. Anything near that means the
    # worse weights Optuna samples never finish one, lap_time() returns NaN and
    # every trial collapses onto the same penalty.
    p.add_argument("--steps", type=int, default=900)
    p.add_argument("--integrator", type=str, default="rk4", choices=["rk4", "euler"])

    # Initial pose
    p.add_argument("--s0", type=float, default=20.0)
    p.add_argument("--n0", type=float, default=0.0)
    p.add_argument("--psi0", type=float, default=0.0)
    p.add_argument("--vx0", type=float, default=3.0)

    # Friction split
    p.add_argument("--mu_track", type=float, default=1.00, help="Plant/track μ via set_sim_mu")
    p.add_argument("--mu_mpc", type=float, default=1.00, help="MPC μ used in p-vector")

    # Scalarization weights
    p.add_argument("--w_lt", type=float, default=1.0, help="Weight for lap time in objective")
    p.add_argument("--w_cost", type=float, default=0.0, help="Weight for mean solver cost in objective")
    p.add_argument("--w_abs_n", type=float, default=0.0, help="Weight for mean solver lateral error in objective")
    p.add_argument("--w_abs_v", type=float, default=0.0, help="Weight for mean solver velocity error in objective")

    # Optuna
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--study_name", type=str, default="mpc_cost_autotune")
    p.add_argument("--storage", type=str, default=None, help="Optuna storage URL, e.g. sqlite:///study.db")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pruner", action="store_true", help="Enable MedianPruner")

    # visualization
    p.add_argument(
        "--no_vis_optimization_history",
        dest="vis_optimization_history",
        action="store_false",
        help="Disable visualization of the optimization history."
    )
    p.add_argument("--vis_parallel", action="store_true",
                   help="parallel-coordinates plot of the sampled weights vs metrics")
    args = p.parse_args()

    # Build scenario with your initial pose baked in
    scn = Scenario(track=args.track,
                   racecar_version=args.racecar_version,
                   floor=args.floor, refline=args.refline)
    scn.init_pose_frenet = np.array([args.s0, args.n0, args.psi0], dtype=float)
    scn.init_vx0 = float(args.vx0)

    objective = MPCOptunaObjective(
        scn=scn,
        dt=args.dt,
        steps=args.steps,
        integrator=args.integrator,
        mu_track=args.mu_track,
        mu_mpc=args.mu_mpc,
        w_lt=args.w_lt,
        w_cost=args.w_cost,
        w_abs_n=args.w_abs_n,
        w_abs_v=args.w_abs_v,
        seed=args.seed,
    )

    # Create study
    pruner = optuna.pruners.MedianPruner() if args.pruner else optuna.pruners.NopPruner()
    sampler = optuna.samplers.TPESampler(seed=args.seed)

    if args.storage:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            load_if_exists=True,
            direction="minimize",
            pruner=pruner,
            sampler=sampler,
        )
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            direction="minimize",
            pruner=pruner,
            sampler=sampler,
        )

    print("=== Starting optimization ===")
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    print("\n=== Best Trial ===")
    bt = study.best_trial
    print(f"Value (J): {bt.value:.6f}")
    print("Weights:")
    for k, v in bt.params.items():
        print(f"  {k}: {v:.6g}")
    print("Metrics:")
    print(f"  lap_time:        {bt.user_attrs.get('lap_time')}")
    print(f"  mean_cost:       {bt.user_attrs.get('mean_cost')}")
    print(f"  mean_abs_n:      {bt.user_attrs.get('mean_abs_n')}")
    print(f"  mean_abs_v_err:  {bt.user_attrs.get('mean_abs_v_err')}")

    if args.vis_optimization_history:
        try:
            from optuna.visualization.matplotlib import \
                plot_optimization_history
            ax = plot_optimization_history(study)
            ax.set_title("Optuna Optimization History")
            ax.set_xlabel("trial")
            ax.set_ylabel("objective J")
            ax.set_ylim(0.1, 10)
            fig = ax.get_figure()
            fig.tight_layout()
            # fig.savefig("optuna_history.png")
            import matplotlib.pyplot as plt
            plt.show()
        except Exception:
            pass

    # --- Parallel coordinates plot over all completed trials ---
    if args.vis_parallel:
        # 1) Choose parameter axes in a stable, human-readable order
        param_names = PARAM_ORDER
        param_limits = {k: (spec["low"], spec["high"] * 1.5) for k, spec in PARAM_SPECS.items()}

        # 2) Collect successful trials
        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        if len(trials) >= 2:
            # Parameters
            P = np.array([[t.params.get(name, np.nan) for name in param_names] for t in trials], dtype=float)

            # Metrics from user attrs
            lt = np.array([t.user_attrs.get("lap_time", np.nan) for t in trials], dtype=float)
            man = np.array([t.user_attrs.get("mean_abs_n", np.nan) for t in trials], dtype=float)
            mc = np.array([t.user_attrs.get("mean_cost", np.nan) for t in trials], dtype=float)

            # Objective (what Optuna minimizes) — the “parameter optimization cost”
            J = np.array([t.value for t in trials], dtype=float)  # same as objective_J if you logged it

            # Robust caps (to keep scales readable)
            lt_cap = 1.5 * np.nanmedian(lt) if np.isfinite(np.nanmedian(lt)) else np.nanmax(lt)
            mc_cap = 1.5 * np.nanmedian(mc) if np.isfinite(np.nanmedian(mc)) else np.nanmax(mc)
            man_cap = 1.5 * np.nanmedian(man) if np.isfinite(np.nanmedian(man)) else np.nanmax(man)
            J_cap = 1.5 * np.nanmedian(J) if np.isfinite(np.nanmedian(J)) else np.nanmax(J)

            lt = np.clip(lt, 0, lt_cap)
            mc = np.clip(mc, 0, mc_cap)
            man = np.clip(man, 0, man_cap)
            J = np.clip(J, 0, J_cap)

            # Stack metrics: (Lap time, |n|, Solver cost, Objective J)
            metrics_matrix = np.column_stack([lt, man, mc, J])
            metric_names = ["Lap time [s]", "Mean |n| [m]", "Solver cost", "Obj J"]
            metrics_limits = {
                "Lap time [s]": (float(np.nanmin(lt)), float(np.nanmax(lt)) * 1.5),
                "Mean |n| [m]": (float(np.nanmin(man)), float(np.nanmax(man)) * 1.5),
                "Solver cost": (float(np.nanmin(mc)), float(np.nanmax(mc)) * 1.5),
                "Obj J": (float(np.nanmin(J)), float(np.nanmax(J)) * 1.5),
            }

            # Drop rows with NaNs
            ok = np.isfinite(P).all(axis=1) & np.isfinite(metrics_matrix).all(axis=1)
            P = P[ok]
            metrics_matrix = metrics_matrix[ok]
            J = J[ok]

            fig = plot_parallel_params_vs_metrics(
                params_matrix=P,
                param_names=param_names,
                metrics_matrix=metrics_matrix,
                metric_names=metric_names,
                color_by=J,                 # color scale = objective J
                cmap="viridis",
                alpha=0.5,
                lw=1.0,
                fig_size=(14, 4),
                n_y_ticks=5,
                title="Tunable parameters vs performance (color = Obj J)",
                fixed_param_limits=param_limits,
                fixed_metrics_limits=metrics_limits,
            )
            fig.gca().set_yticks([])
            fig.gca().set_yticklabels([])
            import matplotlib.pyplot as plt
            plt.show()


if __name__ == "__main__":
    main()
