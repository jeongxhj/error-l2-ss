"""Multi-lap benchmark for one controller, on one track, with fixed metrics.

Same harness for every algorithm, so the numbers are comparable:

    python -m forzaeth_mpc.sim.cli.run_laps --algo stmpc --laps 20
    python -m forzaeth_mpc.sim.cli.run_laps --algo kmpc  --laps 20
    python -m forzaeth_mpc.sim.cli.run_laps --algo lbmpc --laps 20

Writes <out>/<algo>.json and, with --plot, <out>/<algo>_*.png.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from tqdm import tqdm

from forzaeth_mpc.sim.tracks import REFLINES
from forzaeth_mpc.sim.core.scenario import Scenario
from forzaeth_mpc.sim.metrics.metrics import per_lap_times, unwrap_s
from forzaeth_mpc.sim.waypoints import WP

S, N, VX = 0, 1, 3  # state indices, identical across the three models


def control_period(algo: str, scn: Scenario, args) -> float:
    """1 / MPC_freq of the algorithm's own config, unless --dt overrides it."""
    if args.dt:
        return float(args.dt)
    from forzaeth_mpc.config import (load_KMPC_config_ros, load_LBMPC_config_ros,
                                     load_TOMPC_config_ros)
    cfg = {"stmpc": lambda: scn.stmpc_config,
           "kmpc": lambda: load_KMPC_config_ros(scn.racecar_version),
           "lbmpc": lambda: load_LBMPC_config_ros(scn.racecar_version),
           "tompc": lambda: load_TOMPC_config_ros(scn.racecar_version)}[algo]()
    return 1.0 / float(cfg.MPC_freq)


def build(algo: str, scn: Scenario, args):
    """Returns the sim adapter for `algo`. Adapters own their own plant."""
    from forzaeth_mpc.sim.core import controllers as C
    dt = control_period(algo, scn, args)
    if algo == "stmpc":
        mpc = scn.make_mpc(s0=args.s0, n0=args.n0, psi0=args.psi0, dt=dt)
        return C.ControllerMPC(mpc, scn.track_len, dt)
    if algo == "kmpc":
        return C.ControllerKMPC(scn.make_kmpc(dt=dt), scn.track_len, dt)
    if algo == "tompc":
        mpc = scn.make_tompc(s0=args.s0, n0=args.n0, psi0=args.psi0, dt=dt)
        return C.ControllerTOMPC(mpc, scn.track_len, dt)
    if algo == "lbmpc":
        mpc = scn.make_lbmpc(s0=args.s0, n0=args.n0, psi0=args.psi0, dt=dt,
                             lbmpc_path=args.lbmpc_path, gp_mode=args.gp_mode,
                             stochastic_mode=args.stochastic_mode)
        return C.ControllerLBMPC(mpc, scn.track_len, dt, learn=not args.no_learning)
    raise ValueError(algo)


def frenet_to_xy(waypoints: np.ndarray, s: np.ndarray, n: np.ndarray, track_len: float):
    """Project (s, n) onto the reference line, offset along its normal.

    Two traps live at s = 0, and both used to bend the track around the origin:

    - The waypoint list closes, so s runs 0 .. track_len with the same point at both ends.
      `np.interp(period=)` needs one period of data and no repeat, so the duplicate has
      to go.
    - An unwrapped heading is not periodic. It gains 2*pi per lap, so interpolating it
      periodically sweeps it backwards over the last segment, and the normal turns a full
      circle inside the last few centimetres. Interpolate the tangent vector instead.
    """
    ws, wx, wy = waypoints[:, WP.S], waypoints[:, WP.X], waypoints[:, WP.Y]
    if np.isclose(ws[-1], track_len):
        ws, wx, wy = ws[:-1], wx[:-1], wy[:-1]
    # central differences that wrap, so the tangent at the start line is not one-sided
    tx = np.roll(wx, -1) - np.roll(wx, 1)
    ty = np.roll(wy, -1) - np.roll(wy, 1)
    norm = np.hypot(tx, ty)
    smod = np.mod(s, track_len)
    interp = lambda f: np.interp(smod, ws, f, period=track_len)
    # the normal of a unit tangent (tx, ty) is (-ty, tx), which is +n to the left
    return (interp(wx) - n * interp(ty / norm),
            interp(wy) + n * interp(tx / norm))


def track_bounds_xy(waypoints: np.ndarray, track_len: float):
    """The left and right track edges as closed (x, y) polylines.

    `frenet_to_xy` puts +n to the left of the raceline, which is the same sign the
    constraints use, so the right edge takes -d_right.
    """
    s_wp = waypoints[:, WP.S]
    lx, ly = frenet_to_xy(waypoints, s_wp, waypoints[:, WP.D_LEFT], track_len)
    rx, ry = frenet_to_xy(waypoints, s_wp, -waypoints[:, WP.D_RIGHT], track_len)
    close = lambda a: np.append(a, a[0])
    return (close(lx), close(ly)), (close(rx), close(ry))


def run(algo: str, scn: Scenario, args) -> dict:
    ctrl = build(algo, scn, args)
    # The control period is the controller's own, not a harness choice: KMPC's plant
    # is a discrete model that advances exactly 1/MPC_freq per call, and the other two
    # solve a horizon discretised at 1/MPC_freq. Stepping at anything else silently
    # runs the controller at a rate it was not tuned for.
    dt = ctrl.dt
    x = scn.make_x0_full(ctrl.mpc, args.s0, args.n0, args.psi0, args.vx0)

    # plant friction vs MPC friction: the mismatch LBMPC's GP is supposed to absorb
    if hasattr(ctrl.mpc, "set_sim_mu"):
        ctrl.mpc.set_sim_mu(float(args.mu_track))

    qdict = weights_from_config(ctrl.mpc)

    video = None
    if args.video:
        from forzaeth_mpc.sim.plotting.video import SimVideo
        os.makedirs(args.out, exist_ok=True)
        video = SimVideo(scn.waypoints, scn.track_len,
                         f"{algo.upper()} — {args.track} / {args.racecar_version}",
                         os.path.join(args.out, f"{algo}_run"), fps=args.video_fps)

    xs, us, costs, stats, solve_ms = [], [], [], [], []
    laps_done, target_s = 0, None
    for k in tqdm(range(args.max_steps)):
        ctrl.pin_state(x)
        ctrl.set_online_params(scn.waypoints, qdict, float(args.mu_mpc),
                               track_len_fallback=scn.track_len)
        t0 = time.perf_counter()
        u0, sd, cost, status = ctrl.solve()
        solve_ms.append((time.perf_counter() - t0) * 1e3)

        x = ctrl.step_plant(x, u0)
        if not np.all(np.isfinite(x[:ctrl.nx])):
            print(f"[{algo}] diverged at step {k}: non-finite state")
            break

        xs.append(x.copy())
        us.append(np.asarray(u0, dtype=float).copy())
        costs.append(float(cost))
        stats.append(int(status))

        X = np.asarray(xs)
        if target_s is None:
            target_s = X[0, S]
        laps_done = int(np.floor((unwrap_s(X[:, S], scn.track_len)[-1] - target_s)
                                 / scn.track_len))
        if video is not None and k % args.video_every == 0:
            video.add(X, sd, dt, laps=laps_done)

        if laps_done >= args.laps:
            break

    if video is not None:
        video.close()

    X = np.asarray(xs)
    return summarize(algo, args, scn, ctrl, dt, X, np.asarray(us),
                     np.asarray(costs), np.asarray(stats), np.asarray(solve_ms))


def weights_from_config(mpc) -> dict:
    """Cost weights straight off whichever config the controller carries.

    Each controller's p-vector builder falls back to its own config for missing
    keys, so only the ones the algorithm actually has are passed.
    """
    cfg = (getattr(mpc, "stmpc_config", None) or getattr(mpc, "kmpc_config", None)
           or getattr(mpc, "lbmpc_config", None) or getattr(mpc, "tompc_config", None))
    q = {}
    for key, attr in (("q_adv", "qadv"), ("q_v", "qv"), ("q_n", "qn"),
                      ("q_alpha", "qalpha"), ("q_jerk", "qjerk"), ("q_ac", "qac"),
                      ("q_ddelta", "qddelta"),
                      ("q_time", "qtime"), ("q_vy", "qvy"),
                      ("q_v_terminal", "qv_terminal"),
                      ("terminal_multiplier", "terminal_multiplier")):
        if hasattr(cfg, attr):
            q[key] = float(getattr(cfg, attr))
    return q


def summarize(algo, args, scn, ctrl, dt, X, U, C, St, T) -> dict:
    if X.size == 0:
        return {"algo": algo, "steps": 0, "error": "no steps completed"}
    laps = per_lap_times(X, dt, scn.track_len)
    n_abs = np.abs(X[:, N])
    gp_pts = gp_datapoints(ctrl)
    return {
        "algo": algo,
        "track": args.track, "car": args.racecar_version, "floor": args.floor,
        "dt": dt, "mpc_freq_hz": round(1.0 / dt, 3),
        # the kinematic model has no tire model, so friction does not enter it
        "mu_track": None if algo == "kmpc" else args.mu_track,
        "mu_mpc": None if algo == "kmpc" else args.mu_mpc,
        # the traversal time of the last horizon: TOMPC's actual objective
        "horizon_time_s": getattr(ctrl.mpc, "horizon_time", None),
        "steps": int(X.shape[0]),
        "sim_time_s": float(X.shape[0] * dt),
        "laps_completed": int(len(laps)),
        "lap_times_s": [round(float(v), 4) for v in laps],
        "lap_best_s": float(np.min(laps)) if laps.size else None,
        "lap_mean_s": float(np.mean(laps)) if laps.size else None,
        "lap_std_s": float(np.std(laps)) if laps.size else None,
        "n_mean_abs_m": float(np.mean(n_abs)),
        "n_rms_m": float(np.sqrt(np.mean(X[:, N] ** 2))),
        "n_max_abs_m": float(np.max(n_abs)),
        "vx_mean_mps": float(np.mean(X[:, VX])),
        "vx_max_mps": float(np.max(X[:, VX])),
        "cost_mean": float(np.nanmean(C)),
        "solver_ok_frac": float(np.mean(np.isin(St, (0, 2)))),
        "status_counts": {str(int(s)): int(c) for s, c in
                          zip(*np.unique(St, return_counts=True))},
        "solve_ms_mean": float(np.mean(T)),
        "solve_ms_p99": float(np.percentile(T, 99)),
        "gp_datapoints": gp_pts,
        "learning": None if algo != "lbmpc" else (not args.no_learning),
        "gp_mode": None if algo != "lbmpc" else args.gp_mode,
        "stochastic_mode": None if algo != "lbmpc" else (
            args.stochastic_mode or ctrl.mpc.lbmpc_config.gp_params.stochastic_mode),
        "_X": X, "_C": C, "_U": U,  # stripped before json, kept for plotting/--dump_states
    }


def gp_datapoints(ctrl):
    """Size of the LBMPC residual dataset, or None for the nominal controllers."""
    fn = getattr(ctrl, "gp_num_datapoints", None)
    return int(fn()) if fn is not None else None


def plot(res: dict, out_dir: str, waypoints: np.ndarray, track_len: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X, algo = res["_X"], res["algo"]
    s_unw = unwrap_s(X[:, S], track_len)
    t = np.arange(X.shape[0]) * res["dt"]

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"{algo.upper()} — {res['track']} / {res['car']} — "
                 f"{res['laps_completed']} laps")

    xy_x, xy_y = frenet_to_xy(waypoints, X[:, S], X[:, N], track_len)
    sc = ax[0, 0].scatter(xy_x, xy_y, c=X[:, VX], s=3, cmap="viridis")
    (lx, ly), (rx, ry) = track_bounds_xy(waypoints, track_len)
    ax[0, 0].plot(lx, ly, "-", color="0.35", lw=1.1, label="track bounds")
    ax[0, 0].plot(rx, ry, "-", color="0.35", lw=1.1)
    ax[0, 0].plot(waypoints[:, 0], waypoints[:, 1], "k--", lw=0.6, label="raceline")
    ax[0, 0].set_aspect("equal"), ax[0, 0].legend(loc="best", fontsize=8)
    ax[0, 0].set_title("driven path, coloured by $v_x$")
    fig.colorbar(sc, ax=ax[0, 0], label="$v_x$ [m/s]")

    if res["lap_times_s"]:
        ax[0, 1].bar(np.arange(1, len(res["lap_times_s"]) + 1), res["lap_times_s"])
        ax[0, 1].axhline(res["lap_mean_s"], color="r", ls="--",
                         label=f"mean {res['lap_mean_s']:.3f} s")
        ax[0, 1].legend(fontsize=8)
    ax[0, 1].set_title("lap times"), ax[0, 1].set_xlabel("lap")
    ax[0, 1].set_ylabel("s")

    ax[1, 0].plot(s_unw, X[:, N], lw=0.7)
    ax[1, 0].axhline(0, color="k", lw=0.5)
    ax[1, 0].set_title(f"lateral deviation (max |n| = {res['n_max_abs_m']:.3f} m)")
    ax[1, 0].set_xlabel("s travelled [m]"), ax[1, 0].set_ylabel("n [m]")

    ax[1, 1].plot(t, X[:, VX], lw=0.7)
    ax[1, 1].set_title(f"speed (mean {res['vx_mean_mps']:.2f} m/s)")
    ax[1, 1].set_xlabel("t [s]"), ax[1, 1].set_ylabel("$v_x$ [m/s]")

    fig.tight_layout()
    path = os.path.join(out_dir, f"{algo}_summary.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", nargs="+", default=["stmpc"],
                    choices=["stmpc", "kmpc", "lbmpc", "tompc"])
    ap.add_argument("--track", default="THETRACK")
    ap.add_argument("--racecar_version", default="NUC5")
    ap.add_argument("--floor", default="dubi")
    ap.add_argument("--refline", default="iqp", choices=sorted(REFLINES))
    ap.add_argument("--laps", type=int, default=20)
    ap.add_argument("--dt", type=float, default=None,
                    help="control period; default is 1/MPC_freq from the algorithm's own config")
    ap.add_argument("--max_steps", type=int, default=20000)
    ap.add_argument("--mu_track", type=float, default=1.0, help="plant friction")
    ap.add_argument("--mu_mpc", type=float, default=1.0, help="friction the MPC assumes")
    ap.add_argument("--s0", type=float, default=20.0)
    ap.add_argument("--n0", type=float, default=0.0)
    ap.add_argument("--psi0", type=float, default=0.0)
    ap.add_argument("--vx0", type=float, default=3.0)
    ap.add_argument("--no_learning", action="store_true",
                    help="lbmpc only: nominal residual MPC, no online GP updates")
    ap.add_argument("--gp_mode", default="learn", choices=["learn", "use_learnt_model"],
                    help="lbmpc only: overrides gp_params.data_processing_mode")
    ap.add_argument("--stochastic_mode", default=None,
                    choices=["disabled", "open_loop", "lqr", "riccati"],
                    help="lbmpc only: overrides gp_params.stochastic_mode (ZoRO)")
    ap.add_argument("--lbmpc_path", default="/tmp/lbmpc_sim")
    ap.add_argument("--out", default="/tmp/mpc_laps")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--video", action="store_true",
                    help="record the run to <out>/<algo>_run.mp4 (gif without ffmpeg)")
    ap.add_argument("--video_every", type=int, default=1,
                    help="one frame every N control steps (default 1)")
    ap.add_argument("--video_fps", type=int, default=20,
                    help="frames per second (default 20: real time at 20 Hz control)")
    ap.add_argument("--dump_states", action="store_true",
                    help="save the closed-loop state and input trajectories as .npy, for "
                         "before/after comparison across a refactor")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.lbmpc_path, exist_ok=True)

    scn = Scenario(track=args.track, racecar_version=args.racecar_version,
                   floor=args.floor, refline=args.refline)
    for algo in args.algo:
        res = run(algo, scn, args)
        if args.plot and res.get("steps"):
            res["plot"] = plot(res, args.out, scn.waypoints, scn.track_len)
        if args.dump_states:
            np.save(os.path.join(args.out, f"{algo}_states.npy"), res["_X"])
            np.save(os.path.join(args.out, f"{algo}_inputs.npy"), res["_U"])
        res.pop("_X", None), res.pop("_C", None), res.pop("_U", None)
        with open(os.path.join(args.out, f"{algo}.json"), "w") as f:
            json.dump(res, f, indent=2)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
