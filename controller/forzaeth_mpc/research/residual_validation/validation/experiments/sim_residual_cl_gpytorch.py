#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from forzaeth_mpc.sim.plotting.vis_run_simple_rollout import \
    plot_closed_loop_summary
from l4acados.models.pytorch_models.gpytorch_models.gpytorch_gp import (
    BatchIndependentInducingPointGPModel, BatchIndependentMultitaskGPModel)
from l4acados.models.pytorch_models.gpytorch_models.gpytorch_residual_model import \
    GPyTorchResidualModel
from l4acados.models.pytorch_models.pytorch_feature_selector import \
    PyTorchFeatureSelector
from research.residual_validation.mpc_core.MPC.l4a_STMPC import STMPC_L4A
from research.residual_validation.mpc_core.MPC.STMPC import STMPC
from research.residual_validation.mpc_core.residual.gp_interface import train_gp_model
from research.residual_validation.validation.viz.plotting import plot_dict_kept_dropped
from forzaeth_mpc.config import (CarConfig, PacejkaTireConfig, STMPCConfig,
                        TrailingConfig, load_car_config_ros,
                        load_pacejka_tire_config_ros, load_STMPC_config_ros,
                        load_trailing_config_ros)
from forzaeth_mpc.sim.tracks import load_track
from research.residual_validation.mpc_core.utils.indices import InputIndex, StateIndex

PLOT_DIR = Path(os.environ.get("RESIDUAL_PLOT_DIR", "residual_plots"))


def savefig(name: str) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_DIR / f"{name}.png", dpi=140, bbox_inches="tight")
    print(f"[plot] {PLOT_DIR / (name + '.png')}")


def dyn_from_ctrl(ctrl: STMPC, mu_nom: float, mu_real: float, bias=(0.0, 0.0, 0.05)):
    import casadi as ca
    x = ctrl.model.x
    u = ctrl.model.u
    p = ctrl.model.p
    f = ctrl.model.f_expl_expr
    f_fun = ca.Function("f_fun", [x, u, p], [f])
    pvec = np.array(ctrl.model_params.p, dtype=float).reshape(-1)

    idx_vx = StateIndex.VELOCITY_V_X.value
    idx_vy = StateIndex.VELOCITY_V_Y.value
    idx_wz = StateIndex.YAW_RATE.value
    b = np.array(bias, dtype=float)

    def dyn_nom(t, xv, uv):
        pvec[12] = float(mu_nom)
        return np.array(f_fun(xv, uv, pvec)).reshape(-1)

    def dyn_act(t, xv, uv):
        pvec[12] = float(mu_real)
        dx = np.array(f_fun(xv, uv, pvec)).reshape(-1)
        dx[idx_vx] += b[0]
        dx[idx_vy] += b[1]
        dx[idx_wz] += b[2]
        return dx

    return dyn_nom, dyn_act


def integrate_one_step(dyn,
                       x_plant: np.ndarray,
                       dt: float,
                       method: str = "rk4",
                       noise_vars: Optional[Sequence[float]] = None,
                       noise_indices: Optional[Sequence[int]] = None,
                       rng: Optional[np.random.Generator] = None,
                       ) -> np.ndarray:
    """
    Integrate one step of length dt for an augmented plant state:
      x_plant = [ x(0:nx), u0, u1 ], with nx = len(x_plant) - 2.
    `dyn(t, x, u)` returns xdot for the nx model states. Inputs u are held constant.
    """
    x_plant = np.asarray(x_plant, dtype=float)
    if x_plant.ndim != 1 or x_plant.size < 3:
        raise ValueError("x_plant must be 1D and contain at least [x..., u0, u1].")

    nx = x_plant.size - 2
    x = x_plant[:nx]
    u = x_plant[nx:]

    def f(xx):
        # xx is a concatenated vector; we only integrate the state part
        x_local = xx[:nx]
        return np.asarray(dyn(0.0, x_local, u), dtype=float).reshape(-1)

    if method.lower() == "rk4":
        k1 = f(np.r_[x, u])
        k2 = f(np.r_[x + 0.5 * dt * k1, u])
        k3 = f(np.r_[x + 0.5 * dt * k2, u])
        k4 = f(np.r_[x + dt * k3, u])
        x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    else:
        # forward Euler
        x_next = x + dt * f(np.r_[x, u])

    # Add gaussian noise
    if noise_vars is not None and noise_indices is not None:
        if rng is None:
            rng = np.random.default_rng()
        for idx, var in zip(noise_indices, noise_vars):
            x_next[idx] += rng.normal(0.0, np.sqrt(var))

    x_plant_next = x_plant.copy()
    x_plant_next[:nx] = x_next
    return x_plant_next


def lap_times_from_s(s_series: np.ndarray, dt: float, track_length: float) -> List[float]:
    """
    Compute per-lap times by unwrapping s and detecting crossings of s0 + k*track_length.
    Assumes uniform sampling with step dt.
    Returns a list of lap durations [t_lap1, t_lap2, ...] for all fully completed laps.
    """
    s_series = np.asarray(s_series, dtype=float)
    if s_series.ndim != 1 or len(s_series) < 2:
        return []

    # Unwrap s after the cut
    unwrapped = s_series.copy()
    offset = 0.0
    for i in range(1, len(unwrapped)):
        ds = s_series[i] - s_series[i - 1]
        if ds < -0.5 * track_length:
            offset += track_length
        unwrapped[i] += offset

    s0 = unwrapped[0]
    t = np.arange(len(unwrapped), dtype=float) * dt

    total_prog = unwrapped[-1] - s0
    n_laps = int(np.floor(total_prog / track_length))
    if n_laps <= 0:
        return []

    # Find lap-crossing times
    cross_times = []
    for k in range(1, n_laps + 1):
        thresh = s0 + k * track_length
        idx = np.searchsorted(unwrapped, thresh)
        if 0 < idx < len(unwrapped):
            s_prev, s_curr = unwrapped[idx - 1], unwrapped[idx]
            t_prev, t_curr = t[idx - 1], t[idx]
            if s_curr != s_prev:
                frac = (thresh - s_prev) / (s_curr - s_prev)
                cross_times.append(t_prev + frac * (t_curr - t_prev))

    if not cross_times:
        return []

    # Lap durations relative to the first start-line crossing
    lap_times = [cross_times[0]]
    for i in range(1, len(cross_times)):
        lap_times.append(cross_times[i] - cross_times[i - 1])

    return lap_times


def compute_lateral_deviation(x_hist: np.ndarray) -> Tuple[float, float]:
    """
    Compute mean and standard deviation of the lateral deviation [m]
    from the Frenet-state trajectories.

    Assumes the lateral deviation 'n' is the 2nd state (index 1).
    """
    if x_hist.ndim != 2 or x_hist.shape[1] < 2:
        raise ValueError("x_hist must be (N, nx) and contain the lateral deviation at index 1.")

    lateral_dev = x_hist[:, 1]
    mean_dev = float(np.mean(np.abs(lateral_dev)))
    std_dev = float(np.std(lateral_dev))
    return mean_dev, std_dev


def summarize_times_ms(times_ms):
    arr = np.asarray(times_ms, dtype=float)
    if arr.size == 0:
        return {"mean": np.nan, "median": np.nan, "std": np.nan, "p95": np.nan, "max": np.nan, "N": 0}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=0)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "N": int(arr.size),
    }


def update_dictionary(
    Z_list,
    Y_list,
    z_feat,                 # (nz,)  already built (e.g., [x, u])
    y_error,                # (ny,)  already built (e.g., B_inv @ (x_plant - x_pred_1))
    Z_vis_kept,
    Z_vis_dropped,
    idx_feat,               # indices INTO z_feat used for spacing (e.g., idx_gp_inputs)
    m_max,
    min_dist,
    ls=None,                # optional lengthscales for spacing; shape (len(idx_feat),) or None
):
    """
    Decide whether to add a new sample (z_feat, y_error) to the dictionary.
    - Uses nearest-neighbor distance in the subspace z_feat[idx_feat] (optionally scaled by ls).
    - If too close (< min_dist): reject.
    - Else, if capacity available: append.
    - Else, replace the most redundant point (smallest NN distance in candidate set).
    Appends full rows to Z_list and Y_list unchanged.
    """

    # -- sanitize inputs
    z_feat = np.asarray(z_feat, dtype=float).reshape(-1)
    y_error = np.asarray(y_error, dtype=float).reshape(-1)
    idx_feat = np.asarray(idx_feat, dtype=int).reshape(-1)
    if idx_feat.size == 0:
        raise ValueError("idx_feat must contain at least one index.")
    if ls is not None:
        ls = np.asarray(ls, dtype=float).reshape(-1)
        if ls.size != idx_feat.size:
            raise ValueError(f"ls must have shape ({idx_feat.size},), got {ls.shape}.")

    # subspace helpers
    def _subspace(Zfull):      # (M, nz) -> (M, k)
        Zk = Zfull[:, idx_feat]
        return Zk / ls if ls is not None else Zk

    def _subspace_vec(zfull):  # (nz,) -> (k,)
        zk = zfull[idx_feat]
        return zk / ls if ls is not None else zk

    # -- empty dictionary: add directly
    if len(Z_list) == 0:
        Z_list.append(z_feat.copy())
        Y_list.append(y_error.copy())
        Z_vis_kept.append(z_feat.copy())
        return Z_list, Y_list, Z_vis_kept, Z_vis_dropped

    # -- stack existing
    Z = np.vstack(Z_list)             # (M, nz)
    Zk = _subspace(Z)                  # (M, k)
    zk = _subspace_vec(z_feat)         # (k,)

    # distance to nearest existing point in subspace
    d2 = np.sum((Zk - zk[None, :])**2, axis=1)  # (M,)
    dmin = float(np.sqrt(np.min(d2)))

    # -- too close -> reject
    if dmin < min_dist:
        Z_vis_dropped.append(z_feat.copy())
        return Z_list, Y_list, Z_vis_kept, Z_vis_dropped

    # -- capacity available -> append
    if len(Z_list) < m_max:
        Z_list.append(z_feat.copy())
        Y_list.append(y_error.copy())
        Z_vis_kept.append(z_feat.copy())
        return Z_list, Y_list, Z_vis_kept, Z_vis_dropped

    # -- full: candidate replace (drop most redundant point)
    Zcand = np.vstack([Z, z_feat[None, :]])     # (M+1, nz)
    Zk_c = _subspace(Zcand)                    # (M+1, k)

    # pairwise NN distances (no self)
    G = Zk_c @ Zk_c.T
    nrm = np.sum(Zk_c**2, axis=1, keepdims=True)
    D2 = nrm + nrm.T - 2.0 * G
    np.fill_diagonal(D2, np.inf)
    nn = np.sqrt(np.min(D2, axis=1))            # (M+1,)

    drop_idx = int(np.argmin(nn))               # smallest NN distance -> most redundant

    if drop_idx == len(Z_list):
        # new point is least informative -> drop it
        Z_vis_dropped.append(z_feat.copy())
        return Z_list, Y_list, Z_vis_kept, Z_vis_dropped
    else:
        # replace that existing point with the new one
        Z_vis_dropped.append(Z_list[drop_idx].copy())
        Z_list[drop_idx] = z_feat.copy()
        Y_list[drop_idx] = y_error.copy()
        Z_vis_kept.append(z_feat.copy())
        return Z_list, Y_list, Z_vis_kept, Z_vis_dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gp_iters", type=int, default=300)
    ap.add_argument("--gp_lr", type=float, default=0.05)
    ap.add_argument("--bias_vx", type=float, default=1.0)
    ap.add_argument("--bias_vy", type=float, default=0.0)
    ap.add_argument("--bias_wz", type=float, default=0.0)
    ap.add_argument("--track", default="THETRACK")
    ap.add_argument("--racecar_version", default="NUC5")
    ap.add_argument("--floor", default="dubi")
    ap.add_argument("--cl_iter", type=int, default=650)
    ap.add_argument("--lengthscale", default=2.9)
    ap.add_argument("--outputscale", default=0.000155)
    ap.add_argument("--gp_noise", default=1e-4)
    ap.add_argument("--noise_vx", default=1e-5)
    ap.add_argument("--noise_vy", default=1e-5)
    ap.add_argument("--noise_wz", default=1e-5)
    ap.add_argument("--fitc_m", type=int, default=10, help="Number of inducing points along the horizon")
    ap.add_argument("--fitc_place", choices=["equal", "kmeans"], default="equal",
                    help="How to place inducing points on the predicted trajectory")

    args = ap.parse_args()

    # === Waypoints & configs ===
    waypoints, _track_length = load_track(args.track)
    refline_xy = waypoints[:, 0:2]
    left_dist = waypoints[:, 5]
    right_dist = waypoints[:, 6]

    stmpc_cfg: STMPCConfig = load_STMPC_config_ros(args.racecar_version)
    car_cfg: CarConfig = load_car_config_ros(args.racecar_version)
    tire_cfg: PacejkaTireConfig = load_pacejka_tire_config_ros(args.racecar_version, args.floor)
    trailing_cfg: TrailingConfig = load_trailing_config_ros(args.racecar_version)

    mu_nom = getattr(stmpc_cfg, "mu_nom", 1.0)
    mu_real = 0.8
    dt = 1.0 / float(stmpc_cfg.MPC_freq)
    pose_frenet = np.array([15.0, 0.0, 0.0])
    init_vel = 4.0
    cl_iter = args.cl_iter

    # === dictionary settings ===
    m_max = 50
    min_dist = 0.15

    # === 1) Nominal controller (STMPC) ===
    ctrl_nom = STMPC(
        pose_frenet=pose_frenet,
        waypoints_data=waypoints,
        racecar_version=args.racecar_version,
        stmpc_config=stmpc_cfg,
        car_config=car_cfg,
        tire_config=tire_cfg,
        trailing_config=trailing_cfg,
        controller_frequency=float(stmpc_cfg.MPC_freq),
        using_gokart=False,
    )

    # Create the two integration functions
    dyn_nom, dyn_act = dyn_from_ctrl(
        ctrl_nom, mu_nom=float(mu_nom), mu_real=float(mu_real), bias=(
            args.bias_vx, args.bias_vy, args.bias_wz))

    # Iterate with a simple plant to have a better mpc solution than the first one
    x_hist = []
    u_hist = []
    status_hist = []
    x_pred_seq_hist = []
    x_plant = np.array([
        pose_frenet[0], pose_frenet[1], pose_frenet[2],
        init_vel, 0.0, 0.0, 0.0, 0.0,   # vx, vy, delta, yaw_rate, accel
        0.0, 0.0                   # jerk, derDelta (helpers)
    ], dtype=float)

    idx_vx = StateIndex.VELOCITY_V_X.value
    idx_vy = StateIndex.VELOCITY_V_Y.value
    idx_wz = StateIndex.YAW_RATE.value
    idx_noise = [idx_vx, idx_vy, idx_wz]
    # StateIndex.INPUT_JERK=8 / INPUT_DSTEERING=9 were added alongside this script
    # (race_stack 232a6d313) and dropped again by a merge revert (race_stack 94ea48a79).
    # They are just the input slots of the [x, u] feature vector.
    idx_input_jerk = ctrl_nom.model.n_x + InputIndex.JERK.value
    idx_input_dsteer = ctrl_nom.model.n_x + InputIndex.D_STEERING_ANGLE.value
    idx_gp_inputs = [idx_vx, idx_vy, idx_wz, idx_input_jerk, idx_input_dsteer]
    idx_gp_outputs = idx_noise
    dim = idx_input_dsteer + 1

    # Create the selection matrix B
    B = np.zeros((len(idx_gp_outputs), dim), dtype=float)
    for r, c in enumerate(idx_gp_outputs):
        B[r, c] = 1.0
    B_inv = B.copy()

    sigma_vx = getattr(args, "noise_vx", 0.01)
    sigma_vy = getattr(args, "noise_vy", 0.01)
    sigma_wz = getattr(args, "noise_wz", 0.01)
    noise_vars = [sigma_vx, sigma_vy, sigma_wz]

    # Lists of FULL (nx-wide) samples; we'll slice to 3 cols after stacking.
    Z_list, Y_list = [], []
    Z_vis_kept = []
    Z_vis_dropped = []
    times_nom_ms = []

    for k in range(cl_iter):
        start_time = time.time()
        # 1) Pin solver's initial state to current plant (first 8 states only)
        x0_solver = x_plant[:ctrl_nom.model.n_x].copy()
        ctrl_nom.acados_solver.set(0, "lbx", x0_solver)
        ctrl_nom.acados_solver.set(0, "ubx", x0_solver)

        for i in range(ctrl_nom.stmpc_config.N + 1):
            idx = np.abs(waypoints[:, 4] - ctrl_nom.mpc_sd[i, 0] % ctrl_nom.spline.track_length).argmin()
            target_v_speed = waypoints[idx, 2]
            online_parameters = np.array([target_v_speed,
                                          stmpc_cfg.qadv,
                                          stmpc_cfg.qv,
                                          stmpc_cfg.qn,
                                          stmpc_cfg.qalpha,
                                          stmpc_cfg.qjerk,
                                          stmpc_cfg.qddelta,
                                          stmpc_cfg.alat_max,
                                          stmpc_cfg.a_min,
                                          stmpc_cfg.a_max,
                                          stmpc_cfg.track_safety_margin,
                                          0,
                                          mu_nom,
                                          stmpc_cfg.terminal_multiplier],
                                         dtype=np.float64)
            ctrl_nom.acados_solver.set(i, "p", online_parameters)

        # 2) Solve MPC
        status = ctrl_nom.acados_solver.solve()
        status_hist.append(status)

        if not (status == 0 or status == 2):  # 0 is sucess 2 is max iter, necesary for SQP
            # if the solver fails, apply warm start again
            # the initial warm start is a bit rough, but it does the job
            ctrl_nom.apply_warm_start(pose_frenet=[x_plant[0], x_plant[1], x_plant[2]])
            print(f"[MPC Controller] Solver failed with status {status}. Applying warm start again.")

        # 3) Read optimal control (stage 0): u = [jerk, ddelta]
        u0 = ctrl_nom.acados_solver.get(0, "u").astype(float)
        u_hist.append(u0.copy())

        # Collect full horizon state predictions
        H = ctrl_nom.stmpc_config.N
        nx = ctrl_nom.model.n_x
        nu = ctrl_nom.model.n_u
        x_pred = np.zeros((H + 1, nx))
        for i in range(H + 1):
            x_pred[i, :] = ctrl_nom.acados_solver.get(i, "x")
        x_pred_seq_hist.append(x_pred)

        ctrl_nom.mpc_sd = np.array([
            ctrl_nom.acados_solver.get(j, "x")[:2] for j in range(ctrl_nom.stmpc_config.N + 1)
        ])

        elapsed_time = (time.time() - start_time) * 1e3
        times_nom_ms.append(elapsed_time)

        # >>> GP: collect features and the first predicted state (k+1|k)
        x_pred_1 = np.hstack([x_pred[1, :], np.zeros(nu)])
        z_feat = np.hstack([x0_solver[:nx], u0])

        # 4) Apply control to the "plant":
        # Put current control into the helper states (indices 8: jerk, 9: derDelta)
        x_plant[8] = u0[0]  # jerk = d(accel)/dt from MPC
        x_plant[9] = u0[1]  # derDelta = d(delta)/dt from MPC

        # One-step forward Euler integration of your nonlinear dynamics
        x_plant = integrate_one_step(dyn_act, x_plant, dt, noise_vars=noise_vars, noise_indices=idx_noise)

        y_error = B_inv @ (x_plant - x_pred_1)
        Z_list, Y_list, Z_vis_kept, Z_vis_dropped = update_dictionary(
            Z_list, Y_list, z_feat, y_error, Z_vis_kept, Z_vis_dropped, idx_gp_inputs, m_max, min_dist)

        # 5) Log plant state
        x_hist.append(x_plant.copy())

        # (optional) simple timing print
        # if (k % 20) == 0:
        print(f"[{k:04d}] status={status}, u0={u0}, elapsed={elapsed_time:.1f} ms")

    # Visualize closed loop states and inputs
    x_pred_seq_hist = np.asarray(x_pred_seq_hist)
    x_hist = np.array(x_hist)
    u_hist = np.array(u_hist)
    fig = plot_closed_loop_summary(
        refline_xy=refline_xy,
        left_dist=left_dist,
        right_dist=right_dist,
        x_hist=x_hist,
        u_hist=u_hist
    )
    savefig("gpytorch_cl_nominal")
    plt.show()

    plot_dict_kept_dropped(
        Z_vis_kept,
        Z_vis_dropped,
        dims=(
            idx_vx,
            idx_vy,
            idx_wz),
        title="Dictionary: kept vs dropped")
    savefig("gpytorch_dictionary_kept_dropped")
    plt.show()

    x_hist_nom = x_hist

    z_train = np.vstack(Z_list).astype(np.float32)
    y_train = np.vstack(Y_list).astype(np.float32)
    z_train_tensor = torch.from_numpy(z_train)
    y_train_tensor = torch.from_numpy(y_train)
    nout = y_train.shape[1]
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=nout)
    gp_model = BatchIndependentMultitaskGPModel(z_train_tensor, y_train_tensor, likelihood)

    training_iterations = 200
    rng_seed = 456
    gp_model, likelihood = train_gp_model(
        gp_model, torch_seed=rng_seed, training_iterations=training_iterations
    )
    gp_model.eval()
    likelihood.eval()
    residual_model = GPyTorchResidualModel(gp_model)

    ctrl_res = STMPC_L4A(
        pose_frenet=pose_frenet,
        waypoints_data=waypoints,
        racecar_version=args.racecar_version,
        stmpc_config=stmpc_cfg,
        car_config=car_cfg,
        tire_config=tire_cfg,
        trailing_config=trailing_cfg,
        controller_frequency=float(stmpc_cfg.MPC_freq),
        using_gokart=False,
        residual_model=residual_model,
    )

    x_hist = []
    u_hist = []
    status_hist = []
    x_pred_seq_hist = []
    times_res_ms = []

    x_plant = np.array([
        pose_frenet[0], pose_frenet[1], pose_frenet[2],
        init_vel, 0.0, 0.0, 0.0, 0.0,   # vx, vy, delta, yaw_rate, accel
        0.0, 0.0                   # jerk, derDelta (helpers)
    ], dtype=float)

    for k in range(cl_iter):
        start_time = time.time()
        x0_solver = x_plant[:ctrl_res.model.n_x].copy()
        ctrl_res.l4a_solver.set(0, "lbx", x0_solver)
        ctrl_res.l4a_solver.set(0, "ubx", x0_solver)

        for i in range(ctrl_res.stmpc_config.N + 1):

            idx = np.abs(waypoints[:, 4] - ctrl_res.mpc_sd[i, 0] % ctrl_res.spline.track_length).argmin()
            target_v_speed = waypoints[idx, 2]
            online_parameters = np.array([target_v_speed,
                                          stmpc_cfg.qadv,
                                          stmpc_cfg.qv,
                                          stmpc_cfg.qn,
                                          stmpc_cfg.qalpha,
                                          stmpc_cfg.qjerk,
                                          stmpc_cfg.qddelta,
                                          stmpc_cfg.alat_max,
                                          stmpc_cfg.a_min,
                                          stmpc_cfg.a_max,
                                          stmpc_cfg.track_safety_margin,
                                          0,
                                          mu_nom,
                                          stmpc_cfg.terminal_multiplier],
                                         dtype=np.float64)
            ctrl_res.l4a_solver.set(i, "p", online_parameters)

        # 2) Solve MPC
        try:
            status = ctrl_res.l4a_solver.solve()
        except Exception as e:
            msg = str(e)
            m = re.search(r'status\s+(\d+)', msg)
            status = int(m.group(1)) if m else -1
            ctrl_res.apply_warm_start(pose_frenet=[x_plant[0], x_plant[1], x_plant[2]])
            print(f"[MPC Controller] Solver failed with status {status}. Applying warm start again.")

        status_hist.append(status)

        # 3) Read optimal control (stage 0): u = [jerk, ddelta]
        u0 = ctrl_res.l4a_solver.get(0, "u").astype(float)
        u_hist.append(u0.copy())

        # Collect full horizon state predictions
        H = ctrl_res.stmpc_config.N
        nx = ctrl_res.model.n_x
        x_pred_full = np.zeros((H + 1, nx))
        for i in range(H + 1):
            x_pred_full[i, :] = ctrl_res.l4a_solver.get(i, "x")
        x_pred_seq_hist.append(x_pred_full)

        ctrl_res.mpc_sd = np.array([
            ctrl_res.l4a_solver.get(j, "x")[:2] for j in range(ctrl_res.stmpc_config.N + 1)
        ])

        elapsed_time = (time.time() - start_time) * 1e3
        times_res_ms.append(elapsed_time)

        # 4) Apply control to the "plant":
        # Put current control into the helper states (indices 8: jerk, 9: derDelta)
        x_plant[8] = u0[0]  # jerk = d(accel)/dt from MPC
        x_plant[9] = u0[1]  # derDelta = d(delta)/dt from MPC

        # One-step forward Euler integration of your nonlinear dynamics
        x_plant = integrate_one_step(dyn_act, x_plant, dt, noise_vars=noise_vars, noise_indices=idx_noise)

        # 5) Log plant state
        x_hist.append(x_plant.copy())

        # (optional) simple timing print
        # if (k % 20) == 0:
        print(f"[{k:04d}] status={status}, u0={u0}, elapsed={elapsed_time:.1f} ms")

    # Visualize closed loop states and inputs
    x_pred_seq_hist = np.asarray(x_pred_seq_hist)
    x_hist = np.array(x_hist)
    u_hist = np.array(u_hist)
    fig = plot_closed_loop_summary(
        refline_xy=refline_xy,
        left_dist=left_dist,
        right_dist=right_dist,
        x_hist=x_hist,
        u_hist=u_hist
    )
    savefig("gpytorch_cl_lbmpc")
    plt.show()

    x_hist_res = x_hist

    # plot_gp_training_fit(Z, Y, models, likelihoods, device=device, titles=("vx", "vy", "wz"))
    # plt.show()

    # plot_residual_surfaces_vy_wz(Z, Y)
    # plt.show()

    # plot_residual_surfaces_vy_wz_colored_by_vx(Z, Y)
    # plt.show()

    # === PERFORMANCE COMPARISON (lap times & lateral deviation) ===
    L = float(ctrl_nom.spline.track_length)   # track length
    s_nom = x_hist_nom[:, 0]                  # s is the first state
    s_res = x_hist_res[:, 0]

    lap_times_nom = lap_times_from_s(s_nom, dt, L)
    lap_times_res = lap_times_from_s(s_res, dt, L)

    mean_dev_nom, std_dev_nom = compute_lateral_deviation(x_hist_nom)
    mean_dev_res, std_dev_res = compute_lateral_deviation(x_hist_res)

    def fmt_laps(laps):
        return " / ".join(f"{t:.3f}s" for t in laps) if laps else "—"

    print("\n=== PERFORMANCE COMPARISON ===")
    print(f"Track length: {L:.2f} m")
    print(f"Nominal MPC -> laps finished: {len(lap_times_nom)} | lap times: {fmt_laps(lap_times_nom)} | "
          f"mean lateral deviation: {mean_dev_nom:.3f} ± {std_dev_nom:.3f} m")
    print(f"LB-MPC      -> laps finished: {len(lap_times_res)} | lap times: {fmt_laps(lap_times_res)} | "
          f"mean lateral deviation: {mean_dev_res:.3f} ± {std_dev_res:.3f} m")

    stats_nom = summarize_times_ms(times_nom_ms)
    stats_res = summarize_times_ms(times_res_ms)

    print("\n=== SOLVER TIMING (ms) ===")
    print(f"Nominal MPC -> N={stats_nom['N']}, mean={stats_nom['mean']:.1f}, "
          f"median={stats_nom['median']:.1f}, std={stats_nom['std']:.1f}, "
          f"p95={stats_nom['p95']:.1f}, max={stats_nom['max']:.1f}")

    print(f"LB-MPC      -> N={stats_res['N']}, mean={stats_res['mean']:.1f}, "
          f"median={stats_res['median']:.1f}, std={stats_res['std']:.1f}, "
          f"p95={stats_res['p95']:.1f}, max={stats_res['max']:.1f}")

    print("[done] residual MPC validation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
