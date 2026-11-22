import time
import re
from typing import Union

import numpy as np
from forzaeth_mpc.utils.splinify import track_coords_from_waypoints
import torch
from forzaeth_mpc.utils.frenet_converter import FrenetConverter
from l4acados.controllers import ResidualLearningMPC, ZeroOrderGPMPC
from forzaeth_mpc.lbmpc.acados_settings import build_ocp_for_l4acados
from forzaeth_mpc.stmpc.online_params import MU_IDX, build_online_params
from forzaeth_mpc.lbmpc.residual_model import (initialize_residual_model,
                                               persist_learnt_gp_model)
from forzaeth_mpc.lbmpc.solver_artifacts import build_solver_build_config
from forzaeth_mpc.lbmpc.offline_viz_logger import OfflineMPCLogger
from forzaeth_mpc.utils.indices import InputIndex, StateIndex
from forzaeth_mpc.utils.indices import single_track_x0
from forzaeth_mpc.config import (CarConfig, LBMPCConfig, PacejkaTireConfig,
                        TrailingConfig)
from scipy.integrate import solve_ivp
from forzaeth_mpc.utils.splinify import SplineTrack

try:
    from casadi import DM
    _HAS_DM = True
except Exception:
    _HAS_DM = False
    DM = None


def build_offline_logger(
    lbmpc_path: str,
    ocp,
    horizon: int,
    horizon_requested: bool,
    use_zoro: bool,
    enabled: bool,
):
    if horizon_requested and not use_zoro:
        print("[offline_viz] Horizon logging requested, but use_zoro is False. Disabling horizon logging.")

    if not enabled:
        return None

    try:
        nx = int(ocp.dims.nx)
        nu = int(ocp.dims.nu)
        state_names = [e.name.lower() for e in sorted(StateIndex, key=lambda x: x.value)]
        input_names = [e.name.lower() for e in sorted(InputIndex, key=lambda x: x.value)]

        return OfflineMPCLogger(
            root_dir=lbmpc_path,
            nx=nx,
            nu=nu,
            N=int(horizon),
            state_names=state_names,
            input_names=input_names,
            enable=True,
            flush_every=25,
            horizon_stride=4,
        )
    except Exception as exc:
        print(f"[offline_viz] logger disabled: {exc}")
        return None

class LBMPC:
    def __init__(self,
                 pose_frenet: np.ndarray,
                 waypoints_data: np.ndarray,
                 racecar_version: str,
                 lbmpc_config: LBMPCConfig,
                 car_config: CarConfig,
                 tire_config: PacejkaTireConfig,
                 trailing_config: TrailingConfig,
                 controller_frequency: float,
                 using_gokart: bool,
                 lbmpc_path: str,
                 params_path: str) -> None:

        # Store configs
        self.racecar_version = racecar_version
        self.lbmpc_config = lbmpc_config
        self.car_config = car_config
        self.tire_config = tire_config
        self.trailing_config = trailing_config
        self.controller_frequency = controller_frequency
        self.using_gokart = using_gokart

        # TODO: FORCE MODEL MISMATCH TO TEST GP
        # self.tire_config.friction_coeff = 0.5
        # self.car_config.m = 10

        # Initialize parameters
        self._mpc_init_params()

        # Initialize solver with waypoints
        self._mpc_initialize_solver(pose_frenet, waypoints_data)

        # Initialize Residual model
        self.lbmpc_path = lbmpc_path
        self.use_gp = self.lbmpc_config.gp_params.use_gp
        self.stochastic_mode = self.lbmpc_config.gp_params.stochastic_mode
        self.use_zoro = self.stochastic_mode != "disabled"
        self.enable_offline_logging = self.lbmpc_config.gp_params.enable_offline_logging
        self.log_offline_horizon = (
            self.lbmpc_config.gp_params.log_offline_horizon and self.use_zoro
        )

        # Build the B matrix, B_d_pinv and nw
        idx = np.asarray(self.lbmpc_config.gp_params.unc_jac_G_mat_idx, dtype=float)
        self.B = np.diag(idx)
        self.B = self.B[:, ~np.all(self.B == 0, axis=0)]
        self.B_d_pinv = np.linalg.pinv(self.B)
        self._nw_gp = self.B.shape[1]

        if self.use_gp:
            residual_setup = initialize_residual_model(
                model=self.model,
                gp_cfg=self.lbmpc_config.gp_params,
                lbmpc_path=self.lbmpc_path,
                nw_gp=self._nw_gp,
                racecar_version=self.racecar_version,
            )
            self.residual_gp = residual_setup.residual_gp
            self._input_feature_selector = residual_setup.input_feature_selector
            self.use_inducing_gp = residual_setup.use_inducing_gp
            self.max_num_data_points = residual_setup.max_num_data_points
            self._nx_gp = residual_setup.nx_gp
            self._nu_gp = residual_setup.nu_gp
            self.mode = residual_setup.mode
            self.learnt_gp_model_path = residual_setup.learnt_gp_model_path
            self._gp_warmup_done = residual_setup.gp_warmup_done
        else:
            self.residual_gp = None
            self._input_feature_selector = None
            self.use_inducing_gp = False
            self.max_num_data_points = 0
            self._nx_gp = 0
            self._nu_gp = 0
            self.mode = None
            self.learnt_gp_model_path = None
            self.B = None

        self.learning_enabled = bool(
            getattr(self.lbmpc_config.gp_params, "learning_enabled", False)
        )
        learning_unavailable_reason = self._learning_unavailable_reason()
        if self.learning_enabled and learning_unavailable_reason is not None:
            print(
                "[LBMPC] learning_enabled was requested at startup, "
                f"but online learning is unavailable because {learning_unavailable_reason}. "
                "Starting with learning disabled."
            )
            self.learning_enabled = False
        self.lbmpc_config.gp_params.learning_enabled = self.learning_enabled

        solver_build_config = build_solver_build_config(
            lbmpc_path=self.lbmpc_path,
            params_path=params_path,
        )

        model_cls = ZeroOrderGPMPC if self.use_zoro else ResidualLearningMPC
        model_name = "ZeroOrderGPMPC" if self.use_zoro else "ResidualLearningMPC"
        print(
            f"[LBMPC] Using {model_name} "
            f"(stochastic_mode={self.stochastic_mode})"
        )
        active_residual_model = self.residual_gp if self._is_gp_runtime_active() else None
        self.l4a_solver = model_cls(
            ocp=self.ocp,
            B=self.B,
            residual_model=active_residual_model,
            use_cython=True,
            path_json_ocp=solver_build_config.path_json_ocp,
            path_json_sim=solver_build_config.path_json_sim,
            build_c_code=solver_build_config.build_c_code,
        )

        self.offlog = build_offline_logger(
            lbmpc_path=self.lbmpc_path,
            ocp=self.ocp,
            horizon=self.lbmpc_config.N,
            horizon_requested=self.lbmpc_config.gp_params.log_offline_horizon,
            use_zoro=self.use_zoro,
            enabled=self.enable_offline_logging,
        )

        # Keep a handle to the sim solver and set its default horizon
        if hasattr(self.l4a_solver, "sim_solver"):
            self.sim_solver = self.l4a_solver.sim_solver
            # Typically Ts is in ocp.solver_options.Tsim
            self.sim_solver.set("T", float(self.ocp.solver_options.Tsim))
        else:
            self.sim_solver = None

        # Apply warm start
        self._apply_warm_start(pose_frenet=pose_frenet)

        self._prev_state_for_residual = None
        self._prev_input_for_residual = None
        self.last_status = 0
        self.x_pred_nom = None
        self.pred_x = None
        self.previous_pred_x = None
        self.previous_pred_u = None
        self.count = 0

    def _learning_unavailable_reason(self) -> Union[str, None]:
        if not self.use_gp:
            return "gp_params.use_gp is false"
        if self.residual_gp is None:
            return "the residual GP model is not initialized"
        if getattr(self, "mode", None) != "learn":
            return (
                "gp_params.data_processing_mode is "
                f"'{getattr(self, 'mode', None)}' instead of 'learn'"
            )
        return None

    def _is_online_learning_active(self) -> bool:
        return self._learning_unavailable_reason() is None and self.learning_enabled

    def _is_gp_runtime_active(self) -> bool:
        if not (self.use_gp and self.residual_gp is not None):
            return False
        if getattr(self, "mode", None) == "use_learnt_model":
            return True
        if getattr(self, "mode", None) == "learn":
            return getattr(self, "_gp_warmup_done", False)
        return False

    def _sync_solver_residual_model(self) -> None:
        if not hasattr(self, "l4a_solver"):
            return

        if self._is_gp_runtime_active():
            self.l4a_solver.residual_model = self.residual_gp
            self.l4a_solver.has_residual_model = True
            return

        self.l4a_solver.has_residual_model = False
        if hasattr(self.l4a_solver, "residual_fun"):
            self.l4a_solver.residual_fun.fill(0.0)
        if hasattr(self.l4a_solver, "residual_jac"):
            self.l4a_solver.residual_jac.fill(0.0)

    def enable_learning(self, source: str = "runtime request") -> bool:
        learning_unavailable_reason = self._learning_unavailable_reason()
        if learning_unavailable_reason is not None:
            print(
                f"[LBMPC] Ignoring learning enable request from {source}: "
                f"{learning_unavailable_reason}."
            )
            return False

        if self.learning_enabled:
            print(f"[LBMPC] Online learning was already enabled ({source}).")
            return False

        self.learning_enabled = True
        self.lbmpc_config.gp_params.learning_enabled = True
        print(f"[LBMPC] Online learning enabled ({source}).")
        return True

    def set_learning_enabled(self, enabled: bool, source: str = "runtime request") -> bool:
        enabled = bool(enabled)
        if enabled:
            return self.enable_learning(source=source)

        if not self.learning_enabled:
            return False

        self.learning_enabled = False
        self.lbmpc_config.gp_params.learning_enabled = False
        print(f"[LBMPC] Online learning disabled ({source}).")
        return True

    def _mpc_init_params(self) -> None:
        # MPC parameters
        self.sampling_time_mpc = 1 / self.lbmpc_config.MPC_freq
        self.comput_time = 0.01  # Initial guess, updated with measured computation times
        self.t_delay = self.lbmpc_config.t_delay # + self.comput_time
        self.ma_alpha = self.sampling_time_mpc / (self.sampling_time_mpc + 5.0)  # Moving average window of 1.0 seconds
        # buffer to save last control inputs
        self.buffer_len = 20
        self.buffer_prev_solutions = np.zeros((2, self.buffer_len), dtype=np.float64)
        self.buf_filled = 0
        self.delay_sim = self.lbmpc_config.delay_sim

        # Steering angle buffer
        buf_size = 2
        self.steering_angle_buf = np.zeros(buf_size)

        # Initial state
        self.mpc_sd = np.zeros((self.lbmpc_config.N + 1, 2))
        self.u0 = np.zeros(2)
        self.fre_s = 0
        self.previous_frenet_s = 0
        self.comp_time = 0
        self.overtake_d = self.lbmpc_config.overtake_d
        self.loop_rate = self.controller_frequency

        # Gap control variables
        self.gap = None
        self.gap_should = None
        self.gap_error = None
        self.gap_actual = None
        self.v_diff = None
        self.i_gap = 0

        # First time that exceeds certain velocity
        self.first_time_done = False
        self._gp_warmup_done = False

    def _mpc_initialize_solver(self, pose_frenet: np.ndarray, waypoints_data: np.ndarray) -> None:
        """Initialize solver with waypoints data (no ROS dependency)"""

        # Extract waypoint data - assuming format: [x, y, vx, kappa, s, d_left, d_right]
        x = waypoints_data[:, 0]
        y = waypoints_data[:, 1]
        # vx_ref = np.full_like(waypoints_data[:, 2], 10.0, dtype=float)  # waypoints_data[:, 2]
        kapparef = waypoints_data[:, 3]
        self.s_ref = waypoints_data[:, 4]
        d_left = waypoints_data[:, 5]
        d_right = waypoints_data[:, 6]

        # Initialize frenet converter
        self.fren_conv = FrenetConverter(x, y)

        coords_path = self._transform_waypoints_to_coords_from_array(waypoints_data=waypoints_data)
        self.spline = SplineTrack(coords_direct=coords_path)

        # Initialize lap counter
        self.nr_laps = 0

        # Setup acados solver
        (self.ocp, self.model, self.constraint, self.model_params) = build_ocp_for_l4acados(
            self.s_ref, kapparef, d_left, d_right,
            self.lbmpc_config, self.car_config, self.tire_config
        )

        self.set_sim_mu(self.tire_config.friction_coeff)

        self.kappa = kapparef
        self.prev_acc = 0

    def main_loop(
            self,
            state: str,
            position_in_map: np.ndarray,
            waypoint_array_in_map: np.ndarray,
            speed_now: float,
            opponent: Union[np.ndarray, None],
            position_in_map_frenet: np.ndarray,
            alpha: float,
            learning_based_state: np.ndarray,
            track_length: float,
            compute_time: float,
            in_start_bool: bool = False):
        """
        Returns:
            speed, acceleration, jerk, steering_angle, states(list|None), status(int)
        """

        self.state = state
        self.position_in_map = position_in_map
        self.waypoint_array_in_map = waypoint_array_in_map
        self.speed_now = speed_now
        self.opponent = opponent
        self.position_in_map_frenet = position_in_map_frenet
        self.alpha = alpha
        self.vel_y = learning_based_state[0]
        self.yaw_rate = learning_based_state[1]
        if self.using_gokart:
            self.measured_acc = learning_based_state[2]
            self.measured_steer = learning_based_state[3]
        else:
            self.measured_steer = self.steering_angle_buf[-1]
            self.measured_acc = getattr(self, "prev_acc", 0.0)
        self.track_length = self.spline.track_length
        self.comp_time = compute_time

        _t0 = time.perf_counter()
        init_time = 0.0
        residual_time = 0.0
        rest_time = 0.0

        # Heading correction
        center_car_s = self.position_in_map_frenet[0] + self.car_config.lr * np.cos(self.alpha)
        deriv_center = self.spline.get_derivative(center_car_s)
        # alpha is relative to the track at the car's s: shift it by the track's turn up to the center of mass
        deriv_s = self.spline.get_derivative(self.position_in_map_frenet[0])
        alpha_center = self.alpha + np.arctan2(deriv_s[1], deriv_s[0]) - np.arctan2(deriv_center[1], deriv_center[0])
        alpha_center = alpha_center % (2 * np.pi)
        if alpha_center > np.pi:
            alpha_center -= 2 * np.pi

        # Lap counter
        if self.position_in_map_frenet[0] < self.previous_frenet_s - 1:
            self.nr_laps += 1
        self.previous_frenet_s = self.position_in_map_frenet[0]
        current_pos_s = self.position_in_map_frenet[0] + self.nr_laps * self.track_length

        self.fre_s = current_pos_s
        self.fre_d = self.position_in_map_frenet[1]
        self.fre_alpha = alpha_center

        x0 = single_track_x0(s=self.fre_s, n=self.fre_d, theta=self.fre_alpha,
                                v_x=speed_now, v_y=self.vel_y, yaw_rate=self.yaw_rate,
                                delta=self.measured_steer, accel=self.measured_acc)

        _t1 = time.perf_counter()
        init_time = _t1 - _t0

        if speed_now > 3.0 and not self.first_time_done:
            self.first_time_done = True
            print("First time done!")

        # Simulated delay compensation
        self.propagated_x = self._propagate_time_delay(x0) if self.delay_sim else x0

        res = None
        measured_residual = None
        last_pred_residual = None
        self.nominal_pred_x = None
        self.nomres_pred_x = None
        online_learning_active = self._is_online_learning_active()
        if (
            self.use_gp
            and self.residual_gp is not None
            and (
                getattr(self, "mode", None) == "use_learnt_model"
                or online_learning_active
            )
        ):
            t_res_start = time.perf_counter()
            res = self._compute_residual(new_state=x0)
            if (
                online_learning_active
                and res is not None
                and (self.last_status == 0 or self.last_status == 2)
                and self.nr_laps >= 2
            ):
                self.count += 1
            else:
                self.count = 0

            if (
                online_learning_active
                and res is not None
                and (self.last_status == 0 or self.last_status == 2)
                and self.nr_laps >= 2
                and self.count >= 1
            ):
                z_train, residual = res
                self.residual_gp.record_datapoint(z_train, residual)
                persist_learnt_gp_model(
                    model_path=self.learnt_gp_model_path,
                    residual_gp=self.residual_gp,
                    input_feature_selection=self.lbmpc_config.gp_params.input_feature_selection,
                    residual_dimension=self._nw_gp,
                    use_inducing_gp=self.use_inducing_gp,
                    num_inducing_points=self.lbmpc_config.gp_params.num_inducing_points,
                    racecar_version=self.racecar_version,
                )
                self.count = 0
                self._gp_warmup_done = True
            residual_time += time.perf_counter() - t_res_start

        if self.pred_x is not None and res is not None:
            # measured residual r_k = B^+ (x_k - f_nominal(x_{k-1}, u_{k-1}))
            _, measured_residual = res

            # Predicted residual {mpc first solution - nominal(x_prev, u_prev)}
            nominal_pred_x = self._predict_next_state_nominal(
                self._prev_state_for_residual,
                self._prev_input_for_residual)  # (self.previous_pred_x, self.previous_pred_u)
            nomres_pred_x = self._predict_next_state_with_gp(self._prev_state_for_residual,
                                                             self._prev_input_for_residual)
            last_pred_residual = self.B_d_pinv @ (nomres_pred_x - nominal_pred_x)
            self.nominal_pred_x = nominal_pred_x
            self.nomres_pred_x = nomres_pred_x

        if (measured_residual is not None) and (last_pred_residual is not None):
            mr = np.asarray(measured_residual, dtype=np.float64).reshape(-1)
            pr = np.asarray(last_pred_residual, dtype=np.float64).reshape(-1)

            residual_dim = min(mr.size, pr.size, self._res_abs_err_sum.size)
            if residual_dim > 0:
                abs_err = np.abs(pr[:residual_dim] - mr[:residual_dim])
                self._res_abs_err_sum[:residual_dim] += abs_err
                self._res_err_count += 1

        viz_information = None
        if (x0 is not None) and (self.nomres_pred_x is not None):
            # If nominal_pred_x was not computed this step, fill with NaNs to keep size consistent
            if self.nominal_pred_x is None:
                nominal_pred_x_for_viz = np.full_like(self.nomres_pred_x, np.nan, dtype=np.float64)
            else:
                nominal_pred_x_for_viz = np.asarray(self.nominal_pred_x, dtype=np.float64).reshape(-1)

            state_for_viz = np.asarray(x0, dtype=np.float64).reshape(-1)
            pred_for_viz = np.asarray(self.nomres_pred_x, dtype=np.float64).reshape(-1)

            # Visualize the real one step mismatch without propagating delay
            viz_information = np.concatenate(
                [state_for_viz, pred_for_viz, nominal_pred_x_for_viz]
            )

        rest_start = time.perf_counter()

        self._sync_solver_residual_model()

        # Update inducing points
        self._update_inducing_points_along_horizon()

        # Fix initial state
        self.l4a_solver.set(0, "lbx", self.propagated_x)
        self.l4a_solver.set(0, "ubx", self.propagated_x)

        # Trailing control
        if (self.state == "TRAILING") and (self.opponent is not None):
            self.trailing_command = self._trailing_controller(self.waypoint_array_in_map[0, 2])
        else:
            self.i_gap = 0  # reset integral if not trailing

        for i in range(self.lbmpc_config.N + 1):
            if (self.state == "TRAILING") and (self.opponent is not None):
                target_v_speed = self.trailing_command
            else:
                idx = np.abs(self.waypoint_array_in_map[:, 4] - (self.mpc_sd[i, 0] % self.track_length)).argmin()
                target_v_speed = self.waypoint_array_in_map[idx, 2]
                self.overtake_d = 0
                if (self.state == "OVERTAKE") and (self.opponent is not None):
                    self.overtake_d = self.waypoint_array_in_map[idx, 3]

            if in_start_bool:
                alat_max = self.lbmpc_config.alat_max_start
                a_min = self.lbmpc_config.a_min_start
                a_max = self.lbmpc_config.a_max_start
                qv = self.lbmpc_config.qv_start
                qjerk = self.lbmpc_config.qjerk_start
            else:
                alat_max = self.lbmpc_config.alat_max
                a_min = self.lbmpc_config.a_min
                a_max = self.lbmpc_config.a_max
                qv = self.lbmpc_config.qv
                qjerk = self.lbmpc_config.qjerk

            # qv/qjerk and the three limits already carry the start-mode swap above;
            # the delay steps additionally pin the steering rate by inflating its weight.
            weights = {"q_v": qv, "q_jerk": qjerk}
            if i < self.lbmpc_config.steps_delay:
                weights["q_ddelta"] = 1e6 * self.lbmpc_config.qddelta
            online_parameters = build_online_params(
                self.lbmpc_config, self.tire_config,
                target_v=target_v_speed,
                overtake_d=self.overtake_d,
                mu=self.tire_config.friction_coeff,
                weights=weights,
                alat_max=alat_max, a_min=a_min, a_max=a_max)

            self.l4a_solver.set(i, "p", online_parameters)

            if i < self.lbmpc_config.N:
                self.l4a_solver.set(i, "lbu", np.array([self.lbmpc_config.ddelta_min, -50]))
                self.l4a_solver.set(i, "ubu", np.array([self.lbmpc_config.ddelta_max, 50]))
                self.l4a_solver.set(i + 1, "lbx",
                                    np.array([self.lbmpc_config.v_min,
                                              self.lbmpc_config.delta_min,
                                              a_min]))
                self.l4a_solver.set(i + 1, "ubx",
                                    np.array([self.lbmpc_config.v_max,
                                              self.lbmpc_config.delta_max,
                                              a_max]))

        # Solve OCP
        try:
            status = self.l4a_solver.solve()
        except Exception as e:
            msg = str(e)
            m = re.search(r'status\s+(\d+)', msg)
            status = int(m.group(1)) if m else -1
            print(
                f"[MPC Controller] Solver failed with message={msg}.\n Status={status}.\n Applying warm start again.")
            self._apply_warm_start(
                pose_frenet=[
                    self.fre_s,
                    self.fre_d,
                    self.fre_alpha],
                actual_speed=self.speed_now,
                actual_acc=self.measured_acc)

        self.last_status = status

        # Actual cost of the problem, in case wanting to log
        # current_cost = self.l4a_solver.get_cost()

        self.u0 = self.l4a_solver.get(0, "u")
        self.prev_acc = self.l4a_solver.get(0, "x")[StateIndex.ACCEL.value]
        delayed_index = self.lbmpc_config.steps_delay + 1  # if not self.delay_sim else 1
        # delayed_index = delayed_index if speed_now > 2.0 else delayed_index + 3
        self.pred_x = self.l4a_solver.get(delayed_index, "x")

        self.steering_angle = self.pred_x[StateIndex.STEERING_ANGLE_DELTA.value]
        self.speed = self.pred_x[StateIndex.VELOCITY_V_X.value]
        self.acceleration = self.prev_acc + self.u0[InputIndex.JERK.value] / self.controller_frequency
        self.prev_acc = self.acceleration
        self.jerk = self.u0[InputIndex.JERK.value]

        # Metrics calculation purposes
        self.previous_pred_x = self.l4a_solver.get(delayed_index - 1, "x")
        self.previous_pred_u = self.l4a_solver.get(delayed_index - 1, "u")

        # Update steering buffer
        self.steering_angle_buf[1:] = self.steering_angle_buf[:-1]
        self.steering_angle_buf[0] = self.steering_angle

        self.mpc_sd = np.array([
            self.l4a_solver.get(j, "x")[:2] for j in range(self.lbmpc_config.N + 1)
        ])

        self._prev_state_for_residual = x0.copy()
        self._prev_input_for_residual = np.zeros(2) # self.u0.copy()    # [ddelta, jerk]

        # # Comput time
        # # Comput time (total + per-section)
        # _now = time.perf_counter()
        # rest_time = _now - rest_start          # everything after rest_start
        # _elapsed = _now - _t0                  # full main_loop duration
        _elapsed = 0.0

        # Update computational time calculation with moving average
        self.comput_time = self.comput_time * self.ma_alpha + _elapsed * (1 - self.ma_alpha)
        if self._gp_warmup_done:
            self.t_delay = float(self.lbmpc_config.t_delay) + 0.01
        else:
            self.t_delay = float(self.lbmpc_config.t_delay) # + float(self.comput_time)

        # Shift all solutions and att the new one
        self.buffer_prev_solutions[:, 1:] = self.buffer_prev_solutions[:, :-1]
        self.buffer_prev_solutions[:, 0] = np.asarray(self.u0, dtype=np.float64).reshape(2,)
        self.buf_filled = min(self.buf_filled + 1, self.buffer_len)

        # Comput time
        # Comput time (total + per-section)
        _now = time.perf_counter()
        rest_time = _now - rest_start          # everything after rest_start
        _elapsed = _now - _t0                  # full main_loop duration

        if not hasattr(self, "_timing_sum"):
            self._timing_sum = 0.0
            self._timing_count = 0
            self._timing_max = 0.0
            self._timing_last_report = time.perf_counter()
            self._timing_init_sum = 0.0
            self._timing_residual_sum = 0.0
            self._timing_rest_sum = 0.0

        # --- Residual performance accumulators (lazy init) ---
        if not hasattr(self, "_res_abs_err_sum"):
            self._res_abs_err_sum = np.zeros(self._nw_gp, dtype=np.float64)
            self._res_err_count = 0
            self._res_last_lap = -1

        self._timing_sum += _elapsed
        self._timing_init_sum += init_time
        self._timing_residual_sum += residual_time
        self._timing_rest_sum += rest_time
        self._timing_count += 1

        if _elapsed > self._timing_max:
            self._timing_max = _elapsed

        # Print once per lap
        if self.nr_laps > getattr(self, "_timing_last_lap", -1):
            denom = max(1, self._timing_count)

            avg_ms = (self._timing_sum / denom) * 1e3
            max_ms = self._timing_max * 1e3
            init_ms = (self._timing_init_sum / denom) * 1e3
            residual_ms = (self._timing_residual_sum / denom) * 1e3
            rest_ms = (self._timing_rest_sum / denom) * 1e3

            print(
                f"[LBMPC] Lap {self.nr_laps}: "
                f"avg = {avg_ms:.2f} ms "
                f"(init={init_ms:.2f}, residual={residual_ms:.2f}, rest={rest_ms:.2f}) "
                f"max = {max_ms:.2f} ms over {self._timing_count} iters"
            )

            if self._res_err_count > 0:
                mae = self._res_abs_err_sum / float(self._res_err_count)
                residual_terms = ", ".join(
                    f"r[{idx}]={value:.4e}" for idx, value in enumerate(mae)
                )
                print(f"[Residual] Lap {self.nr_laps}: MAE | {residual_terms} (based on {self._res_err_count} samples)")
            else:
                print(f"[Residual] Lap {self.nr_laps}: no residual data available this lap.")

            # Reset residual counters for the next lap
            self._res_abs_err_sum[:] = 0.0
            self._res_err_count = 0
            self._res_last_lap = self.nr_laps
            # ---------------------------------------------

            # reset timing stats for the next lap
            self._timing_sum = 0.0
            self._timing_init_sum = 0.0
            self._timing_residual_sum = 0.0
            self._timing_rest_sum = 0.0
            self._timing_count = 0
            self._timing_max = 0.0
            self._timing_last_lap = self.nr_laps

        # ---- offline logging (best-effort, never break control loop) ----
        if getattr(self, "offlog", None) is not None:
            try:
                # gp size best-effort
                gp_n = None
                try:
                    gp_n = int(self.residual_gp.gp_model.train_inputs[0].shape[0])
                except Exception:
                    gp_n = None

                x_nom = self._predict_next_state_nominal(self.previous_pred_x, self.previous_pred_u)
                self.offlog.log_step(
                    lap=self.nr_laps,
                    status=self.last_status,
                    x_meas=x0,  # Actual measured state, corresponding to time td
                    x_nom_delayed=self.nominal_pred_x,  # Nominal prediction corresponding to time td
                    x_pred_delayed=self.nomres_pred_x,  # Total prediction corresponding to time td
                    x_meas_prop=self.propagated_x,  # Propagated measured state corresponding to the actual t = 0
                    x_pred=self.pred_x,  # Propagated measure
                    x_nom=x_nom,
                    u0=self.u0,
                    t_total=_elapsed, t_init=init_time, t_residual=residual_time, t_rest=rest_time,
                    r_meas=measured_residual,
                    r_pred=last_pred_residual,
                    gp_num_points=gp_n
                )

                if self.log_offline_horizon:
                    cov_arr = getattr(self.l4a_solver, "covariances_array", None)
                    idx_lh_t = getattr(self.ocp.zoro_description, "idx_lh_t", [])
                    idx_uh_t = getattr(self.ocp.zoro_description, "idx_uh_t", [])
                    self.offlog.log_horizon(
                        solver=self.l4a_solver,
                        lap=self.nr_laps,
                        covariances_array=cov_arr,
                        idx_lh_t=list(idx_lh_t) if idx_lh_t is not None else [],
                        idx_uh_t=list(idx_uh_t) if idx_uh_t is not None else [],
                    )

                self.offlog.tick()
            except Exception as e:
                print(f"[offline_viz] log failed: {e}")

        states_list = None
        if status == 0 or status == 2:
            states_list = []
            for node in range(self.lbmpc_config.N + 1):
                result = self.l4a_solver.get(node, "x").tolist()
                for item in result:
                    states_list.append(item)
            return self.speed, self.acceleration, self.jerk, self.steering_angle, states_list, status, viz_information
        else:
            return 0.0, 0.0, 0.0, self.measured_steer, states_list, status, viz_information

    def _apply_warm_start(self, pose_frenet: np.ndarray, actual_speed=1.0, actual_acc=1.0):
        """
        Generates and applies a simple initial guess considering constant inputs.
        """
        nx = int(self.model.n_x)
        nu = int(self.model.n_u)
        N = self.lbmpc_config.N

        x0 = np.zeros(nx)
        x0[StateIndex.POS_ON_CENTER_LINE_S.value] = pose_frenet[0]
        x0[StateIndex.MIN_DIST_TO_CENTER_LINE_N.value] = pose_frenet[1]
        x0[StateIndex.ORIENTATION_THETA.value] = pose_frenet[2]
        x0[StateIndex.VELOCITY_V_X.value] = max(actual_speed, 1.0)
        x0[StateIndex.ACCEL.value] = actual_acc  # Maybe 0.0

        const_steer_vel = 0.0  # to figure out
        u0 = np.zeros(nu)
        u0[InputIndex.D_STEERING_ANGLE.value] = const_steer_vel

        warm_start = np.zeros((N + 1, nx))
        warm_start[0, :] = x0
        x_prev = warm_start[0, :]
        for i in range(1, N + 1):
            # TODO: do it with residual _predict_next_state_with_gp
            x_next = self._predict_next_state_nominal(x_prev, u0)
            warm_start[i, :] = x_next
            self.l4a_solver.set(i, "x", x_next)
            if i < N:
                self.l4a_solver.set(i, "u", u0)
            x_prev = x_next

    # TODO: Shouldn't we use the previous results?
    def _update_warm_start(self) -> None:
        """
        Propagate states along the horizon with the current inputs,
        using the learnt model (nominal + GP).
        """
        for i in range(self.lbmpc_config.N + 1):
            current_state = self.l4a_solver.get(i, "x")
            if i < self.lbmpc_config.N:
                current_input = self.l4a_solver.get(i, "u")
            else:
                current_input = np.zeros(self.model.n_u)

            # TODO: Use also the residual
            # updated_state = self._predict_next_state_with_gp(current_state, current_input)
            updated_state = self._predict_next_state_nominal(current_state, current_input)
            self.l4a_solver.set(i, "x", updated_state)

            if i < self.lbmpc_config.N - 1:
                next_inputs = self.l4a_solver.get(i + 1, "u")
                self.l4a_solver.set(i, "u", next_inputs)

    def _propagate_time_delay(self, x0: np.ndarray) -> np.ndarray:
        """
        Forward propagate current state over self.t_delay with discrete steps of the nominal model.
        """
        x = x0
        n_steps = int(np.floor(self.t_delay / self.sampling_time_mpc))
        n_steps = min(n_steps, self.buffer_len, self.buf_filled)
        if x[StateIndex.VELOCITY_V_X.value] > 1.5:
            for j in range(n_steps - 1, -1, -1):
                x = self._predict_next_state_nominal( # _predict_next_state_with_gp(
                    x,
                    self.buffer_prev_solutions[:, j]
                )

        # Saturations as before
        if abs(x[StateIndex.STEERING_ANGLE_DELTA.value]) > self.constraint.delta_max:
            x[StateIndex.STEERING_ANGLE_DELTA.value] = np.sign(
                x[StateIndex.STEERING_ANGLE_DELTA.value]) * self.constraint.delta_max
        if abs(x[StateIndex.VELOCITY_V_X.value]) > self.constraint.v_x_max:
            x[StateIndex.VELOCITY_V_X.value] = np.sign(x[StateIndex.VELOCITY_V_X.value]) * self.constraint.v_x_max
        if abs(x[StateIndex.VELOCITY_V_X.value]) < self.constraint.v_x_min:
            x[StateIndex.VELOCITY_V_X.value] = self.constraint.v_x_min

        return x

    def _compute_residual(self, new_state: np.ndarray):
        """
        Compute residual y = B^+ (x_meas(k) - f(x(k-1), u(k-1))).
        new_state must be the measured state at time k (same dimension as model state).
        Returns (x_train, residual) or None.
        """
        if self._prev_state_for_residual is None or self._prev_input_for_residual is None:
            return None

        x_prev = self._prev_state_for_residual
        u_prev = self._prev_input_for_residual

        # 1-step prediction with nominal model
        x_pred_nom = self._predict_next_state_nominal(x_prev, u_prev)

        # Training input: [x_prev, u_prev] (full vector)
        z_train = np.concatenate((x_prev, u_prev))

        # Residual in selected coordinates
        residual = self.B_d_pinv @ (new_state - x_pred_nom)

        return z_train, residual

    def _trailing_controller(self, global_speed: float) -> float:
        # opponent: [s, d, v_s, v_d, has_los]
        self.gap = (self.opponent[0] - self.position_in_map_frenet[0]) % self.track_length
        self.gap_actual = self.gap

        if self.trailing_config.trailing_mode:
            self.gap_should = self.trailing_config.trailing_gap
        else:
            self.gap_should = self.position_in_map_frenet[2] * self.trailing_config.trailing_gap

        self.gap_error = self.gap_should - self.gap_actual
        self.v_diff = self.position_in_map_frenet[2] - self.opponent[2]
        self.i_gap = np.clip(self.i_gap + self.gap_error / self.loop_rate, -10.0, 10.0)

        p_value = self.gap_error * self.trailing_config.trailing_p_gain
        d_value = self.v_diff * self.trailing_config.trailing_d_gain
        i_value = self.i_gap * self.trailing_config.trailing_i_gain

        trailing_cmd = np.clip(self.opponent[2] - p_value - i_value - d_value, 0.0, global_speed)
        if not self.opponent[4] and self.gap_actual > self.gap_should:
            trailing_cmd = max(self.trailing_config.blind_trailing_speed, trailing_cmd)

        return trailing_cmd

    def _transform_waypoints_to_coords_from_array(self, waypoints_data: np.ndarray) -> np.ndarray:
        """Build track coordinates (left boundary, centerline, right boundary).

        Shared with the other controllers, see ``utils/splinify.py``. This used to be a
        textually identical copy of STMPC's.
        """
        return track_coords_from_waypoints(waypoints_data)

    def set_sim_mu(self, mu: float) -> None:
        """
        Update ONLY the simulator's parameter vector used in _dynamics_of_car().
        Does NOT touch the MPC solver's stage parameters.
        """
        # get current p as a flat numpy array
        p = np.array(self.model_params.p, dtype=float).reshape(-1).copy()
        p[MU_IDX] = float(mu)

        # write it back in the same type the model expects
        if _HAS_DM and isinstance(self.model_params.p, DM):
            self.model_params.p = DM(p)
        else:
            self.model_params.p = p

    def _update_inducing_points_along_horizon(self) -> None:
        """
        Update inducing point locations along the current open-loop MPC trajectory.

        Should be called once per MPC solve if using an inducing-point GP.
        """
        if not (
            self.use_inducing_gp
            and self.residual_gp is not None
            and self._is_gp_runtime_active()
        ):
            return

        gp_model = self.residual_gp.gp_model
        if not hasattr(gp_model, "num_inducing_points"):
            return

        # Choose evenly spaced indices along horizon
        inducing_indices = np.linspace(
            0,
            self.lbmpc_config.N,
            gp_model.num_inducing_points,
            dtype=int,
            endpoint=False,
        )

        states = np.array([self.l4a_solver.get(i, "x") for i in inducing_indices])
        inputs = np.array([self.l4a_solver.get(i, "u") for i in inducing_indices])
        xu = np.hstack((states, inputs))  # (M, nx+nu)

        gp_model.train()
        gp_model.initialize(
            **{
                "covar_module.inducing_points": self._input_feature_selector(
                    torch.Tensor(xu)
                )
            }
        )
        gp_model.eval()

    def _predict_next_state_nominal(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        One-step nominal prediction using either acados sim_solver (if available)
        or the continuous model with Euler discretization.
        """
        # Use acados sim solver created by ResidualLearningMPC
        self.sim_solver.set("x", x.astype(float))
        self.sim_solver.set("u", u.astype(float))
        self.sim_solver.set("p", np.array(self.model_params.p, dtype=float))
        status = self.sim_solver.solve()  # TODO: Use status to print if something is wrong at least
        x_next = self.sim_solver.get("x")
        return np.array(x_next, dtype=float).reshape(-1)

    def _predict_next_state_with_gp(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Learnt prediction:
            x_next ≈ f_nominal(x,u) + B * μ_residual(x,u)
        Only active if use_gp and warm-up already finished.
        """
        x_next_nom = self._predict_next_state_nominal(x, u)

        # Make sure self._gp_warmup_done is actually being set to True somewhere in main_loop!
        if not (self.use_gp and self.residual_gp is not None and getattr(self, "_gp_warmup_done", False)):
            return x_next_nom

        # 1. Prepare the input as a float32 PyTorch tensor
        xu = np.concatenate((x, u))[None, :]  # shape (1, nx+nu)
        xu_tensor = torch.tensor(xu, dtype=torch.float32)

        # 2. Get the GP prediction safely without tracking gradients
        with torch.no_grad():
            # Apply the configured feature selector before querying the GP.
            xu_selected = self._input_feature_selector(xu_tensor)

            # Query the underlying GPyTorch model
            gp_pred = self.residual_gp.gp_model(xu_selected)

            # Extract the mean of the prediction and convert back to numpy
            mu_residual = gp_pred.mean.detach().cpu().numpy().reshape(-1)  # (nw,)

        # 3. Project the residual back into the full state space
        delta_x = self.B @ mu_residual  # (nx,)
        return x_next_nom + delta_x
