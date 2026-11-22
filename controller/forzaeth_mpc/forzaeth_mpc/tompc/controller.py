"""Time-optimal MPC controller.

Same shape as ``stmpc/controller.py`` -- same constructor arguments, same ``main_loop``
signature and return tuple -- so ``controller_manager.py`` dispatches to it the same way
it dispatches to STMPC and LB-MPC. What differs is inside:

- the OCP is the spatial, minimum-time one in ``spatial_model.py``
- while racing there is no reference speed. ``waypoint_array_in_map[:, 2]``, the global
  planner's speed profile, is not read at all except while trailing
- the actuation delay is exact rather than a fixed node count: elapsed time is a state,
  so the applied command comes from the first node whose clock has passed ``t_delay``
"""

from typing import Union

import numpy as np

from forzaeth_mpc.config import (CarConfig, PacejkaTireConfig, TOMPCConfig,
                                 TrailingConfig)
from forzaeth_mpc.tompc.acados_settings import acados_settings
from forzaeth_mpc.tompc.online_params import (MU_IDX, build_online_params,
                                              racing_weights, trailing_weights)
from forzaeth_mpc.utils.frenet_converter import FrenetConverter
from forzaeth_mpc.utils.indices import (InputIndex, SpatialStateIndex,
                                        StateIndex)
from forzaeth_mpc.utils.splinify import (SplineTrack,
                                         track_coords_from_waypoints)

try:
    from casadi import DM
    _HAS_DM = True
except Exception:
    _HAS_DM = False
    DM = None


class TOMPC:
    def __init__(self,
                 pose_frenet: np.ndarray,
                 waypoints_data: np.ndarray,
                 racecar_version: str,
                 tompc_config: TOMPCConfig,
                 car_config: CarConfig,
                 tire_config: PacejkaTireConfig,
                 trailing_config: TrailingConfig,
                 controller_frequency: float,
                 using_gokart: bool) -> None:

        self.racecar_version = racecar_version
        self.tompc_config = tompc_config
        self.car_config = car_config
        self.tire_config = tire_config
        self.trailing_config = trailing_config
        self.controller_frequency = controller_frequency
        self.using_gokart = using_gokart

        self.mpc_init_params()
        self.mpc_initialize_solver(pose_frenet, waypoints_data)

    def mpc_init_params(self) -> None:
        self.t_MPC = 1 / self.tompc_config.MPC_freq
        self.t_delay = self.tompc_config.t_delay
        # node spacing along the path, set for real once the model is built
        self.ds = self.tompc_config.s_horizon / self.tompc_config.N

        buf_size = 2
        self.steering_angle_buf = np.zeros(buf_size)

        self.mpc_sd = np.zeros((self.tompc_config.N + 1, 2))
        self.u0 = np.zeros(2)
        self.fre_s = 0
        self.previous_frenet_s = 0
        self.comp_time = 0
        self.overtake_d = self.tompc_config.overtake_d
        self.loop_rate = self.tompc_config.MPC_freq
        self.prev_acc = 0.0
        # the traversal time of the last solved horizon, in seconds. This is the
        # objective the controller actually minimises, so it is the number to log.
        self.horizon_time = float("nan")

        # Gap control variables
        self.gap = None
        self.gap_should = None
        self.gap_error = None
        self.gap_actual = None
        self.v_diff = None
        self.i_gap = 0

    def mpc_initialize_solver(self, pose_frenet: np.ndarray, waypoints_data: np.ndarray) -> None:
        """Initialize solver with waypoints data (no ROS dependency).

        ``waypoints_data`` is the canonical (N, >=7) array with columns
        [x, y, vx, kappa, s, d_left, d_right]. Column 2, the speed profile, is only
        used while trailing.
        """
        x = waypoints_data[:, 0]
        y = waypoints_data[:, 1]
        kapparef = waypoints_data[:, 3]
        self.s_ref = waypoints_data[:, 4]
        d_left = waypoints_data[:, 5]
        d_right = waypoints_data[:, 6]

        self.fren_conv = FrenetConverter(x, y)
        coords_path = track_coords_from_waypoints(waypoints_data)
        self.spline = SplineTrack(coords_direct=coords_path)

        self.nr_laps = 0

        (self.constraint, self.model, self.acados_solver, self.model_params) = acados_settings(
            self.s_ref, kapparef, d_left, d_right,
            self.tompc_config, self.car_config, self.tire_config)
        self.ds = self.model.ds

        self.set_sim_mu(self.tire_config.friction_coeff)
        self.apply_warm_start(pose_frenet=pose_frenet)

        self.kappa = kapparef

    def apply_warm_start(self, pose_frenet: np.ndarray, v_x: float = 1.0) -> None:
        """Seeds every node with a constant-speed straight run along the path.

        The spatial horizon makes this simpler than STMPC's Euler roll-out: the node
        spacing is a known distance, so at a constant speed the whole trajectory is
        analytic. ``v_x`` is floored at ``v_min`` because a warm start below it is
        outside the feasible set and near the singularity of ``1/sdot``.
        """
        v_x = max(float(v_x), self.tompc_config.v_min)
        for i in range(self.tompc_config.N + 1):
            x = np.zeros(self.model.n_x)
            x[SpatialStateIndex.POS_ON_CENTER_LINE_S] = pose_frenet[0] + i * self.ds
            x[SpatialStateIndex.MIN_DIST_TO_CENTER_LINE_N] = pose_frenet[1]
            x[SpatialStateIndex.ORIENTATION_THETA] = pose_frenet[2]
            x[SpatialStateIndex.VELOCITY_V_X] = v_x
            x[SpatialStateIndex.TIME] = i * self.ds / v_x
            self.acados_solver.set(i, "x", x)
            if i < self.tompc_config.N:
                self.acados_solver.set(i, "u", np.zeros(self.model.n_u))

    def delay_node(self) -> int:
        """The horizon node whose command to apply, given the actuation delay.

        The horizon is discretised in metres, so "the node ``t_delay`` seconds ahead" is
        not a fixed node count -- at twice the speed it is half as many nodes. Elapsed
        time is a state, though, so the answer is read off the solved trajectory rather
        than estimated: the first node whose clock has passed ``t_delay``. That is exact
        even while braking or accelerating, where ``distance = speed * time`` is not.

        STMPC uses a fixed ``steps_delay`` node count, which is only right at one speed.
        """
        for i in range(self.tompc_config.N + 1):
            if self.acados_solver.get(i, "x")[SpatialStateIndex.TIME] >= self.t_delay:
                return i
        return self.tompc_config.N

    def main_loop(self,
                  state: str,
                  position_in_map: np.ndarray,
                  waypoint_array_in_map: np.ndarray,
                  speed_now: float,
                  opponent: Union[np.ndarray, None],
                  position_in_map_frenet: np.ndarray,
                  alpha: float,
                  single_track_state: np.ndarray,
                  track_length: float,
                  compute_time: float,
                  in_start_bool: bool):
        """
        Args:
            single_track_state: ``[v_y, yaw_rate, accel, steering_angle]``, the last two
                only measured on the gokart.
            in_start_bool: accepted for signature parity with STMPC and LB-MPC. There is
                no separate start mode: the acceleration limits already describe what the
                car can do off the line, and a minimum-time objective will use them.

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
        self.vel_y = single_track_state[0]
        self.yaw_rate = single_track_state[1]
        if self.using_gokart:
            self.measured_acc = single_track_state[2]
            self.measured_steer = single_track_state[3]
        else:
            self.measured_steer = self.steering_angle_buf[-1]
            self.measured_acc = self.prev_acc
        self.track_length = self.spline.track_length
        self.comp_time = compute_time

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

        self.fre_s = self.position_in_map_frenet[0] + self.nr_laps * self.track_length
        self.fre_d = self.position_in_map_frenet[1]
        self.fre_alpha = alpha_center

        # Built by index, not positionally: the state vector has nine entries and two
        # neighbouring pairs of them (v_y/yaw_rate, delta/accel) are easy to transpose.
        x0 = np.zeros(self.model.n_x)
        x0[SpatialStateIndex.POS_ON_CENTER_LINE_S] = self.fre_s
        x0[SpatialStateIndex.MIN_DIST_TO_CENTER_LINE_N] = self.fre_d
        x0[SpatialStateIndex.ORIENTATION_THETA] = self.fre_alpha
        # clipped into the feasible set: below v_min the spatial model is singular, and
        # pinning x0 outside its own bounds makes the QP infeasible at stage 0
        x0[SpatialStateIndex.VELOCITY_V_X] = np.clip(
            speed_now, self.tompc_config.v_min, self.tompc_config.v_max)
        x0[SpatialStateIndex.VELOCITY_V_Y] = self.vel_y
        x0[SpatialStateIndex.YAW_RATE] = self.yaw_rate
        x0[SpatialStateIndex.STEERING_ANGLE_DELTA] = self.measured_steer
        x0[SpatialStateIndex.ACCEL] = self.measured_acc
        x0[SpatialStateIndex.TIME] = 0.0  # the horizon's clock starts now

        self.acados_solver.set(0, "lbx", x0)
        self.acados_solver.set(0, "ubx", x0)

        # Trailing control
        trailing = (self.state == "TRAILING") and (self.opponent is not None)
        if trailing:
            self.trailing_command = self.trailing_controller(self.waypoint_array_in_map[0, 2])
            weights = trailing_weights(self.tompc_config)
        else:
            self.i_gap = 0  # reset integral if not trailing
            weights = racing_weights(self.tompc_config)

        # zeroed every cycle: a stale offset from a finished overtake must not survive
        # into the next state
        self.overtake_d = 0.0

        for i in range(self.tompc_config.N + 1):
            if trailing:
                # holding a gap and minimising time are contradictory objectives, so
                # while trailing the gap controller owns the speed and the time term is
                # switched off by trailing_weights()
                target_v_speed = self.trailing_command
            else:
                # No reference speed while racing: the time-optimal objective is the
                # speed policy. The raceline is still read for the overtake offset.
                target_v_speed = 0.0
                if (self.state == "OVERTAKE") and (self.opponent is not None):
                    idx = np.abs(self.waypoint_array_in_map[:, 4]
                                 - (self.mpc_sd[i, 0] % self.track_length)).argmin()
                    self.overtake_d = self.waypoint_array_in_map[idx, 3]

            self.acados_solver.set(i, "p", build_online_params(
                self.tompc_config, self.tire_config,
                target_v=target_v_speed,
                overtake_d=self.overtake_d,
                mu=self.tire_config.friction_coeff,
                weights=weights))

            if i < self.tompc_config.N:
                self.acados_solver.set(i, "lbu", np.array(
                    [self.tompc_config.ddelta_min, self.tompc_config.jerk_min]))
                self.acados_solver.set(i, "ubu", np.array(
                    [self.tompc_config.ddelta_max, self.tompc_config.jerk_max]))
                # not i, since the MPC needs the pinned initial state
                self.acados_solver.set(i + 1, "lbx", np.array(
                    [self.tompc_config.v_min, self.tompc_config.delta_min, self.tompc_config.a_min]))
                self.acados_solver.set(i + 1, "ubx", np.array(
                    [self.tompc_config.v_max, self.tompc_config.delta_max, self.tompc_config.a_max]))

        status = self.acados_solver.solve()
        if not (status == 0 or status == 2):
            self.apply_warm_start(pose_frenet=[self.fre_s, self.fre_d, self.fre_alpha],
                                  v_x=speed_now)

        self.u0 = self.acados_solver.get(0, "u")
        self.applied_node = self.delay_node()
        self.pred_x = self.acados_solver.get(self.applied_node, "x")

        self.steering_angle = self.pred_x[SpatialStateIndex.STEERING_ANGLE_DELTA]
        self.speed = self.pred_x[SpatialStateIndex.VELOCITY_V_X]
        self.acceleration = (self.acados_solver.get(0, "x")[SpatialStateIndex.ACCEL]
                             + self.u0[InputIndex.JERK] / self.controller_frequency)
        self.prev_acc = self.acceleration
        self.jerk = self.u0[InputIndex.JERK]
        # the last state's clock is the traversal time of the whole horizon
        self.horizon_time = float(
            self.acados_solver.get(self.tompc_config.N, "x")[SpatialStateIndex.TIME])

        self.steering_angle_buf[1:] = self.steering_angle_buf[:-1]
        self.steering_angle_buf[0] = self.steering_angle

        self.mpc_sd = np.array([
            self.acados_solver.get(j, "x")[:2] for j in range(self.tompc_config.N + 1)
        ])

        states_list = None
        if status == 0 or status == 2:
            # the first eight entries only: elapsed time is appended to the single-track
            # layout, so dropping it leaves exactly what the shared MPC viz node reads
            states_list = []
            for node_i in range(self.tompc_config.N + 1):
                states_list.extend(self.acados_solver.get(node_i, "x")[:8].tolist())
            return self.speed, self.acceleration, self.jerk, self.steering_angle, states_list, status
        return 0.0, 0.0, 0.0, self.measured_steer, states_list, status

    def _dynamics_of_car(self, t, x0) -> list:
        """Time-domain derivatives, for forward propagation and for the simulator.

        The OCP integrates over distance, but anything that closes the loop against this
        controller advances in *time*, so this is the time-domain model. The layout is
        the single-track one, ``[x(8), jerk, ddelta]``, not the spatial one.

        Mind the index collision: slot 8 is the jerk input in this layout and the
        horizon clock in the spatial one, so both enums are named explicitly here
        rather than written as literals.
        """
        n_x_plant = int(SpatialStateIndex.TIME)  # the spatial state minus its clock
        u = np.zeros(self.model.n_u)
        u[InputIndex.JERK] = x0[StateIndex.INPUT_JERK]
        u[InputIndex.D_STEERING_ANGLE] = x0[StateIndex.INPUT_DSTEERING]
        xdot = self.model.f_expl_time_func(np.asarray(x0[:n_x_plant], dtype=float), u,
                                           self.model_params.p)
        # the two input slots are held constant over the step
        return [float(v) for v in np.asarray(xdot).reshape(-1)] + [0.0] * self.model.n_u

    def trailing_controller(self, global_speed: float) -> float:
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

    def set_sim_mu(self, mu: float) -> None:
        """Update ONLY the parameter vector used by ``_dynamics_of_car``.

        Does NOT touch the MPC solver's stage parameters, which is what makes an
        offline friction mismatch between plant and controller possible.
        """
        p = np.array(self.model_params.p, dtype=float).reshape(-1).copy()
        p[MU_IDX] = float(mu)
        if _HAS_DM and isinstance(self.model_params.p, DM):
            self.model_params.p = DM(p)
        else:
            self.model_params.p = p


