from typing import Union

import numpy as np
from forzaeth_mpc.utils.frenet_converter import FrenetConverter
from forzaeth_mpc.config import (CarConfig, PacejkaTireConfig, STMPCConfig,
                        TrailingConfig)
from scipy.integrate import solve_ivp
from forzaeth_mpc.stmpc.acados_settings import acados_settings
from forzaeth_mpc.stmpc.online_params import MU_IDX, build_online_params
from forzaeth_mpc.utils.indices import (InputIndex, StateIndex,
                                         single_track_x0)
from forzaeth_mpc.utils.splinify import (SplineTrack,
                                         track_coords_from_waypoints)

try:
    from casadi import DM
    _HAS_DM = True
except Exception:
    _HAS_DM = False
    DM = None


class STMPC:
    def __init__(self,
                 pose_frenet: np.ndarray,
                 waypoints_data: np.ndarray,
                 racecar_version: str,
                 stmpc_config: STMPCConfig,
                 car_config: CarConfig,
                 tire_config: PacejkaTireConfig,
                 trailing_config: TrailingConfig,
                 controller_frequency: float,
                 using_gokart: bool) -> None:

        # Store configs
        self.racecar_version = racecar_version
        self.stmpc_config = stmpc_config
        self.car_config = car_config
        self.tire_config = tire_config
        self.trailing_config = trailing_config
        self.controller_frequency = controller_frequency
        self.using_gokart = using_gokart

        # Initialize parameters
        self.mpc_init_params()

        # Initialize solver with waypoints
        self.mpc_initialize_solver(pose_frenet, waypoints_data)

    def mpc_init_params(self) -> None:
        # MPC parameters
        self.t_MPC = 1 / self.stmpc_config.MPC_freq
        self.t_delay = self.stmpc_config.t_delay + self.t_MPC

        # Steering angle buffer
        buf_size = 2
        self.steering_angle_buf = np.zeros(buf_size)

        # Initial state
        self.mpc_sd = np.zeros((self.stmpc_config.N + 1, 2))
        self.u0 = np.zeros(2)
        self.fre_s = 0
        self.previous_frenet_s = 0
        self.comp_time = 0
        self.overtake_d = self.stmpc_config.overtake_d
        self.loop_rate = 40

        # Gap control variables
        self.gap = None
        self.gap_should = None
        self.gap_error = None
        self.gap_actual = None
        self.v_diff = None
        self.i_gap = 0

    def mpc_initialize_solver(self, pose_frenet: np.ndarray, waypoints_data: np.ndarray) -> None:
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

        coords_path = self.transform_waypoints_to_coords_from_array(waypoints_data=waypoints_data)
        self.spline = SplineTrack(coords_direct=coords_path)

        # Initialize lap counter
        self.nr_laps = 0

        # Setup acados solver
        (self.constraint, self.model, self.acados_solver, self.model_params) = acados_settings(
            self.s_ref, kapparef, d_left, d_right,
            self.stmpc_config, self.car_config, self.tire_config
        )

        self.set_sim_mu(self.tire_config.friction_coeff)

        # Apply warm start
        self.apply_warm_start(pose_frenet=pose_frenet)

        self.kappa = kapparef
        self.prev_acc = 0

    def apply_warm_start(self, pose_frenet: np.ndarray):
        """Applies a warm start to the MPC solver.

        Args:
            pose_frenet (np.ndarray): _description_
        """
        warm_start_trajectory = self.get_warm_start(pose_frenet=pose_frenet, const_acc=1, const_steer_vel=0.0)
        for i in range(self.stmpc_config.N + 1):
            self.acados_solver.set(i, "x", warm_start_trajectory[i][:self.model.n_x])
            if i < self.stmpc_config.N:
                self.acados_solver.set(i, "u", warm_start_trajectory[i][self.model.n_x:])

    def main_loop(
            self,
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
            self.measured_acc = getattr(self, "prev_acc", 0.0)
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
        current_pos_s = self.position_in_map_frenet[0] + self.nr_laps * self.track_length

        self.fre_s = current_pos_s
        self.fre_d = self.position_in_map_frenet[1]
        self.fre_alpha = alpha_center

        x0 = single_track_x0(s=self.fre_s, n=self.fre_d, theta=self.fre_alpha,
                                v_x=speed_now, v_y=self.vel_y, yaw_rate=self.yaw_rate,
                                delta=self.measured_steer, accel=self.measured_acc)

        # Delay compensation
        self.t_delay = 0.0
        propagated_x = x0

        # Fix initial state
        self.acados_solver.set(0, "lbx", propagated_x)
        self.acados_solver.set(0, "ubx", propagated_x)

        # Trailing control
        if (self.state == "TRAILING") and (self.opponent is not None):
            self.trailing_command = self.trailing_controller(self.waypoint_array_in_map[0, 2])
        else:
            self.i_gap = 0  # reset integral if not trailing

        for i in range(self.stmpc_config.N + 1):
            if (self.state == "TRAILING") and (self.opponent is not None):
                target_v_speed = self.trailing_command
            else:
                idx = np.abs(self.waypoint_array_in_map[:, 4] - (self.mpc_sd[i, 0] % self.track_length)).argmin()
                target_v_speed = self.waypoint_array_in_map[idx, 2]
                self.overtake_d = 0
                if (self.state == "OVERTAKE") and (self.opponent is not None):
                    self.overtake_d = self.waypoint_array_in_map[idx, 3]

            # during the delay steps the steering rate is pinned by inflating its weight
            weights = {"q_ddelta": 1e6 * self.stmpc_config.qddelta} \
                if i < self.stmpc_config.steps_delay else None
            online_parameters = build_online_params(
                self.stmpc_config, self.tire_config,
                target_v=target_v_speed,
                overtake_d=self.overtake_d,
                mu=self.tire_config.friction_coeff,
                weights=weights)

            self.acados_solver.set(i, "p", online_parameters)

            if i < self.stmpc_config.N:
                self.acados_solver.set(i, "lbu", np.array([self.stmpc_config.ddelta_min, -50]))
                self.acados_solver.set(i, "ubu", np.array([self.stmpc_config.ddelta_max, 50]))
                self.acados_solver.set(i + 1, "lbx",
                                       np.array([self.stmpc_config.v_min,
                                                 self.stmpc_config.delta_min,
                                                 self.stmpc_config.a_min]))
                self.acados_solver.set(i + 1, "ubx",
                                       np.array([self.stmpc_config.v_max,
                                                 self.stmpc_config.delta_max,
                                                 self.stmpc_config.a_max]))

        # Solve OCP
        status = self.acados_solver.solve()
        if not (status == 0 or status == 2):
            # Re-apply warm start
            self.apply_warm_start(pose_frenet=[self.fre_s, self.fre_d, self.fre_alpha])

        # Actual cost of the problem, in case wanting to log
        # current_cost = self.acados_solver.get_cost()

        # Output
        self.u0 = self.acados_solver.get(0, "u")
        self.prev_acc = self.acados_solver.get(0, "x")[StateIndex.ACCEL.value]
        delayed_index = self.stmpc_config.steps_delay + 1
        self.pred_x = self.acados_solver.get(delayed_index, "x")

        self.steering_angle = self.pred_x[StateIndex.STEERING_ANGLE_DELTA]
        self.speed = self.pred_x[StateIndex.VELOCITY_V_X]
        self.acceleration = (self.prev_acc
                             + self.u0[InputIndex.JERK] / self.controller_frequency)
        self.prev_acc = self.acceleration
        self.jerk = self.u0[InputIndex.JERK]

        # Update steering buffer
        self.steering_angle_buf[1:] = self.steering_angle_buf[:-1]
        self.steering_angle_buf[0] = self.pred_x[StateIndex.STEERING_ANGLE_DELTA]

        self.mpc_sd = np.array([
            self.acados_solver.get(j, "x")[:2] for j in range(self.stmpc_config.N + 1)
        ])

        states_list = None
        if status == 0 or status == 2:
            states_list = []
            for node in range(self.stmpc_config.N + 1):
                result = self.acados_solver.get(node, "x").tolist()
                for item in result:
                    states_list.append(item)
            return self.speed, self.acceleration, self.jerk, self.steering_angle, states_list, status
        else:
            return 0.0, 0.0, 0.0, self.measured_steer, states_list, status

    def get_warm_start(self, pose_frenet: np.ndarray, const_acc: float, const_steer_vel: float) -> np.array:
        """
        Returns a warm start trajectory for the MPC. This is done by propagating the current state with a constant acceleration and steering angle.

        Input:  const_acc   : Constant acceleration
                const_steer : Constant steering angle

        Returns: np.array    : Warm start trajectory
        """
        warm_start = np.zeros((self.stmpc_config.N + 1, self.model.n_x + self.model.n_u))
        # TODO setting the velocity to current actual velocity might remove slowing down
        warm_start[0, StateIndex.POS_ON_CENTER_LINE_S] = pose_frenet[0]
        warm_start[0, StateIndex.MIN_DIST_TO_CENTER_LINE_N] = pose_frenet[1]
        warm_start[0, StateIndex.ORIENTATION_THETA] = pose_frenet[2]
        warm_start[0, StateIndex.VELOCITY_V_X] = 1.0
        warm_start[0, StateIndex.ACCEL] = const_acc
        warm_start[0, StateIndex.INPUT_DSTEERING] = const_steer_vel
        for i in range(1, self.stmpc_config.N + 1):
            xdot = self._dynamics_of_car(0, warm_start[i - 1])
            warm_start[i] = warm_start[i - 1] + np.array(xdot) / self.stmpc_config.MPC_freq  # ugly euler integration
        return warm_start

    def _dynamics_of_car(self, t, x0) -> list:
        '''
        Used for forward propagation. This function takes the dynamics from the acados model.
        '''

        n_x, n_u = self.model.n_x, self.model.n_u
        u = np.zeros(n_u)
        u[InputIndex.JERK] = x0[StateIndex.INPUT_JERK]
        u[InputIndex.D_STEERING_ANGLE] = x0[StateIndex.INPUT_DSTEERING]
        xdot = self.model.f_expl_func(np.asarray(x0[:n_x], dtype=float), u,
                                      self.model_params.p)
        # the two input slots are held constant over the step
        return [float(v) for v in np.asarray(xdot).reshape(-1)] + [0.0] * n_u

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

    def transform_waypoints_to_coords_from_array(self, waypoints_data: np.ndarray) -> np.ndarray:
        """Build track coordinates (left boundary, centerline, right boundary).

        Kept as a method because callers use it as one; the implementation is shared
        with the other controllers in ``utils/splinify.py``.
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


if __name__ == "__main__":
    # Call main_loop the way controller_manager does. The simulator never takes this path:
    # its harness pins the state by index, while main_loop assembles it from named
    # measurements, which is where yaw_rate and delta were once swapped.
    from forzaeth_mpc.sim.core.scenario import Scenario

    scn = Scenario(track="THETRACK", racecar_version="NUC5", floor="dubi")
    mpc = scn.make_mpc(s0=20.0, n0=0.0, psi0=0.0, dt=1.0 / scn.stmpc_config.MPC_freq)
    wp = scn.waypoints
    V_Y, YAW_RATE, ACCEL, DELTA = -0.11, 0.77, 1.9, 0.13   # no two of them equal
    out = mpc.main_loop(
        state="GB_TRACK",
        position_in_map=np.array([[wp[200, 0], wp[200, 1], 0.0]]),
        waypoint_array_in_map=wp,
        speed_now=3.3,
        opponent=None,
        position_in_map_frenet=np.array([wp[200, 4], 0.05, 0.0, 3.3]),
        alpha=0.0,
        single_track_state=np.array([V_Y, YAW_RATE, ACCEL, DELTA]),
        track_length=scn.track_len,
        compute_time=0.0,
        in_start_bool=False,
    )
    status = out[5]
    x0 = mpc.acados_solver.get(0, "x")
    assert abs(x0[StateIndex.VELOCITY_V_Y] - V_Y) < 1e-12, x0
    assert abs(x0[StateIndex.YAW_RATE] - YAW_RATE) < 1e-12, (
        f"the yaw_rate slot holds {x0[StateIndex.YAW_RATE]}, measured {YAW_RATE}")
    # using_gokart is False here, so the steering angle comes from the buffer, which
    # starts at zero, and not from the measurement
    assert abs(x0[StateIndex.STEERING_ANGLE_DELTA]) < 1e-12, x0
    assert status in (0, 2), status
    print("OK: main_loop puts each measurement in its own slot")
