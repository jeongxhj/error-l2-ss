from typing import Union

import numpy as np
from forzaeth_mpc.utils.frenet_converter import FrenetConverter
from forzaeth_mpc.config import (CarConfig, PacejkaTireConfig, STMPCConfig,
                        TrailingConfig)
from scipy.integrate import solve_ivp
# NOTE: use this tree's own acados_settings, not forzaeth_mpc.stmpc's. The production
# model moved the 8 Pacejka coefficients online (race_stack 8382de7d0, 2026-04-13),
# growing p from 14 to 22 entries, while these experiments and l4a_acados_settings
# still build/set the 14-entry p vector.
from research.residual_validation.mpc_core.ocp.acados_settings import acados_settings
from research.residual_validation.mpc_core.utils.indices import StateIndex
from forzaeth_mpc.utils.splinify import SplineTrack

try:
    from casadi import DM
    _HAS_DM = True
except Exception:
    _HAS_DM = False
    DM = None

# TODO clean up original single_track_mpc.py to isolate Standalone controller and ROS interface


class STMPC:
    def __init__(self,
                 pose_frenet: np.ndarray,
                 waypoints_data: np.ndarray,  # Replace ROS waypoints with numpy array
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
            single_track_state: np.ndarray):
        """
        Devuelve:
            speed, acceleration, jerk, steering_angle, states(list|None), status(int)
        """

        # === Actualiza entradas ===
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

        # === Corrige alpha (ángulo respecto a tangente en el centro del coche) ===
        center_car_s = self.position_in_map_frenet[0] + self.car_config.lr * np.cos(self.alpha)
        deriv_center = self.spline.get_derivative(center_car_s)
        alpha_center = self.position_in_map[2] - np.arctan2(deriv_center[1], deriv_center[0])
        alpha_center = alpha_center % (2 * np.pi)
        if alpha_center > np.pi:
            alpha_center -= 2 * np.pi

        # === Conteo de vueltas ===
        if self.position_in_map_frenet[0] < self.previous_frenet_s - 1:
            self.nr_laps += 1
        self.previous_frenet_s = self.position_in_map_frenet[0]
        current_pos_s = self.position_in_map_frenet[0] + self.nr_laps * self.track_length

        self.fre_s = current_pos_s
        self.fre_d = self.position_in_map_frenet[1]
        self.fre_alpha = alpha_center

        x0 = np.array([
            self.fre_s, self.fre_d, self.fre_alpha,
            speed_now, self.vel_y, self.measured_steer, self.yaw_rate, self.measured_acc
        ], dtype=np.float64)

        # Compensación de retardo (desactivada como en el original)
        self.t_delay = 0.0
        propagated_x = x0

        # Fija estado inicial del OCP
        self.acados_solver.set(0, "lbx", propagated_x)
        self.acados_solver.set(0, "ubx", propagated_x)

        # Control de "trailing"
        if (self.state == "TRAILING") and (self.opponent is not None):
            self.trailing_command = self.trailing_controller(self.waypoint_array_in_map[0, 2])
        else:
            self.i_gap = 0  # reset integral si no está trailing

        # Parámetros online y cotas por etapa
        for i in range(self.stmpc_config.N + 1):
            if (self.state == "TRAILING") and (self.opponent is not None):
                target_v_speed = self.trailing_command
            else:
                # la columna 4 es s (como en el original)
                idx = np.abs(self.waypoint_array_in_map[:, 4] - (self.mpc_sd[i, 0] % self.track_length)).argmin()
                target_v_speed = self.waypoint_array_in_map[i, 2]  # [idx, 2]
                self.overtake_d = 0
                if (self.state == "OVERTAKE") and (self.opponent is not None):
                    # columna 3: overtake lateral d
                    self.overtake_d = self.waypoint_array_in_map[idx, 3]

            if i < self.stmpc_config.steps_delay:
                multiplier = 1e6
                online_parameters = np.array([
                    target_v_speed,
                    self.stmpc_config.qadv,
                    self.stmpc_config.qv,
                    self.stmpc_config.qn,
                    self.stmpc_config.qalpha,
                    self.stmpc_config.qjerk,
                    multiplier * self.stmpc_config.qddelta,
                    self.stmpc_config.alat_max,
                    self.stmpc_config.a_min,
                    self.stmpc_config.a_max,
                    self.stmpc_config.track_safety_margin,
                    self.overtake_d,
                    self.tire_config.friction_coeff,
                    self.stmpc_config.terminal_multiplier
                ], dtype=np.float64)
            else:
                online_parameters = np.array([
                    target_v_speed,
                    self.stmpc_config.qadv,
                    self.stmpc_config.qv,
                    self.stmpc_config.qn,
                    self.stmpc_config.qalpha,
                    self.stmpc_config.qjerk,
                    self.stmpc_config.qddelta,
                    self.stmpc_config.alat_max,
                    self.stmpc_config.a_min,
                    self.stmpc_config.a_max,
                    self.stmpc_config.track_safety_margin,
                    self.overtake_d,
                    self.tire_config.friction_coeff,
                    self.stmpc_config.terminal_multiplier
                ], dtype=np.float64)

            self.acados_solver.set(i, "p", online_parameters)

            if i < self.stmpc_config.N:
                self.acados_solver.set(i, "lbu", np.array([-50, self.stmpc_config.ddelta_min]))
                self.acados_solver.set(i, "ubu", np.array([50, self.stmpc_config.ddelta_max]))
                # Cotas de (v, delta, a) en el estado i+1
                self.acados_solver.set(i + 1, "lbx",
                                       np.array([self.stmpc_config.v_min,
                                                 self.stmpc_config.delta_min,
                                                 self.stmpc_config.a_min]))
                self.acados_solver.set(i + 1, "ubx",
                                       np.array([self.stmpc_config.v_max,
                                                 self.stmpc_config.delta_max,
                                                 self.stmpc_config.a_max]))

        # Resolver OCP
        status = self.acados_solver.solve()
        if not (status == 0 or status == 2):
            # Re-aplica warm start como en el original
            self.apply_warm_start(pose_frenet=[self.fre_s, self.fre_d, self.fre_alpha])

        # Coste actual (si lo quieres usar / loguear)
        # current_cost = self.acados_solver.get_cost()

        # Salida
        self.u0 = self.acados_solver.get(0, "u")
        # Índice del estado "ACCEL" (coincide con StateIndex.ACCEL.value)
        self.prev_acc = self.acados_solver.get(0, "x")[StateIndex.ACCEL.value]
        delayed_index = self.stmpc_config.steps_delay + 1
        self.pred_x = self.acados_solver.get(delayed_index, "x")

        self.steering_angle = self.pred_x[5]
        self.speed = self.pred_x[3]
        # Integración de Euler por diferencia de frecuencias
        self.acceleration = self.prev_acc + self.u0[0] / self.controller_frequency
        self.prev_acc = self.acceleration
        self.jerk = self.u0[0]

        # Actualiza buffer de dirección
        self.steering_angle_buf[1:] = self.steering_angle_buf[:-1]
        self.steering_angle_buf[0] = self.pred_x[5]

        # Para visualización (trayectoria MPC en (s,d))
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
        warm_start = np.zeros((self.stmpc_config.N + 1, 10))  # TODO: hardcoded state space dimension
        # TODO setting the velocity to current actual velocity might remove slowing down
        warm_start[0] = np.array([pose_frenet[0], pose_frenet[1], pose_frenet[2],
                                 1.0, 0.0, 0.0, 0.0, const_acc, 0, const_steer_vel])
        for i in range(1, self.stmpc_config.N + 1):
            xdot = self._dynamics_of_car(0, warm_start[i - 1])
            warm_start[i] = warm_start[i - 1] + np.array(xdot) / self.stmpc_config.MPC_freq  # ugly euler integration
        return warm_start

    def update_warm_start(self) -> None:
        """
        Propaga los estados actuales y re-aplica el input anterior (Euler).
        """
        for i in range(self.stmpc_config.N + 1):
            current_state = self.acados_solver.get(i, "x")
            if i < self.stmpc_config.N:
                current_input = self.acados_solver.get(i, "u")
            else:
                current_input = np.zeros(self.model.n_u)

            state_with_input = np.concatenate((current_state, current_input))
            xdot = self._dynamics_of_car(0, state_with_input)
            updated_state = current_state + np.array(xdot)[:self.model.n_x] / self.controller_frequency
            self.acados_solver.set(i, "x", updated_state)

            if i < self.stmpc_config.N - 1:
                next_inputs = self.acados_solver.get(i + 1, "u")
                self.acados_solver.set(i, "u", next_inputs)

    def propagate_time_delay(self, states: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """
        RK45 forward propagation usando el modelo de acados.
        """
        x0 = np.concatenate((states, [inputs[0]], [0]), axis=0)
        # Puedes elegir usar self.stmpc_config.t_delay; en el original acaban forzando 0.0
        self.t_delay = 0.0

        solution = solve_ivp(
            self._dynamics_of_car,
            t_span=[0, self.t_delay],
            y0=x0,
            method='RK45',
            atol=1e-4,
            rtol=1e-4
        )

        sol = [x[-1] for x in solution.y]

        # Satura ángulo de dirección
        if abs(sol[5]) > self.constraint.delta_max:
            sol[5] = np.sign(sol[5]) * self.constraint.delta_max
        # Satura velocidad longitudinal
        if abs(sol[3]) > self.constraint.v_x_max:
            sol[3] = np.sign(sol[3]) * self.constraint.v_x_max
        # Velocidad mínima
        if abs(sol[3]) < self.constraint.v_x_min:
            sol[3] = self.constraint.v_x_min

        return np.array(sol)[:-2]

    def _dynamics_of_car(self, t, x0) -> list:
        '''
        Used for forward propagation. This function takes the dynamics from the acados model.
        '''

        s = x0[0]
        n = x0[1]
        theta = x0[2]
        v_x = x0[3]
        v_y = x0[4]
        delta = x0[5]
        yaw_rate = x0[6]
        accel = x0[7]
        jerk = x0[8]
        derDelta = x0[9]
        xdot = self.model.f_expl_func(
            s, n, theta, v_x, v_y, delta, yaw_rate, accel, jerk, derDelta, self.model_params.p)

        xdot = [float(xdot[0]), float(xdot[1]), float(xdot[2]), float(xdot[3]),
                float(xdot[4]), float(xdot[5]), float(xdot[6]), float(xdot[7]), 0.0, 0.0]

        return xdot

    def trailing_controller(self, global_speed: float) -> float:
        # opponent: [s, d, v_s, v_d, has_los] (mismo convenio que original)
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
        """
        Build track coordinates (left boundary, centerline, right boundary) from an
        ndarray of waypoints.

        Parameters
        ----------
        waypoints_data : np.ndarray
            Shape (N, >= 7) with columns:
            [x, y, vx, kappa, s, d_left, d_right]

        Returns
        -------
        res_coords : np.ndarray
            Shape (3, N, 2), stacked as:
            [boundaries_left, centerline, boundaries_right]
            where each slice is (N, 2) with XY points.
        """
        if waypoints_data.ndim != 2 or waypoints_data.shape[1] < 7:
            raise ValueError(
                "waypoints_data must be (N, >=7): [x, y, vx, kappa, s, d_left, d_right]"
            )

        # Extract columns we need
        xy = waypoints_data[:, :2].astype(float)            # (N, 2)
        d_left = waypoints_data[:, 5].astype(float).copy()  # (N,)
        d_right = waypoints_data[:, 6].astype(float).copy()  # (N,)

        # If the track is closed (last == first), drop the duplicated last point
        if np.allclose(xy[0], xy[-1], atol=1e-9):
            xy = xy[:-1]
            d_left = d_left[:-1]
            d_right = d_right[:-1]

        N = xy.shape[0]
        if N == 0:
            raise ValueError("Empty waypoint array after processing.")
        if N == 1:
            # Degenerate: just replicate centerline to both sides
            boundaries_left = xy.copy()
            boundaries_right = xy.copy()
            return np.stack([boundaries_left, xy, boundaries_right], axis=0)

        # Forward segment directions to next point
        next_xy = np.roll(xy, -1, axis=0)
        seg = next_xy - xy  # (N, 2)

        # If open track, the last segment should not wrap to the first
        open_track = not np.allclose(xy[0], xy[-1], atol=1e-9)
        if open_track:
            seg[-1] = xy[-1] - xy[-2]

        # Normalize robustly; fill invalid with previous valid or [1, 0]
        norms = np.linalg.norm(seg, axis=1)
        eps = 1e-12
        dir_xy = np.zeros_like(seg)
        valid = norms > eps
        dir_xy[valid] = (seg[valid].T / norms[valid]).T

        for i in range(N):
            if not valid[i]:
                dir_xy[i] = dir_xy[i - 1] if i > 0 else np.array([1.0, 0.0])

        # Left normal (+90° rotation): [-dy, dx]
        nL = np.column_stack((-dir_xy[:, 1], dir_xy[:, 0]))  # (N, 2)

        # Boundaries
        boundaries_left = xy + nL * d_left[:, None]
        boundaries_right = xy - nL * d_right[:, None]

        # Stack as required by SplineTrack(coords_direct=...)
        res_coords = np.stack([boundaries_left, xy, boundaries_right], axis=0)  # (3, N, 2)
        return res_coords

    MU_IDX = 12  # keep in class if you prefer

    def set_sim_mu(self, mu: float) -> None:
        """
        Update ONLY the simulator's parameter vector used in _dynamics_of_car().
        Does NOT touch the MPC solver's stage parameters.
        """
        # get current p as a flat numpy array
        p = np.array(self.model_params.p, dtype=float).reshape(-1).copy()
        p[self.MU_IDX] = float(mu)

        # write it back in the same type the model expects
        if _HAS_DM and isinstance(self.model_params.p, DM):
            self.model_params.p = DM(p)
        else:
            self.model_params.p = p
