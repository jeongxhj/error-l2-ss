#! /usr/bin/env python3
from typing import Tuple
import numpy as np
from scipy.integrate import solve_ivp
from forzaeth_mpc.utils.frenet_converter import FrenetConverter
from forzaeth_mpc.utils.splinify import SplineTrack
from forzaeth_mpc.config import KMPCConfig, CarConfig, TrailingConfig
from forzaeth_mpc.kmpc.acados_settings import acados_settings


class Kinematic_MPC_Controller:
    def __init__(self, waypoints_data: np.ndarray, racecar_version: str, kmpc_config: KMPCConfig, car_config: CarConfig, trailing_config: TrailingConfig) -> None:
        """
        Initialise MPC object.

        Input:  conf_file   : String containing the path to the param_config.yaml file
        """
        # Init the parameters
        self.racecar_version = racecar_version
        self.kmpc_config: KMPCConfig = kmpc_config
        self.car_config: CarConfig = car_config
        self.trailing_config: TrailingConfig = trailing_config
        self.mpc_init_params()

        # Init the solver with waypoints data (no ROS dependency)
        self.mpc_initialize_solver(waypoints_data)

    def mpc_init_params(self) -> None:
        # MPC Params
        self.t_MPC = 1 / self.kmpc_config.MPC_freq
        self.bound_inflation = self.kmpc_config.track_safety_margin
        # time delay propagation
        self.t_delay = self.kmpc_config.t_delay + self.t_MPC

        # seering angle buffer size
        buf_size = 2
        self.steering_angle_buf = np.zeros(buf_size)

        # Initial state
        self.mpc_sd = np.zeros((self.kmpc_config.N + 1, 2))
        self.u0 = np.zeros(2)
        self.fre_s = 0
        # Initial time
        self.conp_time = 0
        # Initial speed
        self.speed = 0

        # Initial overtake lateral distance
        self.overtake_d = self.kmpc_config.overtake_d
        self.loop_rate = self.kmpc_config.MPC_freq
        self.gap = None
        self.gap_should = None
        self.gap_error = None
        self.gap_actual = None
        self.v_diff = None
        self.i_gap = 0

    def mpc_initialize_solver(self, waypoints_data: np.ndarray) -> None:
        """Initialises the controller from a waypoints array (no ROS dependency).

        Input: waypoints_data: np.ndarray: Shape (N, 7) with columns
            [x_m, y_m, vx_mps, kappa_radpm, s_m, d_left, d_right].
        """

        x = waypoints_data[:, 0]
        y = waypoints_data[:, 1]
        self.fren_conv = FrenetConverter(x, y)

        # on f track trajectory is 81.803 m long.
        d_left, coords_path, d_right = self._transform_waypoints_to_coords_from_array(waypoints_data)

        self.spline = SplineTrack(coords_direct=coords_path)

        self.nr_laps = 0

        kapparef = waypoints_data[:, 3]
        vx_ref = waypoints_data[:, 2]
        self.s_ref = waypoints_data[:, 4]
        self.constraint, self.model, self.acados_solver, self.model_params = acados_settings(
            self.s_ref, kapparef, vx_ref, d_left, d_right, self.kmpc_config, self.car_config)

        self.kappa = kapparef


    def apply_warm_start(self, pose_frenet: np.ndarray):
        """Applies a warm start to the MPC solver.

        Args:
            pose_frenet (np.ndarray): _description_
        """
        warm_start_trajectory = self.get_warm_start(pose_frenet=pose_frenet, const_v=1, const_steer_vel=0.0)
        for i in range(self.kmpc_config.N + 1):
            self.acados_solver.set(i, "x", warm_start_trajectory[i][:self.model.n_x])
            if i < self.kmpc_config.N:
                self.acados_solver.set(i, "u", warm_start_trajectory[i][self.model.n_x:])

    def main_loop(
            self,
            state,
            position_in_map,
            waypoint_array_in_map,
            speed_now,
            opponent,
            position_in_map_frenet,
            alpha,
            acc_now,
            track_length,
            compute_time):
        """
        Main loop for the kinematic model predictive controller (MPC).

        This function updates the controller's parameters, computes the initial state,
        sets up the MPC problem, dynamically adjusts constraints and target parameters,
        solves the optimal control problem (OCP), and retrieves the solution.

        Args:
            state (str): Current state of the controller, e.g., "TRAILING" or "OVERTAKE".
            position_in_map (np.ndarray): Current position of the vehicle in the map.
            waypoint_array_in_map (np.ndarray): Array of waypoints in the map.
            speed_now (float): Current speed of the vehicle.
            opponent (object): Information about the opponent vehicle, if any.
            position_in_map_frenet (np.ndarray): Current position of the vehicle in Frenet coordinates.
            alpha (float): Current orientation angle of the vehicle.
            acc_now (float): Current acceleration of the vehicle.
            track_length (float): Total length of the track.
            compute_time (float): Time taken for computation.

        Returns:
            tuple: A tuple containing:
                - speed (float): Predicted speed of the vehicle.
                - acceleration (float): Predicted acceleration of the vehicle.
                - jerk (float): Predicted jerk of the vehicle.
                - steering_angle (float): Predicted steering angle of the vehicle.
                - states (list or None): Predicted states of the vehicle over the MPC horizon, or None if the solver fails.

        Notes:
            - The function dynamically adjusts the target speed, weight parameters, and constraints
                based on the current state and opponent information.
            - If the solver fails, a warm start is applied to attempt recovery.
            - Visualization data is generated for the predicted positions along the MPC horizon.
            - Some parameters are hardcoded and should ideally be configurable via a YAML file.
        """
        # Updating parameters from manager
        self.state = state
        self.position_in_map = position_in_map
        self.waypoint_array_in_map = waypoint_array_in_map
        self.speed_now = speed_now
        self.opponent = opponent
        self.position_in_map_frenet = position_in_map_frenet
        self.alpha = alpha
        self.acc_now = acc_now
        self.track_length = track_length
        self.conp_time = compute_time
        # update initial state from sensors
        track_length = self.spline.track_length - 0.1  # Needed as values are not exact
        center_car_s = self.position_in_map_frenet[0] + self.car_config.lr * np.cos(self.alpha)
        deriv_center = self.spline.get_derivative(center_car_s)
        # alpha is relative to the track at the car's s: shift it by the track's turn up to the center of mass
        deriv_s = self.spline.get_derivative(self.position_in_map_frenet[0])
        alpha_center = self.alpha + np.arctan2(deriv_s[1], deriv_s[0]) - np.arctan2(deriv_center[1], deriv_center[0])
        # make aplha_center between -pi and pi
        alpha_center = alpha_center % (2 * np.pi)
        if alpha_center > np.pi:
            alpha_center = alpha_center - 2 * np.pi

        if self.position_in_map_frenet[0] < 0.2 and self.fre_s // track_length != self.nr_laps:
            self.nr_laps = self.fre_s // track_length

        current_pos_s = self.position_in_map_frenet[0] + self.nr_laps * self.spline.track_length

        self.fre_s = current_pos_s
        self.fre_d = self.position_in_map_frenet[1]
        self.fre_alpha = alpha_center

        x0 = np.array([self.fre_s, self.fre_d, self.fre_alpha, self.speed,
                      self.steering_angle_buf[-1]], dtype=np.float64)

        # time delat compensation
        self.t_delay = self.conp_time
        propagated_x = self.propagate_time_delay(x0, self.u0)
        propagated_x = x0
        # set the initial state for the mpc
        self.acados_solver.set(0, "lbx", propagated_x)
        self.acados_solver.set(0, "ubx", propagated_x)

        # dynamically change the target speed, weight parameters and constraints
        for i in range(self.kmpc_config.N + 1):
            # head to head racing
            # for the trailing, MPC just needs to change the reference speed (get from the trailing controller)
            if (self.state == "TRAILING" and (self.opponent is not None)):  # Trailing controller
                target_v_speed = self.trailing_controller(self.waypoint_array_in_map[0, 2])
            else:
                idx = np.abs(self.waypoint_array_in_map[:,4]  - self.mpc_sd[i,0]%self.spline.track_length).argmin()
                target_v_speed=self.waypoint_array_in_map[idx,2]
                self.overtake_d = 0
                # head to head overtaking
                if (self.state == "OVERTAKE" and (self.opponent is not None)):
                    self.overtake_d = self.waypoint_array_in_map[idx, 3]

            online_parameters = np.array([target_v_speed,
                                          self.kmpc_config.qadv,
                                          self.kmpc_config.qv,
                                          self.kmpc_config.qn,
                                          self.kmpc_config.qalpha,
                                          self.kmpc_config.qac,
                                          self.kmpc_config.qddelta,
                                          self.kmpc_config.alat_max,
                                          self.kmpc_config.track_safety_margin,
                                          self.overtake_d],
                                         dtype=np.float64)

            self.acados_solver.set(i, "p", online_parameters)

            if i < self.kmpc_config.N:
                self.acados_solver.set(i, "lbu", np.array([self.kmpc_config.a_min, self.kmpc_config.ddelta_min]))
                self.acados_solver.set(i, "ubu", np.array([self.kmpc_config.a_max, self.kmpc_config.ddelta_max]))
                # do not change i+1 to i since mpc needs the initial state
                self.acados_solver.set(i + 1, "lbx", np.array([self.kmpc_config.v_min, self.kmpc_config.delta_min]))
                self.acados_solver.set(i + 1, "ubx", np.array([self.kmpc_config.v_max, self.kmpc_config.delta_max]))

        # Solve OCP
        status = self.acados_solver.solve()
        if not (status == 0 or status == 2): #0 is sucess 2 is max iter, necesary for SQP
            # if the solver fails, apply warm start again
            # the initial warm start is a bit rough, but it does the job
            print("Solver failed, applying warm start")
            self.apply_warm_start(pose_frenet=[self.fre_s, self.fre_d, self.fre_alpha])

        # get solution
        # time delay step for the prediction
        # todo make this a parameter in the yaml file
        # TODO: remove hardcoding!!!
        time_delay_step = 3
        self.u0 = self.acados_solver.get(0, "u")
        self.pred_x = self.acados_solver.get(time_delay_step, "x")
        self.steering_angle = self.pred_x[4]
        self.speed = self.pred_x[3]
        self.acceleration = 0 * self.u0[0]
        self.jerk = 0
        # steering buffer
        # shift the elements in the buffer to the right and add the predicted steering angle
        self.steering_angle_buf[1:] = self.steering_angle_buf[:-1]
        self.steering_angle_buf[0] = self.pred_x[-1]
        self.states = None

        ##### visualization #####
        # Creating waypoint array with predicted positions
        self.mpc_sd = np.array([
            self.acados_solver.get(j, "x")[:2] for j in range(self.kmpc_config.N + 1)
        ])
        if status == 0 or status == 2: #0 is success 2 is max iter, necesary for SQP
            states = []
            for node in range(self.kmpc_config.N + 1):
                result = self.acados_solver.get(node, "x").tolist()
                for item in result:
                    states.append(item)
            self.states = states
        return self.speed, self.acceleration, self.jerk, self.steering_angle, self.states

    #############
    # Utilities #
    #############
    def get_warm_start(self, pose_frenet: np.ndarray, const_v: float, const_steer_vel: float) -> np.array:
        """
        Returns a warm start trajectory for the MPC. This is done by propagating the current state with a constant velocity and steering angle.

        Input:  const_v   : Constant velocity
                const_steer : Constant steering angle

        Returns: np.array    : Warm start trajectory
        """
        warm_start = np.zeros((self.kmpc_config.N + 1, 7))
        warm_start[0] = np.array([pose_frenet[0], pose_frenet[1], pose_frenet[2],
                                 1, const_steer_vel, 0, 0])
        for i in range(1, self.kmpc_config.N + 1):
            der_state = self._dynamics_of_car(0, warm_start[i - 1])
            warm_start[i] = warm_start[i-1] + np.array(der_state) / self.kmpc_config.MPC_freq  # ugly euler integration
        return warm_start

    def _transform_waypoints_to_coords_from_array(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Helper function to store the waypoints array into the right format such that they can be used for initialisation of SplineTrack.

        Input: data: np.ndarray: Shape (N, >=7) with columns [x_m, y_m, vx_mps, kappa_radpm, s_m, d_left, d_right].

        Returns: Tuple: Stores the waypoints as follows: Left boundaries, reference path, right boundaries
        """
        waypoints = data[:, :2]
        d_left = data[:, 5]
        d_right = data[:, 6]
        waypoints = waypoints[:-1] # remove last point as it is equal to the first point
        track_direction = np.roll(waypoints, -1, axis=0) - waypoints
        track_direction = track_direction / np.linalg.norm(track_direction, axis=1)[:, None]
        # obtain left and right boundaries by shifting the waypoints in the direction perpendicular to the track direction
        # and multiplying with the distance to the left and right boundaries
        boundaries_left = waypoints + np.column_stack((-track_direction[:, 1], track_direction[:, 0])) * d_left[:-1, None]
        boundaries_right = waypoints + np.column_stack((track_direction[:, 1], -track_direction[:, 0])) * d_right[:-1, None]
        res_coords = np.stack([boundaries_left, waypoints, boundaries_right], axis=0)

        return d_left, res_coords, d_right

    def propagate_time_delay(self, states: np.array, inputs: np.array) -> np.array:
        """
        RK45 forward propagation. Calls the function _dynamics_of_car which contains the dynamics of the car.

        Input:  np.array    : Holds the current state which needs to be propagated.
                inputs      : Holds the current inputs from the MPC.

        Returns: np.array   : Propagated state without input.

        """

        # Initial condition on the ODE
        x0 = np.concatenate((states, inputs), axis=0)
        # forward propagation
        solution = solve_ivp(
            self._dynamics_of_car,
            t_span=[
                0,
                self.t_delay],
            y0=x0,
            method='RK45',
            atol=1e-8,
            rtol=1e-8)

        solution = [x[-1] for x in solution.y]

        # Constraint on max. steering angle
        if abs(solution[4]) > self.model.delta_max:
            solution[4] = np.sign(solution[4]) * self.model.delta_max

        # Constraint on max. vel
        if abs(solution[3]) > self.constraint.v_max:
            solution[3] = np.sign(solution[3]) * self.constraint.v_max

        # On minimum speed
        if abs(solution[3]) < self.constraint.v_min:
            solution[3] = self.constraint.v_min

        # Only get the state as solution of where the car will be in t_delay seconds
        # not including the input
        return np.array(solution)[:-2]

    def _dynamics_of_car(self, t, x0) -> list:
        '''
        Used for forward propagation. This function takes the dynamics from the acados model.
        '''

        s = x0[0]
        n = x0[1]
        alpha = x0[2]
        v = x0[3]
        delta = x0[4]
        derv = x0[5]
        derDelta = x0[6]

        xdot = self.model.f_expl_func(s, n, alpha, v, delta, derv, derDelta, self.model_params.p)

        # last two states are 0 assuming constant input
        xdot = [float(xdot[0]), float(xdot[1]), float(xdot[2]), float(xdot[3]), float(xdot[4]), 0, 0]

        return xdot

    def trailing_controller(self, global_speed):
        self.gap = (self.opponent[0] - self.position_in_map_frenet[0]) % self.track_length  # gap to opponent
        # prev_gap = self.gap if self.gap_actual is None else self.gap_actual
        self.gap_actual = self.gap
        if self.trailing_config.trailing_mode:
            self.gap_should = self.trailing_config.trailing_gap
        else:
            self.gap_should = self.position_in_map_frenet[2] * self.trailing_config.trailing_gap

        self.gap_error = self.gap_should - self.gap_actual
        self.v_diff = self.position_in_map_frenet[2] - self.opponent[2]
        self.i_gap = np.clip(self.i_gap + self.gap_error / self.loop_rate, -10, 10)

        p_value = self.gap_error * self.trailing_config.trailing_p_gain
        d_value = self.v_diff * self.trailing_config.trailing_d_gain
        i_value = self.i_gap * self.trailing_config.trailing_i_gain

        self.trailing_command = np.clip(self.opponent[2] - p_value - i_value - d_value, 0, global_speed)
        if not self.opponent[4] and self.gap_actual > self.gap_should:
            self.trailing_command = max(self.trailing_config.blind_trailing_speed, self.trailing_command)

        return self.trailing_command