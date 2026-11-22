#
# Copyright 2019 Gianluca Frison, Dimitris Kouzoupis, Robin Verschueren,
# Andrea Zanelli, Niels van Duijkeren, Jonathan Frey, Tommaso Sartor,
# Branimir Novoselnik, Rien Quirynen, Rezart Qelibari, Dang Doan,
# Jonas Koenemann, Yutao Chen, Tobias Schöls, Jonas Schlagenhauf, Moritz Diehl
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

import types

import casadi as cs
import numpy as np
from casadi import MX, Function, cos, fmod, interpolant, sin, vertcat
from forzaeth_mpc.config import CarConfig, PacejkaTireConfig, STMPCConfig

# author: Daniel Kloeser


def single_track_dynamics(s, n, theta, v_x, v_y, yaw_rate, delta, accel,
                          derDelta, jerk, *, kappa, tire, mu, car_config: CarConfig,
                          load_transfer: bool):
    """Time-domain derivatives of the single-track model with a Pacejka tire.

    Split out of ``bicycle_model`` so that the spatial, time-optimal reformulation in
    ``forzaeth_mpc/tompc/`` runs the same tire and chassis equations instead of a third
    copy of them. Two copies of the state enum and two copies of this model are what the
    package README lists as the source of its worst silent bugs.

    Every argument is a CasADi symbol or expression, except the two that select the
    equations:

    Args:
        kappa: reference curvature at ``s``. The caller owns the spline and the
            wrap-around, since the spatial model indexes it differently.
        tire: the eight Pacejka coefficients in the order ``(Bf, Cf, Df, Ef, Br, Cr,
            Dr, Er)``. Online symbols, not baked-in floats.
        mu: friction coefficient, also an online symbol.
        load_transfer: put the longitudinal load transfer into the axle loads.

    Returns:
        A namespace with the individual derivatives and ``f_expl``, the 8-row vector in
        the canonical state order ``[s, n, theta, v_x, v_y, yaw_rate, delta, accel]``.
    """
    lr, lf, m, Iz, h_cg = car_config.lr, car_config.lf, car_config.m, car_config.Iz, car_config.h_cg
    Bf, Cf, Df, Ef, Br, Cr, Dr, Er = tire
    g = 9.81

    # side slip angle
    v_0 = 0.5
    alpha_f = (cs.atan2((-v_y - (lf * yaw_rate)), v_x + v_0) + delta)
    alpha_r = (cs.atan2((-v_y + (lr * yaw_rate)), v_x + v_0))

    # load transfer
    if load_transfer:
        F_zf = m * ((-accel * h_cg) + (g * lr)) / (lf + lr)
        F_zr = m * ((accel * h_cg) + (g * lf)) / (lf + lr)
    else:
        F_zf = m * g * lr / (lf + lr)
        F_zr = m * g * lf / (lf + lr)

    # lateral tire forces
    F_yf = mu * Df * F_zf * sin(Cf * cs.atan(Bf * alpha_f - Ef * (Bf * alpha_f - cs.atan(Bf * alpha_f))))
    F_yr = mu * Dr * F_zr * sin(Cr * cs.atan(Br * alpha_r - Er * (Br * alpha_r - cs.atan(Br * alpha_r))))

    # motor force
    F_motor = accel * m
    motor_split_front = 0.5
    motor_split_rear = 1 - motor_split_front
    F_xf = motor_split_front * F_motor
    F_xr = motor_split_rear * F_motor

    # dynamics
    s_dot = ((v_x * cos(theta)) - (v_y * sin(theta))) / (1 - kappa * n)
    n_dot = v_x * sin(theta) + v_y * cos(theta)
    theta_dot = yaw_rate - (kappa * s_dot)
    v_x_dot = (1 / m) * (
        F_xr
        + F_xf * cos(delta)
        - F_yf * sin(delta)
        + m * v_y * yaw_rate
    )
    v_y_dot = (1 / m) * (
        F_yr
        + F_xf * sin(delta)
        + F_yf * cos(delta)
        - m * v_x * yaw_rate
    )
    yaw_rate_dot = (1 / Iz) * ((-F_yr * lr) + (F_yf * lf) * cos(delta))

    # delta inputs for smoother control
    delta_dot = derDelta
    accel_dot = jerk

    return types.SimpleNamespace(
        s_dot=s_dot, n_dot=n_dot, theta_dot=theta_dot,
        v_x_dot=v_x_dot, v_y_dot=v_y_dot, yaw_rate_dot=yaw_rate_dot,
        delta_dot=delta_dot, accel_dot=accel_dot,
        alpha_f=alpha_f, alpha_r=alpha_r,
        # the kinematic lateral-acceleration approximation, kept because the
        # ``correct_v_y_dot`` flag still selects it on some cars
        a_lat_kinematic=v_x * v_x * cs.tan(delta) / (lr + lf),
        f_expl=vertcat(s_dot, n_dot, theta_dot, v_x_dot, v_y_dot,
                       yaw_rate_dot, delta_dot, accel_dot),
    )


def bicycle_model(s_coords: list, kapparef: list, d_left: list, d_right: list,
                  stmpc_config: STMPCConfig, car_config: CarConfig, tire_config: PacejkaTireConfig,
                  model_name: str = "Spatialbicycle_model"):
    """The single-track model, shared by STMPC and LB-MPC.

    `model_name` is the acados codegen name. It must differ per consumer: acados exports to
    a cwd-relative `c_generated_code/<name>*`, so two controllers sharing a name in one
    working directory silently load each other's compiled solver -- which surfaced as
    `run_laps --algo stmpc kmpc lbmpc` dying with "mismatching dimension for field lbx with
    dimension 8 (you have 5)".
    """
    # define structs
    constraint = types.SimpleNamespace()
    model = types.SimpleNamespace()


    length = len(s_coords)
    pathlength = s_coords[-1]

    # copy loop to beginning and end
    s_coords = np.concatenate([s_coords[:-1] - pathlength, s_coords, pathlength + s_coords[1:length]])
    kapparef = np.concatenate([kapparef[:-1], kapparef, kapparef[1:length]])

    d_left = np.concatenate([d_left[:-1], d_left, d_left[1:length]])
    d_right = np.concatenate([d_right[:-1], d_right, d_right[1:length]])

    # compute spline interpolations
    kapparef_s = interpolant("kapparef_s", "bspline", [s_coords], kapparef)
    left_bound_s = interpolant("left_bound_s", "bspline", [s_coords], d_left)
    right_bound_s = interpolant("right_bound_s", "bspline", [s_coords], d_right)

    # CasADi Model
    # set up states & controls
    s = MX.sym("s")
    n = MX.sym("n")
    theta = MX.sym("theta")
    v_x = MX.sym("v_x")
    v_y = MX.sym("v_y")
    yaw_rate = MX.sym("yaw_rate")
    delta = MX.sym("delta")
    accel = MX.sym("accel")
    x = vertcat(s, n, theta, v_x, v_y, yaw_rate, delta, accel)
    n_x = x.size()[0]

    # controls
    derDelta = MX.sym("derDelta")
    jerk = MX.sym("jerk")
    u = vertcat(derDelta, jerk)
    n_u = u.size()[0]

    # xdot
    s_dot = MX.sym("sdot")
    n_dot = MX.sym("ndot")
    theta_dot = MX.sym("thetadot")
    v_x_dot = MX.sym("v_x_dot")
    v_y_dot = MX.sym("v_y_dot")
    yaw_rate_dot = MX.sym("yaw_ratedot")
    delta_dot = MX.sym("deltadot")
    accel_dot = MX.sym("accel_dot")

    # algebraic variables
    z = vertcat([])

    # parameters
    freq = stmpc_config.MPC_freq

    lr = car_config.lr
    lf = car_config.lf
    m = car_config.m
    Iz = car_config.Iz
    h_cg = car_config.h_cg

    friction_coeff = tire_config.friction_coeff
    # Online, matching stmpc/bicycle_model.py: baking these in meant re-generating C
    # code to retune a tire, and it is what kept LBMPC's p vector at 14 entries while
    # STMPC moved to 22 (race_stack 8382de7d0).
    Bf = MX.sym("Bf")
    Cf = MX.sym("Cf")
    Df = MX.sym("Df")
    Ef = MX.sym("Ef")
    Br = MX.sym("Br")
    Cr = MX.sym("Cr")
    Dr = MX.sym("Dr")
    Er = MX.sym("Er")
    g = 9.81

    # # online parameters
    V_target = MX.sym('V_target')
    weight_adv = MX.sym('weight_adv')
    weight_qn = MX.sym('weight_qn')
    weight_qv = MX.sym('weight_qv')
    weight_qalpha = MX.sym('weight_qalpha')
    weight_qjerk = MX.sym('weight_qjerk')
    weight_ddelta = MX.sym('weight_ddelta')
    a_lat_max = MX.sym('a_lat_max')
    a_long_min = MX.sym('a_long_min')
    a_long_max = MX.sym('a_long_max')
    safety_margin = MX.sym('safety_margin')
    overtake_d = MX.sym('overtake_d')

    # --- NEW: μ as an online parameter ---
    MU = MX.sym('MU')
    terminal_multiplier = MX.sym('terminal_multiplier')

    p = vertcat(V_target, weight_adv, weight_qv, weight_qn, weight_qalpha, weight_qjerk, weight_ddelta,  # weights
                a_lat_max, a_long_min, a_long_max, safety_margin, overtake_d, MU, terminal_multiplier,
                Bf, Cf, Df, Ef, Br, Cr, Dr, Er)  # lateral acceleration constraints

    # calculate the index
    s_mod = fmod(s, pathlength)

    # single track model definition with pacejka tire model, shared with the spatial
    # reformulation in forzaeth_mpc/tompc/
    dyn = single_track_dynamics(
        s, n, theta, v_x, v_y, yaw_rate, delta, accel, derDelta, jerk,
        kappa=kapparef_s(s_mod), tire=(Bf, Cf, Df, Ef, Br, Cr, Dr, Er), mu=MU,
        car_config=car_config, load_transfer=stmpc_config.load_transfer)
    s_dot = dyn.s_dot
    v_x_dot = dyn.v_x_dot
    v_y_dot = dyn.v_y_dot

    # continuous dynamics
    f_expl = dyn.f_expl
    # Takes the vectors, not loose scalars: a positional scalar signature silently
    # accepts a caller that has delta and yaw_rate the wrong way round, which is exactly
    # what happened when the two single-track models were merged.
    f_expl_func = Function('f_expl_func', [x, u, p], [f_expl])

    # Define initial conditions
    model.x0 = np.zeros(n_x)

    model.cost_expr_ext_cost_0 = weight_qn * (n - overtake_d)**2 + weight_qalpha * theta**2 + weight_qv * \
        (v_x - V_target)**2 + weight_qjerk * jerk**2 + weight_ddelta * derDelta**2
    model.cost_expr_ext_cost = weight_qn * (n - overtake_d)**2 + weight_qalpha * theta**2 + weight_qv * \
        (v_x - V_target)**2 + weight_qjerk * jerk**2 + weight_ddelta * derDelta**2
    model.cost_expr_ext_cost_e = terminal_multiplier * \
        (weight_qn * (n - overtake_d)**2 + weight_qalpha * theta**2 + weight_qv * (v_x - V_target)**2)
    if stmpc_config.vy_minimization:
        model.cost_expr_ext_cost_0 += 100 * v_y**2
        model.cost_expr_ext_cost += 100 * v_y**2
        model.cost_expr_ext_cost_e += terminal_multiplier * (100 * v_y**2)
    if stmpc_config.adv_maximization:
        model.cost_expr_ext_cost_0 -= weight_adv * s_dot / freq
        model.cost_expr_ext_cost -= weight_adv * s_dot / freq
        model.cost_expr_ext_cost_e -= terminal_multiplier * weight_adv * s_dot / freq

    # constraint on lateral errors
    n_right_bound = n + right_bound_s(s_mod) - safety_margin
    n_left_bound = n - left_bound_s(s_mod) + safety_margin

    # nonlinear constraints
    if stmpc_config.correct_v_y_dot:
        a_lat = v_y_dot
    else:
        a_lat = dyn.a_lat_kinematic
    constraint.pathlength = pathlength
    # TODO: introduce asymmetric constraints (now you can just over constrain breaking) with a_long_min
    if stmpc_config.combined_constraints == "ellipse":
        combined_constraints = (a_lat / a_lat_max)**2 + (v_x_dot / a_long_max)**2
    elif stmpc_config.combined_constraints == "diamond":
        combined_constraints = cs.fabs(a_lat / a_lat_max) + cs.fabs(v_x_dot / a_long_max)
    else:
        combined_constraints = 0
    # TODO fix the n constraint with a single constraint
    # NOTE: easiest when non-terminal constraints have the extra constraints
    # only at the end, otherwise need to change acados_settings.py
    constraint.expr = vertcat(
        n_right_bound,
        n_left_bound,
        combined_constraints,
        s_dot
    )
    constraint.expr_e = vertcat(
        n_right_bound,
        n_left_bound
    )

    # read the parameters range from the yaml file

    # state bounds
    constraint.delta_min = stmpc_config.delta_min  # minimum steering angle [rad]
    constraint.delta_max = stmpc_config.delta_max  # maximum steering angle [rad]
    constraint.v_x_min = stmpc_config.v_min  # minimum velocity [m/s]
    constraint.v_x_max = stmpc_config.v_max  # maximum velocity [m/s]
    constraint.a_min = stmpc_config.a_min  # minimum velocity change rate
    constraint.a_max = stmpc_config.a_max  # maximum velocity change rate

    # input bounds
    constraint.ddelta_min = stmpc_config.ddelta_min  # minimum change rate of stering angle [rad/s]
    constraint.ddelta_max = stmpc_config.ddelta_max  # maximum change rate of steering angle [rad/s]
    constraint.jerk_min = stmpc_config.jerk_min  # minimum jerk [m/s^3]
    constraint.jerk_max = stmpc_config.jerk_max  # maximum jerk [m/s^3]

    # Define model struct
    params = types.SimpleNamespace()
    # model.f_impl_expr = xdot - f_expl
    model.f_expl_expr = f_expl
    model.x = x
    model.n_x = n_x
    # model.xdot = xdot
    model.u = u
    model.n_u = n_u
    model.z = z
    model.p = p
    model.name = model_name
    model.params = params
    model.kappa = kapparef_s
    model.f_expl_func = f_expl_func
    model.left_bound_s = left_bound_s
    model.right_bound_s = right_bound_s
    model.track_max = stmpc_config.track_max_width
    # model.safety_width = stmpc_config.track_safety_margin
    return model, constraint, params


if __name__ == "__main__":
    # Order probe. `delta_dot = derDelta` and `accel_dot = jerk` are the only two rows of
    # f_expl that are a bare input, so feeding distinguishable inputs and reading the
    # derivative back through the enum pins the state row order, the input order and the
    # enum against the model at once. No acados build, runs in under a second.
    #
    # This is the check that makes the state permutation safe to do at all: greps catch
    # stale references, but only this catches the enum and the model drifting apart.
    import casadi as _cs
    import numpy as _np

    from forzaeth_mpc.config import (load_car_config_ros,
                                     load_pacejka_tire_config_ros, load_STMPC_config_ros)
    from forzaeth_mpc.utils.indices import InputIndex as _II, StateIndex as _SI

    n = 32
    # the builder does array arithmetic on these, so ndarrays not lists
    s_coords = _np.linspace(0.0, 10.0, n)
    # A curved reference, not a flat one: casadi's bspline fit returns nan for an
    # all-zero value vector, so a zero curvature makes kapparef_s(s) nan and every
    # derivative that touches it nan too.
    kappa = 0.2 * _np.sin(2 * _np.pi * s_coords / s_coords[-1])
    ones = _np.ones(n)
    model, _constraint, _params = bicycle_model(
        s_coords, kappa, ones, ones,
        load_STMPC_config_ros("NUC5"), load_car_config_ros("NUC5"),
        load_pacejka_tire_config_ros("NUC5", "dubi"))

    f = _cs.Function("f", [model.x, model.u, model.p], [model.f_expl_expr])
    x = _np.zeros(model.n_x); x[_SI.VELOCITY_V_X] = 3.0
    u = _np.zeros(model.n_u)
    u[_II.JERK] = 7.0
    u[_II.D_STEERING_ANGLE] = 11.0
    xdot = _np.asarray(f(x, u, _np.ones(model.p.shape[0]))).reshape(-1)

    assert xdot[_SI.ACCEL] == 7.0, (
        f"accel_dot must be the jerk input; got {xdot[_SI.ACCEL]} at index "
        f"{int(_SI.ACCEL)} -- the enum and the model disagree")
    assert xdot[_SI.STEERING_ANGLE_DELTA] == 11.0, (
        f"delta_dot must be the steering-rate input; got "
        f"{xdot[_SI.STEERING_ANGLE_DELTA]} at index "
        f"{int(_SI.STEERING_ANGLE_DELTA)} -- the enum and the model disagree")

    print(f"OK: n_x={model.n_x} n_u={model.n_u} n_p={model.p.shape[0]} | "
          f"delta@{int(_SI.STEERING_ANGLE_DELTA)} yaw_rate@{int(_SI.YAW_RATE)} "
          f"accel@{int(_SI.ACCEL)} | jerk@{int(_II.JERK)} "
          f"ddelta@{int(_II.D_STEERING_ANGLE)}")
