"""Spatial, time-optimal reformulation of the single-track model.

Follows the minimum-time NMPC of

    R. Verschueren, M. Zanon, R. Quirynen, M. Diehl, "Time-optimal race car driving
    using an online exact Hessian based nonlinear MPC algorithm", ECC 2016,

on top of the path-parametric single-track model of

    D. Kloeser, T. Schoels, T. Sartor, A. Zanelli, G. Frison, M. Diehl, "NMPC for Racing
    Using a Singularity-Free Path-Parametric Model with Obstacle Avoidance", IFAC 2020,

which is the model ``forzaeth_mpc/stmpc/`` already runs.

The idea
--------
STMPC integrates over time and asks the car to track a speed profile the global planner
computed offline. The time-optimal formulation integrates over *distance* and asks for
the fastest way through the next ``s_horizon`` metres, with no reference speed at all.

Change the independent variable from ``t`` to the path coordinate ``s``. For any state
``x`` with time derivative ``xdot``, the chain rule gives

    dx/ds = xdot / sdot,   sdot = (v_x cos(theta) - v_y sin(theta)) / (1 - kappa * n)

and elapsed time becomes a state that integrates as ``dt/ds = 1 / sdot``. The horizon
then covers a fixed *distance*, and the traversal time of that distance is
``sum_i (ds / sdot_i)`` -- which is the stage cost. Minimising it is minimising lap
time directly, rather than minimising the tracking error against a precomputed proxy
for it.

What it costs
-------------
``1 / sdot`` is singular at both ends, and the two ends fail differently.

At ``sdot -> 0`` the cost blows up. A hard lower bound ``v_x >= v_min > 0`` and a hard
lower bound ``sdot >= s_dot_min > 0`` keep it away. There is no ``fmax`` guard inside
the model on purpose: it would flatten the gradient exactly where the solver needs it
to push the speed back up.

At ``sdot -> infinity`` the cost goes to *zero*, and that end is the dangerous one. It
is reached not by driving fast but by driving sideways: ``sdot`` has ``1 - kappa * n``
in its denominator, so a large enough ``n`` towards the centre of curvature makes the
predicted traversal time vanish. The track boundary is what rules that state out, but
it is a *soft* constraint here as everywhere else in this stack -- so left alone, the
solver buys the singularity and pays the slack penalty for it. Observed directly: the
first working build drove ``n`` to 8 m on a 1 m track and reported a horizon time near
zero. The hard upper bound ``sdot <= s_dot_max`` is what closes it; it bounds the
reward, so no amount of slack is worth paying for. That one constraint row is
deliberately *not* in ``idxsh``.

State and input
---------------
``x = [s, n, theta, v_x, v_y, yaw_rate, delta, accel, t]``, ``u = [ddelta, jerk]``. The
first eight entries are the single-track layout unchanged -- see ``SpatialStateIndex``.
``s`` stays a state with ``ds/ds = 1`` so the curvature spline can stay in the model.
"""

import types

import casadi as cs
import numpy as np
from casadi import MX, Function, fmod, interpolant, vertcat

from forzaeth_mpc.config import CarConfig, PacejkaTireConfig, TOMPCConfig
from forzaeth_mpc.stmpc.bicycle_model import single_track_dynamics


def spatial_model(s_coords: list, kapparef: list, d_left: list, d_right: list,
                  tompc_config: TOMPCConfig, car_config: CarConfig,
                  tire_config: PacejkaTireConfig,
                  model_name: str = "Spatialtimeoptimal_model"):
    """Builds the spatial time-optimal model, its cost and its constraints.

    ``model_name`` is the acados codegen name and must differ from every other
    controller's: acados exports to a cwd-relative ``c_generated_code/<name>*``, so two
    controllers sharing a name in one working directory silently load each other's
    compiled solver.
    """
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

    # states, see SpatialStateIndex
    s = MX.sym("s")
    n = MX.sym("n")
    theta = MX.sym("theta")
    v_x = MX.sym("v_x")
    v_y = MX.sym("v_y")
    yaw_rate = MX.sym("yaw_rate")
    delta = MX.sym("delta")
    accel = MX.sym("accel")
    t = MX.sym("t")
    x = vertcat(s, n, theta, v_x, v_y, yaw_rate, delta, accel, t)
    n_x = x.size()[0]

    # controls
    derDelta = MX.sym("derDelta")
    jerk = MX.sym("jerk")
    u = vertcat(derDelta, jerk)
    n_u = u.size()[0]

    z = vertcat([])

    # Pacejka coefficients stay online, as in the single-track model: retuning a tire
    # must not mean re-generating C code.
    Bf, Cf, Df, Ef = MX.sym("Bf"), MX.sym("Cf"), MX.sym("Df"), MX.sym("Ef")
    Br, Cr, Dr, Er = MX.sym("Br"), MX.sym("Cr"), MX.sym("Dr"), MX.sym("Er")
    MU = MX.sym("MU")

    # online parameters, layout in online_params.py
    V_target = MX.sym("V_target")
    weight_qtime = MX.sym("weight_qtime")
    weight_qv = MX.sym("weight_qv")
    weight_qn = MX.sym("weight_qn")
    weight_qalpha = MX.sym("weight_qalpha")
    weight_qvy = MX.sym("weight_qvy")
    weight_qjerk = MX.sym("weight_qjerk")
    weight_ddelta = MX.sym("weight_ddelta")
    a_lat_max = MX.sym("a_lat_max")
    a_long_min = MX.sym("a_long_min")
    a_long_max = MX.sym("a_long_max")
    safety_margin = MX.sym("safety_margin")
    overtake_d = MX.sym("overtake_d")
    terminal_multiplier = MX.sym("terminal_multiplier")
    weight_qv_terminal = MX.sym("weight_qv_terminal")

    p = vertcat(V_target, weight_qtime, weight_qv, weight_qn, weight_qalpha, weight_qvy,
                weight_qjerk, weight_ddelta,
                a_lat_max, a_long_min, a_long_max, safety_margin, overtake_d, MU,
                terminal_multiplier, weight_qv_terminal,
                Bf, Cf, Df, Ef, Br, Cr, Dr, Er)

    s_mod = fmod(s, pathlength)

    # the same tire and chassis equations STMPC and LB-MPC run
    dyn = single_track_dynamics(
        s, n, theta, v_x, v_y, yaw_rate, delta, accel, derDelta, jerk,
        kappa=kapparef_s(s_mod), tire=(Bf, Cf, Df, Ef, Br, Cr, Dr, Er), mu=MU,
        car_config=car_config, load_transfer=tompc_config.load_transfer)

    # --- the spatial transform -------------------------------------------------- #
    # dx/ds = xdot / sdot, and dt/ds = 1/sdot. No saturation on the division: the
    # hard bounds v_x >= v_min > 0 and sdot >= s_dot_min > 0 are what keep it finite,
    # and a fmax guard here would zero the gradient exactly where the solver needs it.
    s_dot = dyn.s_dot
    dt_ds = 1.0 / s_dot

    f_expl = vertcat(
        1.0,                        # ds/ds
        dyn.n_dot * dt_ds,
        dyn.theta_dot * dt_ds,
        dyn.v_x_dot * dt_ds,
        dyn.v_y_dot * dt_ds,
        dyn.yaw_rate_dot * dt_ds,
        dyn.delta_dot * dt_ds,
        dyn.accel_dot * dt_ds,
        dt_ds,                      # dt/ds
    )
    f_expl_func = Function("f_expl_func", [x, u, p], [f_expl])

    # The time-domain derivatives of the same state, for warm starts and for closing
    # the loop in the simulator, which advances in time and not in distance.
    f_expl_time = dyn.f_expl
    f_expl_time_func = Function(
        "f_expl_time_func",
        [vertcat(s, n, theta, v_x, v_y, yaw_rate, delta, accel), u, p], [f_expl_time])

    model.x0 = np.zeros(n_x)

    # --- cost ------------------------------------------------------------------- #
    # acados sums the EXTERNAL stage cost over the shooting nodes without scaling it by
    # the step. Multiplying the time term by ds makes the sum the horizon traversal
    # time in seconds, so weight_qtime = 1 means "one unit of cost per second" and
    # every other weight is read as a trade against seconds.
    ds = tompc_config.s_horizon / tompc_config.N

    tracking = (weight_qn * (n - overtake_d)**2
                + weight_qalpha * theta**2
                + weight_qvy * v_y**2
                + weight_qv * (v_x - V_target)**2)
    regularization = weight_qjerk * jerk**2 + weight_ddelta * derDelta**2

    model.cost_expr_ext_cost_0 = weight_qtime * dt_ds * ds + tracking + regularization
    model.cost_expr_ext_cost = weight_qtime * dt_ds * ds + tracking + regularization
    # Terminal: no input, and a bonus on the exit speed. The horizon ends mid-track, so
    # without it the last node has no reason to leave the car fast for the metres the
    # controller cannot see, and it trades exit speed for a cheaper final stage.
    model.cost_expr_ext_cost_e = (terminal_multiplier * tracking
                                  - weight_qv_terminal * v_x)

    # --- constraints ------------------------------------------------------------ #
    n_right_bound = n + right_bound_s(s_mod) - safety_margin
    n_left_bound = n - left_bound_s(s_mod) + safety_margin

    # The friction ellipse. A minimum-time objective sits on whichever limit binds, so
    # unlike a tracking controller this one is load-bearing, not a safety net.
    a_lat = dyn.v_y_dot
    a_long = dyn.v_x_dot
    if tompc_config.combined_constraints == "ellipse":
        combined_constraints = (a_lat / a_lat_max)**2 + (a_long / a_long_max)**2
    elif tompc_config.combined_constraints == "diamond":
        combined_constraints = cs.fabs(a_lat / a_lat_max) + cs.fabs(a_long / a_long_max)
    else:
        combined_constraints = MX(0)

    constraint.pathlength = pathlength
    # Row order matters: acados_settings softens rows 0..2 and leaves row 3, the sdot
    # bound, hard. See the module docstring for why that row cannot be soft.
    constraint.expr = vertcat(
        n_right_bound,
        n_left_bound,
        combined_constraints,
        s_dot,
    )
    constraint.expr_e = vertcat(
        n_right_bound,
        n_left_bound,
        s_dot,
    )
    constraint.n_soft = 3
    """How many leading rows of ``expr`` are soft. The rest are hard."""
    constraint.n_soft_e = 2

    # state bounds
    constraint.delta_min = tompc_config.delta_min
    constraint.delta_max = tompc_config.delta_max
    constraint.v_x_min = tompc_config.v_min
    constraint.v_x_max = tompc_config.v_max
    constraint.a_min = tompc_config.a_min
    constraint.a_max = tompc_config.a_max
    # the two bounds that keep 1/sdot finite and its reward bounded
    constraint.s_dot_min = tompc_config.s_dot_min
    constraint.s_dot_max = tompc_config.s_dot_max

    # input bounds
    constraint.ddelta_min = tompc_config.ddelta_min
    constraint.ddelta_max = tompc_config.ddelta_max
    constraint.jerk_min = tompc_config.jerk_min
    constraint.jerk_max = tompc_config.jerk_max

    params = types.SimpleNamespace()
    model.f_expl_expr = f_expl
    model.x = x
    model.n_x = n_x
    model.u = u
    model.n_u = n_u
    model.z = z
    model.p = p
    model.name = model_name
    model.params = params
    model.kappa = kapparef_s
    model.f_expl_func = f_expl_func
    model.f_expl_time_func = f_expl_time_func
    model.left_bound_s = left_bound_s
    model.right_bound_s = right_bound_s
    model.track_max = tompc_config.track_max_width
    model.ds = ds
    return model, constraint, params


if __name__ == "__main__":
    # Two probes, neither of which needs an acados build.
    #
    # 1. The state order, the same way stmpc/bicycle_model.py pins its own: feed
    #    distinguishable inputs and read the derivative back through the enum.
    # 2. The spatial transform itself: dx/ds must equal xdot/sdot row by row against the
    #    time-domain model, and dt/ds must equal 1/sdot. That is the one thing that is
    #    genuinely new here, and getting it wrong gives a plausible controller that
    #    optimises the wrong thing.
    from forzaeth_mpc.config import (load_car_config_ros,
                                     load_pacejka_tire_config_ros,
                                     load_TOMPC_config_ros)
    from forzaeth_mpc.tompc.online_params import build_online_params
    from forzaeth_mpc.utils.indices import InputIndex as II
    from forzaeth_mpc.utils.indices import SpatialStateIndex as SI

    n_wp = 32
    s_coords = np.linspace(0.0, 10.0, n_wp)
    # A real curvature profile, not a flat one: casadi's bspline fit returns nan for an
    # all-zero value vector, so a flat track silently makes every assertion below
    # compare nan against nan. It also exercises the curvature coupling in sdot.
    kappa = 0.2 * np.sin(2 * np.pi * s_coords / s_coords[-1])
    ones = np.ones(n_wp)
    tompc_config = load_TOMPC_config_ros("NUC5")
    car_config = load_car_config_ros("NUC5")
    tire_config = load_pacejka_tire_config_ros("NUC5", "dubi")
    model, constraint, _params = spatial_model(
        s_coords, kappa, ones, ones, tompc_config, car_config, tire_config)

    p_val = build_online_params(tompc_config, tire_config, target_v=3.0,
                                overtake_d=0.0, mu=1.0)
    assert model.p.shape[0] == len(p_val), (model.p.shape[0], len(p_val))

    x = np.zeros(model.n_x)
    x[SI.VELOCITY_V_X] = 3.0
    x[SI.ORIENTATION_THETA] = 0.1
    x[SI.VELOCITY_V_Y] = -0.2
    x[SI.YAW_RATE] = 0.3
    u = np.zeros(model.n_u)
    u[II.JERK] = 7.0
    u[II.D_STEERING_ANGLE] = 11.0

    dxds = np.asarray(model.f_expl_func(x, u, p_val)).reshape(-1)
    xdot = np.asarray(model.f_expl_time_func(x[:8], u, p_val)).reshape(-1)
    s_dot = xdot[SI.POS_ON_CENTER_LINE_S]

    assert dxds[SI.POS_ON_CENTER_LINE_S] == 1.0, "ds/ds must be exactly 1"
    assert abs(dxds[SI.TIME] - 1.0 / s_dot) < 1e-12, (
        f"dt/ds must be 1/sdot; got {dxds[SI.TIME]} vs {1.0 / s_dot}")
    for i in range(1, 8):
        assert abs(dxds[i] - xdot[i] / s_dot) < 1e-12, (
            f"row {i} ({SI(i).name}): dx/ds {dxds[i]} != xdot/sdot {xdot[i] / s_dot}")

    # the input rows of the time-domain model are still the bare inputs, which pins the
    # enum and the input order against the shared dynamics
    assert xdot[SI.ACCEL] == 7.0, f"accel_dot must be the jerk input, got {xdot[SI.ACCEL]}"
    assert xdot[SI.STEERING_ANGLE_DELTA] == 11.0, (
        f"delta_dot must be the steering-rate input, got {xdot[SI.STEERING_ANGLE_DELTA]}")

    # the traversal time of the horizon at a constant sdot is s_horizon / sdot, which is
    # what the cost has to be worth when qtime = 1
    assert abs(model.ds * tompc_config.N - tompc_config.s_horizon) < 1e-9

    print(f"OK: n_x={model.n_x} n_u={model.n_u} n_p={model.p.shape[0]} | "
          f"t@{int(SI.TIME)} | ds={model.ds:.3f} m over {tompc_config.s_horizon} m | "
          f"dx/ds == xdot/sdot on all {model.n_x} rows")
