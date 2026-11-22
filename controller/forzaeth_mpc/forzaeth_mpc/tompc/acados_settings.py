"""acados OCP for the time-optimal MPC.

Mirrors ``stmpc/acados_settings.py``: same solver options, same slack layout, same
nonlinear-constraint plumbing. Three things differ, all of them because the independent
variable is distance and not time:

- ``solver_options.tf`` is the horizon *length in metres*, so the shooting grid is a
  distance grid and ``tf / N`` is the node spacing ``ds``.
- the lower bound on the ``sdot`` constraint row is ``s_dot_min > 0``, not 0. That row
  was a formality for STMPC; here it is what keeps ``1/sdot`` finite.
- the terminal node keeps its input bounds off, since the terminal cost has no input.
"""

from typing import Tuple

import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

from forzaeth_mpc.config import CarConfig, PacejkaTireConfig, TOMPCConfig
from forzaeth_mpc.utils.indices import SpatialStateIndex

from .online_params import default_parameter_values
from .spatial_model import spatial_model


def acados_settings(s0, kapparef, d_left, d_right,
                    tompc_config: TOMPCConfig, car_config: CarConfig,
                    tire_config: PacejkaTireConfig) -> Tuple:
    ocp = AcadosOcp()

    model, constraint, params = spatial_model(s0, kapparef, d_left, d_right,
                                              tompc_config, car_config, tire_config)

    model_ac = AcadosModel()
    model_ac.f_expl_expr = model.f_expl_expr
    model_ac.con_h_expr_0 = constraint.expr
    model_ac.con_h_expr = constraint.expr
    model_ac.con_h_expr_e = constraint.expr_e
    model_ac.x = model.x
    model_ac.u = model.u
    model_ac.z = model.z
    model_ac.p = model.p
    model_ac.name = model.name
    ocp.model = model_ac

    p = default_parameter_values(tompc_config, tire_config)
    params.p = p
    ocp.parameter_values = p

    nx = model.n_x
    nu = model.n_u

    # Only the leading rows are soft. The trailing sdot row stays hard: it is what
    # bounds the reward of the 1/sdot cost, and a soft bound can be bought with slack.
    ns = constraint.n_soft
    ns_e = constraint.n_soft_e

    ocp.dims.N = tompc_config.N
    ocp.dims.nx = nx
    ocp.dims.nu = nu
    ocp.dims.ny = 0
    ocp.dims.ny_e = 0

    # external cost, defined in spatial_model.py
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = model.cost_expr_ext_cost
    ocp.model.cost_expr_ext_cost_e = model.cost_expr_ext_cost_e

    # Slack penalty is Zl*sl**2 + zl*sl (and the same for the upper side).
    # at stage 0
    ocp.cost.Zl_0 = tompc_config.Zl * np.ones((ns,))
    ocp.cost.Zu_0 = tompc_config.Zu * np.ones((ns,))
    ocp.cost.zl_0 = tompc_config.zl * np.ones((ns,))
    ocp.cost.zu_0 = tompc_config.zu * np.ones((ns,))
    # from 1 to N-1
    ocp.cost.Zl = tompc_config.Zl * np.ones((ns,))
    ocp.cost.Zu = tompc_config.Zu * np.ones((ns,))
    ocp.cost.zl = tompc_config.zl * np.ones((ns,))
    ocp.cost.zu = tompc_config.zu * np.ones((ns,))
    # at stage N
    ocp.cost.Zl_e = tompc_config.Zl * np.ones((ns_e,))
    ocp.cost.Zu_e = tompc_config.Zu * np.ones((ns_e,))
    ocp.cost.zl_e = tompc_config.zl * np.ones((ns_e,))
    ocp.cost.zu_e = tompc_config.zu * np.ones((ns_e,))

    # input constraint. Not set on the terminal node: the terminal cost has no input,
    # so bounding one there only adds a decision variable the QP has to carry.
    ocp.constraints.lbu_0 = np.array([constraint.ddelta_min, constraint.jerk_min])
    ocp.constraints.ubu_0 = np.array([constraint.ddelta_max, constraint.jerk_max])
    ocp.constraints.idxbu_0 = np.array([0, 1])

    ocp.constraints.lbu = np.array([constraint.ddelta_min, constraint.jerk_min])
    ocp.constraints.ubu = np.array([constraint.ddelta_max, constraint.jerk_max])
    ocp.constraints.idxbu = np.array([0, 1])

    # state constraint. v_x_min must be strictly positive, see spatial_model.py.
    state_constraint_min = np.array([
        constraint.v_x_min,
        constraint.delta_min,
        constraint.a_min,
    ])
    state_constraint_max = np.array([
        constraint.v_x_max,
        constraint.delta_max,
        constraint.a_max,
    ])
    state_constraint_index = np.array([
        SpatialStateIndex.VELOCITY_V_X.value,
        SpatialStateIndex.STEERING_ANGLE_DELTA.value,
        SpatialStateIndex.ACCEL.value])

    ocp.constraints.lbx_0 = state_constraint_min
    ocp.constraints.ubx_0 = state_constraint_max
    ocp.constraints.idxbx_0 = state_constraint_index

    ocp.constraints.lbx = state_constraint_min
    ocp.constraints.ubx = state_constraint_max
    ocp.constraints.idxbx = state_constraint_index

    ocp.constraints.lbx_e = state_constraint_min
    ocp.constraints.ubx_e = state_constraint_max
    ocp.constraints.idxbx_e = state_constraint_index

    # Nonlinear constraints: [right bound, left bound, combined accel, sdot].
    # Unlike every other bound here, sdot is bounded on BOTH sides and hard on both:
    # the lower bound keeps 1/sdot finite, the upper bound keeps its reward finite.
    Nonlinear_constraint_min = np.array([
        0,
        -model.track_max,
        0,
        constraint.s_dot_min])
    Nonlinear_constraint_max = np.array([
        model.track_max,
        0,
        1,
        constraint.s_dot_max])
    # soften the leading rows only, see the note on ns above
    soft_index = np.arange(ns)
    soft_index_e = np.arange(ns_e)

    # at stage 0
    ocp.constraints.lh_0 = Nonlinear_constraint_min
    ocp.constraints.uh_0 = Nonlinear_constraint_max
    ocp.constraints.idxsh_0 = soft_index
    # from stage 1 to N-1
    ocp.constraints.lh = Nonlinear_constraint_min
    ocp.constraints.uh = Nonlinear_constraint_max
    ocp.constraints.idxsh = soft_index
    # at stage N. expr_e keeps the sdot row too, and it is hard there as well.
    ocp.constraints.lh_e = np.array([0, -model.track_max, constraint.s_dot_min])
    ocp.constraints.uh_e = np.array([model.track_max, 0, constraint.s_dot_max])
    ocp.constraints.idxsh_e = soft_index_e

    # set to make the initial condition necessary (otherwise lbx_0 overwrites this)
    ocp.constraints.x0 = model.x0

    # The horizon is a distance. acados calls it tf and slices it into N steps; here
    # those steps are metres, so ds = s_horizon / N.
    ocp.solver_options.tf = tompc_config.s_horizon
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP_WITH_FEASIBLE_QP"
    # NOTE: as in STMPC, do not believe the acados warning; "EXACT" makes the solver fail
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.regularize_method = "PROJECT"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
    # Ten substeps, where the time-domain controllers use three. A distance grid means
    # the *time* a node spans is speed-dependent: 0.3 m is 37 ms at 8 m/s but 600 ms at
    # 0.5 m/s, and the Pacejka model is stiff enough that RK4 diverges over the latter.
    # It does not diverge quietly -- it puts a NaN in the QP data, and HPIPM returns
    # status 3 on the first QP of the first SQP iteration, at every single call.
    # `v_min` is the other half of this: it bounds how long a node can span.
    ocp.solver_options.sim_method_num_steps = 10
    ocp.solver_options.tol = 1e-2
    ocp.solver_options.print_level = 0
    ocp.solver_options.nlp_solver_max_iter = 3  # For SQP methods, RTI is limited to 1 anyway

    acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_tompc.json")

    return constraint, model, acados_solver, params
