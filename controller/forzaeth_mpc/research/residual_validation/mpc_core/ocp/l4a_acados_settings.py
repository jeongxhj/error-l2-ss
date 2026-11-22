"""
Build the same AcadosOcp as in `acados_settings.py`, but ready for L4acados.
Nothing about the nominal *dynamics*, *constraints* or *costs* changes.
We simply:
- keep the external cost and nonlinear constraints exactly as before,
- keep the same solver options (SQP + GAUSS_NEWTON),
- return the AcadosOcp *and* the nominal model/constraint structs so L4acados
can wrap it with a residual model later.
"""
from typing import Tuple

import numpy as np
from acados_template import AcadosModel, AcadosOcp
from research.residual_validation.mpc_core.models.bicycle_model import bicycle_model
from forzaeth_mpc.config import CarConfig, PacejkaTireConfig, STMPCConfig
from research.residual_validation.mpc_core.utils.indices import StateIndex


def get_parameters(cfg: STMPCConfig, tire_config: PacejkaTireConfig) -> np.ndarray:
    params = np.array([
        2,  # initial velocity (placeholder, overwritten online)
        cfg.qadv,
        cfg.qv,
        cfg.qn,
        cfg.qalpha,
        cfg.qjerk,
        cfg.qddelta,
        cfg.alat_max,
        cfg.a_min,
        cfg.a_max,
        cfg.track_safety_margin,
        cfg.overtake_d,
        tire_config.friction_coeff,
        cfg.terminal_multiplier,
    ], dtype=float)
    return params


def build_ocp_for_l4acados(
    s0, kapparef, d_left, d_right,
        stmpc_config: STMPCConfig, car_config: CarConfig, tire_config: PacejkaTireConfig
) -> Tuple[AcadosOcp, object, object, object]:
    """
    Returns the *same* OCP as `acados_settings.acados_settings`, but without
    instantiating the solver. This OCP is then consumable by L4acados.


    Returns
    -------
    ocp : AcadosOcp
    model : types.SimpleNamespace (symbolic model with fields used in STMPC)
    constraint : types.SimpleNamespace
    params : types.SimpleNamespace (with `.p` = parameter vector)
    """
    ocp = AcadosOcp()

    # Export nominal model, constraints & params *exactly* as before
    model, constraint, params = bicycle_model(
        s0, kapparef, d_left, d_right, stmpc_config, car_config, tire_config
    )

    # Wrap into AcadosModel (unchanged from your original file)
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

    # Parameters (same order & meaning)
    p = get_parameters(stmpc_config, tire_config)
    params.p = p
    ocp.parameter_values = p

    # Dimensions
    nx = model.n_x
    nu = model.n_u
    nsbx = 0
    nsbu = 0
    nsh = constraint.expr.shape[0]
    nsh_e = constraint.expr_e.shape[0]
    ns = nsh + nsbx + nsbu

    ocp.dims.N = stmpc_config.N
    ocp.dims.nx = nx
    ocp.dims.nu = nu
    ocp.dims.ny = 0
    ocp.dims.ny_e = 0

    # External cost (identical expressions)
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = model.cost_expr_ext_cost
    ocp.model.cost_expr_ext_cost_e = model.cost_expr_ext_cost_e

    # Slack weights (same values/shape as original)
    Zl, Zu, zl, zu = stmpc_config.Zl, stmpc_config.Zu, stmpc_config.zl, stmpc_config.zu
    ocp.cost.Zl_0 = stmpc_config.Zl * np.ones((nsh,))
    ocp.cost.Zu_0 = stmpc_config.Zu * np.ones((nsh,))
    # must have in the cost function, but not used (quadratic cost is enough to handle the constraints)
    ocp.cost.zl_0 = stmpc_config.zl * np.ones((nsh,))
    ocp.cost.zu_0 = stmpc_config.zu * np.ones((nsh,))
    # from 1 to N-1
    ocp.cost.Zl = stmpc_config.Zl * np.ones((ns,))
    ocp.cost.Zu = stmpc_config.Zu * np.ones((ns,))
    ocp.cost.zl = stmpc_config.zl * np.ones((ns,))
    ocp.cost.zu = stmpc_config.zu * np.ones((ns,))
    # at stage N
    ocp.cost.Zl_e = stmpc_config.Zl * np.ones((nsh_e,))
    ocp.cost.Zu_e = stmpc_config.Zu * np.ones((nsh_e,))
    ocp.cost.zl_e = stmpc_config.zl * np.ones((nsh_e,))
    ocp.cost.zu_e = stmpc_config.zu * np.ones((nsh_e,))

    # Input bounds (same)
    ocp.constraints.lbu_0 = np.array([constraint.jerk_min, constraint.ddelta_min])
    ocp.constraints.ubu_0 = np.array([constraint.jerk_max, constraint.ddelta_max])
    ocp.constraints.idxbu_0 = np.array([0, 1])

    ocp.constraints.lbu = np.array([constraint.jerk_min, constraint.ddelta_min])
    ocp.constraints.ubu = np.array([constraint.jerk_max, constraint.ddelta_max])
    ocp.constraints.idxbu = np.array([0, 1])

    # State bounds (same)
    state_min = np.array([constraint.v_x_min, constraint.delta_min, constraint.a_min])
    state_max = np.array([constraint.v_x_max, constraint.delta_max, constraint.a_max])
    idx = np.array([
        StateIndex.VELOCITY_V_X.value,
        StateIndex.STEERING_ANGLE_DELTA.value,
        StateIndex.ACCEL.value,
    ])
    ocp.constraints.lbx_0 = state_min
    ocp.constraints.ubx_0 = state_max
    ocp.constraints.idxbx_0 = idx

    ocp.constraints.lbx = state_min
    ocp.constraints.ubx = state_max
    ocp.constraints.idxbx = idx

    ocp.constraints.lbx_e = state_min
    ocp.constraints.ubx_e = state_max
    ocp.constraints.idxbx_e = idx

    # Nonlinear path + combined-acc constraints (same limits & ordering)
    Nonlin_min = np.array([0, -model.track_max, 0])
    Nonlin_max = np.array([model.track_max, 0, 1])
    idxh = np.arange(nsh)

    ocp.constraints.lh_0 = Nonlin_min
    ocp.constraints.uh_0 = Nonlin_max
    ocp.constraints.idxsh_0 = idxh

    ocp.constraints.lh = Nonlin_min
    ocp.constraints.uh = Nonlin_max
    ocp.constraints.idxsh = idxh

    ocp.constraints.lh_e = Nonlin_min[:nsh_e]
    ocp.constraints.uh_e = Nonlin_max[:nsh_e]
    ocp.constraints.idxsh_e = idxh[:nsh_e]

    # Initial condition must be enforced
    ocp.constraints.x0 = model.x0

    ocp.solver_options.exact_hess_constr = 0
    ocp.solver_options.exact_hess_cost = 0
    ocp.solver_options.exact_hess_dyn = 0
    # ocp.solver_options.ext_cost_num_hess = True
    # ocp.solver_options.sim_method_jac_reuse = 1
    # ocp.solver_options.qp_solver_warm_start = 1
    # ocp.solver_options.as_rti_level = 1

    # Solver options (kept as in your original)
    ocp.solver_options.tf = stmpc_config.N / stmpc_config.MPC_freq
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"  # L4acados assumes SQP w/ GN
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.regularize_method = "PROJECT"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 3
    ocp.solver_options.tol = 1e-1
    ocp.solver_options.print_level = 0
    ocp.solver_options.nlp_solver_max_iter = 1

    return ocp, model, constraint, params
