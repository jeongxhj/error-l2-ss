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
from acados_template import AcadosModel, AcadosOcp, ZoroDescription
from forzaeth_mpc.stmpc.bicycle_model import bicycle_model
from forzaeth_mpc.stmpc.online_params import default_parameter_values
from forzaeth_mpc.utils.indices import InputIndex, StateIndex
from forzaeth_mpc.config import CarConfig, LBMPCConfig, PacejkaTireConfig


def build_ocp_for_l4acados(
    s0, kapparef, d_left, d_right,
        lbmpc_config: LBMPCConfig, car_config: CarConfig, tire_config: PacejkaTireConfig
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
        s0, kapparef, d_left, d_right, lbmpc_config, car_config, tire_config,
        # its own codegen name, so LB-MPC's OCP and nominal sim solver do not collide with
        # STMPC's in a shared c_generated_code/
        model_name="LBMPCbicycle_model",
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
    p = default_parameter_values(lbmpc_config, tire_config)
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

    ocp.dims.N = lbmpc_config.N
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
    Zl, Zu, zl, zu = lbmpc_config.Zl, lbmpc_config.Zu, lbmpc_config.zl, lbmpc_config.zu
    ocp.cost.Zl_0 = lbmpc_config.Zl * np.ones((nsh,))
    ocp.cost.Zu_0 = lbmpc_config.Zu * np.ones((nsh,))
    # must have in the cost function, but not used (quadratic cost is enough to handle the constraints)
    ocp.cost.zl_0 = lbmpc_config.zl * np.ones((nsh,))
    ocp.cost.zu_0 = lbmpc_config.zu * np.ones((nsh,))
    # from 1 to N-1
    ocp.cost.Zl = lbmpc_config.Zl * np.ones((ns,))
    ocp.cost.Zu = lbmpc_config.Zu * np.ones((ns,))
    ocp.cost.zl = lbmpc_config.zl * np.ones((ns,))
    ocp.cost.zu = lbmpc_config.zu * np.ones((ns,))
    # at stage N
    ocp.cost.Zl_e = lbmpc_config.Zl * np.ones((nsh_e,))
    ocp.cost.Zu_e = lbmpc_config.Zu * np.ones((nsh_e,))
    ocp.cost.zl_e = lbmpc_config.zl * np.ones((nsh_e,))
    ocp.cost.zu_e = lbmpc_config.zu * np.ones((nsh_e,))

    # Input bounds (same)
    ocp.constraints.lbu_0 = np.array([constraint.ddelta_min, constraint.jerk_min])
    ocp.constraints.ubu_0 = np.array([constraint.ddelta_max, constraint.jerk_max])
    ocp.constraints.idxbu_0 = np.array([0, 1])

    ocp.constraints.lbu = np.array([constraint.ddelta_min, constraint.jerk_min])
    ocp.constraints.ubu = np.array([constraint.ddelta_max, constraint.jerk_max])
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

    # Nonlinear path + combined-acc constraints + forward advancement
    Nonlin_min = np.array([0, -model.track_max, 0, 0])
    Nonlin_max = np.array([model.track_max, 0, 1, 10_000])
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
    ocp.solver_options.Tsim = 1 / lbmpc_config.MPC_freq
    ocp.solver_options.tf = lbmpc_config.N / lbmpc_config.MPC_freq
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

    stochastic_mode = lbmpc_config.gp_params.stochastic_mode
    if stochastic_mode != "disabled":
        ocp = add_zoro_description(ocp, lbmpc_config, stochastic_mode)

    return ocp, model, constraint, params


def add_zoro_description(ocp, lbmpc_config, stochastic_mode):
    """
    Attaches the ZoroDescription object required by the ZeroOrderGPMPC class.
    Configures it to tighten the specific track constraints defined in build_ocp.
    """
    # Create the description object
    zoro_description = ZoroDescription()

    # 1. Uncertainty Configuration
    # This matrix P0 represents initial state uncertainty diagonal
    zoro_description.P0_mat = np.diag([1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 5e-3, 0.0, 0.0])

    # Identify which noise matrix from the external GP affects the dynamics
    # Assuming G matrix maps noise to states.
    idx = np.asarray(lbmpc_config.gp_params.unc_jac_G_mat_idx, dtype=float)
    unc_jac_G_mat = np.diag(idx)
    unc_jac_G_mat = unc_jac_G_mat[:, ~np.all(unc_jac_G_mat == 0, axis=0)]
    nw = unc_jac_G_mat.shape[1]
    if nw == 0:
        raise ValueError(
            "stochastic_mode is enabled, but unc_jac_G_mat_idx does not activate any disturbance channel."
        )

    zoro_description.W_mat = np.diag(np.zeros((nw,), dtype=float))
    zoro_description.unc_jac_G_mat = unc_jac_G_mat

    nx = ocp.dims.nx
    nu = ocp.dims.nu
    if stochastic_mode == "open_loop":
        zoro_description.fdbk_K_mat = np.zeros((nu, nx), dtype=float)
    elif stochastic_mode == "lqr":
        # Calculated in scripts/build_k.py
        zoro_description.fdbk_K_mat = np.array([
            [0.0, 6.831301, 9.928423, 0.0, 0.825643, 0.33575, 13.06181, 0.0],
            [0.0, 0.0, 0.0, 11.83216, 0.0, 0.0, 0.0, 4.864599],
        ], dtype=float)
    elif stochastic_mode == "riccati":
        zoro_description.fdbk_K_mat = np.zeros((nu, nx), dtype=float)
        zoro_description.feedback_optimization_mode = "RICCATI_CONSTANT_COST"

        Q = np.zeros((nx, nx))
        R = np.zeros((nu, nu))
        S = np.zeros((nu, nx))

        qn = float(lbmpc_config.qn)
        qalpha = float(lbmpc_config.qalpha)
        qv = float(lbmpc_config.qv)

        qjerk = float(lbmpc_config.qjerk)
        qddelta = float(lbmpc_config.qddelta)

        # Lateral deviation minimization (only if available)
        qvy = 100.0 if getattr(lbmpc_config, "vy_minimization", False) else 0.0

        # State penalties
        Q[StateIndex.MIN_DIST_TO_CENTER_LINE_N, StateIndex.MIN_DIST_TO_CENTER_LINE_N] = qn
        Q[StateIndex.ORIENTATION_THETA, StateIndex.ORIENTATION_THETA] = qalpha
        Q[StateIndex.VELOCITY_V_X, StateIndex.VELOCITY_V_X] = qv
        Q[StateIndex.VELOCITY_V_Y, StateIndex.VELOCITY_V_Y] = qvy

        # Input penalties
        R[InputIndex.D_STEERING_ANGLE, InputIndex.D_STEERING_ANGLE] = qddelta
        R[InputIndex.JERK, InputIndex.JERK] = qjerk
        tm = float(lbmpc_config.terminal_multiplier) if hasattr(lbmpc_config, "terminal_multiplier") else 1.0
        Qe = tm * Q.copy()
        eps = 1e-6
        # written by name but read positionally before, which under the old input order
        # (jerk first) put qjerk into the ddelta slot
        for _i in (InputIndex.D_STEERING_ANGLE, InputIndex.JERK):
            R[_i, _i] = max(R[_i, _i], eps)

        zoro_description.riccati_Q_const = Q
        zoro_description.riccati_R_const = R
        zoro_description.riccati_S_const = S
        zoro_description.riccati_Q_const_e = Qe
    else:
        raise ValueError(
            f"Unsupported stochastic_mode '{stochastic_mode}'. "
            "Expected one of ['open_loop', 'lqr', 'riccati']."
        )

    # 2. Scaling
    zoro_description.backoff_scaling_gamma = 1.0  # 2-sigma confidence -> approx 95% | 1-sigma confidence -> 68%

    # 3. Define WHICH constraints to tighten

    # Our constraints are:
    # h[0]: [0, track_max]  <-- We need to lower the Upper Bound (uh)
    # h[1]: [-track_max, 0] <-- We need to raise the Lower Bound (lh)
    # h[2]: [0, 1]          <-- Ignore (probably not affected by spatial uncertainty)

    zoro_description.idx_uh_t = [1]  # Tighten lower bound of h[0]
    zoro_description.idx_lh_t = [0]  # Tighten upper bound of h[1]

    # Also tighten state bounds if necessary
    zoro_description.idx_lbx_t = []
    zoro_description.idx_ubx_t = []

    # 4. Flags expected by the ZOMPC class
    zoro_description.input_P0_diag = True
    zoro_description.input_P0 = False
    zoro_description.input_W_diag = True
    zoro_description.input_W_add_diag = True
    zoro_description.output_P_matrices = True

    # Attach to OCP
    ocp.zoro_description = zoro_description

    return ocp
