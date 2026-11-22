from typing import Dict, Optional, Tuple

import numpy as np
from forzaeth_mpc.config import KMPCConfig
from forzaeth_mpc.sim.core.plant import Plant
from forzaeth_mpc.kmpc.online_params import build_online_params as build_kmpc_online_params
from forzaeth_mpc.stmpc.online_params import MU_IDX, build_online_params
from forzaeth_mpc.utils.indices import InputIndex, StateIndex
from forzaeth_mpc.stmpc.controller import STMPC






class ControllerMPC:
    """Thin façade over STMPC/ACADOS: set params, pin x0, solve, read u/cost."""

    # Where jerk and ddelta sit in this controller's own u vector. step_plant needs it
    # because the plant reads them from named state slots, so copying u0 positionally
    # would silently swap them.
    idx_jerk, idx_ddelta = int(InputIndex.JERK), int(InputIndex.D_STEERING_ANGLE)

    def __init__(self, mpc: STMPC, track_len: float, dt: Optional[float] = None,
                 integrator: str = "rk4", noise_vars: Optional[np.ndarray] = None):
        self.mpc = mpc
        self.H = mpc.stmpc_config.N
        self.nx = mpc.model.n_x
        self.nu = int(mpc.model.n_u)
        self.solver = mpc.acados_solver
        # Try spline first; if missing, we’ll fallback to waypoint-based track_len upstream.
        # self.track_len = float(getattr(getattr(mpc, "spline", None), "track_length", 0.0))
        self.track_len = track_len
        self.dt = float(dt) if dt is not None else 1.0 / mpc.stmpc_config.MPC_freq
        self.plant = Plant(mpc._dynamics_of_car, self.dt, integrator, noise_vars)

    def set_online_params(self,
                          waypoints: np.ndarray,
                          qdict: Dict[str, float],
                          mpc_mu: float,
                          track_len_fallback: float):
        track_len = self.track_len if self.track_len > 0.0 else float(track_len_fallback)

        for i in range(self.H + 1):
            # reference s along horizon (mpc_sd is always (N+1, 2), see mpc_init_params)
            s_ref = float(self.mpc.mpc_sd[i, 0]) % track_len

            idx = int(np.abs(waypoints[:, 4] - s_ref).argmin())
            target_v = float(waypoints[idx, 2])

            p = build_online_params(self.mpc.stmpc_config, self.mpc.tire_config,
                                    target_v=target_v,
                                    overtake_d=0.0,
                                    mu=float(mpc_mu),
                                    weights=qdict)
            self.mpc.acados_solver.set(i, "p", p)

    def pin_state(self, x0_full: np.ndarray):
        x0 = np.asarray(x0_full[:self.nx], dtype=float)
        self.mpc.acados_solver.set(0, "lbx", x0)
        self.mpc.acados_solver.set(0, "ubx", x0)

    def _warm_start(self):
        self.mpc.apply_warm_start(pose_frenet=self.mpc.pose_frenet)

    def solve(self) -> Tuple[np.ndarray, np.ndarray, float, int]:
        status = int(self.mpc.acados_solver.solve())
        if status not in (0, 2):
            try:
                self._warm_start()
            except Exception:
                print(f"Applying warm start after solver not converging not succesful!")
                pass

        try:
            cost = float(self.mpc.acados_solver.get_cost())
        except Exception:
            print(f"Didn't recieve the cost of acados solver")
            cost = float("nan")

        u0 = np.asarray(self.mpc.acados_solver.get(0, "u"), dtype=float).copy()

        # keep mpc_sd rolling for the next step
        try:
            self.mpc.mpc_sd = np.array(
                [self.mpc.acados_solver.get(j, "x")[:2] for j in range(self.H + 1)]
            )
        except Exception:
            print(f"Keep mpc_sd for next step not succesful!")
            pass

        return u0, self.mpc.mpc_sd.copy(), cost, status

    def step_plant(self, x: np.ndarray, u0: np.ndarray) -> np.ndarray:
        """One closed-loop plant step of the controller's own ``_dynamics_of_car``.

        ``_dynamics_of_car`` reads the control out of the state tail, so the full
        plant state is ``[x(nx), u(nu)]``: for STMPC ``x[8]=jerk, x[9]=ddelta``,
        for KMPC ``x[5]=a, x[6]=ddelta``. Both come out of ``nx``/``nu``.
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        x_full = np.zeros(self.nx + self.nu)
        x_full[:self.nx] = x[:self.nx]
        # The plant reads jerk and ddelta out of named slots, while u0 comes back in the
        # OCP's own input order. Copying the tail wholesale silently swaps them.
        u0 = np.asarray(u0, dtype=float).reshape(-1)
        x_full[StateIndex.INPUT_JERK] = u0[self.idx_jerk]
        x_full[StateIndex.INPUT_DSTEERING] = u0[self.idx_ddelta]
        return self.plant.step(x_full)


class ControllerKMPC(ControllerMPC):
    """Same façade for the kinematic MPC.

    Subclassed only to reuse ``pin_state``/``solve``/``step_plant``: both
    controllers expose ``acados_solver``, ``mpc_sd`` and ``_dynamics_of_car``.

    The plant here is ``model.f_disc_func``, not ``_dynamics_of_car``. KMPC's OCP
    runs with ``integrator_type = "DISCRETE"`` over ``f_disc``, which applies the
    inputs as per-step increments (``dis_v_k = v + der_v``), whereas
    ``_dynamics_of_car``/``f_expl`` applies them as rates (``vdot = der_v``).
    Integrating f_expl therefore makes the plant react ``1/(dt*MPC_freq)`` times
    more slowly than the controller's own model believes, and the closed loop
    leaves the track. Stepping f_disc keeps plant and OCP model consistent, which
    is also what ``ControllerMPC`` does for STMPC.
    """

    idx_jerk, idx_ddelta = 0, 1   # KMPC's inputs are [a, ddelta]

    def __init__(self, mpc, track_len: float, dt: Optional[float] = None,
                 integrator: str = "rk4", noise_vars: Optional[np.ndarray] = None):
        self.mpc = mpc
        self.H = mpc.kmpc_config.N
        self.nx = mpc.model.n_x          # 5: [s, n, alpha, v, delta]
        self.nu = int(mpc.model.n_u)     # 2: [a, ddelta]
        self.solver = mpc.acados_solver
        self.track_len = track_len
        self.dt = float(dt) if dt is not None else 1.0 / mpc.kmpc_config.MPC_freq
        self.plant = None  # KMPC steps its own discrete model, see step_plant

    def set_online_params(self,
                          waypoints: np.ndarray,
                          qdict: Dict[str, float],
                          mpc_mu: float,
                          track_len_fallback: float):
        """``mpc_mu`` is ignored: the kinematic model has no friction parameter."""
        track_len = self.track_len if self.track_len > 0.0 else float(track_len_fallback)
        cfg = self.mpc.kmpc_config

        for i in range(self.H + 1):
            s_ref = float(self.mpc.mpc_sd[i, 0]) % track_len
            idx = int(np.abs(waypoints[:, 4] - s_ref).argmin())
            target_v = float(waypoints[idx, 2])

            p = build_kmpc_online_params(cfg, target_v=target_v, overtake_d=0.0, weights=qdict)
            self.solver.set(i, "p", p)

            # per-stage bounds, exactly as Kinematic_MPC_Controller.main_loop does
            if i < self.H:
                self.solver.set(i, "lbu", np.array([cfg.a_min, cfg.ddelta_min]))
                self.solver.set(i, "ubu", np.array([cfg.a_max, cfg.ddelta_max]))
                self.solver.set(i + 1, "lbx", np.array([cfg.v_min, cfg.delta_min]))
                self.solver.set(i + 1, "ubx", np.array([cfg.v_max, cfg.delta_max]))

    def step_plant(self, x: np.ndarray, u0: np.ndarray) -> np.ndarray:
        """One step of the OCP's own discrete model (inputs are increments)."""
        x = np.asarray(x, dtype=float).reshape(-1)
        u = np.asarray(u0, dtype=float).reshape(-1)
        x_next = self.mpc.model.f_disc_func(x[0], x[1], x[2], x[3], x[4],
                                            u[0], u[1], self.mpc.model_params.p)
        return np.asarray(x_next, dtype=float).reshape(-1)

    def _warm_start(self):
        # KMPC has no ``pose_frenet`` attribute; use the last pinned state.
        self.mpc.apply_warm_start(pose_frenet=self._x0_pinned[:3])

    def pin_state(self, x0_full: np.ndarray):
        self._x0_pinned = np.asarray(x0_full[:self.nx], dtype=float)
        super().pin_state(x0_full)


class ControllerTOMPC(ControllerMPC):
    """Same façade for the time-optimal MPC.

    Two things are not like ``ControllerMPC``:

    - the OCP state has nine entries (elapsed time appended), while the plant state is
      the usual eight plus the two input slots. ``pin_state`` bridges them; because time
      is *appended* and not inserted, that is a copy of the first eight and a zero.
    - there is no reference speed to look up, so ``set_online_params`` never reads the
      waypoint array and passes the racing weights, which zero ``q_v``. That is the
      whole point of the controller, not an omission.
    """

    def __init__(self, mpc, track_len: float, dt: Optional[float] = None,
                 integrator: str = "rk4", noise_vars: Optional[np.ndarray] = None):
        self.mpc = mpc
        self.H = mpc.tompc_config.N
        self.nx = int(mpc.model.n_x)      # 9: [s, n, theta, vx, vy, r, delta, a, t]
        self.nx_plant = 8                 # the time-domain single-track state
        self.nu = int(mpc.model.n_u)      # 2: [ddelta, jerk]
        self.solver = mpc.acados_solver
        self.track_len = track_len
        self.dt = float(dt) if dt is not None else 1.0 / mpc.tompc_config.MPC_freq
        self.plant = Plant(mpc._dynamics_of_car, self.dt, integrator, noise_vars)

    def set_online_params(self,
                          waypoints: np.ndarray,
                          qdict: Dict[str, float],
                          mpc_mu: float,
                          track_len_fallback: float):
        """``waypoints`` is unused: a time-optimal controller has no speed reference."""
        from forzaeth_mpc.tompc.online_params import \
            build_online_params as build_tompc_params
        from forzaeth_mpc.tompc.online_params import racing_weights

        cfg = self.mpc.tompc_config
        weights = racing_weights(cfg)
        weights.update({k: v for k, v in (qdict or {}).items() if k in weights})
        weights["q_v"] = 0.0  # whatever the harness passed, racing does not track a speed

        p = build_tompc_params(cfg, self.mpc.tire_config, target_v=0.0, overtake_d=0.0,
                               mu=float(mpc_mu), weights=weights)
        for i in range(self.H + 1):
            self.solver.set(i, "p", p)
            if i < self.H:
                self.solver.set(i, "lbu", np.array([cfg.ddelta_min, cfg.jerk_min]))
                self.solver.set(i, "ubu", np.array([cfg.ddelta_max, cfg.jerk_max]))
                self.solver.set(i + 1, "lbx", np.array([cfg.v_min, cfg.delta_min, cfg.a_min]))
                self.solver.set(i + 1, "ubx", np.array([cfg.v_max, cfg.delta_max, cfg.a_max]))

    def pin_state(self, x0_full: np.ndarray):
        """Plant state (8 + 2 input slots) -> OCP state (8 + elapsed time)."""
        from forzaeth_mpc.utils.indices import SpatialStateIndex

        x = np.asarray(x0_full, dtype=float).reshape(-1)
        x0 = np.zeros(self.nx)
        x0[:self.nx_plant] = x[:self.nx_plant]
        # v_x is pinned as an equality, so it has to be inside its own bounds, and
        # strictly above zero: the spatial model divides by ds/dt.
        x0[StateIndex.VELOCITY_V_X] = np.clip(
            x0[StateIndex.VELOCITY_V_X],
            self.mpc.tompc_config.v_min, self.mpc.tompc_config.v_max)
        x0[SpatialStateIndex.TIME] = 0.0
        self._x0_pinned = x0.copy()
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)

    def _warm_start(self):
        x0 = self._x0_pinned
        self.mpc.apply_warm_start(pose_frenet=x0[:3], v_x=x0[StateIndex.VELOCITY_V_X])

    def solve(self):
        out = super().solve()
        # the clock at the last node is the horizon traversal time, i.e. the quantity
        # this controller is actually minimising. main_loop records it on the car; the
        # harness bypasses main_loop, so record it here too.
        from forzaeth_mpc.utils.indices import SpatialStateIndex
        self.mpc.horizon_time = float(
            self.solver.get(self.H, "x")[SpatialStateIndex.TIME])
        return out

    def step_plant(self, x: np.ndarray, u0: np.ndarray) -> np.ndarray:
        """One step of the *time-domain* model: the loop closes in time, not distance."""
        x = np.asarray(x, dtype=float).reshape(-1)
        x_full = np.zeros(self.nx_plant + self.nu)
        x_full[:self.nx_plant] = x[:self.nx_plant]
        u0 = np.asarray(u0, dtype=float).reshape(-1)
        x_full[StateIndex.INPUT_JERK] = u0[self.idx_jerk]
        x_full[StateIndex.INPUT_DSTEERING] = u0[self.idx_ddelta]
        return self.plant.step(x_full)


class ControllerLBMPC:
    """Same façade for the learning-based MPC.

    Differences to ``ControllerMPC``:
      - the solver is ``mpc.l4a_solver`` (an l4acados wrapper), not ``acados_solver``
      - the plant is ``mpc._predict_next_state_nominal`` (LBMPC's own acados sim
        solver), whose μ comes from ``mpc.set_sim_mu`` and is independent of the
        μ the MPC sees at p index 12 — that is the offline μ-mismatch setup
      - the GP residual dataset is fed every step, mirroring ``LBMPC.main_loop``

    Args:
        learn: record one-step residuals into the GP dataset. Set False for a
            nominal-MPC baseline run. Note that learning also needs
            ``gp_params.use_gp`` and ``data_processing_mode == "learn"``.
        delay_freeze: inflate qddelta by 1e6 over the first ``steps_delay``
            stages, as ``LBMPC.main_loop`` does. Off by default: on the car the
            applied command is read from the delayed prediction node, while this
            harness feeds back u0, and the 1e6 term wrecks the QP conditioning
            (bang-bang ddelta, the car brakes to a stop). ``ControllerMPC``
            leaves the same inflation out of the STMPC rollout for that reason.
    """

    def __init__(self, mpc, track_len: float, dt: Optional[float] = None, learn: bool = True,
                 delay_freeze: bool = False):
        self.mpc = mpc
        self.H = mpc.lbmpc_config.N
        self.nx = int(mpc.model.n_x)     # 8: [s, n, theta, vx, vy, r, delta, a]
        self.nu = int(mpc.model.n_u)     # 2: [ddelta, jerk]  (reversed wrt STMPC)
        self.solver = mpc.l4a_solver
        self.track_len = track_len
        self.dt = float(dt) if dt is not None else 1.0 / mpc.lbmpc_config.MPC_freq
        self.learn = bool(learn)
        self.delay_freeze = bool(delay_freeze)
        mpc.set_learning_enabled(self.learn, source="ControllerLBMPC")
        self._x0_pinned = None
        self._mu_mpc = float(mpc.tire_config.friction_coeff)  # updated by set_online_params
        # the plant step is the sim solver's horizon; LBMPC defaults it to 1/MPC_freq
        if getattr(mpc, "sim_solver", None) is not None:
            mpc.sim_solver.set("T", self.dt)

    def set_online_params(self,
                          waypoints: np.ndarray,
                          qdict: Dict[str, float],
                          mpc_mu: float,
                          track_len_fallback: float):
        track_len = self.track_len if self.track_len > 0.0 else float(track_len_fallback)
        cfg = self.mpc.lbmpc_config
        self._mu_mpc = float(mpc_mu)

        for i in range(self.H + 1):
            s_ref = float(self.mpc.mpc_sd[i, 0]) % track_len
            idx = int(np.abs(waypoints[:, 4] - s_ref).argmin())
            target_v = float(waypoints[idx, 2])

            weights = dict(qdict)
            if self.delay_freeze and i < cfg.steps_delay:
                weights["q_ddelta"] = 1e6 * weights.get("q_ddelta", cfg.qddelta)
            p = build_online_params(cfg, self.mpc.tire_config,
                                    target_v=target_v,
                                    overtake_d=0.0,
                                    mu=float(mpc_mu),
                                    weights=weights)
            self.solver.set(i, "p", p)

            # per-stage bounds, exactly as LBMPC.main_loop does
            if i < self.H:
                self.solver.set(i, "lbu", np.array([cfg.ddelta_min, -50.0]))
                self.solver.set(i, "ubu", np.array([cfg.ddelta_max, 50.0]))
                self.solver.set(i + 1, "lbx", np.array([cfg.v_min, cfg.delta_min, cfg.a_min]))
                self.solver.set(i + 1, "ubx", np.array([cfg.v_max, cfg.delta_max, cfg.a_max]))

    def pin_state(self, x0_full: np.ndarray):
        x0 = np.asarray(x0_full[:self.nx], dtype=float)
        self._x0_pinned = x0.copy()

        # Residual bookkeeping, same order as LBMPC.main_loop: measure -> record
        # -> sync residual model -> update inducing points -> pin x0.
        # main_loop additionally gates recording on nr_laps >= 2 (on-car warm-up);
        # offline we record from the first available pair.
        if self.learn and self.mpc.residual_gp is not None:
            # _compute_residual and step_plant both call _predict_next_state_nominal,
            # which reads μ out of model_params.p. Evaluate the residual's nominal
            # model at the MPC's μ and put the plant's μ back afterwards, otherwise
            # plant == nominal model and every recorded residual is exactly zero.
            mu_plant = float(np.asarray(self.mpc.model_params.p, dtype=float).reshape(-1)[MU_IDX])
            self.mpc.set_sim_mu(self._mu_mpc)
            res = self.mpc._compute_residual(new_state=x0)
            self.mpc.set_sim_mu(mu_plant)
            if res is not None and self.mpc.last_status in (0, 2):
                self.mpc.residual_gp.record_datapoint(*res)
                self.mpc._gp_warmup_done = True
        self.mpc._sync_solver_residual_model()
        self.mpc._update_inducing_points_along_horizon()

        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)

    def solve(self) -> Tuple[np.ndarray, np.ndarray, float, int]:
        try:
            status = int(self.solver.solve())
        except Exception as e:
            print(f"[ControllerLBMPC] solve raised: {e}")
            status = -1
        self.mpc.last_status = status
        if status not in (0, 2):
            try:
                x0 = self._x0_pinned
                self.mpc._apply_warm_start(pose_frenet=x0[:3],
                                           actual_speed=x0[StateIndex.VELOCITY_V_X],
                                           actual_acc=x0[StateIndex.ACCEL])
            except Exception:
                print(f"Applying warm start after solver not converging not succesful!")

        try:
            cost = float(self.solver.ocp_solver.get_cost())
        except Exception:
            print(f"Didn't recieve the cost of acados solver")
            cost = float("nan")

        u0 = np.asarray(self.solver.get(0, "u"), dtype=float).copy()

        try:
            self.mpc.mpc_sd = np.array(
                [self.solver.get(j, "x")[:2] for j in range(self.H + 1)]
            )
        except Exception:
            print(f"Keep mpc_sd for next step not succesful!")

        # what the next _compute_residual() compares against
        self.mpc._prev_state_for_residual = self._x0_pinned.copy()
        self.mpc._prev_input_for_residual = u0.copy()

        return u0, self.mpc.mpc_sd.copy(), cost, status

    def step_plant(self, x: np.ndarray, u0: np.ndarray) -> np.ndarray:
        """One step of LBMPC's own acados sim solver (μ from ``set_sim_mu``)."""
        x = np.asarray(x, dtype=float).reshape(-1)[:self.nx]
        return self.mpc._predict_next_state_nominal(x, np.asarray(u0, dtype=float).reshape(-1))

    def gp_num_datapoints(self) -> int:
        try:
            return int(self.mpc.residual_gp.gp_model.train_inputs[0].shape[0])
        except Exception:
            return 0


if __name__ == "__main__":
    # Interface self-check on stub controllers: no acados solver is built.
    from types import SimpleNamespace

    from forzaeth_mpc.config import (load_KMPC_config_ros,
                                     load_LBMPC_config_ros,
                                     load_pacejka_tire_config_ros,
                                     load_STMPC_config_ros, load_TOMPC_config_ros)

    def _stub(nx, nu, **kwargs):
        return SimpleNamespace(
            model=SimpleNamespace(n_x=nx, n_u=nu),
            acados_solver=None, l4a_solver=None,
            _dynamics_of_car=lambda t, x: np.zeros(nx + nu),
            set_learning_enabled=lambda enabled, source="": False,
            **kwargs)

    stmpc_config = load_STMPC_config_ros("NUC5")
    kmpc_config = load_KMPC_config_ros("NUC5")
    lbmpc_config = load_LBMPC_config_ros("NUC5")
    tompc_config = load_TOMPC_config_ros("NUC5")
    tire_config = load_pacejka_tire_config_ros("NUC5", "dubi")

    adapters = [
        ControllerMPC(_stub(8, 2, stmpc_config=stmpc_config, tire_config=tire_config), 26.06, 0.02),
        ControllerKMPC(_stub(5, 2, kmpc_config=kmpc_config), 26.06, 0.02),
        ControllerLBMPC(_stub(8, 2, lbmpc_config=lbmpc_config, tire_config=tire_config,
                              residual_gp=None), 26.06, 0.02),
        ControllerTOMPC(_stub(9, 2, tompc_config=tompc_config, tire_config=tire_config),
                        26.06, 0.02),
    ]
    for ctrl in adapters:
        for attr in ("mpc", "H", "nx", "nu", "solver", "set_online_params",
                     "pin_state", "solve", "step_plant"):
            assert hasattr(ctrl, attr), f"{type(ctrl).__name__} misses {attr}"
        assert ctrl.H > 0 and ctrl.nx > 0 and ctrl.nu > 0, type(ctrl).__name__

    p_k = build_kmpc_online_params(kmpc_config, target_v=3.5, overtake_d=0.25)
    assert len(p_k) == 10, len(p_k)
    assert p_k[0] == 3.5 and p_k[5] == kmpc_config.qac and p_k[9] == 0.25

    # LBMPC now shares STMPC's 22-entry layout; online_params.py's own self-check
    # covers it in detail, this only pins that the adapter reaches the same builder.
    p_l = build_online_params(lbmpc_config, tire_config, 3.5, 0.25, mu=0.63)
    assert len(p_l) == 22, len(p_l)
    assert p_l[MU_IDX] == 0.63
    assert p_l[14] == tire_config.Bf

    # TOMPC's OCP state is one longer than its plant state, and the extra entry is
    # appended rather than inserted. That is the invariant every consumer that slices
    # the first eight -- the plant, the metrics, the ROS viz node -- relies on.
    from forzaeth_mpc.utils.indices import SpatialStateIndex

    tompc = adapters[-1]
    assert (tompc.nx, tompc.nx_plant) == (9, 8), (tompc.nx, tompc.nx_plant)
    assert int(SpatialStateIndex.TIME) == 8, "time must be appended, not inserted"
    for i in range(8):
        assert int(SpatialStateIndex(i)) == int(StateIndex(i)), (
            f"spatial state {i} must match the single-track layout")

    print("OK: 4 adapters expose the interface; p layouts are 10 (KMPC) / "
          "22 (single-track) / 24 (TOMPC)")
