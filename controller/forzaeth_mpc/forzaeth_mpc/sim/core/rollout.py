from __future__ import annotations

import time
from typing import Callable, Dict, Optional

import numpy as np
from forzaeth_mpc.sim.core.controllers import ControllerMPC
from forzaeth_mpc.sim.core.plant import Plant
from forzaeth_mpc.utils.indices import InputIndex, StateIndex

try:
    from tqdm.auto import tqdm
    _safe_log = tqdm.write
except Exception:
    _safe_log = print


class RolloutRunner:
    """
    Closed-loop rollout with optional prediction capture.

    - Writes control into the input slots of the plant state (see StateIndex)
    - Integrates with Plant (RK4/Euler), which reads u back out of those slots.
    - Decoupled μ via optional set_sim_mu_fn() callable
    - Optionally records full (H+1, nx) predicted state sequence per step.
    """

    def __init__(self, controller: ControllerMPC, plant: Plant, track_len: float):
        self.ctrl = controller
        self.plant = plant
        self.track_len = float(track_len)

    def run(self,
            x0_full: np.ndarray,
            waypoints: np.ndarray,
            iterations: int,
            qdict: Dict[str, float],
            mpc_mu: float,
            set_sim_mu_fn: Optional[callable] = None,
            *,
            capture_predictions: bool = False,
            log_fn: Callable[[str], None] = _safe_log):
        """
        Returns:
            X: (K, x_full)
            U: (K, nu)
            C: (K,) cost
            S: (K,) status
            X_pred_seq: (K, H+1, nx) or None if capture_predictions=False
        """
        x_hist, u_hist, cost_hist, status_hist = [], [], [], []
        x_pred_seq_hist = [] if capture_predictions else None

        # Optionally set PLANT μ (mpc.set_sim_mu(mu_track))
        if set_sim_mu_fn is not None:
            try:
                set_sim_mu_fn()
            except Exception:
                log_fn(f"Setting simulation mu not succesful!")
                pass

        x = np.asarray(x0_full, dtype=float).copy()
        H = self.ctrl.H
        nx = self.ctrl.nx
        nu = int(getattr(self.ctrl.mpc.model, "n_u", 2))

        start_time = time.time()
        for k in range(int(iterations)):
            # MPC: pin x0, set online params (includes μ_mpc), solve
            self.ctrl.pin_state(x)
            self.ctrl.set_online_params(waypoints, qdict, mpc_mu, track_len_fallback=self.track_len)
            u0, _, cost, status = self.ctrl.solve()

            # (optional) capture the whole horizon
            if capture_predictions:
                solver = self.ctrl.mpc.acados_solver
                x_pred_full = np.zeros((H + 1, nx), dtype=float)
                for i in range(H + 1):
                    x_pred_full[i, :] = solver.get(i, "x")
                x_pred_seq_hist.append(x_pred_full)

            # Inject control into state helpers (jerk, ddelta)
            if x.shape[0] > 9 and u0.shape[0] >= 2:
                x[StateIndex.INPUT_JERK] = float(u0[InputIndex.JERK])
                x[StateIndex.INPUT_DSTEERING] = float(u0[InputIndex.D_STEERING_ANGLE])

            # Integrate one step with the real plant
            x = self.plant.step(x)

            x_hist.append(x.copy())
            u_hist.append(u0.copy())
            cost_hist.append(float(cost))
            status_hist.append(int(status))

            elapsed_time = time.time() - start_time
            log_fn(f"[{k:04d}] status={status}, u0={u0}, elapsed={elapsed_time*1e3:.1f} ms")
            start_time = time.time()

        X = np.asarray(x_hist)
        U = np.asarray(u_hist)
        C = np.asarray(cost_hist, dtype=float)
        S = np.asarray(status_hist, dtype=int)
        X_pred_seq = np.asarray(x_pred_seq_hist) if capture_predictions else None
        return X, U, C, S, X_pred_seq
