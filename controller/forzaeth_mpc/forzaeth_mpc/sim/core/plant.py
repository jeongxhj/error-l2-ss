from __future__ import annotations

from typing import Callable, Literal, Optional

import numpy as np
from forzaeth_mpc.sim.utils.integration import DynFun, euler_step, rk4_step


class Plant:
    """
    Wrapper around your real plant using the shared integration utilities.

    IMPORTANT:
      - dyn_fun must be like mpc._dynamics_of_car(t, x), reading u out of the
        input slots of the state vector (StateIndex.INPUT_JERK / INPUT_DSTEERING).
      - Controls are assumed to be already written into the state before calling step().
    """

    def __init__(
        self,
        dyn_fun: Callable[[float, np.ndarray], np.ndarray],
        dt: float,
        integrator: Literal["rk4", "euler"] = "rk4",
        noise_vars: Optional[np.ndarray] = None,
    ):
        # Vector field f_state(x) = dx/dt, given controls stored in x.
        self._f_state = lambda x: np.asarray(dyn_fun(0.0, x), dtype=float)
        self.dt = float(dt)
        self.integrator = integrator.lower()

        # Optional noise configuration
        self.noise_vars = (
            np.asarray(noise_vars, dtype=float)
            if noise_vars is not None
            else None
        )

        # Adapter to match the utils signature f(x, u) -> dx/dt (we ignore u).
        self._f_ignore_u: DynFun = lambda x, u: self._f_state(x)

        # Pick the stepper from the shared utils.
        self._step_fn = rk4_step if self.integrator == "rk4" else euler_step

    def step(self, x: np.ndarray) -> np.ndarray:
        """
        Advance one step using the selected integrator.
        Assumes the control is already in the state's input slots.
        Optionally adds Gaussian noise to the new state.
        """
        x = np.asarray(x, dtype=float)
        x_next = self._step_fn(self._f_ignore_u, x, None, self.dt)

        # Add Gaussian noise if configured
        if self.noise_vars is not None:
            stds = np.sqrt(self.noise_vars)
            noise = np.random.normal(0.0, stds, size=x_next.shape)
            x_next += noise

        return x_next

    def update_noise_vars(self, new_noise_vars: Optional[np.ndarray]) -> None:
        """
        Update the Gaussian noise variances.

        Args:
            new_noise_vars (np.ndarray | None):
                New variances for each state. If None, disables noise.

        Examples:
            >>> # Change noise level during simulation
            >>> plant.update_noise_vars(np.array([1e-3, 1e-3, 1e-4, 0, 0, 0, 0]))

            >>> # Disable noise completely
            >>> plant.update_noise_vars(None)
        """
        if new_noise_vars is None:
            self.noise_vars = None
            return

        new_noise_vars = np.asarray(new_noise_vars, dtype=float)
        if np.any(new_noise_vars < 0):
            raise ValueError("Noise variances must be non-negative.")
        self.noise_vars = new_noise_vars
