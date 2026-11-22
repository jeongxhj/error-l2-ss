from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

State = np.ndarray
Control = Optional[np.ndarray]
DynFun = Callable[[State, Control], State]


def euler_step(f: DynFun, x: State, u: Control, dt: float) -> State:
    """
    Forward Euler step: x_{k+1} = x_k + dt * f(x_k, u_k)
    f must return dx/dt with same shape as x.
    """
    dx = np.asarray(f(x, u), dtype=float)
    return x + dt * dx


def rk4_step(f: DynFun, x: State, u: Control, dt: float) -> State:
    """
    Classic Runge–Kutta 4th order step.
    f(x, u) must return dx/dt with same shape as x.
    """
    k1 = np.asarray(f(x, u), dtype=float)
    k2 = np.asarray(f(x + 0.5 * dt * k1, u), dtype=float)
    k3 = np.asarray(f(x + 0.5 * dt * k2, u), dtype=float)
    k4 = np.asarray(f(x + dt * k3, u), dtype=float)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
