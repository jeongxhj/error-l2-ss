from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple, Union
import numpy as np


@dataclass(frozen=True)
class WpCols:
    """Column indices for the canonical waypoint array shape (N, 7)."""
    X: int = 0
    Y: int = 1
    VX: int = 2
    KAPPA: int = 3
    S: int = 4
    D_LEFT: int = 5
    D_RIGHT: int = 6


WP = WpCols()


def track_length_from_waypoints(waypoints: np.ndarray) -> float:
    """
    Best-effort track length from s column; if not reliable, falls back to
    last s or to XY arc-length.
    """
    s = np.asarray(waypoints[:, WP.S], dtype=float)
    if s.size == 0:
        return 0.0
    if np.all(np.diff(s) >= 0.0) and s[-1] > 0.0:
        return float(s[-1])
    # fallback to XY curve length
    xy = waypoints[:, [WP.X, WP.Y]]
    ds = np.hypot(*(np.diff(xy, axis=0).T))
    return float(np.sum(ds))
