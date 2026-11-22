from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _minmax(arr: np.ndarray) -> Tuple[float, float]:
    vmin = float(np.nanmin(arr)) if np.isfinite(arr).any() else 0.0
    vmax = float(np.nanmax(arr)) if np.isfinite(arr).any() else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def plot_parallel_params_vs_metrics(
    params_matrix: np.ndarray,             # shape: (n_trials, n_params)
    param_names: Sequence[str],
    metrics_matrix: np.ndarray,            # shape: (n_trials, n_metrics)
    metric_names: Sequence[str],
    *,
    color_by: Optional[np.ndarray] = None,  # shape: (n_trials,), e.g. mean_cost
    cmap: str = "viridis",
    alpha: float = 0.45,
    lw: float = 1.0,
    fig_size: Tuple[int, int] = (12, 4),
    n_y_ticks: int = 5,
    title: Optional[str] = None,
    fixed_param_limits: Optional[Dict[str, Tuple[float, float]]] = None,
    fixed_metrics_limits: Optional[Dict[str, Tuple[float, float]]] = None,
) -> plt.Figure:
    """
    Draws a parallel-coordinates chart with one vertical axis per column:
    [all parameters | all metrics]. Each Optuna trial is a polyline.

    - Each axis is independently scaled to [0,1] for plotting,
      but tick labels show original values.
    - Lines are colored by 'color_by' (e.g. mean_cost) with a colorbar.
    """
    # Concatenate params + metrics
    cols = list(param_names) + list(metric_names)
    X = np.concatenate([params_matrix, metrics_matrix], axis=1)  # (N, D)
    N, D = X.shape
    xs = np.arange(D)

    # Normalize per-column; store min/max for ticks
    mins = np.zeros(D)
    maxs = np.zeros(D)
    Xn = np.zeros_like(X, dtype=float)
    for j in range(D):
        name = cols[j]
        if fixed_param_limits and name in fixed_param_limits:
            vmin, vmax = fixed_param_limits[name]
        elif fixed_metrics_limits and name in fixed_metrics_limits:
            vmin, vmax = fixed_metrics_limits[name]
        else:
            vmin, vmax = _minmax(X[:, j])
        mins[j], maxs[j] = vmin, vmax
        Xn[:, j] = (X[:, j] - vmin) / (vmax - vmin)

    # Figure
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    ax.set_xlim([xs[0], xs[-1]])
    ax.set_ylim([0.0, 1.0])
    ax.set_xticks(xs)
    ax.set_xticklabels(cols, rotation=0, fontsize=11)
    ax.set_ylabel("normalized per-axis", fontsize=9)

    # Vertical axes
    for x in xs:
        ax.vlines(x, 0, 1, color="lightgray", lw=1.0, zorder=0)

    # Colors
    if color_by is None:
        colors = [None] * N
        sm = None
    else:
        c = np.asarray(color_by, dtype=float)
        cmin, cmax = _minmax(c)
        c_norm = (c - cmin) / (cmax - cmin)
        colors = plt.get_cmap(cmap)(np.clip(c_norm, 0, 1))
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array(c)
        sm.set_clim(cmin, cmax)

    # Plot polylines
    for i in range(N):
        yi = Xn[i, :]
        ax.plot(xs, yi, color=(colors[i] if color_by is not None else "C0"),
                alpha=alpha, lw=lw)

    # Axis tick labels per vertical axis (show real values)
    for j, x in enumerate(xs):
        vmin, vmax = mins[j], maxs[j]
        ticks = np.linspace(0.0, 1.0, n_y_ticks)
        vals = vmin + ticks * (vmax - vmin)
        for t, v in zip(ticks, vals):
            ax.text(x - 0.02, t, f"{v:,.0f}" if abs(v) >= 10 else f"{v:.2f}",
                    ha="right", va="center", fontsize=8, color="dimgray")

    # Colorbar
    if sm is not None:
        cbar = fig.colorbar(sm, ax=ax, shrink=0.85)
        cbar.set_label(metric_names[-1] if len(metric_names) else "color", fontsize=9)
        # If you always want cost on the colorbar label:
        # cbar.set_label("Cost", fontsize=9)

    ax.grid(False)
    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig
