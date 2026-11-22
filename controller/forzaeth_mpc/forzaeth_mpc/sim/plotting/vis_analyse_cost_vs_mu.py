from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_mu_sensitivity_violin(delta_mu,
                               lap_times,
                               mean_abs_lat_dev,
                               mean_abs_vel_err,
                               mean_cost=None,
                               colors=("tab:blue", "tab:green", "tab:red", "tab:purple")):
    """
    Plot violin + scatter + mean line for metrics vs Δμ (track − MPC).
    Supports an optional 4th metric: mean_cost (objective).

    Inputs can be 1D (one sample per Δμ) or 2D (samples x μ values).
    In 2D, each column is treated as the group for a given Δμ.

    Returns:
        fig: matplotlib Figure
    """
    def group_by_mu(arr):
        a = np.asarray(arr)
        if a.ndim == 1:
            # one value per mu -> each group is a single-element array
            return [np.array([v]) for v in a]
        # 2D: decide grouping by columns (samples x mus), otherwise by rows
        if a.shape[1] <= a.shape[0]:
            return [a[:, j] for j in range(a.shape[1])]
        return [a[j, :] for j in range(a.shape[0])]

    lt_groups = group_by_mu(lap_times)
    lat_groups = group_by_mu(mean_abs_lat_dev)
    vel_groups = group_by_mu(mean_abs_vel_err)
    groups_all = [lt_groups, lat_groups, vel_groups]
    titles_all = ["Lap time [s]", "Mean |n| [m]", "Mean |vx−v*| [m/s]"]

    if mean_cost is not None:
        cost_groups = group_by_mu(mean_cost)
        # mean_cost goes on top (as in the original figure ordering)
        groups_all = [cost_groups] + groups_all
        titles_all = ["Mean cost (objective)"] + titles_all

    # x/labels
    delta_mu = np.asarray(delta_mu, dtype=float)
    n_mu = len(lt_groups)
    labels = [f"{dm:+.2f}" for dm in delta_mu]
    x = np.arange(1, n_mu + 1)

    # figure with one row per metric panel
    fig, axes = plt.subplots(len(groups_all), 1, figsize=(8.5, 12), sharex=True)
    if len(groups_all) == 1:
        axes = [axes]

    # choose colors for however many panels we drew
    cols = list(colors)
    if len(cols) < len(groups_all):
        cols += [f"C{i}" for i in range(len(groups_all) - len(cols))]

    # draw panels
    for ax, groups, col, yl in zip(axes, groups_all, cols, titles_all):
        # violins
        v = ax.violinplot(groups, positions=x, showmeans=False,
                          showmedians=False, showextrema=False)
        for b in v['bodies']:
            b.set_facecolor(col)
            b.set_alpha(0.4)
            b.set_edgecolor("black")
            b.set_linewidth(0.8)

        # scatter points (with a tiny jitter if multiple samples)
        rng = np.random.default_rng(42)
        for i, g in enumerate(groups, start=1):
            g = np.asarray(g, dtype=float)
            if g.size == 1:
                ax.scatter([i], g, color=col, s=40, zorder=3)
            else:
                jitter = (rng.random(g.size) - 0.5) * 0.25
                ax.scatter(i + jitter, g, color=col, alpha=0.7, s=20, zorder=3)

        # group means
        means = [float(np.mean(np.asarray(g, dtype=float))) for g in groups]
        ax.plot(x, means, marker='o', color=col, linewidth=2, zorder=4)

        ax.set_ylabel(yl)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(True, linestyle=':', alpha=0.6)

    axes[-1].set_xlabel("Δμ (track − MPC nominal)")
    fig.tight_layout()
    return fig
