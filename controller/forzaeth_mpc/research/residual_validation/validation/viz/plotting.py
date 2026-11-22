# plotting.py
from typing import Optional, Union

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch
from research.residual_validation.mpc_core.utils.indices import StateIndex

from ...mpc_core.utils.indices import IDX


def plot_dict_kept_dropped(Z_kept, Z_dropped, dims=None, title="Dictionary events", save_path=None, three_d=False):
    """
    Overlay plot of kept vs dropped points.
    dims: tuple of indices to plot (i,j[,k]); defaults to first two dims.
    three_d: set True to use 3D when dims has length 3.
    """
    if isinstance(Z_kept, list) and len(Z_kept) > 0:
        Zk = np.vstack(Z_kept)
    elif isinstance(Z_kept, np.ndarray):
        Zk = Z_kept
    else:
        Zk = np.empty((0, 3))
    if isinstance(Z_dropped, list) and len(Z_dropped) > 0:
        Zd = np.vstack(Z_dropped)
    elif isinstance(Z_dropped, np.ndarray):
        Zd = Z_dropped
    else:
        Zd = np.empty((0, 3))

    # pick dims
    nz = max(Zk.shape[1] if Zk.size else 0, Zd.shape[1] if Zd.size else 0)
    if dims is None:
        if nz < 3:
            raise ValueError(f"Need at least 3 dims to plot 3D, got nz={nz}. Provide dims=(i,j,k).")
        dims = (0, 1, 2)
    if len(dims) != 3:
        raise ValueError(f"dims must be a tuple of 3 indices, got {dims}")

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')

    if Zk.size:
        ax.scatter(Zk[:, dims[0]], Zk[:, dims[1]], Zk[:, dims[2]],
                   s=18, alpha=0.9, label="kept", depthshade=True)
    if Zd.size:
        ax.scatter(Zd[:, dims[0]], Zd[:, dims[1]], Zd[:, dims[2]],
                   s=20, alpha=0.9, marker="x", label="dropped", depthshade=False)

    ax.set_xlabel(f"z[{dims[0]}]")
    ax.set_ylabel(f"z[{dims[1]}]")
    ax.set_zlabel(f"z[{dims[2]}]")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
        print(f"Saved: {save_path}")
    else:
        plt.show()
