"""ROS-free loader for stack_master's global_waypoints.json track files."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np

from .waypoints import WP, track_length_from_waypoints

_REPO_ROOT = Path(__file__).resolve().parents[2]

REFLINES = {
    "iqp": "global_traj_wpnts_iqp",
    "sp": "global_traj_wpnts_sp",
    "mintime": "global_traj_wpnts_mintime",
}

# Column order matches WP (x_m, y_m, vx_mps, kappa_radpm, s_m, d_left, d_right).
_FIELDS = ("x_m", "y_m", "vx_mps", "kappa_radpm", "s_m", "d_left", "d_right")


def _resolve_path(track: str) -> Path:
    """A path to a json file, a name in the tracks directory, or a race_stack map name."""
    p = Path(track)
    if p.is_file():
        return p
    tracks_dir = Path(os.environ.get("FORZAETH_MPC_TRACKS", _REPO_ROOT / "tracks"))
    # the second form is the layout of stack_master/maps, so pointing
    # FORZAETH_MPC_TRACKS at it needs no copy and no rename
    for candidate in (tracks_dir / f"{track}.json",
                      tracks_dir / track / "global_waypoints.json"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve track {track!r}: no {track}.json and no "
        f"{track}/global_waypoints.json in {tracks_dir}")


def load_track(track: str, refline: str = "iqp") -> Tuple[np.ndarray, float]:
    """Load a track's waypoints straight from its global_waypoints.json.

    Returns the canonical (N,7) array (see waypoints.WP for columns) and the
    track length in meters.
    """
    path = _resolve_path(track)
    with open(path, "r") as f:
        d = json.load(f)
    key = REFLINES[refline]
    if key not in d:
        have = [k for k, v in REFLINES.items() if v in d]
        raise KeyError(f"{path.name} has no {refline!r} raceline; it has {have}")
    wpnts = d[key]["wpnts"]
    arr = np.array([[float(wp[k]) for k in _FIELDS] for wp in wpnts], dtype=float)
    return arr, track_length_from_waypoints(arr)


if __name__ == "__main__":
    waypoints, track_len = load_track("THETRACK")
    assert waypoints.shape[1] == 7
    assert np.all(np.diff(waypoints[:, WP.S]) >= 0.0), "s column must be non-decreasing"
    assert track_len > 0
    print(f"OK: THETRACK -> {waypoints.shape[0]} wpnts, track_len={track_len:.3f} m")

    # the (s, n) -> (x, y) projection must not tear at the start line: it used to
    # interpolate an unwrapped heading periodically, which turned the normal a full circle
    # inside the last few centimetres of the lap
    from forzaeth_mpc.sim.cli.run_laps import frenet_to_xy
    s_q = np.linspace(-1.0, 1.0, 201)
    for n_q in (-0.5, 0.0, 0.5):
        x, y = frenet_to_xy(waypoints, s_q, np.full_like(s_q, n_q), track_len)
        jump = np.hypot(np.diff(x), np.diff(y)).max()
        assert jump < 0.05, f"projection tears at s=0 for n={n_q}: {jump:.3f} m"
    print("OK: projection continuous across the start line")

    # the stack_master/maps layout, <name>/global_waypoints.json, must resolve without a
    # copy and without a rename
    import shutil
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "SOMEMAP"
        d.mkdir()
        shutil.copy(_resolve_path("THETRACK"), d / "global_waypoints.json")
        os.environ["FORZAETH_MPC_TRACKS"] = tmp
        wp_sp, len_sp = load_track("SOMEMAP", refline="sp")
        assert wp_sp.shape[1] == 7 and len_sp > 0
        print(f"OK: <name>/global_waypoints.json resolves; sp raceline -> {len_sp:.2f} m")
