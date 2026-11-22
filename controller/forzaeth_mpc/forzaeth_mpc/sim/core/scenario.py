from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from forzaeth_mpc.config import (load_car_config_ros, load_KMPC_config_ros,
                                 load_LBMPC_config_ros,
                                 load_pacejka_tire_config_ros,
                                 load_STMPC_config_ros, load_trailing_config_ros)
from forzaeth_mpc.sim.tracks import load_track
from forzaeth_mpc.stmpc.controller import STMPC


class Scenario:
    """
    Loads a track + configs and can build a ready-to-use MPC object.

    - waypoints: canonical (N,7) array
    - track_len: float
    - make_mpc(dt): builds your STMPC using your real constructor
    - make_kmpc / make_lbmpc: the other two controllers, same track and configs
    - make_x0_full(mpc, s0,n0,psi0,vx0): returns a robust x0_full vector
    """

    def __init__(self,
                 track: str,
                 racecar_version: str,
                 floor: str,
                 refline: str = "iqp"):
        self.track = track
        self.racecar_version = racecar_version
        self.floor = floor
        self.refline_kind = refline

        self.waypoints, self.track_len = load_track(track, refline=refline)

        # configs (same as in your scripts)
        self.stmpc_config = load_STMPC_config_ros(racecar_version)
        self.car_config = load_car_config_ros(racecar_version)
        self.tire_config = load_pacejka_tire_config_ros(racecar_version, floor)
        self.trailing_config = load_trailing_config_ros(racecar_version)

    # ---- Builders -------------------------------------------------------- #
    def make_mpc(self,
                 s0: float,
                 n0: float,
                 psi0: float,
                 dt: float):
        pose_frenet = np.array([s0, n0, psi0], dtype=float)
        mpc = STMPC(
            pose_frenet=pose_frenet,
            waypoints_data=self.waypoints,
            racecar_version=self.racecar_version,
            stmpc_config=self.stmpc_config,
            car_config=self.car_config,
            tire_config=self.tire_config,
            trailing_config=self.trailing_config,
            controller_frequency=(1.0 / dt),
            using_gokart=False
        )
        return mpc

    def make_tompc(self, s0: float, n0: float, psi0: float, dt: float):
        """Time-optimal MPC. Same configs as STMPC plus its own schema; the horizon is
        a distance, so ``dt`` only sets the rate the controller is called at."""
        from forzaeth_mpc.config import load_TOMPC_config_ros
        from forzaeth_mpc.tompc import TOMPC
        return TOMPC(
            pose_frenet=np.array([s0, n0, psi0], dtype=float),
            waypoints_data=self.waypoints,
            racecar_version=self.racecar_version,
            tompc_config=load_TOMPC_config_ros(self.racecar_version),
            car_config=self.car_config,
            tire_config=self.tire_config,
            trailing_config=self.trailing_config,
            controller_frequency=(1.0 / dt),
            using_gokart=False,
        )

    def make_kmpc(self, dt: float):
        """Kinematic MPC. Takes no tire config (kinematic model) and no frequency."""
        from forzaeth_mpc.kmpc import Kinematic_MPC_Controller
        return Kinematic_MPC_Controller(
            waypoints_data=self.waypoints,
            racecar_version=self.racecar_version,
            kmpc_config=load_KMPC_config_ros(self.racecar_version),
            car_config=self.car_config,
            trailing_config=self.trailing_config,
        )

    def make_lbmpc(self, s0: float, n0: float, psi0: float, dt: float, lbmpc_path: str,
                   gp_mode: Optional[str] = None,
                   stochastic_mode: Optional[str] = None):
        """Learning-based MPC. `lbmpc_path` is a scratch dir for solver and GP artifacts.

        `stochastic_mode` overrides `gp_params.stochastic_mode` ("disabled" / "open_loop" /
        "lqr" / "riccati"). Every shipped fixture says "disabled", so the ZoRO path is
        never exercised unless this is passed.

        `gp_mode` overrides `gp_params.data_processing_mode`. The shipped fixtures use
        "use_learnt_model", which needs a previously saved GP; with none on disk the
        residual model never initializes and LBMPC silently degrades to nominal MPC.
        Pass "learn" to learn from scratch.
        """
        from forzaeth_mpc.lbmpc import LBMPC
        from forzaeth_mpc.config.paths import config_root
        lbmpc_config = load_LBMPC_config_ros(self.racecar_version)
        if gp_mode is not None:
            lbmpc_config.gp_params.data_processing_mode = gp_mode
        if stochastic_mode is not None:
            lbmpc_config.gp_params.stochastic_mode = stochastic_mode
        return LBMPC(
            pose_frenet=np.array([s0, n0, psi0], dtype=float),
            waypoints_data=self.waypoints,
            racecar_version=self.racecar_version,
            lbmpc_config=lbmpc_config,
            car_config=self.car_config,
            tire_config=self.tire_config,
            trailing_config=self.trailing_config,
            controller_frequency=(1.0 / dt),
            using_gokart=False,
            lbmpc_path=lbmpc_path,
            params_path=str(config_root() / self.racecar_version),
        )

    def make_x0_full(self,
                     mpc,
                     s0: float, n0: float, psi0: float, vx0: float) -> np.ndarray:
        nx_full = int(getattr(mpc.model, "n_x_full", 10))
        x0 = np.zeros((nx_full,), dtype=float)
        x0[0], x0[1], x0[2], x0[3] = s0, n0, psi0, vx0
        return x0
