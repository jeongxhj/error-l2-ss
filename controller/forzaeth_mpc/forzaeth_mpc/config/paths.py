"""Where the sim and the module self-tests find a car's yamls: ``config_root()/<car>/``.

The race stack does not use this: its ROS layer (forza_config_ros) resolves the files in
stack_master/config. Point FORZAETH_MPC_CONFIG at such a config tree to run the sim.
"""
import os
from pathlib import Path

from forza_config import (load_car_config, load_model, load_pacejka_tire_config, load_trailing_config,  # noqa: F401
                          strict_safe_load)

_ENV_VAR = 'FORZAETH_MPC_CONFIG'


def config_root() -> Path:
    """Directory holding the per-car config dirs (NUC5/, DEFAULT/, ...), from FORZAETH_MPC_CONFIG,
    e.g. <race_stack>/stack_master/config."""
    root = os.environ.get(_ENV_VAR)
    if not root:
        raise RuntimeError(f"Set {_ENV_VAR} to a config tree, e.g. <race_stack>/stack_master/config")
    return Path(root)


def load_car_config_ros(racecar_version: str):
    return load_car_config(config_root() / racecar_version / 'car_model.yaml')


def load_pacejka_tire_config_ros(racecar_version: str, floor: str):
    return load_pacejka_tire_config(config_root() / racecar_version / 'pacejka' / floor / 'default.yaml')


def load_trailing_config_ros(racecar_version: str):
    return load_trailing_config(config_root() / racecar_version / 'trailing_params.yaml')
