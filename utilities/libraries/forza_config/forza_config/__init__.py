"""ForzaETH car configs: pydantic schemas and path-based loaders, no ROS.

The values live with the stack, this package only knows their shape. Every loader
takes the file's path; where the files are (``<stack_master>/config/<car>/...``) is the
stack's convention, and its ROS layer resolves it::

    load_car_config(f"{config_dir}/NUC5/car_model.yaml")
    load_pacejka_tire_config(f"{config_dir}/NUC5/pacejka/dubi/default.yaml")
"""
from .archive import write_archived
from .car import CarConfig, load_car_config
from .steering_lut import load_steering_lut, save_steering_lut
from .tire import PacejkaTireConfig, load_pacejka_tire_config, save_pacejka_tire_config
from .trailing import TrailingConfig, load_trailing_config
from .yaml_io import StrictLoader, load_model, strict_safe_load

__all__ = [
    "CarConfig", "load_car_config",
    "PacejkaTireConfig", "load_pacejka_tire_config", "save_pacejka_tire_config",
    "TrailingConfig", "load_trailing_config",
    "load_steering_lut", "save_steering_lut",
    "StrictLoader", "strict_safe_load", "load_model", "write_archived",
]
