from __future__ import annotations

from typing import Tuple, TYPE_CHECKING

from forzaeth_mpc.config import STMPCConfig, load_STMPC_config_ros
from pydantic import BaseModel, ConfigDict

from .paths import config_root, load_model

if TYPE_CHECKING:
    from dynamic_reconfigure.parameter_generator_catkin import ParameterGenerator


class STMPCConfigDyn(BaseModel):
    """Single track MPC dynamic reconfigure configuration class"""
    model_config = ConfigDict(extra='forbid', use_attribute_docstrings=True)

    # Cost function settings #
    ##########################
    qadv: Tuple[float, float]
    """Limits on the advancement cost"""
    qv: Tuple[float, float]
    """Limits on the velocity cost"""
    qv_start: Tuple[float, float]
    """Limits on the velocity cost on the start mode"""
    qn: Tuple[float, float]
    """Limits on the lateral deviation cost"""
    qalpha: Tuple[float, float]
    """Limits on the heading deviation cost"""
    qjerk: Tuple[float, float]
    """Limits on the jerk cost"""
    qjerk_start: Tuple[float, float]
    """Limits on the jerk cost on the start mode"""
    qddelta: Tuple[float, float]
    """Limits on the steering angle rate cost"""
    alat_max: Tuple[float, float]
    """Limits on the maximum lateral acceleration"""
    alat_max_start: Tuple[float, float]
    """Limits on the maximum lateral acceleration on the start mode"""
    a_min: Tuple[float, float]
    """Limits on the minimum acceleration (Maximum Breaking)"""
    a_min_start: Tuple[float, float]
    """Limits on the minimum acceleration on the start mode"""
    a_max: Tuple[float, float]
    """Limits on the maximum acceleration"""
    a_max_start: Tuple[float, float]
    """Limits on the maximum acceleration on the start mode"""
    track_safety_margin: Tuple[float, float]
    """Limits on the track safety margin"""
    terminal_multiplier: Tuple[float, float]
    """Weight multiplier for the cost of the last state"""


def create_STMPC_dynamic_parameters(gen: ParameterGenerator):
    """Create dynamic reconfigure parameters for the single track MPC controller

    Args:
        gen (ParameterGenerator): The dynamic reconfigure parameter generator
    """
    default_car = "DEFAULT"
    # load default parameters from the yaml file
    MPCconfig: STMPCConfig = load_STMPC_config_ros(
        racecar_version=default_car)  # using the default_car version by default

    # load the dynamic reconfigure parameters
    config = load_model(STMPCConfigDyn, config_root() / default_car / 'single_track_mpc_dyn_params.yaml')

    # check dyn parameters are available in the normal config
    for key in config.model_dump().keys():
        if key not in MPCconfig.model_dump().keys():
            raise ValueError(
                f"Key <{key}> is not available in the normal config. Please only try to dynamically reconfigure only the available parameters.")

    # create the dynamic reconfigure parameters
    for key, value in config.model_dump().items():
        gen.add(name=key,
                paramtype='double',
                level=0,
                description=f"{config.model_fields[key].description}",
                default=MPCconfig.model_dump()[key],
                min=value[0],
                max=value[1],
                )

    return gen
