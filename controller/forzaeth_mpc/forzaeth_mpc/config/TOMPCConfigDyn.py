from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from pydantic import BaseModel, ConfigDict

from .paths import config_root, load_model
from .TOMPCConfig import TOMPCConfig, load_TOMPC_config_ros

if TYPE_CHECKING:
    from dynamic_reconfigure.parameter_generator_catkin import ParameterGenerator


class TOMPCConfigDyn(BaseModel):
    """Time-optimal MPC dynamic reconfigure configuration class"""
    model_config = ConfigDict(extra='forbid', use_attribute_docstrings=True)

    # Cost function settings #
    ##########################
    qtime: Tuple[float, float]
    """Limits on the traversal time cost"""
    qv: Tuple[float, float]
    """Limits on the velocity tracking cost, active while trailing"""
    qn: Tuple[float, float]
    """Limits on the lateral deviation cost"""
    qalpha: Tuple[float, float]
    """Limits on the heading deviation cost"""
    qvy: Tuple[float, float]
    """Limits on the lateral velocity cost"""
    qjerk: Tuple[float, float]
    """Limits on the jerk cost"""
    qddelta: Tuple[float, float]
    """Limits on the steering angle rate cost"""
    qv_terminal: Tuple[float, float]
    """Limits on the terminal speed bonus"""
    terminal_multiplier: Tuple[float, float]
    """Weight multiplier for the cost of the last state"""
    alat_max: Tuple[float, float]
    """Limits on the maximum lateral acceleration"""
    a_min: Tuple[float, float]
    """Limits on the minimum acceleration (Maximum Breaking)"""
    a_max: Tuple[float, float]
    """Limits on the maximum acceleration"""
    v_max: Tuple[float, float]
    """Limits on the maximum velocity. With no reference speed to track this is the
    knob that caps the pace"""
    track_safety_margin: Tuple[float, float]
    """Limits on the track safety margin"""


def create_TOMPC_dynamic_parameters(gen: ParameterGenerator):
    """Create dynamic reconfigure parameters for the time-optimal MPC controller

    Args:
        gen (ParameterGenerator): The dynamic reconfigure parameter generator
    """
    default_car = "DEFAULT"
    # load default parameters from the yaml file
    MPCconfig: TOMPCConfig = load_TOMPC_config_ros(
        racecar_version=default_car)  # using the default_car version by default

    # load the dynamic reconfigure parameters
    config = load_model(TOMPCConfigDyn, config_root() / default_car / 'time_optimal_mpc_dyn_params.yaml')

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
