from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from forzaeth_mpc.config import load_KMPC_config_ros, KMPCConfig
from typing import Tuple, TYPE_CHECKING

from .paths import config_root, load_model

if TYPE_CHECKING:
    from dynamic_reconfigure.parameter_generator_catkin import ParameterGenerator

class KMPCConfigDyn(BaseModel):
    """Kinematic MPC dynamic reconfigure configuration class"""
    model_config = ConfigDict(extra='forbid', use_attribute_docstrings=True)

    # Cost function settings #
    ##########################
    qadv: Tuple[float, float]
    """Limits on the advancement cost"""
    qv: Tuple[float, float]
    """Limits on the velocity cost"""
    qn: Tuple[float, float]
    """Limits on the lateral deviation cost"""
    qalpha: Tuple[float, float]
    """Limits on the heading deviation cost"""
    qac: Tuple[float, float]
    """Limits on the acceleration cost"""
    qddelta: Tuple[float, float]
    """Limits on the steering angle rate cost"""
    alat_max: Tuple[float, float]
    """Limits on the maximum lateral acceleration"""
    a_min: Tuple[float, float]
    """Limits on the minimum acceleration (Maximum Breaking)"""
    a_max: Tuple[float, float]
    """Limits on the maximum acceleration"""
    v_min: Tuple[float, float]
    """Limits on the minimum velocity"""
    v_max: Tuple[float, float]
    """Limits on the maximum velocity"""
    track_safety_margin: Tuple[float, float]
    """Limits on the track safety margin"""


def create_KMPC_dynamic_parameters(gen: ParameterGenerator):
    """Create dynamic reconfigure parameters for the kinematic MPC controller

    Args:
        gen (ParameterGenerator): The dynamic reconfigure parameter generator
    """
    default_car = "DEFAULT"
    # load default parameters from the yaml file
    MPCconfig: KMPCConfig = load_KMPC_config_ros(racecar_version=default_car) # using the default_car version by default

    # load the dynamic reconfigure parameters
    config = load_model(KMPCConfigDyn, config_root() / default_car / 'kinematic_mpc_dyn_params.yaml')

    # check dyn parameters are available in the normal config
    for key in config.model_dump().keys():
        if key not in MPCconfig.model_dump().keys():
            raise ValueError(f"Key <{key}> is not available in the normal config. Please only try to dynamically reconfigure only the available parameters.")

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
