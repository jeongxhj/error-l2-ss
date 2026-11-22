import logging
from pydantic import BaseModel, ConfigDict

from .paths import config_root, load_model

class KMPCConfig(BaseModel):
    """Kinematic MPC configuration class"""
    model_config = ConfigDict(extra='forbid', use_attribute_docstrings=True)

    # general parameters
    N: int
    """Prediction horizon"""
    t_delay: float
    """Delay in seconds accounted for propagation before the MPC is applied"""
    steps_delay: int
    """Delays for the steering angle"""
    MPC_freq: int
    """Frequency of the MPC controller"""
    track_safety_margin: float
    """Safety margin for the track"""
    track_max_width: float
    """Maximum width of the track"""
    overtake_d: float
    """Distance to overtake"""


    # Cost function settings #
    qac: float
    """acceleration cost weight"""
    qddelta: float
    """steering angle rate cost weight"""
    qadv: float
    """advancement cost weight"""
    qn: float
    """lateral deviation cost weight"""
    qalpha: float
    """heading deviation cost weight"""
    qv: float
    """velocity tracking cost weight"""

    Zl: float
    """quadratic coefficient state slack variable"""
    Zu: float
    """quadratic coefficient input slack variable"""
    zl: float
    """linear coefficient state slack variable"""
    zu: float
    """linear coefficient input slack variable"""

    # Model constraints
    # state bounds
    delta_min: float
    """Minimum steering angle"""
    delta_max: float
    """Maximum steering angle"""
    v_min: float
    """Minimum velocity"""
    v_max: float
    """Maximum velocity"""

    # input bounds
    ddelta_min: float
    """Minimum steering angle rate"""
    ddelta_max: float
    """Maximum steering angle rate"""
    a_min: float
    """Minimum acceleration (Maximum Breaking)"""
    a_max: float
    """Maximum acceleration"""

    # nonlinear constraint
    alat_max: float
    """Maximum lateral acceleration"""


def load_KMPC_config_ros(racecar_version: str) -> KMPCConfig:
    """Loads the MPC config from the yaml file

    Args:
        racecar_version (str): a car name
    """

    return load_model(KMPCConfig, config_root() / racecar_version / 'kinematic_mpc_params.yaml')
