

from pydantic import BaseModel, ConfigDict

from .paths import config_root, load_model


class STMPCConfig(BaseModel):
    """Single track MPC configuration class"""
    model_config = ConfigDict(extra='forbid', use_attribute_docstrings=True)

    # general parameters
    N: int
    """Prediction horizon"""
    t_delay: float
    """Delay in seconds accounted for propagation before the MPC is applied"""
    steps_delay: int
    """Delays for the steering angle"""
    use_steering_servo: bool
    """Model the steering servo as a first order lag of ``car_config.tau_steer``, which
    makes the first input a commanded steering *angle* instead of a steering rate.

    False keeps the historical model, where the input is a rate and the servo is assumed
    instantaneous; the delay is then compensated by publishing the angle of the state
    ``steps_delay + 1`` nodes ahead, which only holds at one ``MPC_freq`` and has to
    absorb the servo lag as well as the transport delay. True sends the commanded angle
    the solver decided on, and leaves ``steps_delay`` covering the transport delay
    alone. The cost weights are not interchangeable between the two: ``q_ddelta``
    penalises the steering rate in both cases, but the reachable rates differ, so a
    weight set tuned for one wants retuning for the other."""
    MPC_freq: int
    """Frequency of the MPC controller"""
    track_safety_margin: float
    """Safety margin for the track"""
    track_max_width: float
    """Maximum width of the track"""
    overtake_d: float
    """MPC Overtake path lateral tracking distance"""

    # Cost function settings #
    qjerk: float
    """jerk cost weight"""
    qjerk_start: float
    """jerk cost weight on the yeet on the start mode"""
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
    qv_start: float
    """velocity tracking cost weight on the yeet on the start mode"""

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
    a_min: float
    """Minimum acceleration (Maximum Breaking)"""
    a_min_start: float
    """Minimum acceleration (Maximum Breaking) on the yeet on the start mode"""
    a_max: float
    """Maximum acceleration"""
    a_max_start: float
    """Maximum acceleration on the yeet on the start mode"""

    # input bounds
    ddelta_min: float
    """Minimum steering angle rate"""
    ddelta_max: float
    """Maximum steering angle rate"""
    jerk_min: float
    """Minimum jerk"""
    jerk_max: float
    """Maximum jerk"""

    # nonlinear constraint
    alat_max: float
    """Maximum lateral acceleration"""
    alat_max_start: float
    """Maximum lateral acceleration on the yeet on the start mode"""

    # Cost Flags
    vy_minimization: bool
    """If true, minimize the lateral velocity"""
    adv_maximization: bool
    """If true, maximize the advancement"""

    # constraints flags
    combined_constraints: str
    """ellipse/diamond/anything else for None"""

    # model flags
    load_transfer: bool
    """If true, load transfer is considered"""
    correct_v_y_dot: bool
    """If true, the lateral velocity derivative is from the model and not approximated"""

    # Terminal multiplier
    terminal_multiplier: float


def load_STMPC_config_ros(racecar_version: str) -> STMPCConfig:
    """Loads the MPC config from the yaml file

    Args:
        racecar_version (str): a car name

    Returns:
        STMPCConfig: the MPC configuration object
    """

    return load_model(STMPCConfig, config_root() / racecar_version / 'single_track_mpc_params.yaml')
