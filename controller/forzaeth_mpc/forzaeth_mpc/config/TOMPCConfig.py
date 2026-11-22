from pydantic import BaseModel, ConfigDict

from .paths import config_root, load_model


class TOMPCConfig(BaseModel):
    """Time-optimal MPC configuration class.

    The controller is a spatial reformulation: the independent variable is the path
    coordinate ``s``, not time, and the objective is the time the horizon takes to
    traverse. So the horizon is a *distance* (``s_horizon``) and not a duration, and
    there is no velocity reference to track while racing -- ``qv`` and ``V_target``
    only act while trailing an opponent.
    """
    model_config = ConfigDict(extra='forbid', use_attribute_docstrings=True)

    # general parameters
    N: int
    """Number of shooting nodes over the horizon"""
    s_horizon: float
    """Length of the prediction horizon along the path [m]. The node spacing is s_horizon/N"""
    MPC_freq: int
    """Rate at which the controller is called [Hz]"""
    t_delay: float
    """Actuation delay [s]. Converted to a horizon node with the current speed, so the
    applied command is read from the node the car will actually have reached"""
    track_safety_margin: float
    """Safety margin kept away from both track boundaries [m]"""
    track_max_width: float
    """Maximum width of the track [m], the bound on the boundary constraints"""
    overtake_d: float
    """Lateral tracking offset used while overtaking [m]"""
    s_dot_min: float
    """Lower bound on ds/dt [m/s]. The spatial dynamics divide by ds/dt, so this bound
    is what keeps the model non-singular; it must be strictly positive"""
    s_dot_max: float
    """Upper bound on ds/dt [m/s]. ds/dt has (1 - kappa*n) in its denominator, so a
    large enough lateral deviation drives the predicted traversal time to zero -- an
    optimum the solver will otherwise buy by paying the soft boundary penalty. This
    bound is hard, and it is what makes the objective bounded. Set it above the largest
    ds/dt the car can physically reach, roughly 2 * v_max"""

    # Cost function settings #
    qtime: float
    """Weight on the traversal time: the time term of the cost is exactly
    ``qtime * (horizon traversal time in seconds)``. Every other weight is therefore
    read as a trade against ``qtime`` per second. Only the ratios change the optimum,
    so the default is large enough to keep every cost term of order 1, which the QP
    prefers, rather than 1"""
    qv: float
    """Velocity tracking cost weight. Only active while TRAILING, where the target speed
    comes from the gap controller instead of from the time-optimal objective"""
    qn: float
    """Lateral deviation cost weight"""
    qalpha: float
    """Heading deviation cost weight"""
    qvy: float
    """Lateral velocity cost weight. Set to 0 to disable the drift penalty"""
    qjerk: float
    """Jerk cost weight"""
    qddelta: float
    """Steering angle rate cost weight"""
    qv_terminal: float
    """Bonus on the terminal speed, ``- qv_terminal * v_x`` at the last node. Stands in
    for the time still to be spent beyond the horizon: with 0 the controller optimises
    only the distance it can see, and gives up exit speed at the end of it"""
    terminal_multiplier: float
    """Multiplier on the terminal state cost"""

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
    """Minimum steering angle [rad]"""
    delta_max: float
    """Maximum steering angle [rad]"""
    v_min: float
    """Minimum velocity [m/s]. Must be strictly positive: the spatial model is singular
    at standstill"""
    v_max: float
    """Maximum velocity [m/s]. With no velocity reference to track, this bound and the
    acceleration limits are what set the pace"""
    a_min: float
    """Minimum acceleration, i.e. maximum braking [m/s^2]"""
    a_max: float
    """Maximum acceleration [m/s^2]"""

    # input bounds
    ddelta_min: float
    """Minimum steering angle rate [rad/s]"""
    ddelta_max: float
    """Maximum steering angle rate [rad/s]"""
    jerk_min: float
    """Minimum jerk [m/s^3]"""
    jerk_max: float
    """Maximum jerk [m/s^3]"""

    # nonlinear constraint
    alat_max: float
    """Maximum lateral acceleration [m/s^2]"""

    # constraints flags
    combined_constraints: str
    """ellipse/diamond/anything else for None. The combined longitudinal-lateral
    acceleration limit. A time-optimal objective sits on whichever limit binds, so this
    is the constraint that decides the lap time"""

    # model flags
    load_transfer: bool
    """If true, longitudinal load transfer is considered in the axle loads"""


def load_TOMPC_config_ros(racecar_version: str) -> TOMPCConfig:
    """Loads the time-optimal MPC config from the yaml file

    Args:
        racecar_version (str): a car name

    Returns:
        TOMPCConfig: the MPC configuration object
    """

    return load_model(TOMPCConfig, config_root() / racecar_version / 'time_optimal_mpc_params.yaml')
