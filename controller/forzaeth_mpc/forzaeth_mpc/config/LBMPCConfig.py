
from typing import Dict, List, Optional, Union

from pydantic import (BaseModel, ConfigDict, field_validator,
                      model_validator)

from .paths import config_root, load_model

NestedFloats = Union[List[float], List[List[float]]]


class GPParamsConfig(BaseModel):
    """Configuration for the GP residual model"""
    model_config = ConfigDict(extra='forbid')

    use_gp: bool = True
    stochastic_mode: str = "disabled"
    enable_offline_logging: bool = True
    log_offline_horizon: bool = False
    use_inducing_gp: bool = False
    max_num_data_points: int = 2000

    # Mandatory: same name as in YAML
    unc_jac_G_mat_idx: List[int]
    input_feature_selection: List[int]

    data_processing_mode: str = "learn"
    """GP operating mode: 'learn' updates and saves the model, 'use_learnt_model' loads the last saved one."""
    learning_enabled: bool = False
    """Runtime gate for online learning in 'learn' mode. Keeps GP setup intact, but only starts recording once enabled."""

    # Only relevant when use_inducing_gp is True
    num_inducing_points: int = 10

    # Free-form hyperparameters, matched by string keys
    hyperparameters: Optional[Dict[str, NestedFloats]] = None

    @field_validator("data_processing_mode")
    @classmethod
    def validate_data_processing_mode(cls, value: str) -> str:
        normalized = "use_learnt_model" if value == "use_learnet_model" else value
        allowed = {"learn", "use_learnt_model"}
        if normalized not in allowed:
            raise ValueError(
                "data_processing_mode must be one of ['learn', 'use_learnt_model']"
            )
        return normalized

    @field_validator("stochastic_mode")
    @classmethod
    def validate_stochastic_mode(cls, value: str) -> str:
        normalized = value.strip().lower()
        aliases = {
            "none": "disabled",
            "off": "disabled",
            "zero_feedback": "open_loop",
            "constant_feedback": "lqr",
            "time_varying_feedback": "riccati",
        }
        normalized = aliases.get(normalized, normalized)

        allowed = {"disabled", "open_loop", "lqr", "riccati"}
        if normalized not in allowed:
            raise ValueError(
                "stochastic_mode must be one of ['disabled', 'open_loop', 'lqr', 'riccati']"
            )
        return normalized

    @model_validator(mode="after")
    def validate_mode_consistency(self):
        if not self.use_gp and self.stochastic_mode != "disabled":
            raise ValueError(
                "stochastic_mode must be 'disabled' when use_gp is false"
            )
        return self


class LBMPCConfig(BaseModel):
    """Learning-Based MPC configuration class"""
    model_config = ConfigDict(extra='forbid', use_attribute_docstrings=True)

    # general parameters
    N: int
    """Prediction horizon"""
    t_delay: float
    """Delay in seconds accounted for propagation before the MPC is applied"""
    delay_sim: bool
    """Weather to use simulated model to compensate for delay"""
    steps_delay: int
    """Delays for the steering angle in case of delay_sim = False"""
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

    gp_params: GPParamsConfig
    """Parameters for the GP residual model"""


def load_LBMPC_config_ros(racecar_version: str) -> LBMPCConfig:
    """Loads the MPC config from the yaml file

    Args:
        racecar_version (str): a car name

    Returns:
        LBMPCConfig: the MPC configuration object
    """

    return load_model(LBMPCConfig, config_root() / racecar_version / 'lb_mpc_params.yaml')
