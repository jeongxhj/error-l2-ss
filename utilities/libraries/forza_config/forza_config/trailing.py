from pydantic import BaseModel, ConfigDict

from .yaml_io import PathLike, load_model


class TrailingConfig(BaseModel):
    """Gap controller used by every controller while TRAILING an opponent."""
    model_config = ConfigDict(extra='forbid', use_attribute_docstrings=True)

    allow_accel_trailing: bool
    """Allow accelerating above the raceline speed to close the gap"""
    trailing_mode: bool
    """True: the gap is a distance in meters; False: a time gap in seconds"""
    prioritize_dyn: bool
    """Prefer the dynamic obstacle when several are ahead"""
    trailing_gap: float
    """Target gap, meters or seconds depending on trailing_mode"""
    trailing_p_gain: float
    """P gain of the gap PID"""
    trailing_i_gain: float
    """I gain of the gap PID"""
    trailing_d_gain: float
    """D gain of the gap PID"""
    blind_trailing_speed: float
    """Speed to hold when the opponent is not seen"""


def load_trailing_config(path: PathLike) -> TrailingConfig:
    """``trailing_params.yaml`` at ``path``."""
    return load_model(TrailingConfig, path)
