from pydantic import BaseModel, ConfigDict, Field

from .yaml_io import PathLike, load_model


class CarConfig(BaseModel):
    """Car parameters: geometry, mass, actuator mappings and limits."""
    model_config = ConfigDict(extra='forbid', use_attribute_docstrings=True)

    m: float
    """Mass of the car"""
    Iz: float
    """Moment of inertia of the car"""
    lf: float
    """Distance from the center of mass to the front axle"""
    lr: float
    """Distance from the center of mass to the rear axle"""
    wheelbase: float
    """Wheelbase of the car"""
    width_m: float
    """Width of the car"""
    h_cg: float
    """Height of the center of gravity"""
    a_max: float
    """Maximum acceleration"""
    a_min: float
    """Minimum acceleration (maximum deceleration)"""
    v_max: float
    """Maximum velocity"""
    v_min: float
    """Minimum velocity"""

    C_0d: float
    """Steering Angle-to-Servo Offset"""
    C_d: float
    """Steering Angle-to-Servo Gain"""
    C_acc: float
    """Acceleration to Current Gain"""
    C_dec: float
    """Deceleration to Current Gain"""
    C_R: float
    """Velocity to Current Gain"""
    C_0v: float
    """Velocity-to-erpm Offset"""
    C_v: float
    """Velocity-to-erpm Gain"""

    tau_steer: float = Field(gt=0)
    """Steering servo time constant [s], identified per car.
    Every MPC models the servo as a first order lag on the commanded angle,
    ``delta_dot = (delta_cmd - delta) / tau_steer``, so this is load bearing for the
    controllers, not only for the sysid model. Must be > 0."""
    max_steering_angle: float
    """Maximum Steering Angle in radians"""
    min_radius_curv_m: float
    """Minimum turning radius in meters (wheelbase / tan(max_steering_angle))"""
    max_steering_velocity: float
    """Maximum Steering Velocity"""


def load_car_config(path: PathLike) -> CarConfig:
    """``car_model.yaml`` at ``path``."""
    return load_model(CarConfig, path)
