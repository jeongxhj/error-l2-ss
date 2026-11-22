from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from .archive import write_archived
from .yaml_io import PathLike, load_model


class PacejkaTireConfig(BaseModel):
    """Pacejka magic-formula lateral tyre parameters, front and rear."""
    model_config = ConfigDict(extra='forbid', use_attribute_docstrings=True)

    friction_coeff: float
    """Friction coefficient of the tire"""
    Bf: float
    """Stiffness factor (B) of the FRONT tire"""
    Cf: float
    """Shape factor (C) of the FRONT tire"""
    Df: float
    """Peak value (D) of the FRONT tire"""
    Ef: float
    """Curvature factor (E) of the FRONT tire"""

    Br: float
    """Stiffness factor (B) of the REAR tire"""
    Cr: float
    """Shape factor (C) of the REAR tire"""
    Dr: float
    """Peak value (D) of the REAR tire"""
    Er: float
    """Curvature factor (E) of the REAR tire"""

    floor: str
    """Floor the parameters were identified on"""


def load_pacejka_tire_config(path: PathLike) -> PacejkaTireConfig:
    """A Pacejka yaml at ``path``, e.g. ``<car>/pacejka/<floor>/default.yaml``."""
    return load_model(PacejkaTireConfig, path)


def save_pacejka_tire_config(config: PacejkaTireConfig, folder: PathLike,
                             update_latest: bool = True) -> Path:
    """Write ``folder/archive/<YYYYMMDD>_pacejka.yaml`` and, with ``update_latest``,
    point ``folder/default.yaml`` at it, e.g. ``folder = <car>/pacejka/<floor>``."""
    def write(path: Path):
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(config.model_dump(), fh)
    return write_archived(folder, "pacejka.yaml", write, update_latest)
