"""YAML reading with two guarantees the plain loader does not give: a repeated key is an
error, and a file that does not match its schema names the file."""
from pathlib import Path
from typing import Type, TypeVar, Union

import yaml
from pydantic import BaseModel, ValidationError

PathLike = Union[str, Path]
M = TypeVar("M", bound=BaseModel)


class StrictLoader(yaml.SafeLoader):
    """SafeLoader that rejects a repeated mapping key instead of taking the last one.

    Plain ``yaml.safe_load`` is last-wins and silent. A tyre-constraint flag once came
    to be declared twice in a car's MPC yaml, ``true`` on one line and ``false`` nine
    lines later, so the controller ran a different lateral acceleration than the file
    appeared to say. Pydantic cannot catch that: by the time the schema sees the dict,
    one of the two values is gone.
    """

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.YAMLError(f"duplicate key {key!r} in {key_node.start_mark}")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def strict_safe_load(stream):
    """``yaml.safe_load``, except a repeated key is an error, not last-wins."""
    return yaml.load(stream, Loader=StrictLoader)


def load_model(model: Type[M], path: PathLike) -> M:
    """Read ``path`` as YAML and validate it as ``model``.

    Raises ``FileNotFoundError``, ``yaml.YAMLError`` (malformed or duplicate key),
    ``ValueError`` (the file is not a mapping) or pydantic's ``ValidationError``. The
    ``ValidationError`` is re-raised with the file path in its title, so a missing or
    extra key reads as "1 validation error for CarConfig (<path>) / tau_steer / Field
    required", and its ``errors()`` are the original ones.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        data = strict_safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level, got {type(data).__name__}")
    try:
        return model(**data)
    except ValidationError as exc:
        line_errors = [{k: v for k, v in err.items() if k in ("type", "loc", "input", "ctx")}
                       for err in exc.errors()]
        raise ValidationError.from_exception_data(f"{model.__name__} ({path})", line_errors) from exc
