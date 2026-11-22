# forza_config

Pydantic schemas and path-based loaders for the ForzaETH car configs (Python-only, no ROS).

| File | Schema | Load | Save |
|---|---|---|---|
| `car_model.yaml` | `CarConfig` | `load_car_config(path)` | hand-maintained |
| `pacejka/<floor>/default.yaml` | `PacejkaTireConfig` | `load_pacejka_tire_config(path)` | `save_pacejka_tire_config(cfg, folder)` |
| `LUT/<floor>/default.csv` | numpy array | `load_steering_lut(path)` | `save_steering_lut(table, folder)` |
| `trailing_params.yaml` | `TrailingConfig` | `load_trailing_config(path)` | hand-maintained |

Every schema is `extra='forbid'`: a missing or an extra key stops the load. A repeated
key in a yaml is an error too (`strict_safe_load`), not last-wins. `load_model(Model,
path)` is the generic loader the four above are built on; other packages use it for
their own schemas (forzaeth_mpc does for the MPC params).

## Values live with the stack

This package carries no yaml and does not know where the files are. Every loader takes
a path; every saver takes the folder in which to write the dated file and flip the
`default.*` symlink:

```python
from forza_config import load_car_config, load_pacejka_tire_config, save_pacejka_tire_config

car = load_car_config(f"{config_dir}/NUC5/car_model.yaml")
tire = load_pacejka_tire_config(f"{config_dir}/NUC5/pacejka/dubi/default.yaml")
save_pacejka_tire_config(tire, f"{config_dir}/NUC5/pacejka/dubi")   # -> archive/YYYYMMDD_pacejka.yaml
```

The layout (`stack_master/config/<car>/...`, floors, archives) is the stack's convention,
documented and resolved in its ROS layer. The savers only know "dated file in `archive/`,
symlink named `default`", so every identification stays on disk.

## ROS layers

A ROS package adds the package lookup and the layout, nothing else:

```python
# ROS 1
root = rospkg.RosPack().get_path("stack_master") + "/config"
# ROS 2 (the config folder is not mirrored into the install space, hence the source tree)
root = Path(get_package_share_directory("stack_master")).parents[3] / "src/race_stack/stack_master/config"

def load_car_config_ros(car):
    return load_car_config(f"{root}/{car}/car_model.yaml")

def load_pacejka_tire_config_ros(car, floor):   # the "no such floor" message belongs here too
    return load_pacejka_tire_config(f"{root}/{car}/pacejka/{floor}/default.yaml")
```

Anything that needs `rclpy`/`rospy` (parameter services, dynamic reconfigure) belongs
in that layer, not here.

The ROS 2 race stack's layer is `forza_config_ros` (`utilities/libraries/forza_config_ros`).

## Errors

- `FileNotFoundError`: the file is not there (a wrong floor included).
- `yaml.YAMLError`: malformed yaml or a duplicate key.
- `pydantic.ValidationError`: schema mismatch. The title names the file, so a missing
  key reads `1 validation error for CarConfig (/.../NUC5/car_model.yaml)`.

## Install and test

```bash
pip install -e .                                          # or build it in a colcon workspace
pytest                                                    # on a tree built in tmp_path
FORZA_CONFIG_TREE=<stack_master>/config pytest -k real_tree # every car of a real tree
```

Python >= 3.8, pydantic 2, pyyaml, numpy. `package.xml` only lets colcon and rosdep find the package; the metadata is in
`setup.py` because the ROS humble images pin setuptools 58, which ignores pyproject's `[project]`.
