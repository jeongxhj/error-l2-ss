"""Round trips on files written in tmp_path, plus every car of a real tree when
FORZA_CONFIG_TREE points at a <stack_master>/config directory."""
import os
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from forza_config import (CarConfig, PacejkaTireConfig, load_car_config, load_pacejka_tire_config,
                        load_steering_lut, load_trailing_config, save_pacejka_tire_config,
                        save_steering_lut, strict_safe_load)

CAR = dict(m=3.695, Iz=0.05797, lf=0.170, lr=0.151, wheelbase=0.321, width_m=0.29, h_cg=0.014,
           a_max=3, a_min=-3, v_max=15.0, v_min=-5.0, C_0d=0.48, C_d=-1.1, C_acc=8.29, C_dec=5.77,
           C_R=2.03, C_0v=100, C_v=20, tau_steer=0.15779476, max_steering_angle=0.4189,
           min_radius_curv_m=0.75, max_steering_velocity=3.2)
TIRE = dict(Bf=4.142, Br=4.8721, Cf=0.4375, Cr=2.1651, Df=2.6476, Dr=1.057, Ef=0.2332, Er=1.692,
            floor="dubi", friction_coeff=1.0)
TRAILING = dict(allow_accel_trailing=False, trailing_mode=True, prioritize_dyn=True, trailing_gap=2.0,
                trailing_p_gain=1.0, trailing_i_gain=0.0, trailing_d_gain=0.2, blind_trailing_speed=1.5)


def test_duplicate_key_is_an_error():
    with pytest.raises(yaml.YAMLError, match="duplicate key"):
        strict_safe_load("a: 1\nb: 2\na: 3\n")
    assert strict_safe_load("a: 1\nnested:\n  c: 3\n") == {"a": 1, "nested": {"c": 3}}


def test_car_and_trailing_round_trip(tmp_path):
    (tmp_path / "car_model.yaml").write_text(yaml.dump(CAR))
    (tmp_path / "trailing_params.yaml").write_text(yaml.dump(TRAILING))
    cfg = load_car_config(tmp_path / "car_model.yaml")
    assert cfg.tau_steer == pytest.approx(0.15779476)
    assert cfg.model_dump() == CAR
    assert load_trailing_config(str(tmp_path / "trailing_params.yaml")).trailing_gap == 2.0


def test_schema_errors_name_the_file(tmp_path):
    path = tmp_path / "car_model.yaml"
    path.write_text(yaml.dump({**CAR, "typo_key": 1}))
    with pytest.raises(ValidationError) as exc:
        load_car_config(path)
    assert str(path) in str(exc.value) and "typo_key" in str(exc.value)
    bad = dict(CAR)
    del bad["Iz"]
    path.write_text(yaml.dump(bad))
    with pytest.raises(ValidationError, match="Iz"):
        load_car_config(path)
    path.write_text(yaml.dump({**CAR, "tau_steer": 0.0}))
    with pytest.raises(ValidationError, match="tau_steer"):
        load_car_config(path)
    path.write_text("- just\n- a list\n")
    with pytest.raises(ValueError, match="mapping"):
        load_car_config(path)
    with pytest.raises(FileNotFoundError):
        load_car_config(tmp_path / "missing.yaml")


def test_pacejka_archive(tmp_path):
    folder = tmp_path / "pacejka" / "dubi"
    folder.mkdir(parents=True)
    (folder / "default.yaml").write_text(yaml.dump(TIRE))
    assert load_pacejka_tire_config(folder / "default.yaml").Df == 2.6476

    new = PacejkaTireConfig(**{**TIRE, "Df": 3.0})
    written = save_pacejka_tire_config(new, folder)
    assert written.parent == folder / "archive" and written.name.endswith("_pacejka.yaml")
    default = folder / "default.yaml"
    assert default.is_symlink() and os.readlink(default) == f"archive/{written.name}"
    assert load_pacejka_tire_config(default).Df == 3.0


def test_steering_lut_round_trip(tmp_path):
    table = np.array([[np.nan, 1.0, 2.0], [0.1, 0.5, 1.1], [0.2, 0.9, np.nan]])
    save_steering_lut(table, tmp_path)
    back = load_steering_lut(tmp_path / "default.csv")
    np.testing.assert_array_equal(np.isnan(back), np.isnan(table))
    np.testing.assert_allclose(np.nan_to_num(back), np.nan_to_num(table))


@pytest.mark.skipif("FORZA_CONFIG_TREE" not in os.environ, reason="set FORZA_CONFIG_TREE=<stack_master>/config")
def test_real_tree_loads():
    root = Path(os.environ["FORZA_CONFIG_TREE"])
    cars = sorted(root.glob("*/car_model.yaml"))
    assert cars, f"no car_model.yaml under {root}"
    for path in cars:
        assert isinstance(load_car_config(path), CarConfig)
    for path in root.glob("*/trailing_params.yaml"):
        load_trailing_config(path)
    for path in root.glob("*/pacejka/*/default.yaml"):
        load_pacejka_tire_config(path)
