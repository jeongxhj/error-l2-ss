"""On-disk artifacts of the LB-MPC solver, and the format of a saved GP bundle.

Nothing here imports gpytorch or l4acados. residual_model.py depends on this module;
the reverse edge does not exist.
"""
import hashlib
import os
import re
from dataclasses import dataclass
from glob import glob
from typing import Optional

import torch

from forzaeth_mpc.utils.indices import StateIndex


@dataclass
class SolverBuildConfig:
    build_c_code: bool
    path_json_ocp: str
    path_json_sim: str


STATE_LAYOUT = "-".join(e.name.lower() for e in sorted(StateIndex, key=lambda x: x.value))
"""Fingerprint of the state/input ordering a GP was trained under. Bundles saved under a
different ordering must be rejected, not silently reused."""


def _ensure_parent_dir(file_path: str) -> None:
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def get_build_file_hashes(files):
    solver_code = b""
    for file in files:
        with open(file, "rb") as gen_file:
            solver_code += gen_file.read()

    return hashlib.md5(solver_code).hexdigest()


def check_file_changes(
    files: list, hashfile_name: str, rewrite_hash=False, rewrite_hash_if_changed=True
):
    curr_hash = get_build_file_hashes(files)
    _ensure_parent_dir(hashfile_name)
    try:
        with open(hashfile_name, "r", encoding="utf-8") as hashfile:
            last_hash = hashfile.read()
            hash_changed = last_hash != curr_hash
    except FileNotFoundError:
        hash_changed = True

    if rewrite_hash or (hash_changed and rewrite_hash_if_changed):
        with open(hashfile_name, "w", encoding="utf-8") as hashfile:
            hashfile.write(curr_hash)

    return hash_changed


def _ensure_text_artifact(file_path: str, default_contents: str = "") -> bool:
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    if os.path.exists(file_path):
        return False

    with open(file_path, "w", encoding="utf-8") as artifact_file:
        artifact_file.write(default_contents)
    return True


def ensure_placeholder_gp_bundle(model_path: str) -> bool:
    parent_dir = os.path.dirname(model_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    if os.path.exists(model_path):
        return False

    placeholder_bundle = {
        "artifact_status": "placeholder",
        "train_x": None,
        "train_y": None,
        "metadata": {
            "placeholder": True,
        },
    }
    tmp_path = f"{model_path}.tmp"
    torch.save(placeholder_bundle, tmp_path)
    os.replace(tmp_path, model_path)
    return True


def ensure_runtime_artifacts(lbmpc_path: str) -> None:
    created_artifacts = []

    artifact_specs = (
        (os.path.join(lbmpc_path, "data/hash", "build_c_code_hash.md5"), ""),
        (os.path.join(lbmpc_path, "data/ocp", "lbmpc_ocp_solver_config.json"), "{}\n"),
        (os.path.join(lbmpc_path, "data/ocp", "lbmpc_sim_solver_config.json"), "{}\n"),
    )
    for artifact_path, default_contents in artifact_specs:
        if _ensure_text_artifact(artifact_path, default_contents=default_contents):
            created_artifacts.append(artifact_path)

    learnt_gp_model_path = os.path.join(lbmpc_path, "data", "models", "latest_gp_model.pt")
    if ensure_placeholder_gp_bundle(learnt_gp_model_path):
        created_artifacts.append(learnt_gp_model_path)

    if created_artifacts:
        print(
            "[LBMPC] Created missing runtime artifact(s): "
            + ", ".join(created_artifacts)
        )


def get_invalid_gp_bundle_reason(bundle) -> Optional[str]:
    if bundle is None:
        return "the saved bundle could not be loaded"
    if not isinstance(bundle, dict):
        return "the saved bundle has an unexpected format"

    metadata = bundle.get("metadata") or {}
    if bundle.get("artifact_status") == "placeholder" or metadata.get("placeholder", False):
        return "it is only a first-run placeholder"

    if bundle.get("train_x") is None or bundle.get("train_y") is None:
        return "it does not contain training data"

    if "gp_model_state_dict" not in bundle or "likelihood_state_dict" not in bundle:
        return "it does not contain serialized GP weights"

    saved_layout = metadata.get("state_layout")
    if saved_layout != STATE_LAYOUT:
        return (f"it was trained under a different state layout "
                f"({saved_layout!r} != {STATE_LAYOUT!r})")

    return None


def build_solver_build_config(lbmpc_path: str, params_path: str) -> SolverBuildConfig:
    ensure_runtime_artifacts(lbmpc_path)

    # The two sources whose changes must force a C-code rebuild sit next to this module.
    # They were looked up under "<lbmpc_path>/src/", which only worked while lbmpc_path
    # happened to be the package directory. lbmpc_path now means only "where the
    # generated artifacts go".
    _pkg_dir = os.path.dirname(os.path.abspath(__file__))
    solver_settings_file_name = os.path.join(_pkg_dir, "acados_settings.py")
    # the model lives in stmpc/ now: STMPC and LB-MPC share one single-track model.
    # If this path goes stale, LB-MPC silently reuses a compiled solver that no longer
    # matches its model, and every parity check passes for the wrong reason.
    bicycle_model_file_name = os.path.join(
        os.path.dirname(_pkg_dir), "stmpc", "bicycle_model.py")
    params_file_name = os.path.join(params_path, "lb_mpc_params.yaml")
    mpc_files_to_watch = [
        solver_settings_file_name,
        bicycle_model_file_name,
        params_file_name,
    ]

    build_hash_path = os.path.join(lbmpc_path, "data/hash", "build_c_code_hash.md5")
    path_json_ocp = os.path.join(lbmpc_path, "data/ocp", "lbmpc_ocp_solver_config.json")
    path_json_sim = os.path.join(lbmpc_path, "data/ocp", "lbmpc_sim_solver_config.json")

    build_c_code = check_file_changes(mpc_files_to_watch, build_hash_path)
    build_c_code = build_c_code or not cython_solver_artifacts_available(
        path_json_ocp=path_json_ocp,
        path_json_sim=path_json_sim,
    )

    return SolverBuildConfig(
        build_c_code=build_c_code,
        path_json_ocp=path_json_ocp,
        path_json_sim=path_json_sim,
    )


def cython_solver_artifacts_available(path_json_ocp: str, path_json_sim: str) -> bool:
    for json_path, module_stem in (
        (path_json_ocp, "acados_ocp_solver_pyx"),
        (path_json_sim, "acados_sim_solver_pyx"),
    ):
        code_export_directory = read_code_export_directory(json_path)
        if not code_export_directory:
            return False

        artifact_pattern = os.path.join(code_export_directory, f"{module_stem}*.so")
        if not glob(artifact_pattern):
            return False

    return True


def read_code_export_directory(json_path: str) -> Optional[str]:
    if not os.path.exists(json_path):
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as json_file:
            match = re.search(
                r'"code_export_directory"\s*:\s*"([^"]+)"',
                json_file.read(),
            )
    except OSError:
        return None

    if not match:
        return None

    return os.path.abspath(match.group(1))


def _state_dict_to_cpu(state_dict: dict) -> dict:
    cpu_state_dict = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            cpu_state_dict[key] = value.detach().cpu()
        else:
            cpu_state_dict[key] = value
    return cpu_state_dict


def save_gp_model_bundle(model_path: str, gp_model, metadata: dict = None) -> None:
    train_x = None
    train_inputs = getattr(gp_model, "train_inputs", None)
    if train_inputs is not None and len(train_inputs) > 0:
        train_x = train_inputs[0]

    train_y = getattr(gp_model, "train_targets", None)

    bundle = {
        "gp_model_state_dict": _state_dict_to_cpu(gp_model.state_dict()),
        "likelihood_state_dict": _state_dict_to_cpu(gp_model.likelihood.state_dict()),
        "train_x": train_x.detach().cpu() if torch.is_tensor(train_x) else train_x,
        "train_y": train_y.detach().cpu() if torch.is_tensor(train_y) else train_y,
        "metadata": metadata or {},
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    tmp_path = f"{model_path}.tmp"
    torch.save(bundle, tmp_path)
    os.replace(tmp_path, model_path)


def load_gp_model_bundle(model_path: str):
    if not os.path.exists(model_path):
        return None
    try:
        return torch.load(model_path, map_location="cpu")
    except Exception as exc:
        print(f"[LBMPC] Failed to load GP model bundle from {model_path}: {exc}")
        return None
