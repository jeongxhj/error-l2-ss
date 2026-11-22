#!/usr/bin/env python3
import argparse
from pathlib import Path

import gpytorch
import numpy as np
import torch
from l4acados.models.pytorch_models.gpytorch_models.gpytorch_gp import \
    BatchIndependentMultitaskGPModel
import yaml
from forzaeth_mpc.config.paths import strict_safe_load


DEFAULT_INPUT_FEATURE_SELECTION = np.array(
    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
    dtype=bool,
)

STATE_COLUMNS = [
    "pos_on_center_line_s",
    "min_dist_to_center_line_n",
    "orientation_theta",
    "velocity_v_x",
    "velocity_v_y",
    "yaw_rate",
    "steering_angle_delta",
    "accel",
]

INPUT_DIMENSION = 2


def parse_args():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]

    parser = argparse.ArgumentParser(
        description="Train GP hyperparameters from archived LBMPC data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root / "stack_master" / "config" / "DEFAULT" / "lb_mpc_params.yaml",
        help="LBMPC YAML used to read gp_params.input_feature_selection.",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=script_dir.parent / "data" / "experiments_data",
        help="Directory searched recursively for archived step_log.csv files.",
    )
    parser.add_argument(
        "--offline-step-log",
        type=Path,
        default=script_dir.parent / "data" / "offline_viz" / "csv" / "step_log.csv",
        help="Fallback online log used when experiments_data has no archived runs.",
    )
    parser.add_argument(
        "--model-bundle",
        type=Path,
        default=script_dir.parent / "data" / "models" / "latest_gp_model.pt",
        help="Fallback learnt-model bundle used if no step logs are available.",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "experiments", "offline_viz", "model_bundle"],
        default="auto",
        help="Preferred source for training data.",
    )
    return parser.parse_args()


def load_input_feature_selection(config_path: Path) -> np.ndarray:
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            cfg = strict_safe_load(file) or {}
        input_selection = cfg["gp_params"]["input_feature_selection"]
        mask = np.asarray(input_selection, dtype=bool)
        if mask.size != DEFAULT_INPUT_FEATURE_SELECTION.size:
            raise ValueError(
                f"Expected {DEFAULT_INPUT_FEATURE_SELECTION.size} input features, got {mask.size}."
            )
        return mask
    except Exception as exc:
        print(
            f"[train_gp_hyperparams] Failed to read input_feature_selection from {config_path}: {exc}"
        )
        print("[train_gp_hyperparams] Falling back to the default feature selection.")
        return DEFAULT_INPUT_FEATURE_SELECTION.copy()


def load_data_from_bundle(model_bundle_path: Path, input_mask: np.ndarray):
    if not model_bundle_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {model_bundle_path}")

    bundle = torch.load(model_bundle_path, map_location="cpu")
    z = bundle.get("train_x")
    y = bundle.get("train_y")
    if z is None or y is None:
        raise ValueError(f"Model bundle {model_bundle_path} does not contain train_x/train_y.")

    z = z.detach().cpu().float() if torch.is_tensor(z) else torch.as_tensor(z, dtype=torch.float32)
    y = y.detach().cpu().float() if torch.is_tensor(y) else torch.as_tensor(y, dtype=torch.float32)

    if z.ndim == 1:
        z = z.unsqueeze(-1)
    if y.ndim == 1:
        y = y.unsqueeze(-1)

    # Saved bundles already store the feature-selected train_x. Only apply the mask
    # if the bundle still contains the full [x, u] vector.
    if z.shape[1] == input_mask.size:
        z = z[:, input_mask]

    print(
        f"[train_gp_hyperparams] Using learnt model bundle {model_bundle_path} "
        f"with {z.shape[0]} points."
    )
    return z, y


def _load_structured_csv(csv_path: Path):
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if getattr(data, "dtype", None) is None or data.dtype.names is None:
        return None
    return np.atleast_1d(data)


def load_data_from_step_logs(step_log_paths, input_mask: np.ndarray):
    z_rows = []
    y_rows = []

    residual_columns = [f"r_meas_{idx}" for idx in range(3)]
    meas_state_columns = [f"x_meas_{name}" for name in STATE_COLUMNS]

    for step_log_path in step_log_paths:
        data = _load_structured_csv(step_log_path)
        if data is None or data.size < 2:
            continue

        missing_columns = [
            col for col in (["lap"] + residual_columns + meas_state_columns) if col not in data.dtype.names
        ]
        if missing_columns:
            print(
                f"[train_gp_hyperparams] Skipping {step_log_path}: missing columns {missing_columns}"
            )
            continue

        for idx in range(1, data.shape[0]):
            if int(data["lap"][idx]) != int(data["lap"][idx - 1]):
                continue

            residual = np.asarray(
                [data[col][idx] for col in residual_columns],
                dtype=np.float32,
            )
            if not np.any(np.isfinite(residual)):
                continue

            x_prev = np.asarray(
                [data[col][idx - 1] for col in meas_state_columns],
                dtype=np.float32,
            )

            # This mirrors the current runtime implementation in lbmpc/controller.py,
            # where _prev_input_for_residual is stored as zeros.
            u_prev = np.zeros(INPUT_DIMENSION, dtype=np.float32)
            z_full = np.concatenate((x_prev, u_prev))

            z_rows.append(z_full[input_mask])
            y_rows.append(residual)

    if not z_rows:
        raise ValueError("No valid residual datapoints were found in the provided step logs.")

    z = np.asarray(z_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.float32)

    active_residual_columns = np.any(np.isfinite(y), axis=0)
    if not np.any(active_residual_columns):
        raise ValueError("Step logs were found, but they do not contain any finite measured residuals.")
    y = y[:, active_residual_columns]

    valid_rows = np.all(np.isfinite(z), axis=1) & np.all(np.isfinite(y), axis=1)
    if not np.any(valid_rows):
        raise ValueError("Step logs were found, but none of the rows contain finite training data.")
    z = torch.as_tensor(z[valid_rows], dtype=torch.float32)
    y = torch.as_tensor(y[valid_rows], dtype=torch.float32)

    print(
        f"[train_gp_hyperparams] Using {z.shape[0]} points from {len(step_log_paths)} step log(s)."
    )
    return z, y


def find_step_logs(experiments_dir: Path):
    if not experiments_dir.exists():
        return []
    return sorted(experiments_dir.rglob("step_log.csv"))


def load_training_data(args, input_mask: np.ndarray):
    experiment_step_logs = find_step_logs(args.experiments_dir)

    if args.source in ("auto", "experiments") and experiment_step_logs:
        try:
            return load_data_from_step_logs(experiment_step_logs, input_mask)
        except ValueError as exc:
            if args.source == "experiments":
                raise
            print(f"[train_gp_hyperparams] Skipping experiments_data source: {exc}")

    if args.source == "experiments":
        raise FileNotFoundError(
            f"No step_log.csv files were found under {args.experiments_dir}."
        )

    if args.source in ("auto", "offline_viz") and args.offline_step_log.exists():
        try:
            return load_data_from_step_logs([args.offline_step_log], input_mask)
        except ValueError as exc:
            if args.source == "offline_viz":
                raise
            print(f"[train_gp_hyperparams] Skipping offline_viz source: {exc}")

    if args.source == "offline_viz":
        raise FileNotFoundError(f"Offline log not found: {args.offline_step_log}")

    if args.source in ("auto", "model_bundle") and args.model_bundle.exists():
        return load_data_from_bundle(args.model_bundle, input_mask)

    raise FileNotFoundError(
        "No training data source was found. Checked experiments_data, offline_viz, and model bundle."
    )


def main():
    args = parse_args()
    gp_input_mask = load_input_feature_selection(args.config)
    z, y = load_training_data(args, gp_input_mask)

    n_tasks = y.shape[1]

    # Define likelihood and GP
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks)
    gp_model = BatchIndependentMultitaskGPModel(
        train_x=z,
        train_y=y,
        likelihood=likelihood,
        use_ard=True,
    )

    # Train hyperparameters
    gp_model, likelihood = train_gp_model(gp_model, torch_seed=0)

    # Extract and print hyperparameters
    with torch.no_grad():
        task_noises = likelihood.task_noises.detach().cpu().numpy().tolist()
        lengthscale = (
            gp_model.covar_module.base_kernel.lengthscale
            .squeeze()
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )
        outputscale = (
            gp_model.covar_module.outputscale
            .squeeze()
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

    print("hyperparameters:")
    print(f"  likelihood.task_noises: {task_noises}")
    print("  kernel.lengthscale:")
    for row in lengthscale:
        print(f"    - {row}")
    print(f"  kernel.outputscale: {outputscale}")


def train_gp_model(
    gp_model, torch_seed=None, training_iterations=400, learning_rate=0.02
):
    if torch_seed is not None:
        torch.manual_seed(torch_seed)

    likelihood = gp_model.likelihood
    train_x = gp_model.train_inputs[0]
    train_y = gp_model.train_targets

    # Find optimal model hyperparameters
    gp_model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, gp_model.parameters()), lr=learning_rate
    )  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

    prev_loss = np.inf
    num_loss_below_threshold = 0

    for i in range(training_iterations):
        optimizer.zero_grad()
        # output = likelihood(gp_model(train_x))
        # loss = -mll(output, train_y.reshape((train_y.numel(),)))
        output = gp_model(train_x)
        loss = -mll(output, train_y)

        num_loss_below_threshold = (
            num_loss_below_threshold + 1 if abs(loss - prev_loss) < 5e-4 else 0
        )

        if num_loss_below_threshold > 5:
            print(f"stopping GP optimization early after {i} iterations.")
            break

        prev_loss = loss

        loss.backward()
        if (i + 1) % 20 == 0:
            print("Iter %d/%d - Loss: %.3f" % (i + 1, training_iterations, loss.item()))

        optimizer.step()

    gp_model.eval()
    likelihood.eval()
    return gp_model, likelihood


if __name__ == "__main__":
    main()
