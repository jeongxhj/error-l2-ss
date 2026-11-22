import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_meta(base_dir):
    meta_path = os.path.join(base_dir, "run_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    return None


def main(base_dir, state, step_min, step_max, stride):
    csv_dir = os.path.join(base_dir, "csv")
    out_dir = os.path.join(base_dir, "figs")
    os.makedirs(out_dir, exist_ok=True)

    meta = load_meta(base_dir)
    dt = float(meta["dt"]) if meta and "dt" in meta else 0.05
    N = int(meta["N"]) if meta and "N" in meta else None

    step = pd.read_csv(os.path.join(csv_dir, "step_log.csv"))
    hp = pd.read_csv(os.path.join(csv_dir, "horizon_pred.csv"))

    # filter window
    if step_min is not None:
        step = step[step["step"] >= step_min]
        hp = hp[hp["step"] >= step_min]
    if step_max is not None:
        step = step[step["step"] <= step_max]
        hp = hp[hp["step"] <= step_max]

    # subsample steps for plotting
    if stride is not None and stride > 1:
        keep_steps = step["step"].values[::stride]
        step = step[step["step"].isin(keep_steps)]
        hp = hp[hp["step"].isin(keep_steps)]

    meas_col = f"x_meas_{state}"
    pred_col = f"x_{state}"
    if meas_col not in step.columns:
        raise RuntimeError(f"Missing column {meas_col} in step_log.csv")
    if pred_col not in hp.columns:
        raise RuntimeError(f"Missing column {pred_col} in horizon_pred.csv")

    # base measured
    plt.figure()
    plt.plot(step["t_rel"].values, step[meas_col].values, label=f"meas {state}")
    plt.xlabel("t [s]")
    plt.ylabel(state)
    plt.grid(True)

    # horizon fan: for each step, plot (t0 + k*dt, x_pred(k))
    # Join step->t_rel
    step_t = step.set_index("step")["t_rel"].to_dict()
    for s in sorted(hp["step"].unique()):
        if s not in step_t:
            continue
        t0 = float(step_t[s])
        hps = hp[hp["step"] == s].sort_values("node")
        t_pred = t0 + dt * hps["node"].values.astype(float)
        plt.plot(t_pred, hps[pred_col].values, alpha=0.15)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"horizon_fan_{state}.png"), dpi=180)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="")
    ap.add_argument("--state", required=True, help="state name, e.g. d, s, yaw, velocity_v_x")
    ap.add_argument("--step_min", type=int, default=700)
    ap.add_argument("--step_max", type=int, default=800)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()
    main(args.base_dir, args.state, args.step_min, args.step_max, args.stride)
