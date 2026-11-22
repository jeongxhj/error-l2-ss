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
    N = int(meta["N"]) if meta and "N" in meta else None

    hc = pd.read_csv(os.path.join(csv_dir, "horizon_covdiag.csv"))

    if step_min is not None:
        hc = hc[hc["step"] >= step_min]
    if step_max is not None:
        hc = hc[hc["step"] <= step_max]

    if stride is not None and stride > 1:
        steps = sorted(hc["step"].unique())
        keep = set(steps[::stride])
        hc = hc[hc["step"].isin(keep)]

    col = f"Pdiag_{state}"
    if col not in hc.columns:
        raise RuntimeError(f"Missing column {col} in horizon_covdiag.csv")

    steps = sorted(hc["step"].unique())
    nodes = sorted(hc["node"].unique())
    if N is not None:
        nodes = [n for n in nodes if n <= N]

    M = np.full((len(steps), len(nodes)), np.nan)
    step_to_i = {s: i for i, s in enumerate(steps)}
    node_to_j = {n: j for j, n in enumerate(nodes)}

    for _, r in hc.iterrows():
        i = step_to_i.get(int(r["step"]))
        j = node_to_j.get(int(r["node"]))
        if i is None or j is None:
            continue
        M[i, j] = float(r[col])

    plt.figure()
    plt.imshow(M, aspect="auto", origin="lower")
    plt.xlabel("horizon node")
    plt.ylabel("step index (filtered)")
    plt.title(f"horizon covariance diag heatmap: {state}")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"horizon_cov_heatmap_{state}.png"), dpi=180)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="")
    ap.add_argument("--state", required=True)
    ap.add_argument("--step_min", type=int, default=600)
    ap.add_argument("--step_max", type=int, default=900)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()
    main(args.base_dir, args.state, args.step_min, args.step_max, args.stride)
