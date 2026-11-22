import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def main(csv_dir, out_dir, step_min, step_max, stride):
    os.makedirs(out_dir, exist_ok=True)
    hb = pd.read_csv(os.path.join(csv_dir, "horizon_bounds.csv"))

    if step_min is not None:
        hb = hb[hb["step"] >= step_min]
    if step_max is not None:
        hb = hb[hb["step"] <= step_max]

    steps = sorted(hb["step"].unique())
    if stride and stride > 1:
        steps = steps[::stride]

    plt.figure()
    for s in steps:
        h = hb[hb["step"] == s].sort_values("node")
        if "uh_t0" in h.columns:
            plt.plot(h["node"], h["uh_t0"], alpha=0.15)
    plt.xlabel("node")
    plt.ylabel("uh_t0")
    plt.title("tightened upper bound component t0 (window)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tight_uh_t0_window.png"), dpi=180)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default="csv")
    ap.add_argument("--out_dir", default="figs")
    ap.add_argument("--step_min", type=int, default=400)
    ap.add_argument("--step_max", type=int, default=700)
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()
    main(args.csv_dir, args.out_dir, args.step_min, args.step_max, args.stride)
