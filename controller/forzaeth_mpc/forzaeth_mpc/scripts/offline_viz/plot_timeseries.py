import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def main(csv_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(os.path.join(csv_dir, "step_log.csv"))

    step_min = 400
    step_max = 700

    if step_min is not None:
        df = df[df["step"] >= step_min]
    if step_max is not None:
        df = df[df["step"] <= step_max]

    t = df["t_rel"].values

    # ---- timing ----
    plt.figure()
    plt.plot(t, df["t_total_ms"], label="total")
    plt.plot(t, df["t_init_ms"], label="init")
    plt.plot(t, df["t_residual_ms"], label="residual")
    plt.plot(t, df["t_rest_ms"], label="rest")
    plt.xlabel("t [s]")
    plt.ylabel("time [ms]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "timing_ms.png"), dpi=160)

    # ---- states (meas/prop/pred) for key ones ----
    def plot_triplet(name, fname):
        plt.figure()
        plt.plot(t, df[f"x_meas_{name}"], label=f"meas {name}")
        plt.plot(t, df[f"x_prop_{name}"], label=f"prop {name}")
        plt.plot(t, df[f"x_pred_delay_{name}"], label=f"pred_delay {name}")
        plt.xlabel("t [s]")
        plt.ylabel(name)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=160)

    # adapt these names if your enums differ
    for key in ["s", "d", "yaw", "velocity_v_x", "steering_angle_delta", "accel"]:
        col = f"x_meas_{key}"
        if col in df.columns:
            plot_triplet(key, f"state_{key}.png")

    # ---- inputs ----
    u_cols = [c for c in df.columns if c.startswith("u0_")]
    if u_cols:
        plt.figure()
        for c in u_cols:
            plt.plot(t, df[c], label=c)
        plt.xlabel("t [s]")
        plt.ylabel("u0")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "inputs_u0.png"), dpi=160)

    # ---- residuals ----
    if "r_meas_0" in df.columns:
        plt.figure()
        for i in range(3):
            plt.plot(t, df[f"r_meas_{i}"], label=f"meas r{i}")
            plt.plot(t, df[f"r_pred_{i}"], label=f"pred r{i}", linestyle="--")
        plt.xlabel("t [s]")
        plt.ylabel("residual")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "residuals.png"), dpi=160)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default="csv")
    ap.add_argument("--out_dir", default="figs")
    args = ap.parse_args()
    main(args.csv_dir, args.out_dir)
