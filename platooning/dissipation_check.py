"""Dissipation inequality of the gym longitudinal loop (v_cmd -> v) on the nonlinear JET1 simulation.

The gym speed loop is not LTI: a = sat(kp (v_cmd - v)) with kp = 1/0.3 (accel) or 1/0.1 (brake), |a| <= 5 m/s^2.
With the storage S = tau_acc * v^2 / 2 it should still satisfy, for forward driving,
    S(t) - S(0) <= int_0^t (u y - eps y^2) dt,   u = v_cmd, y = v
with eps = 1 (output strictly passive) without saturation, and eps = 0 (passive) with saturation.
Checked over 20 random piecewise-constant speed command sequences each (0.3 s steps, 9 s), the worst margin
min_t [int(u y - eps y^2) - (S(t) - S(0))] must stay >= 0.
Result (2026-11-22): small steps: eps=0 +0.090, eps=1 -0.0003 (integration error) -> OSP;
large steps: eps=0 +0.063 -> passive, eps=1 -1.96 -> OSP lost under saturation.

Run (needs the ROS env for forza_config_ros / f110_gym):
    source /opt/ros/humble/setup.bash && source ~/ws/install/setup.bash && python3 dissipation_check.py
"""
import contextlib
import io

import numpy as np

with contextlib.redirect_stdout(io.StringIO()):
    import lti_passivity as M          # reuse sim() (gym dynamics, pid, RK4); its own report runs silently on import

TAU_ACC = 0.3                          # S = TAU_ACC * v^2 / 2


def main():
    rng = np.random.default_rng(0)
    for name, amp in [("small steps (no saturation)", 0.4), ("large steps (saturating)   ", 2.5)]:
        worst = {0.0: np.inf, 1.0: np.inf}
        for _ in range(20):
            levels = np.clip(3.0 + rng.uniform(-amp, amp, 30), 0.6, 8.0)
            cmd = lambda t, L=levels: (L[min(int(t / 0.3), 29)], 0.0)
            _, X = M.sim(3.0, cmd, 9.0)
            v = np.r_[3.0, X[:, 3]]
            u = np.array([cmd(k * M.DT)[0] for k in range(len(v))])
            S = TAU_ACC * v**2 / 2
            for eps in worst:
                supply = np.cumsum((u[:-1] * v[:-1] - eps * v[:-1]**2) * M.DT)
                worst[eps] = min(worst[eps], (supply - (S[1:] - S[0])).min())
        print(f"{name}: worst margin  eps=0 (passive): {worst[0.0]:+.4f}   eps=1 (OSP): {worst[1.0]:+.4f}")


if __name__ == "__main__":
    main()
