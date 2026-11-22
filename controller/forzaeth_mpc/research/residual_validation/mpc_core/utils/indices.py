"""State and input layout for this tree's frozen single-track model.

This tree is pinned to the pre-unification layout: 8 states ordered
``[s, n, theta, v_x, v_y, delta, yaw_rate, accel]`` with inputs ``[jerk, ddelta]``, and a
14-entry parameter vector. Production moved to ``[..., yaw_rate, delta, accel]`` with
``[ddelta, jerk]`` and 22 parameters when the two single-track models were merged.

So these enums are deliberately NOT ``forzaeth_mpc.utils.indices``. The two have
same-named members with transposed values, and importing the wrong one is a silent wrong
answer rather than an ImportError -- it would read the steering angle where it means the
yaw rate. Everything under ``research/`` imports from here.
"""
from enum import IntEnum


class StateIndex(IntEnum):
    POS_ON_CENTER_LINE_S = 0
    MIN_DIST_TO_CENTER_LINE_N = 1
    ORIENTATION_THETA = 2
    VELOCITY_V_X = 3
    VELOCITY_V_Y = 4
    STEERING_ANGLE_DELTA = 5
    YAW_RATE = 6
    ACCEL = 7
    # input slots of the [x, u] feature vector
    INPUT_JERK = 8
    INPUT_DSTEERING = 9


class InputIndex(IntEnum):
    JERK = 0
    D_STEERING_ANGLE = 1


IDX = {
    "s": 0, "n": 1, "psi": 2, "vx": 3, "vy": 4, "delta": 5, "wz": 6, "ax": 7,
}
SUBSET_RESIDUAL = ["vx", "vy", "wz"]
