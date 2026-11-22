"""State and input layout of the single-track model.

One module on purpose. There used to be a second copy under ``lbmpc/utils/indicies.py``
whose same-named members held transposed values, so importing the wrong one was a silent
wrong answer rather than an ImportError. ``research/`` keeps its own, deliberately, for
the frozen pre-unification layout it is pinned to.

The order is the model's: see the probe in ``stmpc/bicycle_model.py``, which asserts it
against ``f_expl`` rather than trusting this file.
"""
from enum import IntEnum


class StateIndex(IntEnum):
    POS_ON_CENTER_LINE_S = 0
    MIN_DIST_TO_CENTER_LINE_N = 1
    ORIENTATION_THETA = 2
    VELOCITY_V_X = 3
    VELOCITY_V_Y = 4
    YAW_RATE = 5
    STEERING_ANGLE_DELTA = 6
    ACCEL = 7
    # the input slots of the [x, u] feature vector; dropped by a merge revert in
    # race_stack 94ea48a79, which is why x0[8]/x0[9] used to be literals
    INPUT_JERK = 8
    INPUT_DSTEERING = 9


class InputIndex(IntEnum):
    D_STEERING_ANGLE = 0
    JERK = 1


class SpatialStateIndex(IntEnum):
    """State layout of the time-optimal, spatial model in ``forzaeth_mpc/tompc/``.

    The first eight entries are ``StateIndex`` unchanged, on purpose: elapsed time is
    appended rather than inserted, so ``x[:8]`` of a spatial state is a single-track
    state and every consumer that slices the first eight -- the plant, the metrics, the
    ROS viz node that reads ``8 * stage`` -- needs no remapping.

    ``s`` is still a state even though it is also the independent variable. It
    integrates as ``ds/ds = 1``, which costs one state and lets the curvature spline
    stay inside the model instead of becoming a per-stage online parameter.
    """
    POS_ON_CENTER_LINE_S = 0
    MIN_DIST_TO_CENTER_LINE_N = 1
    ORIENTATION_THETA = 2
    VELOCITY_V_X = 3
    VELOCITY_V_Y = 4
    YAW_RATE = 5
    STEERING_ANGLE_DELTA = 6
    ACCEL = 7
    TIME = 8


def single_track_x0(*, s: float, n: float, theta: float, v_x: float, v_y: float,
                       yaw_rate: float, delta: float, accel: float) -> "np.ndarray":
    """Assemble a single-track state x0 by name.

    Every argument is keyword-only, because the two entries that a caller gets wrong are
    yaw_rate and delta, and a positional list accepts them either way round. The ROS entry
    point of STMPC held them swapped, so the car got its own yaw rate as a steering angle
    while the simulator, which builds x0 by index, ran clean.
    """
    import numpy as np
    x = np.zeros(len(StateIndex) - 2, dtype=np.float64)   # the last two entries are inputs
    x[StateIndex.POS_ON_CENTER_LINE_S] = s
    x[StateIndex.MIN_DIST_TO_CENTER_LINE_N] = n
    x[StateIndex.ORIENTATION_THETA] = theta
    x[StateIndex.VELOCITY_V_X] = v_x
    x[StateIndex.VELOCITY_V_Y] = v_y
    x[StateIndex.YAW_RATE] = yaw_rate
    x[StateIndex.STEERING_ANGLE_DELTA] = delta
    x[StateIndex.ACCEL] = accel
    return x


if __name__ == "__main__":
    import numpy as np
    x = single_track_x0(s=1, n=2, theta=3, v_x=4, v_y=5, yaw_rate=6, delta=7, accel=8)
    assert x.shape == (8,), x.shape
    assert np.array_equal(x, [1, 2, 3, 4, 5, 6, 7, 8]), x
    assert x[StateIndex.YAW_RATE] == 6 and x[StateIndex.STEERING_ANGLE_DELTA] == 7
    assert (StateIndex.YAW_RATE, StateIndex.STEERING_ANGLE_DELTA) == (5, 6)
    assert (InputIndex.D_STEERING_ANGLE, InputIndex.JERK) == (0, 1)
    print("OK: single_track_x0 puts yaw_rate at 5 and delta at 6")
