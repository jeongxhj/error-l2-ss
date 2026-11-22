"""Single source of truth for the time-optimal MPC acados online-parameter (p) vector.

The layout is defined by ``spatial_model()`` in ``spatial_model.py``:

    0  V_target             8  a_lat_max          16 Bf
    1  weight_qtime         9  a_long_min         17 Cf
    2  weight_qv           10  a_long_max         18 Df
    3  weight_qn           11  safety_margin      19 Ef
    4  weight_qalpha       12  overtake_d         20 Br
    5  weight_qvy          13  MU                 21 Cr
    6  weight_qjerk        14  terminal_multiplier 22 Dr
    7  weight_ddelta       15  weight_qv_terminal 23 Er

Anything that calls ``acados_solver.set(i, "p", ...)`` must build it here.

This is deliberately *not* ``stmpc.online_params``'s 22-entry vector. TOMPC has no
advancement weight -- minimising the traversal time is what the advancement term was an
approximation of -- and it has two weights STMPC does not (``qtime``, ``qv_terminal``),
plus the lateral-velocity penalty as a real weight instead of STMPC's hardcoded ``100``
behind a boolean.
"""

import numpy as np

from forzaeth_mpc.config import PacejkaTireConfig, TOMPCConfig

WEIGHT_KEYS = ("q_time", "q_v", "q_n", "q_alpha", "q_vy", "q_jerk", "q_ddelta",
               "q_v_terminal", "terminal_multiplier")

MU_IDX = 13
"""Index of the friction coefficient. Differs from ``stmpc.online_params.MU_IDX``; the
two layouts are not interchangeable, which is why each has its own constant."""

N_PARAMS = 24


def build_online_params(config: TOMPCConfig,
                        tire_config: PacejkaTireConfig,
                        target_v: float,
                        overtake_d: float,
                        mu: float,
                        weights: dict = None,
                        alat_max: float = None,
                        a_min: float = None,
                        a_max: float = None) -> np.ndarray:
    """Builds the 24-entry acados online parameter vector for TOMPC.

    Args:
        config: static TOMPC config (cost weights, accel/margin limits).
        tire_config: Pacejka coefficients.
        target_v: target longitudinal speed. Only has an effect where ``q_v`` is
            non-zero, which while racing it is not -- see ``racing_weights`` and
            ``trailing_weights``.
        overtake_d: lateral reference offset for this stage.
        mu: friction coefficient handed to the MPC model.
        weights: optional per-call overrides of the cost weights, keys out of
            ``WEIGHT_KEYS``; missing keys fall back to ``config``.
        alat_max, a_min, a_max: optional limit overrides.
    """
    w = default_weights(config)
    if weights:
        w.update({k: weights[k] for k in WEIGHT_KEYS if k in weights})

    return np.array([
        target_v,
        w["q_time"],
        w["q_v"],
        w["q_n"],
        w["q_alpha"],
        w["q_vy"],
        w["q_jerk"],
        w["q_ddelta"],
        config.alat_max if alat_max is None else alat_max,
        config.a_min if a_min is None else a_min,
        config.a_max if a_max is None else a_max,
        config.track_safety_margin,
        overtake_d,
        mu,
        w["terminal_multiplier"],
        w["q_v_terminal"],
        tire_config.Bf,
        tire_config.Cf,
        tire_config.Df,
        tire_config.Ef,
        tire_config.Br,
        tire_config.Cr,
        tire_config.Dr,
        tire_config.Er,
    ], dtype=np.float64)


def default_weights(config: TOMPCConfig) -> dict:
    """The cost weights straight off the config, as ``WEIGHT_KEYS``."""
    return {
        "q_time": config.qtime,
        "q_v": config.qv,
        "q_n": config.qn,
        "q_alpha": config.qalpha,
        "q_vy": config.qvy,
        "q_jerk": config.qjerk,
        "q_ddelta": config.qddelta,
        "q_v_terminal": config.qv_terminal,
        "terminal_multiplier": config.terminal_multiplier,
    }


def racing_weights(config: TOMPCConfig) -> dict:
    """Weights for free running: minimise time, ignore the reference speed profile.

    Dropping ``q_v`` is the whole point of the controller. The global planner's speed
    profile is a *precomputed* time-optimal solution for a nominal model; tracking it
    while also minimising time means the two objectives fight, and the tracking term
    wins wherever the planner was conservative.
    """
    w = default_weights(config)
    w["q_v"] = 0.0
    return w


def trailing_weights(config: TOMPCConfig) -> dict:
    """Weights for following an opponent: track the gap controller's speed, no time term.

    Minimum time and holding a gap are contradictory objectives. While trailing the gap
    controller owns the speed, so the time term goes to zero rather than fighting it.
    """
    w = default_weights(config)
    w["q_time"] = 0.0
    w["q_v_terminal"] = 0.0
    return w


def default_parameter_values(config: TOMPCConfig,
                             tire_config: PacejkaTireConfig) -> np.ndarray:
    """The p vector acados is initialised with. Every entry is overwritten online each
    cycle; this only has to be the right length and finite."""
    return build_online_params(config, tire_config,
                               target_v=2.0,
                               overtake_d=config.overtake_d,
                               mu=tire_config.friction_coeff)


if __name__ == "__main__":
    from forzaeth_mpc.config import (load_pacejka_tire_config_ros,
                                     load_TOMPC_config_ros)

    tire_config = load_pacejka_tire_config_ros("NUC5", "dubi")
    PACEJKA = (tire_config.Bf, tire_config.Cf, tire_config.Df, tire_config.Ef,
               tire_config.Br, tire_config.Cr, tire_config.Dr, tire_config.Er)
    config = load_TOMPC_config_ros("NUC5")

    p = build_online_params(config, tire_config, target_v=3.5, overtake_d=0.25, mu=0.63)
    assert len(p) == N_PARAMS, len(p)
    assert p[0] == 3.5
    assert p[1] == config.qtime
    assert p[7] == config.qddelta
    assert p[8] == config.alat_max
    assert p[12] == 0.25
    assert p[MU_IDX] == 0.63, f"index {MU_IDX} must be the friction coefficient"
    assert p[14] == config.terminal_multiplier
    assert p[15] == config.qv_terminal
    assert tuple(p[16:24]) == PACEJKA, "Bf..Er must occupy 16..23 in order"

    # weight override: only index 7 moves
    p_w = build_online_params(config, tire_config, 3.5, 0.25, 0.63,
                              weights={"q_ddelta": 1e6 * config.qddelta})
    assert p_w[7] == 1e6 * config.qddelta
    assert np.array_equal(np.delete(p_w, 7), np.delete(p, 7))

    # limit overrides: only indices 8, 9, 10 move
    p_s = build_online_params(config, tire_config, 3.5, 0.25, 0.63,
                              alat_max=1.0, a_min=-2.0, a_max=3.0)
    assert (p_s[8], p_s[9], p_s[10]) == (1.0, -2.0, 3.0)
    assert np.array_equal(np.delete(p_s, [8, 9, 10]), np.delete(p, [8, 9, 10]))

    # the two modes are mutually exclusive in exactly one place each
    p_race = build_online_params(config, tire_config, 3.5, 0.0, 0.63,
                                 weights=racing_weights(config))
    p_trail = build_online_params(config, tire_config, 3.5, 0.0, 0.63,
                                  weights=trailing_weights(config))
    assert p_race[2] == 0.0 and p_race[1] == config.qtime, "racing: no speed tracking"
    assert p_trail[1] == 0.0 and p_trail[2] == config.qv, "trailing: no time term"

    # the acados init vector is the same layout, just a different target speed
    d = default_parameter_values(config, tire_config)
    assert len(d) == N_PARAMS and d[0] == 2.0 and d[12] == config.overtake_d
    assert d[MU_IDX] == tire_config.friction_coeff

    print(f"OK: {N_PARAMS}-entry TOMPC layout; qtime@1, mu@{MU_IDX}, Bf@16={tire_config.Bf}")
