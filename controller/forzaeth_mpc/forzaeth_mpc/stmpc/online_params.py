"""Single source of truth for the acados online-parameter (p) vector layout.

The layout is defined by ``bicycle_model()`` in ``bicycle_model.py``:

    0  V_target             7  a_lat_max          14 Bf
    1  weight_adv           8  a_long_min         15 Cf
    2  weight_qv            9  a_long_max         16 Df
    3  weight_qn           10  safety_margin      17 Ef
    4  weight_qalpha       11  overtake_d         18 Br
    5  weight_qjerk        12  MU                 19 Cr
    6  weight_ddelta       13  terminal_multiplier 20 Dr
                                                  21 Er

Anything that calls ``acados_solver.set(i, "p", ...)`` must build it here.
"""

import numpy as np

from typing import Union

from forzaeth_mpc.config import LBMPCConfig, PacejkaTireConfig, STMPCConfig

WEIGHT_KEYS = ("q_adv", "q_v", "q_n", "q_alpha", "q_jerk", "q_ddelta", "terminal_multiplier")

MU_IDX = 12
"""Index of the friction coefficient. The same in the 14- and 22-entry layouts, which
is why ``set_sim_mu`` can hardcode it."""


def build_online_params(config: Union[STMPCConfig, LBMPCConfig],
                        tire_config: PacejkaTireConfig,
                        target_v: float,
                        overtake_d: float,
                        mu: float,
                        weights: dict = None,
                        alat_max: float = None,
                        a_min: float = None,
                        a_max: float = None) -> np.ndarray:
    """Builds the 22-entry acados online parameter vector.

    Shared by STMPC and LB-MPC: the two schemas declare the same fields here, and
    since LB-MPC's Pacejka coefficients went online the two layouts are identical.

    Args:
        config: static MPC config (cost weights, accel/margin limits).
        tire_config: Pacejka coefficients.
        target_v: target longitudinal speed for this stage.
        overtake_d: lateral reference offset for this stage.
        mu: friction coefficient handed to the MPC model.
        weights: optional per-call overrides of the cost weights, keys out of
            ``WEIGHT_KEYS``; missing keys fall back to ``config``.
        alat_max, a_min, a_max: optional limit overrides, for LB-MPC's start mode
            (``in_start_bool``) which swaps in the ``*_start`` variants.
    """
    w = {
        "q_adv": config.qadv,
        "q_v": config.qv,
        "q_n": config.qn,
        "q_alpha": config.qalpha,
        "q_jerk": config.qjerk,
        "q_ddelta": config.qddelta,
        "terminal_multiplier": config.terminal_multiplier,
    }
    if weights:
        w.update({k: weights[k] for k in WEIGHT_KEYS if k in weights})

    return np.array([
        target_v,
        w["q_adv"],
        w["q_v"],
        w["q_n"],
        w["q_alpha"],
        w["q_jerk"],
        w["q_ddelta"],
        config.alat_max if alat_max is None else alat_max,
        config.a_min if a_min is None else a_min,
        config.a_max if a_max is None else a_max,
        config.track_safety_margin,
        overtake_d,
        mu,
        w["terminal_multiplier"],
        tire_config.Bf,
        tire_config.Cf,
        tire_config.Df,
        tire_config.Ef,
        tire_config.Br,
        tire_config.Cr,
        tire_config.Dr,
        tire_config.Er,
    ], dtype=np.float64)


def default_parameter_values(config: Union[STMPCConfig, LBMPCConfig],
                             tire_config: PacejkaTireConfig) -> np.ndarray:
    """The p vector acados is initialised with. Every entry is overwritten online each
    cycle; this only has to be the right length and finite."""
    return build_online_params(config, tire_config,
                               target_v=2.0,
                               overtake_d=config.overtake_d,
                               mu=tire_config.friction_coeff)


if __name__ == "__main__":
    from forzaeth_mpc.config import (load_LBMPC_config_ros, load_pacejka_tire_config_ros,
                                     load_STMPC_config_ros)

    tire_config = load_pacejka_tire_config_ros("NUC5", "dubi")
    PACEJKA = (tire_config.Bf, tire_config.Cf, tire_config.Df, tire_config.Ef,
               tire_config.Br, tire_config.Cr, tire_config.Dr, tire_config.Er)

    # The same builder must serve both controllers, at the same 22-entry layout.
    for label, config in (("STMPC", load_STMPC_config_ros("NUC5")),
                          ("LBMPC", load_LBMPC_config_ros("NUC5"))):
        p = build_online_params(config, tire_config, target_v=3.5, overtake_d=0.25, mu=0.63)
        assert len(p) == 22, (label, len(p))
        assert p[0] == 3.5
        assert p[6] == config.qddelta
        assert p[7] == config.alat_max
        assert p[11] == 0.25
        assert p[MU_IDX] == 0.63, f"{label}: index {MU_IDX} must be the friction coefficient"
        assert p[13] == config.terminal_multiplier
        assert tuple(p[14:22]) == PACEJKA, f"{label}: Bf..Er must occupy 14..21 in order"

        # weight override: only index 6 moves
        p_w = build_online_params(config, tire_config, 3.5, 0.25, 0.63,
                                  weights={"q_ddelta": 1e6 * config.qddelta})
        assert p_w[6] == 1e6 * config.qddelta
        assert np.array_equal(np.delete(p_w, 6), np.delete(p, 6))

        # limit overrides, LB-MPC's start mode: only indices 7, 8, 9 move
        p_s = build_online_params(config, tire_config, 3.5, 0.25, 0.63,
                                  alat_max=1.0, a_min=-2.0, a_max=3.0)
        assert (p_s[7], p_s[8], p_s[9]) == (1.0, -2.0, 3.0)
        assert np.array_equal(np.delete(p_s, [7, 8, 9]), np.delete(p, [7, 8, 9]))

        # the acados init vector is the same layout, just a different target speed
        d = default_parameter_values(config, tire_config)
        assert len(d) == 22 and d[0] == 2.0 and d[11] == config.overtake_d
        assert d[MU_IDX] == tire_config.friction_coeff

    print("OK: one 22-entry layout for STMPC and LBMPC; "
          f"mu@{MU_IDX}, Bf@14={tire_config.Bf}")
