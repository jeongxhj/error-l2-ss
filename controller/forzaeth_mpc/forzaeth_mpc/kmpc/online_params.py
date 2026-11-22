"""The acados online-parameter vector of the kinematic MPC.

Ten entries, and no friction: a kinematic model has no tire forces. The single-track
controllers use a different, longer layout, in stmpc/online_params.py.
"""
import numpy as np

from forzaeth_mpc.config import KMPCConfig

WEIGHT_KEYS = ("q_adv", "q_v", "q_n", "q_alpha", "q_ac", "q_ddelta")


def build_online_params(kmpc_config: KMPCConfig,
                             target_v: float,
                             overtake_d: float,
                             weights: dict = None) -> np.ndarray:
    """Builds the 10-entry KMPC acados online parameter vector.

    Layout (see ``kmpc/bicycle_model.py``):
        0 V_target  1 weight_adv  2 weight_qv  3 weight_qn  4 weight_qalpha
        5 weight_qac  6 weight_ddelta  7 a_lat_max  8 safety_margin  9 overtake_d

    There is deliberately no μ slot: KMPC is a kinematic model with no tire
    forces, so the friction coefficient is not applicable.

    Args:
        kmpc_config: static KMPC config (cost weights, alat/margin limits).
        target_v: target speed for this stage.
        overtake_d: lateral reference offset for this stage.
        weights: optional overrides out of ``WEIGHT_KEYS``.
    """
    w = {
        "q_adv": kmpc_config.qadv,
        "q_v": kmpc_config.qv,
        "q_n": kmpc_config.qn,
        "q_alpha": kmpc_config.qalpha,
        "q_ac": kmpc_config.qac,
        "q_ddelta": kmpc_config.qddelta,
    }
    if weights:
        w.update({k: weights[k] for k in WEIGHT_KEYS if k in weights})

    return np.array([
        target_v,
        w["q_adv"],
        w["q_v"],
        w["q_n"],
        w["q_alpha"],
        w["q_ac"],
        w["q_ddelta"],
        kmpc_config.alat_max,
        kmpc_config.track_safety_margin,
        overtake_d,
    ], dtype=np.float64)


def default_parameter_values(cfg: KMPCConfig) -> np.ndarray:
    """Get parameters from param_config.yaml"""
    params = np.array([
        2, # initial velocity, not important it is immediately changed
        cfg.qadv,
        cfg.qv,
        cfg.qn,
        cfg.qalpha,
        cfg.qac,
        cfg.qddelta,
        cfg.alat_max,
        cfg.track_safety_margin,
        cfg.overtake_d,
    ]).astype(float)

    return params


if __name__ == "__main__":
    from forzaeth_mpc.config import load_KMPC_config_ros

    cfg = load_KMPC_config_ros("NUC5")
    p = build_online_params(cfg, target_v=3.5, overtake_d=0.25)
    assert len(p) == 10, len(p)
    assert p[0] == 3.5
    assert p[5] == cfg.qac, "index 5 is the acceleration weight, not the jerk weight"
    assert p[9] == 0.25
    d = default_parameter_values(cfg)
    assert len(d) == 10 and d[9] == cfg.overtake_d
    print("OK: KMPC p layout is 10 entries; q_ac@5, overtake_d@9")
