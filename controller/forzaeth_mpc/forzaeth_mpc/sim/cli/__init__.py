"""Runnable entry points. From the repo root:

    python3 -m forzaeth_mpc.sim.cli.run_laps --algo stmpc kmpc lbmpc --laps 20
    python3 -m forzaeth_mpc.sim.cli.run_single_rollout
    python3 -m forzaeth_mpc.sim.cli.analyze_cost_vs_mu
    python3 -m forzaeth_mpc.sim.cli.tune_mpc_weights
    python3 -m forzaeth_mpc.sim.cli.noise_robustness_analysis
"""

from .analyze_cost_vs_mu import main as analyze_cost_vs_mu
from .noise_robustness_analysis import main as noise_robustness_analysis
from .run_laps import main as run_laps
from .run_single_rollout import main as run_single_rollout
from .tune_mpc_weights import main as tune_mpc_weights

__all__ = [
    "run_laps",
    "run_single_rollout",
    "analyze_cost_vs_mu",
    "tune_mpc_weights",
    "noise_robustness_analysis",
]
