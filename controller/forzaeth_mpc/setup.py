# Metadata lives here, not in pyproject's [project]: the ROS humble images pin setuptools 58,
# which predates PEP 621 and would build a nameless package from pyproject alone.
from setuptools import find_packages, setup

setup(
    name='forzaeth_mpc',
    version='0.1.0',
    description='ForzaETH F1TENTH model predictive controllers (KMPC, STMPC, LB-MPC, TOMPC) and a ROS-free simulator',
    python_requires='>=3.8',
    packages=find_packages(include=['forzaeth_mpc*']),
    # forza_config: https://github.com/ForzaETH/forza_config (a submodule of race_stack)
    install_requires=['numpy', 'scipy', 'casadi', 'pydantic>=2', 'pyyaml', 'forza_config'],
    extras_require={
        # GP residual learning. acados itself and the pinned gpytorch/linear_operator
        # forks are NOT pip-resolvable -- run install/install_acados.sh and
        # install/install_l4acados.sh for those.
        'lbmpc': ['torch'],
        # Offline simulator, tuning and plots.
        'sim': ['matplotlib', 'optuna', 'tqdm'],
    },
)
