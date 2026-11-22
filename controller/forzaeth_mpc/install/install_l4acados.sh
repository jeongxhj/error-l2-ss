#!/usr/bin/env bash
# Installs l4acados (the pinned submodule) with the exo GPyTorch backend LB-MPC needs.
# Run install/install_acados.sh first; acados and acados_template are prerequisites.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# No-op after `git clone --recurse-submodules`; clones l4acados after a plain clone.
git -C "${REPO_DIR}" submodule update --init external/l4acados
L4A_DIR="${REPO_DIR}/external/l4acados"

# The acados Cython interface (controller.py uses use_cython=True) needs Cython at codegen time.
python3 -m pip install "Cython>=0.29"

# torch from PyPI drags ~2.5 GB of nvidia-* CUDA wheels. Most machines here have no GPU,
# so install the CPU build first; gpytorch then sees torch as satisfied and never pulls
# the CUDA one. On a GPU box:
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 ./install/install_l4acados.sh
python3 -c 'import torch' 2>/dev/null || \
  python3 -m pip install --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}" torch

# [gpytorch-exo] pins naefjo/gpytorch and naefjo/linear_operator. LB-MPC does not work
# with the released gpytorch; pip replaces a PyPI copy with the fork.
python3 -m pip install -e "${L4A_DIR}[gpytorch-exo]"
