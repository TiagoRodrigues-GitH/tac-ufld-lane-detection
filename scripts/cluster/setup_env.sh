#!/usr/bin/env bash
# One-time environment on the cluster (Linux, NVIDIA GPU, CUDA 12.x driver). Run from the repository root:
#   bash scripts/cluster/setup_env.sh
# Adapt the two lines marked CLUSTER to the local module system if Python 3.11+ is not the default.
set -euo pipefail
# module load python/3.12 cuda/12.6        # CLUSTER: uncomment / adapt
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "torch>=2.4" --index-url https://download.pytorch.org/whl/cu126   # CLUSTER: match the CUDA driver
python -m pip install -e ".[dev]" gdown
# ImageNet weights of the backbones into ~/.cache/torch (compute nodes often have no internet)
python -c "import torch; from tac_ufld.models.resnet import IMAGENET_WEIGHTS as W; [torch.hub.load_state_dict_from_url(u, map_location='cpu') for u in W.values()]"
python -m tac_ufld doctor                    # environment, GPU and dependency check
python -m pytest -q tests                    # ~2 min on a CPU node; every test must pass
