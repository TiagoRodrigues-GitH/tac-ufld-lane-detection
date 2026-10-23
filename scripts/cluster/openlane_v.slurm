#!/usr/bin/env bash
# OpenLane-V, full experiment: one array task per seed (10 models each), then the report.
#   sbatch scripts/cluster/openlane_v.slurm
#   sbatch --dependency=afterok:<array job id> --export=ALL,REPORT=1 scripts/cluster/openlane_v.slurm
# Estimated per seed on an A100 (40 GB): ~20-30 h (30 epochs x 10 models + evaluation of 23,021 test frames).
#SBATCH --job-name=tacufld-olv
#SBATCH --array=1-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%A_%a.out
# #SBATCH --partition=<gpu partition>       # CLUSTER: adapt
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
. .venv/bin/activate
: "${OPENLANE_ROOT:?export OPENLANE_ROOT (folder prepared by prepare_openlane_v.sh)}"
export PYTHONUTF8=1
if [ "${REPORT:-0}" = "1" ]; then  # every seed is already evaluated: aggregates, tests, figures, report
  python -m tac_ufld run --config configs/cluster/openlane_v.yaml --confirm --resume
else
  python -m tac_ufld run --config configs/cluster/openlane_v.yaml --confirm --resume --seeds "${SLURM_ARRAY_TASK_ID}"
fi
