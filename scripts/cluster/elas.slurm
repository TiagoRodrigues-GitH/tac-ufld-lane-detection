#!/usr/bin/env bash
# ELAS, all 22 scenes: seed 1 first (it runs the hyper-parameter search, shared by every seed), then seeds 2-7 in
# parallel, then the report and the robustness evaluation.
#   J1=$(sbatch --parsable --array=1 scripts/cluster/elas.slurm)
#   J2=$(sbatch --parsable --array=2-7 --dependency=afterok:$J1 scripts/cluster/elas.slurm)
#   sbatch --dependency=afterok:$J2 --export=ALL,REPORT=1 --array=0 scripts/cluster/elas.slurm
# Estimated on an A100: search ~6-10 h, then ~10-15 h per seed.
#SBATCH --job-name=tacufld-elas
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%A_%a.out
# #SBATCH --partition=<gpu partition>       # CLUSTER: adapt
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
. .venv/bin/activate
: "${ELAS_ROOT:?export ELAS_ROOT (dataset_elas_v1 with all 22 scenes)}"
export PYTHONUTF8=1
if [ "${REPORT:-0}" = "1" ]; then
  python -m tac_ufld run --config configs/cluster/elas_full22.yaml --confirm --resume
  python -m tac_ufld robustness --runs results/elas_full22_cluster
else
  python -m tac_ufld run --config configs/cluster/elas_full22.yaml --confirm --resume --seeds "${SLURM_ARRAY_TASK_ID}"
fi
