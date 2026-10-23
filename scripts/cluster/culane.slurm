#!/usr/bin/env bash
# CULane, FULL plan (configs/cluster/culane_full.yaml): 10 models x 7 seeds x 50 epochs. Seed 1 runs the
# hyper-parameter search (on Optuna's own split carved from train) that every seed reuses; seeds 2-7 then run in
# parallel; the last job writes the report and runs the robustness evaluation.
#   J1=$(sbatch --parsable --array=1 scripts/cluster/culane.slurm)
#   J2=$(sbatch --parsable --array=2-7 --dependency=afterok:$J1 scripts/cluster/culane.slurm)
#   sbatch --dependency=afterok:$J2 --export=ALL,REPORT=1 --array=0 scripts/cluster/culane.slurm
# Core plan instead (6 models, 15-epoch fine-tuning): add --export=ALL,CONFIG=configs/culane.yaml to each sbatch.
# Estimated on an A100: search ~8-12 h, then ~25-35 h per seed (each job resumes when resubmitted).
#SBATCH --job-name=tacufld-culane
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%A_%a.out
# #SBATCH --partition=<gpu partition>       # CLUSTER: adapt
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
. .venv/bin/activate
: "${CULANE_ROOT:?export CULANE_ROOT (folder with driver_*/, laneseg_label_w16/, list/)}"
CONFIG="${CONFIG:-configs/cluster/culane_full.yaml}"
RUN="results/$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['name'])" "$CONFIG")"
export PYTHONUTF8=1
if [ "${REPORT:-0}" = "1" ]; then
  python -m tac_ufld run --config "$CONFIG" --confirm --resume
  python -m tac_ufld robustness --runs "$RUN" --workers 8
else
  python -m tac_ufld run --config "$CONFIG" --confirm --resume --seeds "${SLURM_ARRAY_TASK_ID}"
fi
