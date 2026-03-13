#!/bin/bash
#SBATCH -N 1
#SBATCH -t 08:00:00
#SBATCH -J eval_best
#SBATCH -o slurm_logs/eval_best_%A_%a.out
#SBATCH -e slurm_logs/eval_best_%A_%a.err

set -euo pipefail

BASE="${BASE:-${SLURM_SUBMIT_DIR:-$PWD}}"
BASE="$(cd "$BASE" && pwd)"
CONF="${CONF:-$BASE/sweeps/eval_best_runs.tsv}"
RUNS_DIR="${RUNS_DIR:-/ocean/projects/cis260049p/zliu49/cegs_runs}"

echo "BASE=$BASE"
echo "CONF=$CONF"
echo "RUNS_DIR=$RUNS_DIR"
echo "HOST=$(hostname)"
echo "TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "START=$(date -Iseconds)"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate cegs
python -V
nvidia-smi -L 2>/dev/null || true

cd "$BASE"
python -u tools/eval_batch.py \
  --conf "$CONF" \
  --row "$SLURM_ARRAY_TASK_ID" \
  --runs_dir "$RUNS_DIR"

echo "END=$(date -Iseconds)"
