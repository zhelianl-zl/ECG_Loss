#!/bin/bash
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -J cegs_eval
#SBATCH -o slurm_logs/slurm_%x_%A_%a.out
#SBATCH -e slurm_logs/slurm_%x_%A_%a.err

set -euo pipefail
umask 077

BASE="${BASE:-${SLURM_SUBMIT_DIR:-$PWD}}"
BASE="$(cd "$BASE" && pwd)"
CONF="${CONF:-$BASE/sweeps/RunA_eval.tsv}"

DATA_DIR="${DATA_DIR:-/ocean/projects/cis260049p/zliu49/cegs}"
RUNS_DIR="${RUNS_DIR:-/ocean/projects/cis260049p/zliu49/cegs_runs}"

export CEGS_DATA_DIR="$DATA_DIR"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1

echo "BASE=$BASE"
echo "CONF=$CONF"
echo "TASK=${SLURM_ARRAY_TASK_ID}"
echo "DATA_DIR=$DATA_DIR"
echo "RUNS_DIR=$RUNS_DIR"
echo "HOST=$(hostname)"
echo "START=$(date -Iseconds)"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate cegs
unset LD_PRELOAD || true
python -V
nvidia-smi -L 2>/dev/null || true

cd "$BASE"
python -u tools/eval_checkpoints.py \
  --conf "$CONF" \
  --idx "$SLURM_ARRAY_TASK_ID"

echo "END=$(date -Iseconds)"
