#!/bin/bash
#SBATCH -N 1
#SBATCH -t 08:00:00
#SBATCH -J cegs
#SBATCH -o slurm_%x_%A_%a.out
#SBATCH -e slurm_%x_%A_%a.err

set -euo pipefail
umask 077

BASE="${BASE:-$HOME/projects/cegs/ECG_Loss}"
CONF="${CONF:-$BASE/sweeps/cifar100.tsv}"

# IMPORTANT:
# Prefer DATA_DIR/RUNS_DIR exported by scripts/submit_array.sh.
# Only fallback if they are not provided.
if [[ -z "${DATA_DIR:-}" || -z "${RUNS_DIR:-}" ]]; then
  if [[ -n "${SCRATCH:-}" ]]; then
    SCR="$SCRATCH"
  elif [[ -d "/scratch/$USER" ]]; then
    SCR="/scratch/$USER"
  else
    SCR="$HOME/scratch"
  fi
  DATA_DIR="${DATA_DIR:-$SCR/cegs_data}"
  RUNS_DIR="${RUNS_DIR:-$SCR/cegs_runs}"
fi

WANDB_MODE="${WANDB_MODE:-offline}"
AUTO_DOWNLOAD="${CEGS_AUTO_DOWNLOAD:-1}"

RUN="$RUNS_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$RUN" "$DATA_DIR" "$RUNS_DIR"

# Tee logs into the run folder so each task has self-contained logs
exec > >(tee -a "$RUN/out.log") 2> >(tee -a "$RUN/err.log" >&2)

echo "BASE=$BASE"
echo "CONF=$CONF"
echo "RUN=$RUN"
echo "DATA_DIR=$DATA_DIR"
echo "HOST=$(hostname)"
echo "START=$(date -Iseconds)"

# Snapshot code into RUN/src (isolation: logs/models won't collide)
rsync -a --delete \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  --exclude 'logs' --exclude 'models' --exclude 'wandb' \
  "$BASE/" "$RUN/src/"

mkdir -p "$RUN/src/logs" "$RUN/src/models"

COMMIT="$(cd "$BASE" && git rev-parse --short HEAD)"
echo "$COMMIT" > "$RUN/commit.txt"
echo "COMMIT=$COMMIT"

# Make ../data (relative to RUN/src) point to shared DATA_DIR
# RUN/src/.. is RUN, so ../data == RUN/data
ln -sfn "$DATA_DIR" "$RUN/data"

# Let runner know where to put/download dataset
export CEGS_DATA_DIR="$DATA_DIR"
export CEGS_AUTO_DOWNLOAD="$AUTO_DOWNLOAD"

# W&B isolation per task
export WANDB_DIR="$RUN/wandb"
export WANDB_PROJECT="${WANDB_PROJECT:-CEGS}"
export WANDB_GROUP="cegs_${COMMIT}"
export WANDB_NAME="task${SLURM_ARRAY_TASK_ID}_${COMMIT}"
export WANDB_MODE="$WANDB_MODE"

# Activate conda env (Python 3.10)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate cegs
python -V

cd "$RUN/src"
python -u tools/run_from_tsv.py \
  --conf "$CONF" \
  --idx "$SLURM_ARRAY_TASK_ID" \
  --run_dir "$RUN" \
  --commit "$COMMIT" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_mode "$WANDB_MODE"

echo "END=$(date -Iseconds)"