#!/bin/bash
#SBATCH -N 1
#SBATCH -t 08:00:00
#SBATCH -J cegs
#SBATCH -o slurm_%x_%A_%a.out
#SBATCH -e slurm_%x_%A_%a.err

set -euo pipefail
umask 077

# These are provided by scripts/submit_array.sh (or can be exported manually)
BASE="${BASE:-$HOME/projects/cegs/ECG_Loss}"            # main repo with .venv
CONF="${CONF:-$BASE/sweeps/cifar100.tsv}"              # tsv path
DATA_DIR="${DATA_DIR:-$SCRATCH/cegs_data}"             # shared dataset root
RUNS_DIR="${RUNS_DIR:-$SCRATCH/cegs_runs}"             # all runs go here
WANDB_MODE="${WANDB_MODE:-offline}"                    # online/offline/disabled
AUTO_DOWNLOAD="${CEGS_AUTO_DOWNLOAD:-1}"               # 1 enable, 0 disable

# Unique run directory per task
RUN="$RUNS_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$RUN" "$DATA_DIR" "$RUNS_DIR"

# Also tee logs into per-run files so everything is in one folder
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

# Record git commit
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

# Activate venv from BASE (shared env)
source "$BASE/.venv/bin/activate"

cd "$RUN/src"
python -u tools/run_from_tsv.py \
  --conf "$CONF" \
  --idx "$SLURM_ARRAY_TASK_ID" \
  --run_dir "$RUN" \
  --commit "$COMMIT" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_mode "$WANDB_MODE"

echo "END=$(date -Iseconds)"