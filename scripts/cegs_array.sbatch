#!/bin/bash
#SBATCH -N 1
#SBATCH -t 08:00:00
#SBATCH -J cegs
#SBATCH -o slurm_%x_%A_%a.out
#SBATCH -e slurm_%x_%A_%a.err

set -euo pipefail
umask 077

# These are provided by submit_array.sh (or you can export them manually)
BASE="${BASE:-$HOME/projects/cegs/ECG_Loss}"                  # main repo with .venv
CONF="${CONF:-$BASE/sweeps/cifar100.tsv}"                    # tsv path
MAX_PARALLEL="${MAX_PARALLEL:-8}"                            # not used here; used by submit
DATA_DIR="${DATA_DIR:-$SCRATCH/cegs_data}"
RUNS_DIR="${RUNS_DIR:-$SCRATCH/cegs_runs}"
WANDB_MODE="${WANDB_MODE:-offline}"                          # offline recommended on compute nodes

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
ln -sfn "$DATA_DIR" "$RUN/data"

# W&B isolation per task
export WANDB_DIR="$RUN/wandb"
export WANDB_PROJECT="${WANDB_PROJECT:-CEGS}"
export WANDB_GROUP="cegs_${COMMIT}"
export WANDB_NAME="task${SLURM_ARRAY_TASK_ID}_${COMMIT}"
export WANDB_MODE="$WANDB_MODE"

# Activate venv from BASE (shared env)
source "$BASE/.venv/bin/activate"

# Optional: prevent concurrent dataset downloads from corrupting files
# (If compute nodes can't access internet, this will fail; you can ignore or pre-stage data.)
LOCK="$DATA_DIR/.download.lock"
exec 9>"$LOCK"
flock 9 || true
python - <<PY || true
try:
    from torchvision import datasets
    root = "$DATA_DIR"
    # quick no-op: only ensures root exists; actual download handled inside datasets if you call them
    print("torchvision datasets import OK; data_root=", root)
except Exception as e:
    print("Skip torchvision precheck:", e)
PY
flock -u 9 || true

cd "$RUN/src"
python -u tools/run_from_tsv.py \
  --conf "$CONF" \
  --idx "$SLURM_ARRAY_TASK_ID" \
  --run_dir "$RUN" \
  --commit "$COMMIT" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_mode "$WANDB_MODE"

echo "END=$(date -Iseconds)"