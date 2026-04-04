#!/bin/bash
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -J cegs
#SBATCH -o slurm_logs/slurm_%x_%A_%a.out
#SBATCH -e slurm_logs/slurm_%x_%A_%a.err

# GPU selection is controlled at submission time by scripts/submit_array.sh via -p/--gres.
# Examples:
#   GRES=gpu:h100-80:1 PARTITION=GPU-shared ./scripts/submit_array.sh sweeps/imageNet1k.tsv 10
#   GRES=gpu:l40s-48:1 PARTITION=GPU-shared ./scripts/submit_array.sh sweeps/imageNet1k.tsv 10

set -euo pipefail
umask 077

# Resolve repo root robustly even when Slurm runs a staged copy of this script under /var/spool/slurm/...
# Priority: BASE from environment -> SLURM_SUBMIT_DIR (original submission directory) -> current working directory.
BASE="${BASE:-${SLURM_SUBMIT_DIR:-$PWD}}"
BASE="$(cd "$BASE" && pwd)"
CONF="${CONF:-$BASE/sweeps/imageNet1k.tsv}"

# Prefer DATA_DIR/RUNS_DIR exported by submit_array.sh
# If not provided, default to Ocean project space when available (large quota).
if [[ -z "${DATA_DIR:-}" || -z "${RUNS_DIR:-}" ]]; then
  ACCT="${SLURM_JOB_ACCOUNT:-}"
  OCEAN_USER_DIR=""
  if [[ -n "$ACCT" && -d "/ocean/projects/$ACCT" ]]; then
    OCEAN_USER_DIR="/ocean/projects/$ACCT/$USER"
    mkdir -p "$OCEAN_USER_DIR" 2>/dev/null || true
  fi

  if [[ -n "$OCEAN_USER_DIR" && -d "$OCEAN_USER_DIR" && -w "$OCEAN_USER_DIR" ]]; then
    SCR="$OCEAN_USER_DIR"
  elif [[ -n "${SCRATCH:-}" && -d "$SCRATCH" ]]; then
    SCR="$SCRATCH"
  elif [[ -d "/scratch/$USER" ]]; then
    SCR="/scratch/$USER"
  else
    SCR="$HOME/scratch"
    mkdir -p "$SCR" 2>/dev/null || true
  fi

  DATA_DIR="${DATA_DIR:-$SCR/cegs_data}"
  RUNS_DIR="${RUNS_DIR:-$SCR/cegs_runs}"
fi

AUTO_DOWNLOAD="${CEGS_AUTO_DOWNLOAD:-1}"

mkdir -p "$RUNS_DIR" "$DATA_DIR"

# ---- ImageNet: default 32x32 when conf path contains "imagenet32" or "imnet32" (e.g. RunA/RunB *_imagenet32_* / *_imnet32best.tsv); else 224x224 with shared dataset symlinks ----
if [[ "$CONF" == *imagenet32* || "$CONF" == *imnet32* ]]; then
  export IMAGENET_ORIGINAL="${IMAGENET_ORIGINAL:-0}"
  export IMAGENET_RES="${IMAGENET_RES:-32}"
  # Prefer existing dir: IMAGENET_DS_ROOT env, then DATA_DIR/smallimagenet_32, then sibling cegs_data/smallimagenet_32 (PSC: data often in .../cegs_data)
  if [[ -n "${IMAGENET_DS_ROOT:-}" && -d "$IMAGENET_DS_ROOT" ]]; then
    : # already set and valid
  elif [[ -d "$DATA_DIR/smallimagenet_32" ]]; then
    export IMAGENET_DS_ROOT="$DATA_DIR/smallimagenet_32"
  elif [[ -d "$DATA_DIR/../cegs_data/smallimagenet_32" ]]; then
    export IMAGENET_DS_ROOT="$DATA_DIR/../cegs_data/smallimagenet_32"
  else
    export IMAGENET_DS_ROOT="${IMAGENET_DS_ROOT:-$DATA_DIR/smallimagenet_32}"
  fi
  echo "IMAGENET_DS_ROOT=$IMAGENET_DS_ROOT"
  export DL_WORKERS="${DL_WORKERS:-10}"
else
  mkdir -p "$DATA_DIR/imageNet"
  ln -sfn /ocean/datasets/community/imagenet/train "$DATA_DIR/imageNet/train"
  ln -sfn /ocean/datasets/community/imagenet/val   "$DATA_DIR/imageNet/val"
  export IMAGENET_ORIGINAL="${IMAGENET_ORIGINAL:-1}"
  export IMAGENET_TORCHVISION="${IMAGENET_TORCHVISION:-1}"
  export IMAGENET_ROOT="${IMAGENET_ROOT:-$DATA_DIR/imageNet}"
  export DL_WORKERS="${DL_WORKERS:-5}"
fi
# -------------------------------------------------------------------------------------------------------------

RUN="$RUNS_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$RUN" "$DATA_DIR" "$RUNS_DIR"

# per-run logs
exec > >(tee -a "$RUN/out.log") 2> >(tee -a "$RUN/err.log" >&2)

echo "BASE=$BASE"
echo "CONF=$CONF"
echo "RUN=$RUN"
echo "DATA_DIR=$DATA_DIR"
echo "HOST=$(hostname)"
echo "START=$(date -Iseconds)"

# Snapshot code (copy repo into run folder)
rsync -a --delete \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  --exclude 'logs' --exclude 'models' --exclude 'wandb' \
  "$BASE/" "$RUN/src/"

mkdir -p "$RUN/src/logs" "$RUN/src/models"

# If MODELS_SRC is set, copy checkpoints into the run's models dir (for resume)
if [[ -n "${MODELS_SRC:-}" && -d "$MODELS_SRC" ]]; then
  echo "Copying checkpoints from MODELS_SRC=$MODELS_SRC"
  cp "$MODELS_SRC"/*.pt "$RUN/src/models/" 2>/dev/null || true
fi

COMMIT="$(cd "$BASE" && git rev-parse --short HEAD)"
echo "$COMMIT" > "$RUN/commit.txt"
echo "COMMIT=$COMMIT"

# Make ../data point to shared DATA_DIR
ln -sfn "$DATA_DIR" "$RUN/data"

export CEGS_DATA_DIR="$DATA_DIR"
export CEGS_AUTO_DOWNLOAD="$AUTO_DOWNLOAD"

# W&B isolation per task
mkdir -p "$RUN/wandb"
chmod 700 "$RUN/wandb" 2>/dev/null || true
export WANDB_DIR="$RUN/wandb"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-cegs-imageNet1k-pmax}"
export WANDB_PROJECT_DEFAULT="${WANDB_PROJECT_DEFAULT:-$WANDB_PROJECT}"
export WANDB_JOB_TYPE_DEFAULT="${WANDB_JOB_TYPE_DEFAULT:-official}"

# Make W&B run name map 1:1 to Slurm array task
export WANDB_GROUP="${WANDB_GROUP:-j${SLURM_ARRAY_JOB_ID}}"
export WANDB_NAME="${WANDB_NAME:-j${SLURM_ARRAY_JOB_ID}_t${SLURM_ARRAY_TASK_ID}}"
export WANDB_TAGS="${WANDB_TAGS:-},slurm,job${SLURM_JOB_ID},array${SLURM_ARRAY_JOB_ID},task${SLURM_ARRAY_TASK_ID}"

#Make re
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Activate conda env (Python 3.10)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate cegs
unset LD_PRELOAD || true
python -V
echo "GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 1 2>/dev/null || echo n/a)"
nvidia-smi -L 2>/dev/null || true

cd "$RUN/src"
python -u tools/run_from_tsv.py \
  --conf "$CONF" \
  --idx "$SLURM_ARRAY_TASK_ID" \
  --run_dir "$RUN" \
  --commit "$COMMIT"

echo "END=$(date -Iseconds)"