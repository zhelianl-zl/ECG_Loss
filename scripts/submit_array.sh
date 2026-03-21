#!/bin/bash
set -euo pipefail

CONF_PATH="${1:-sweeps/cifar100.tsv}"
# Second arg: concurrent array cap for Slurm --array=START-END%CAP.
#   0 | all | max | unlimited  -> no % cap (schedule all tasks; cluster/QOS still limits actual slots)
#   <positive int>             -> at most CAP tasks running at once
# Omitted defaults to 0 (no artificial throttle).
MAX_PARALLEL="${2:-0}"

# Optional positional overrides:
#   $3: GPU model or full GRES (e.g., "h100-80" or "gpu:h100-80:1")
#   $4: partition (e.g., "GPU-shared" or "GPU")
GPU_ARG="${3:-}"
PARTITION_ARG="${4:-}"

if [[ -n "$GPU_ARG" ]]; then
  if [[ "$GPU_ARG" == gpu:* ]]; then
    export GRES="$GPU_ARG"
  else
    export GRES="gpu:${GPU_ARG}:1"
  fi
fi

if [[ -n "$PARTITION_ARG" ]]; then
  export PARTITION="$PARTITION_ARG"
fi

# Slurm defaults (override via env if needed)
ACCOUNT="${ACCOUNT:-cis260049p}"
PARTITION="${PARTITION:-GPU-shared}"
QOS="${QOS:-}"
GRES="${GRES:-gpu:v100-32:1}"
TIME="${TIME:-1-00:00:00}"


# QOS handling:
# - On Bridges-2, GPU/GPU-shared typically accept QOS=gpu.
# - GPU-dev (A100) often rejects QOS; leave empty to omit --qos.
if [[ -z "$QOS" && "$PARTITION" != "GPU-dev" ]]; then
  QOS="gpu"
fi
QOS_ARGS=()
if [[ -n "$QOS" ]]; then
  QOS_ARGS+=(--qos="$QOS")
fi

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONF="$BASE/$CONF_PATH"
SBATCH_SCRIPT="$BASE/scripts/cegs_array.sbatch"

if [[ ! -f "$CONF" ]]; then
  echo "ERROR: Config TSV not found: $CONF" >&2
  exit 2
fi

# Load wandb config if exists
WCFG="${WANDB_CONFIG:-$BASE/configs/wandb.env}"
if [[ -f "$WCFG" ]]; then
  # shellcheck disable=SC1090
  source "$WCFG"
fi

# Count tasks robustly:
# - allow top meta directives like: #wandb_project=...
# - ignore full-line comments anywhere
# - header is the first non-comment line (or a commented header starting with #dataset\t...)
N=$(awk '
  BEGIN{header_seen=0; n=0}
  /^[[:space:]]*$/ {next}

  # full-line comments
  /^[[:space:]]*#/ {
    # treat a commented header like "#dataset\tseed..." as header
    if (header_seen==0 && $0 ~ /^[[:space:]]*#dataset[[:space:]]*\t/) header_seen=1
    next
  }

  # first non-comment line is header
  {
    if (header_seen==0) {header_seen=1; next}
    n++
  }
  END{print n}
' "$CONF")

if [[ "$N" -le 0 ]]; then
  echo "ERROR: TSV must have header + >=1 data row: $CONF" >&2
  exit 2
fi

ARRAY_RANGE="0-$((N-1))"
mpl="$(printf '%s' "$MAX_PARALLEL" | tr '[:upper:]' '[:lower:]')"
if [[ "$MAX_PARALLEL" == "0" || "$mpl" == "all" || "$mpl" == "max" || "$mpl" == "unlimited" ]]; then
  ARRAY_SPEC="$ARRAY_RANGE"
  MP_DISPLAY="unlimited (no % throttle)"
elif [[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  ARRAY_SPEC="${ARRAY_RANGE}%${MAX_PARALLEL}"
  MP_DISPLAY="$MAX_PARALLEL"
else
  echo "ERROR: second arg must be 0|all|max|unlimited for no throttle, or a positive integer" >&2
  exit 2
fi

# Choose workspace root (prefer Ocean)
OCEAN_USER_DIR="/ocean/projects/${ACCOUNT}/${USER}"
if [[ -d "/ocean/projects/${ACCOUNT}" ]]; then
  mkdir -p "$OCEAN_USER_DIR" 2>/dev/null || true
fi

if [[ -d "$OCEAN_USER_DIR" && -w "$OCEAN_USER_DIR" ]]; then
  SCR="$OCEAN_USER_DIR"
elif [[ -n "${SCRATCH:-}" && -d "$SCRATCH" ]]; then
  SCR="$SCRATCH"
elif [[ -d "/scratch/$USER" ]]; then
  SCR="/scratch/$USER"
else
  SCR="$HOME/scratch"
  mkdir -p "$SCR" 2>/dev/null || true
fi

export BASE CONF
# CIFAR-C (and optional data) under cegs; override with env DATA_DIR if needed
export DATA_DIR="${DATA_DIR:-/ocean/projects/cis260049p/zliu49/cegs}"
export RUNS_DIR="${RUNS_DIR:-$SCR/cegs_runs}"

mkdir -p "$BASE/slurm_logs" "$DATA_DIR" "$RUNS_DIR" || true

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_PROJECT_DEFAULT="${WANDB_PROJECT_DEFAULT:-CEGS}"
export WANDB_JOB_TYPE_DEFAULT="${WANDB_JOB_TYPE_DEFAULT:-official}"

echo "BASE=$BASE"
echo "CONF=$CONF"
echo "Tasks=$N  Array=$ARRAY_SPEC  MaxParallel=$MP_DISPLAY"
echo "ACCOUNT=$ACCOUNT PARTITION=$PARTITION QOS=${QOS:-<none>} GRES=$GRES TIME=$TIME"
echo "DATA_DIR=$DATA_DIR"
echo "RUNS_DIR=$RUNS_DIR"
echo "WANDB_MODE=$WANDB_MODE WANDB_ENTITY=${WANDB_ENTITY:-<empty>} WANDB_PROJECT_DEFAULT=$WANDB_PROJECT_DEFAULT"
echo

# IMPORTANT: --export=ALL ensures BASE/CONF/DATA_DIR/RUNS_DIR propagate to compute nodes
sbatch \
  -A "$ACCOUNT" \
  -p "$PARTITION" \
  "${QOS_ARGS[@]}" \
  --gres="$GRES" \
  -t "$TIME" \
  --export=ALL \
  --array="$ARRAY_SPEC" \
  "$SBATCH_SCRIPT"