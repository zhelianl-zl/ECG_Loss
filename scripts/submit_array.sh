#!/bin/bash
set -euo pipefail

CONF_PATH="${1:-sweeps/cifar100.tsv}"
MAX_PARALLEL="${2:-8}"

# Slurm defaults (override via env if needed)
ACCOUNT="${ACCOUNT:-cis260049p}"
PARTITION="${PARTITION:-GPU-shared}"
QOS="${QOS:-gpu}"
GRES="${GRES:-gpu:v100-32:1}"
TIME="${TIME:-08:00:00}"

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

# Count tasks: first non-empty line is header; ignore later full-line comments
N=$(awk '
  BEGIN{seen=0; n=0}
  /^[[:space:]]*$/ {next}
  {
    if (seen==0) {seen=1; next}     # header
    if ($0 ~ /^[[:space:]]*#/) next # skip comments
    n++
  }
  END{print n}
' "$CONF")

if [[ "$N" -le 0 ]]; then
  echo "ERROR: TSV must have header + >=1 data row: $CONF" >&2
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
export DATA_DIR="${DATA_DIR:-$SCR/cegs_data}"
export RUNS_DIR="${RUNS_DIR:-$SCR/cegs_runs}"

mkdir -p "$BASE/slurm_logs" "$DATA_DIR" "$RUNS_DIR" || true

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_PROJECT_DEFAULT="${WANDB_PROJECT_DEFAULT:-CEGS}"
export WANDB_JOB_TYPE_DEFAULT="${WANDB_JOB_TYPE_DEFAULT:-official}"

echo "BASE=$BASE"
echo "CONF=$CONF"
echo "Tasks=$N  MaxParallel=$MAX_PARALLEL"
echo "ACCOUNT=$ACCOUNT PARTITION=$PARTITION QOS=$QOS GRES=$GRES TIME=$TIME"
echo "DATA_DIR=$DATA_DIR"
echo "RUNS_DIR=$RUNS_DIR"
echo "WANDB_MODE=$WANDB_MODE WANDB_ENTITY=${WANDB_ENTITY:-<empty>} WANDB_PROJECT_DEFAULT=$WANDB_PROJECT_DEFAULT"
echo

# IMPORTANT: --export=ALL ensures BASE/CONF/DATA_DIR/RUNS_DIR propagate to compute nodes
sbatch \
  -A "$ACCOUNT" \
  -p "$PARTITION" \
  --qos="$QOS" \
  --gres="$GRES" \
  -t "$TIME" \
  --export=ALL \
  --array="0-$((N-1))%${MAX_PARALLEL}" \
  "$SBATCH_SCRIPT"
