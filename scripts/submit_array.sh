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

# ---- clean/archive old slurm logs to keep repo clean ----
mkdir -p "$BASE/slurm_logs"
mv "$BASE"/slurm_*.out "$BASE"/slurm_*.err "$BASE"/slurm_logs/" 2>/dev/null || true
# ---------------------------------------------------------

# ---- load wandb config file (optional) ----
WCFG="${WANDB_CONFIG:-$BASE/configs/wandb.env}"
if [[ -f "$WCFG" ]]; then
  # shellcheck disable=SC1090
  source "$WCFG"
fi
# ---------------------------------------------------------

# Count tasks robustly:
# - Treat FIRST non-empty line as header (even if it starts with '#dataset')
# - Ignore later comment lines starting with '#'
mapfile -t LINES < <(awk '
  BEGIN{seen=0}
  /^[[:space:]]*$/ {next}
  {
    if (seen==0) {print; seen=1; next}
    if ($0 ~ /^[[:space:]]*#/) next
    print
  }
' "$CONF")

if (( ${#LINES[@]} < 2 )); then
  echo "ERROR: TSV must have header + >=1 data row: $CONF" >&2
  exit 2
fi

N=$(( ${#LINES[@]} - 1 ))   # exclude header

# Robust workspace selection
OCEAN_ROOT="/ocean/projects/${ACCOUNT}"
OCEAN_USER_DIR="${OCEAN_ROOT}/${USER}"

if [[ -d "$OCEAN_ROOT" ]]; then
  mkdir -p "$OCEAN_USER_DIR" 2>/dev/null || true
fi

if [[ -d "$OCEAN_USER_DIR" && -w "$OCEAN_USER_DIR" ]]; then
  SCR="$OCEAN_USER_DIR"
elif [[ -n "${SCRATCH:-}" && -d "$SCRATCH" ]]; then
  SCR="$SCRATCH"
elif [[ -d "/scratch/$USER" ]]; then
  SCR="/scratch/$USER"
elif [[ -d "$HOME/scratch" ]]; then
  SCR="$HOME/scratch"
else
  SCR="$HOME"
fi

export BASE CONF
export DATA_DIR="${DATA_DIR:-$SCR/cegs_data}"
export RUNS_DIR="${RUNS_DIR:-$SCR/cegs_runs}"
export WANDB_MODE="${WANDB_MODE:-${WANDB_MODE:-offline}}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_PROJECT_DEFAULT="${WANDB_PROJECT_DEFAULT:-CEGS}"
export WANDB_JOB_TYPE_DEFAULT="${WANDB_JOB_TYPE_DEFAULT:-official}"

mkdir -p "$DATA_DIR" "$RUNS_DIR" "$BASE/slurm_logs" || true
chmod 700 "$DATA_DIR" "$RUNS_DIR" 2>/dev/null || true

echo "BASE=$BASE"
echo "CONF=$CONF"
echo "Tasks=$N  MaxParallel=$MAX_PARALLEL"
echo "ACCOUNT=$ACCOUNT PARTITION=$PARTITION QOS=$QOS GRES=$GRES TIME=$TIME"
echo "DATA_DIR=$DATA_DIR"
echo "RUNS_DIR=$RUNS_DIR"
echo "WANDB_MODE=$WANDB_MODE WANDB_ENTITY=${WANDB_ENTITY:-<default>} WANDB_PROJECT_DEFAULT=$WANDB_PROJECT_DEFAULT"
echo

SUBMIT_OUT=$(sbatch \
  -A "$ACCOUNT" \
  -p "$PARTITION" \
  --export=ALL \
  --qos="$QOS" \
  --gres="$GRES" \
  -t "$TIME" \
  --array="0-$((N-1))%${MAX_PARALLEL}" \
  "$SBATCH_SCRIPT" 2>&1) || {
    echo "ERROR: sbatch failed:" >&2
    echo "$SUBMIT_OUT" >&2
    exit 3
  }

echo "$SUBMIT_OUT"