#!/bin/bash
set -euo pipefail

CONF_PATH="${1:-sweeps/cifar100.tsv}"
MAX_PARALLEL="${2:-8}"

# Defaults (override by env or flags if you want)
ACCOUNT="${ACCOUNT:-cis260049p}"
PARTITION="${PARTITION:-GPU-shared}"
QOS="${QOS:-gpu}"
GRES="${GRES:-gpu:v100-32:1}"
TIME="${TIME:-08:00:00}"

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONF="$BASE/$CONF_PATH"
SBATCH_SCRIPT="$BASE/scripts/cegs_array.sbatch"

if [[ ! -f "$CONF" ]]; then
  echo "Config TSV not found: $CONF"
  exit 2
fi

# Count non-comment non-empty lines
mapfile -t LINES < <(grep -v '^\s*#' "$CONF" | awk 'NF')
if (( ${#LINES[@]} < 2 )); then
  echo "TSV must have header + >=1 data row: $CONF"
  exit 2
fi

N=$(( ${#LINES[@]} - 1 ))   # exclude header
echo "BASE=$BASE"
echo "CONF=$CONF"
echo "Tasks=$N  MaxParallel=$MAX_PARALLEL"
echo "ACCOUNT=$ACCOUNT PARTITION=$PARTITION QOS=$QOS GRES=$GRES TIME=$TIME"

export BASE CONF MAX_PARALLEL

SCR="${SCRATCH:-/scratch/$USER}"

export DATA_DIR="${DATA_DIR:-$SCRATCH/cegs_data}"
export RUNS_DIR="${RUNS_DIR:-$SCRATCH/cegs_runs}"
export WANDB_MODE="${WANDB_MODE:-offline}"

sbatch \
  -A "$ACCOUNT" \
  -p "$PARTITION" \
  --qos="$QOS" \
  --gres="$GRES" \
  -t "$TIME" \
  --array="0-$((N-1))%${MAX_PARALLEL}" \
  "$SBATCH_SCRIPT"