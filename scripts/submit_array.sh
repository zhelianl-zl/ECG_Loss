#!/bin/bash
set -euo pipefail

CONF_PATH="${1:-sweeps/cifar100.tsv}"
MAX_PARALLEL="${2:-8}"

# Defaults (override via env if needed)
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

# Count tasks robustly:
# - Treat the FIRST non-empty line as header (even if it starts with '#dataset')
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

# Robust scratch fallback (because $SCRATCH may be undefined on login nodes)
if [[ -n "${SCRATCH:-}" ]]; then
  SCR="$SCRATCH"
elif [[ -d "/scratch/$USER" ]]; then
  SCR="/scratch/$USER"
elif [[ -n "${SLURM_TMPDIR:-}" ]]; then
  SCR="$SLURM_TMPDIR"
elif [[ -n "${TMPDIR:-}" ]]; then
  SCR="$TMPDIR"
else
  SCR="$HOME/scratch"
fi

export BASE CONF MAX_PARALLEL
export DATA_DIR="${DATA_DIR:-$SCR/cegs_data}"
export RUNS_DIR="${RUNS_DIR:-$SCR/cegs_runs}"
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p "$DATA_DIR" "$RUNS_DIR" || true
chmod 700 "$DATA_DIR" "$RUNS_DIR" 2>/dev/null || true

echo "BASE=$BASE"
echo "CONF=$CONF"
echo "Tasks=$N  MaxParallel=$MAX_PARALLEL"
echo "ACCOUNT=$ACCOUNT PARTITION=$PARTITION QOS=$QOS GRES=$GRES TIME=$TIME"
echo "DATA_DIR=$DATA_DIR"
echo "RUNS_DIR=$RUNS_DIR"
echo

# Submit and ALWAYS print the sbatch output
SUBMIT_OUT=$(sbatch \
  -A "$ACCOUNT" \
  -p "$PARTITION" \
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