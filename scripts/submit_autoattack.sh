#!/bin/bash
# Submit AutoAttack evaluation for all 15 tasks (5 datasets x 3 runs each)
# Usage: bash scripts/submit_autoattack.sh
set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONF="$BASE/sweeps/autoattack_eval_v1.tsv"

export BASE CONF
export DATA_DIR="${DATA_DIR:-/ocean/projects/cis260049p/zliu49/cegs}"
export RUNS_DIR="${RUNS_DIR:-/ocean/projects/cis260049p/zliu49/cegs_runs}"
export WANDB_MODE="${WANDB_MODE:-online}"

mkdir -p "$BASE/slurm_logs"

sbatch -A cis260049p -p GPU-shared --gres=gpu:v100-32:1 -t 0-06:00:00 \
  --export=ALL \
  --array=0-29 \
  "$BASE/scripts/eval_array.sbatch"

echo "Submitted autoattack_eval_v1: tasks 0-29 (0-14=eps4, 15-29=eps2)"
