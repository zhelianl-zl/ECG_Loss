#!/bin/bash
# Resume trades from epoch 50 checkpoint, extend to 60 epochs
# Usage: bash scripts/resume_trades_60ep.sh

set -euo pipefail

BASE="/ocean/projects/cis260049p/zliu49/cegs/ECG_Loss"
CONF="$BASE/sweeps/imagenet32_1k_trades_60ep.tsv"
SBATCH="$BASE/scripts/cegs_array.sbatch"

export CONF MODELS_SRC="/ocean/projects/cis260049p/zliu49/cegs_runs/cegs_38435706_1/src/models"
sbatch -A cis260049p -p GPU-shared --gres=gpu:v100-32:1 -t 1-00:00:00 \
  --array=0-0 --export=ALL \
  "$SBATCH"
echo "Submitted trades 60ep resume (from epoch 50, cegs_38435706_1)"
