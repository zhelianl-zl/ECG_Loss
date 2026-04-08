#!/bin/bash
# Resume pgdat from epoch 50 -> 60
# Usage: bash scripts/resume_pgdat_60ep.sh

set -euo pipefail

BASE="/ocean/projects/cis260049p/zliu49/cegs/ECG_Loss"
SBATCH="$BASE/scripts/cegs_array.sbatch"

export CONF="$BASE/sweeps/imagenet32_1k_pgdat_60ep.tsv"
export MODELS_SRC="/ocean/projects/cis260049p/zliu49/cegs_runs/cegs_38435705_0/src/models"

sbatch -A cis260049p -p GPU-shared --gres=gpu:v100-32:1 -t 2-00:00:00 \
  --array=0-0 --export=ALL \
  "$SBATCH"
echo "Submitted pgdat 50->60ep"
