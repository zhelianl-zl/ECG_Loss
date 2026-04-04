#!/bin/bash
# Resume robust training from checkpoints
# Usage: bash scripts/resume_robust.sh

set -euo pipefail

BASE="/ocean/projects/cis260049p/zliu49/cegs/ECG_Loss"
CONF="$BASE/sweeps/imagenet32_1k_robust.tsv"
SBATCH="$BASE/scripts/cegs_array.sbatch"

# pgdat (task 0, resume from epoch 25)
export CONF MODELS_SRC="/ocean/projects/cis260049p/zliu49/cegs_runs/cegs_38376084_0/src/models"
sbatch -A cis260049p -p GPU-shared --gres=gpu:v100-32:1 -t 2-00:00:00 \
  --array=0-0 --export=ALL \
  "$SBATCH"
echo "Submitted pgdat (task 0)"

# trades (task 1, resume from epoch 25)
export CONF MODELS_SRC="/ocean/projects/cis260049p/zliu49/cegs_runs/cegs_38376085_1/src/models"
sbatch -A cis260049p -p GPU-shared --gres=gpu:v100-32:1 -t 2-00:00:00 \
  --array=1-1 --export=ALL \
  "$SBATCH"
echo "Submitted trades (task 1)"

# mart_w10 (task 3, resume from epoch 30)
export CONF MODELS_SRC="/ocean/projects/cis260049p/zliu49/cegs_runs/cegs_38376047_3/src/models"
sbatch -A cis260049p -p GPU-shared --gres=gpu:v100-32:1 -t 2-00:00:00 \
  --array=3-3 --export=ALL \
  "$SBATCH"
echo "Submitted mart_w10 (task 3)"
