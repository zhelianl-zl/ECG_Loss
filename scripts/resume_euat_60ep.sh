#!/bin/bash
# Resume imnet32 euat from epoch 55 -> 60
# Usage: bash scripts/resume_euat_60ep.sh

set -euo pipefail

BASE="/ocean/projects/cis260049p/zliu49/cegs/ECG_Loss"
SBATCH="$BASE/scripts/cegs_array.sbatch"

export CONF="$BASE/sweeps/imagenet32_euat_60ep_resume.tsv"
export MODELS_SRC="/ocean/projects/cis260049p/zliu49/cegs_runs/cegs_38264075_4/src/models"

sbatch -A cis260049p -p GPU-shared --gres=gpu:v100-32:1 -t 1-00:00:00 \
  --array=0-0 --export=ALL \
  "$SBATCH"
echo "Submitted euat 55->60ep"
