#!/bin/bash
# Resume imnet32 valley_v3 w0 from epoch 50 -> 60
# Usage: bash scripts/resume_valley_v3_w0_60ep.sh
#
# Fix: new run uses stop_val=60 -> stage1_epochs=60 -> checkpoint name contains s160,
# but original checkpoints have s150. We pre-copy and rename into a staging dir.

set -euo pipefail

BASE="/ocean/projects/cis260049p/zliu49/cegs/ECG_Loss"
SBATCH="$BASE/scripts/cegs_array.sbatch"
SRC="/ocean/projects/cis260049p/zliu49/cegs_runs/cegs_38432902_2/src/models"
STAGING="/ocean/projects/cis260049p/zliu49/cegs_staging/valley_v3_w0_60ep"

# Create staging dir and copy checkpoints with s150 -> s160 rename
mkdir -p "$STAGING"
for f in "$SRC"/*.pt; do
    fname=$(basename "$f")
    new_fname="${fname//_s150_ecg_/_s160_ecg_}"
    cp "$f" "$STAGING/$new_fname"
done
echo "Staged checkpoints in $STAGING:"
ls "$STAGING"

export CONF="$BASE/sweeps/imagenet32_1k_valley_v3_w0_60ep.tsv"
export MODELS_SRC="$STAGING"

sbatch -A cis260049p -p GPU-shared --gres=gpu:v100-32:1 -t 2-00:00:00 \
  --array=0-0 --export=ALL \
  "$SBATCH"
echo "Submitted valley_v3_w0 50->60ep"
