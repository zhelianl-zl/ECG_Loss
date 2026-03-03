#!/bin/bash
set -euo pipefail

# Usage:
#   DATA_DIR=/path/to/cegs_data ./download_smallimagenet.sh 32
#   DATA_DIR=/path/to/cegs_data ./download_smallimagenet.sh 64
#
# This downloads the official downsampled ImageNet zips (CIFAR-like pickles)
# and extracts them into:
#   $DATA_DIR/SmallImageNet_<RES>x<RES>/

RES="${1:-32}"
DATA_DIR="${DATA_DIR:-$PWD/data}"

OUT_DIR="$DATA_DIR/SmallImageNet_${RES}x${RES}"
ZIP_DIR="$DATA_DIR/smallimagenet_zips"
mkdir -p "$OUT_DIR" "$ZIP_DIR"

TRAIN_ZIP="$ZIP_DIR/Imagenet${RES}_train.zip"
VAL_ZIP="$ZIP_DIR/Imagenet${RES}_val.zip"

echo "[+] Downloading to $ZIP_DIR"
wget -O "$TRAIN_ZIP" "https://image-net.org/data/downsample/Imagenet${RES}_train.zip"
wget -O "$VAL_ZIP"   "https://image-net.org/data/downsample/Imagenet${RES}_val.zip"

echo "[+] Extracting into $OUT_DIR"
unzip -o "$TRAIN_ZIP" -d "$OUT_DIR"
unzip -o "$VAL_ZIP"   -d "$OUT_DIR"

echo "[+] Checking files"
ls -lh "$OUT_DIR"/train_data_batch_1 "$OUT_DIR"/val_data
echo "[OK] Ready: $OUT_DIR"
