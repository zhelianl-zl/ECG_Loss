#!/bin/bash
#SBATCH -N 1
#SBATCH -t 08:00:00
#SBATCH -J cegs
#SBATCH -o slurm_logs/slurm_%x_%A_%a.out
#SBATCH -e slurm_logs/slurm_%x_%A_%a.err

set -euo pipefail
umask 077

BASE="${BASE:-$HOME/projects/cegs/ECG_Loss}"
CONF="${CONF:-$BASE/sweeps/cifar100.tsv}"

# Prefer DATA_DIR/RUNS_DIR exported by submit_array.sh
if [[ -z "${DATA_DIR:-}" || -z "${RUNS_DIR:-}" ]]; then
  SCR="$HOME/scratch"
  DATA_DIR="${DATA_DIR:-$SCR/cegs_data}"
  RUNS_DIR="${RUNS_DIR:-$SCR/cegs_runs}"
fi

mkdir -p "$BASE/slurm_logs" "$DATA_DIR" "$RUNS_DIR"

RUN="$RUNS_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$RUN"

# per-run logs
exec > >(tee -a "$RUN/out.log") 2> >(tee -a "$RUN/err.log" >&2)

echo "BASE=$BASE"
echo "CONF=$CONF"
echo "RUN=$RUN"
echo "DATA_DIR=$DATA_DIR"
echo "HOST=$(hostname)"
echo "START=$(date -Iseconds)"

# Snapshot code
rsync -a --delete \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  --exclude 'logs' --exclude 'models' --exclude 'wandb' \
  "$BASE/" "$RUN/src/"

mkdir -p "$RUN/src/logs" "$RUN/src/models"

COMMIT="$(cd "$BASE" && git rev-parse --short HEAD)"
echo "$COMMIT" > "$RUN/commit.txt"
echo "COMMIT=$COMMIT"

# Make ../data (relative to RUN/src) point to shared DATA_DIR
ln -sfn "$DATA_DIR" "$RUN/data"

export CEGS_DATA_DIR="$DATA_DIR"
export CEGS_AUTO_DOWNLOAD="${CEGS_AUTO_DOWNLOAD:-1}"

# Activate conda env (Python 3.10)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate cegs
unset LD_PRELOAD || true
python -V

# Load W&B config (so job gets the same routing as submit script)
if [[ -f "$BASE/configs/wandb.env" ]]; then
  # shellcheck disable=SC1091
  source "$BASE/configs/wandb.env"
fi

# Auto fallback: if compute node can't reach W&B, switch to offline (training continues)
if [[ "${WANDB_MODE:-online}" == "online" ]]; then
  python - <<'PY' || export WANDB_MODE="offline"
import socket
socket.setdefaulttimeout(2.0)
socket.create_connection(("api.wandb.ai", 443))
print("wandb network ok")
PY
fi
echo "WANDB_MODE=$WANDB_MODE"

# W&B isolation per task (never mix runs)
mkdir -p "$RUN/wandb"
chmod 700 "$RUN/wandb" 2>/dev/null || true
export WANDB_DIR="$RUN/wandb"

cd "$RUN/src"
python -u tools/run_from_tsv.py \
  --conf "$CONF" \
  --idx "$SLURM_ARRAY_TASK_ID" \
  --run_dir "$RUN" \
  --commit "$COMMIT"

echo "END=$(date -Iseconds)"