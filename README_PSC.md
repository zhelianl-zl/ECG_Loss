# PSC / Bridges2 Slurm Array Workflow (1 GPU per task)

## One-time setup
1) Clone repo on PSC:
- `git clone ...`
- `cd ECG_Loss`

2) Create venv once (shared by all jobs):
- `python3 -m venv .venv`
- `source .venv/bin/activate`
- install deps (torch/torchvision, wandb, etc.)

3) W&B:
- Recommended on compute nodes: `WANDB_MODE=offline`
- After jobs finish, sync from login node:
  - `wandb sync $SCRATCH/cegs_runs/<run_id>/wandb`

4) Data (CIFAR-C, ImageNet32) — one-time install:
- C-suite (RunA/RunB) needs CIFAR-10-C and CIFAR-100-C under `$DATA_DIR`. Do this once (e.g. in `cegs_data` next to smallimagenet_32):

```bash
DATA_DIR=/ocean/projects/cis260049p/zliu49/cegs_data   # or your path
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# CIFAR-10-C
curl -L -o CIFAR-10-C.tar "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1"
tar -xf CIFAR-10-C.tar

# CIFAR-100-C
curl -L -o CIFAR-100-C.tar "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar?download=1"
tar -xf CIFAR-100-C.tar
```

- Then set `DATA_DIR` when submitting (or in `submit_array.sh`):  
  `export DATA_DIR=/ocean/projects/cis260049p/zliu49/cegs_data`
- ImageNet32: same idea — put smallimagenet in a dir and set `IMAGENET_DS_ROOT` (and optionally `IMAGENET_RES=32`). No auto-download in the training script.

## Sweep config
Edit `sweeps/cifar100.tsv`:
- First non-comment line is header
- Each subsequent line is one task (one GPU)

## Submit (max parallel GPUs = 8)
- `bash scripts/submit_array.sh sweeps/cifar100.tsv 8`

Override Slurm parameters if needed:
- `ACCOUNT=cis260279 PARTITION=GPU-shared QOS=gpu GRES=gpu:v100-32:1 bash scripts/submit_array.sh sweeps/cifar100.tsv 8`

## Outputs (no collisions)
Each task writes into its own folder:
`$SCRATCH/cegs_runs/cegs_<jobid>_<taskid>/`

Inside:
- `src/` snapshot of code used for this run
- `src/models/` checkpoints (e.g., every 15 epochs)
- `src/logs/` training logs
- `wandb/` isolated W&B files
- `commit.txt`, `cmd.txt`, `config_row.tsv` for reproducibility