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