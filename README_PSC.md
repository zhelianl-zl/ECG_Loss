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
- C-suite (RunA/RunB) needs **CIFAR-10-C** and **CIFAR-100-C** under `$DATA_DIR`. Code looks for `$DATA_DIR/CIFAR-10-C/` and `$DATA_DIR/CIFAR-100-C/`. You do the download + extract once on PSC; then point `DATA_DIR` at that directory (e.g. under `cegs`, same level as repo):

**On PSC (one-time):** pick a data root, e.g. your `cegs` directory (same parent as `ECG_Loss` repo):

```bash
# 数据根目录：和 ECG_Loss 同级的 cegs 下，代码会找 $DATA_DIR/CIFAR-10-C 和 $DATA_DIR/CIFAR-100-C
DATA_DIR=/ocean/projects/cis260049p/zliu49/cegs
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# CIFAR-10-C（下载并解压到当前目录，得到 CIFAR-10-C/）
curl -L -o CIFAR-10-C.tar "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1"
tar -xf CIFAR-10-C.tar

# CIFAR-100-C（同上，得到 CIFAR-100-C/）
curl -L -o CIFAR-100-C.tar "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar?download=1"
tar -xf CIFAR-100-C.tar
```

- **提交任务时** 设置环境变量（或在 `submit_array.sh` 里写死）：  
  `export DATA_DIR=/ocean/projects/cis260049p/zliu49/cegs`  
  这样训练时就会用 `cegs/CIFAR-10-C` 和 `cegs/CIFAR-100-C`，无需改代码。
- ImageNet32：同样手动放数据（如 smallimagenet_32），用 `IMAGENET_DS_ROOT` 指过去即可；训练脚本里不做自动下载。

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