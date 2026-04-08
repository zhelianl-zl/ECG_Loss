# ECG Loss — Equalized Cross-Gating Gradient Scaling

ECG Loss is a per-sample gradient scaling method for deep classification. It up-weights the gradient contribution of hard, uncertain samples relative to confident ones, guided by a confidence gate derived from the model's own output distribution. The result is a smooth, adaptive emphasis on samples near the decision boundary without discarding any training signal.

## Method Overview

For each sample in a minibatch, ECG computes a **confidence gate** from the model's current prediction and uses it to scale the backward gradients:

```
gate_i  = wrong_gate_i × conf_gate_i
scale_i = 1 + λ · gate_i
```

- `wrong_gate_i` — 1 minus the probability assigned to the true class, so mis-classified samples receive a stronger signal.
- `conf_gate_i` — sigmoid sharpening around a threshold τ, which separates low-confidence (uncertain) samples from high-confidence ones.
- `λ` — scaling strength, either fixed or controlled by an **auto-lambda** controller that keeps the mean gate contribution at a target level.
- `τ` — confidence threshold, either fixed or adapted per epoch by **auto\_q\_valley**, which detects the valley between the low- and high-confidence modes of the confidence histogram automatically.

The forward loss (cross-entropy) is unchanged; only the backward gradients are rescaled. An optional `scale_normalize` step keeps the global step size constant while redistributing gradient mass toward harder samples.

## Key Features

- **auto\_q\_valley**: automatic τ scheduling — finds the valley in the per-epoch confidence histogram, no manual threshold tuning required.
- **Auto-lambda**: keeps mean gate strength at a user-defined target, adapting λ to the current training dynamics.
- **Multi-dataset**: CIFAR-10, CIFAR-100, SVHN, BinaryCIFAR-10, ImageNet-32 (SmallImageNet).
- **Robust training**: PGD-AT, TRADES, and MART robust training integrated.
- **Adversarial evaluation**: FGSM, PGD-Linf, and PGD-Linf with random start (per-dataset ε in pixel scale).
- **Corruption evaluation**: CIFAR-C / CIFAR-100-C with configurable corruption list and severity level.
- **TSV-based sweep dispatch**: grid/list experiments via tab-separated config files; each row runs as an independent Slurm array task.

## Repository Structure

```
train.py               # Main training entry point
ecg_loss.py            # ECG loss, confidence gate, auto_q_valley controller
robust_losses.py       # TRADES / MART / PGD-AT loss functions
models.py              # Model definitions (ResNet variants, WideResNet)
run.py                 # Local parallel launcher (multi-GPU, research use)
tools/
  eval_checkpoints.py  # Adversarial & corruption evaluation tool
  run_from_tsv.py      # TSV-to-CLI sweep dispatcher
scripts/
  cegs_array.sbatch    # Slurm array job template (PSC Bridges-2)
sweeps/                # Experiment config TSV files
configs/               # W&B / environment config files
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision wandb
```

ImageNet-32 requires the SmallImageNet pickle dataset. Set the environment variable `IMAGENET_DS_ROOT` to point to the directory containing `train_data_batch_1` ... `train_data_batch_10` and `val_data`.

## Quick Start

**Standard ECG training on CIFAR-10 (60 epochs):**

```bash
python train.py \
  --dataset cifar10 \
  --stop epochs --stop_val 60 \
  --lr 0.01 --momentum 0.9 --batch 128 \
  --loss_stage1 ecg --loss_stage2 ecg \
  --stage1_epochs 60 --stage2_epochs 0 \
  --ecg_conf_type pmax \
  --ecg_lam_start auto_tr_autocap --ecg_lam_end 0.05 \
  --ecg_tau_start auto_q_valley --ecg_tau_end 0.36 \
  --ecg_k_start 20 --ecg_k_end 20 \
  --ecg_schedule linear
```

**TSV-based sweep (recommended for multi-run experiments):**

```bash
# Edit sweeps/your_config.tsv, then dispatch via Slurm:
sbatch --array=0-N scripts/cegs_array.sbatch
```

See [README_PSC.md](README_PSC.md) for PSC Bridges-2 specific setup and the `sweeps/` directory for example configurations.

## Evaluation

```bash
python tools/eval_checkpoints.py \
  --run_dir /path/to/run \
  --attacks fgsm,pgd_linf,pgd_linf_rs \
  --adv_eps 8 --adv_steps 20 \
  --c_corruptions gaussian_noise,shot_noise,... --c_severity 5
```

## Citation

If you use this code, please cite the associated paper (forthcoming).
