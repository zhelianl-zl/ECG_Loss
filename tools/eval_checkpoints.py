#!/usr/bin/env python3
"""
Offline checkpoint evaluator v2 — ADV / C-suite / calibration / LT / ECGS diagnostics.

All adversarial attacks run in **pixel space [0,1]** with correct clamping.
A NormalizeModel wrapper applies per-channel normalization before the real
forward pass, so eps/alpha have their literal pixel-scale meaning.

Usage:

  # TSV batch mode (SLURM array)
  python tools/eval_checkpoints.py --conf sweeps/RunA_eval.tsv --idx 0

  # Single SLURM job
  python tools/eval_checkpoints.py --job 37988109 --task 0 --dataset binaryCifar10 \
      --slurm_logs_dir slurm_logs --wandb_project ecg_binary_pmax

  # Directory of checkpoints
  python tools/eval_checkpoints.py --ckpt_dir /path/to/models/ --dataset cifar10

  # Single checkpoint
  python tools/eval_checkpoints.py --ckpt /path/to/model_epoch60.pt --dataset cifar10
"""

import os
import sys
import glob
import pickle
import random
import argparse
import time
import re
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models import ResNet, BasicBlock, Bottleneck, Wide_ResNet
import torchvision.models as tv_models
from torchvision import datasets, transforms
from torchvision.datasets.vision import VisionDataset
from PIL import Image

try:
    from autoattack import AutoAttack
    HAS_AUTOATTACK = True
except ImportError:
    HAS_AUTOATTACK = False

DEFAULT_RUNS_DIR = "/ocean/projects/cis260049p/zliu49/cegs_runs"

DATASET_STATS: Dict[str, dict] = {
    "cifar10":       {"mean": (0.4914, 0.4822, 0.4465), "std": (0.2471, 0.2435, 0.2616), "C": 10},
    "binaryCifar10": {"mean": (0.4914, 0.4822, 0.4465), "std": (0.2471, 0.2435, 0.2616), "C": 2},
    "cifar100":      {"mean": (0.5071, 0.4867, 0.4408), "std": (0.2675, 0.2565, 0.2761), "C": 100},
    "svhn":          {"mean": (0.0, 0.0, 0.0),          "std": (1.0, 1.0, 1.0),          "C": 10},
    "imageNet":      {"mean": (0.4810, 0.4574, 0.4078), "std": (0.2146, 0.2104, 0.2138), "C": 1000},
    "imageNet224":   {"mean": (0.485, 0.456, 0.406),    "std": (0.229, 0.224, 0.225),    "C": 1000},
}


# ═══════════════════════════════════════════════════════════════════════════
#  NormalizeModel — wraps a raw model so attacks can operate in [0,1]
# ═══════════════════════════════════════════════════════════════════════════

class NormalizeModel(nn.Module):
    """Prepend per-channel normalization so inputs stay in pixel space [0,1]."""

    def __init__(self, model: nn.Module, mean: Tuple[float, ...], std: Tuple[float, ...]):
        super().__init__()
        self.model = model
        self.register_buffer("_mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("_std",  torch.tensor(std,  dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self._mean) / self._std)


# ═══════════════════════════════════════════════════════════════════════════
#  Dataset helpers (test-only, mirrors train.py)
# ═══════════════════════════════════════════════════════════════════════════

class SmallImagenet(VisionDataset):
    """ImageNet-Downsampled (32×32 / 64×64) pickle loader — copied from train.py."""
    train_list = [f"train_data_batch_{i+1}" for i in range(10)]
    val_list = ["val_data"]

    def __init__(self, root="data", size=32, train=True, transform=None, classes=None, shuffle=False):
        super().__init__(root, transform=transform)
        file_list = self.train_list if train else self.val_list
        self.data, self.targets = [], []
        for fn in file_list:
            with open(os.path.join(self.root, fn), "rb") as f:
                entry = pickle.load(f)
            self.data.append(entry["data"].reshape(-1, 3, size, size))
            self.targets.append(entry["labels"])
        self.data = np.vstack(self.data).transpose((0, 2, 3, 1))
        self.targets = np.concatenate(self.targets).astype(int) - 1
        if classes is not None:
            classes = np.array(classes)
            fd, ft = [], []
            for c in classes:
                m = self.targets == c
                fd.append(self.data[m])
                ft.append(self.targets[m])
            self.data = np.vstack(fd)
            self.targets = np.concatenate(ft)
        if shuffle:
            perm = np.arange(len(self.data))
            random.shuffle(perm)
            self.data = self.data[perm]
            self.targets = self.targets[perm]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        if self.transform:
            img = self.transform(img)
        return img, int(self.targets[idx])


class BinaryCIFAR10(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, positive_class="cat", negative_class="dog"):
        cifar = datasets.CIFAR10(root, train=train, download=True, transform=transform)
        pos_idx = cifar.classes.index(positive_class)
        neg_idx = cifar.classes.index(negative_class)
        indices = [i for i, t in enumerate(cifar.targets) if t in (pos_idx, neg_idx)]
        self.data = Subset(cifar, indices)
        self._map = {pos_idx: 1, neg_idx: 0}
        self.targets = [self._map[cifar.targets[i]] for i in indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        return img, self._map.get(label, label)


class CIFAR10C(VisionDataset):
    def __init__(self, root, name="gaussian_noise", severity=5, transform=None):
        super().__init__(root, transform=transform)
        self.data = np.load(os.path.join(root, "CIFAR-10-C", f"{name}.npy"))
        self.targets = np.load(os.path.join(root, "CIFAR-10-C", "labels.npy")).astype(np.int64)
        s, e = (severity - 1) * 10000, severity * 10000
        if len(self.data) >= e:
            self.data, self.targets = self.data[s:e], self.targets[s:e]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        if self.transform:
            img = self.transform(img)
        return img, int(self.targets[idx])


class CIFAR100C(VisionDataset):
    def __init__(self, root, name="gaussian_noise", severity=5, transform=None):
        super().__init__(root, transform=transform)
        self.data = np.load(os.path.join(root, "CIFAR-100-C", f"{name}.npy"))
        self.targets = np.load(os.path.join(root, "CIFAR-100-C", "labels.npy")).astype(np.int64)
        s, e = (severity - 1) * 10000, severity * 10000
        if len(self.data) >= e:
            self.data, self.targets = self.data[s:e], self.targets[s:e]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        if self.transform:
            img = self.transform(img)
        return img, int(self.targets[idx])


# ═══════════════════════════════════════════════════════════════════════════
#  Data loading — returns UNNORMALIZED [0,1] tensors
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_data_root() -> str:
    for var in ("CEGS_DATA_DIR", "DATA_DIR"):
        v = os.environ.get(var, "").strip()
        if v and os.path.isdir(v):
            return v
    fallback = os.path.join(ROOT_DIR, "..", "data")
    return fallback if os.path.isdir(fallback) else "../data"


def build_test_loader(dataset_name: str, batch_size: int = 64):
    """Return (test_loader, num_classes). Images are in [0,1] — no normalization."""
    tf = transforms.ToTensor()
    data_root = _resolve_data_root()

    if dataset_name == "cifar10":
        ds = datasets.CIFAR10(data_root, train=False, download=True, transform=tf)
    elif dataset_name == "binaryCifar10":
        ds = BinaryCIFAR10(data_root, train=False, transform=tf)
    elif dataset_name == "cifar100":
        ds = datasets.CIFAR100(data_root, train=False, download=True, transform=tf)
    elif dataset_name == "svhn":
        ds = datasets.SVHN(data_root, split="test", download=True, transform=tf)
    elif dataset_name == "imageNet":
        imagenet_orig = os.environ.get("IMAGENET_ORIGINAL", "0").lower() in ("1", "true")
        if not imagenet_orig:
            root_dir = os.environ.get("IMAGENET_DS_ROOT", "")
            if not root_dir:
                for candidate in ("SmallImageNet_32x32", "smallimagenet_32",
                                  "smallImageNet_32x32", "imageNet"):
                    p = os.path.join(data_root, candidate)
                    if os.path.isdir(p):
                        root_dir = p
                        break
            if not root_dir:
                raise FileNotFoundError(
                    "IMAGENET_DS_ROOT not set and no known SmallImageNet directory found "
                    f"under {data_root}. Set env_IMAGENET_DS_ROOT in your eval TSV.")
            res = int(os.environ.get("IMAGENET_RES", "32"))
            n_cls = int(os.environ.get("IMAGENET_CLASSES", "1000"))
            ds = SmallImagenet(root=root_dir, size=res, train=False, transform=tf, classes=range(n_cls))
        else:
            valdir = os.path.join(os.environ.get("IMAGENET_ROOT", os.path.join(data_root, "imageNet")), "val")
            ds = datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224), tf]))
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    num_classes = DATASET_STATS.get(dataset_name, {}).get("C", 10)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return loader, num_classes


def _get_norm_params(dataset_name: str) -> Tuple[Tuple, Tuple]:
    imagenet_orig = os.environ.get("IMAGENET_ORIGINAL", "0").lower() in ("1", "true")
    key = "imageNet224" if (dataset_name == "imageNet" and imagenet_orig) else dataset_name
    s = DATASET_STATS.get(key, DATASET_STATS.get(dataset_name, {"mean": (0,0,0), "std": (1,1,1)}))
    return s["mean"], s["std"]


# ═══════════════════════════════════════════════════════════════════════════
#  Model building — matches train.py resetModel exactly
# ═══════════════════════════════════════════════════════════════════════════

def build_model(dataset_name: str, device: str = "cuda") -> Tuple[nn.Module, int]:
    dropout_rate = 0.5
    if dataset_name == "imageNet":
        num_classes, depth, dropout_rate = 1000, 50, 0.3
        use_tv = os.environ.get("IMAGENET_TORCHVISION", "1").lower() in ("1", "true", "yes", "y")
        if use_tv:
            ctor = {18: tv_models.resnet18, 34: tv_models.resnet34,
                    50: tv_models.resnet50, 101: tv_models.resnet101}.get(depth, tv_models.resnet50)
            try:
                m = ctor(weights=None)
            except TypeError:
                m = ctor(pretrained=False)
            in_f = m.fc.in_features
            m.fc = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(in_f, num_classes))
        else:
            m = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, depth, dropout_rate)
    elif dataset_name in ("cifar10", "binaryCifar10"):
        num_classes = 10 if dataset_name == "cifar10" else 2
        m = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, 18, dropout_rate)
    elif dataset_name == "svhn":
        num_classes = 10
        m = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, 18, dropout_rate)
    elif dataset_name == "cifar100":
        num_classes = 100
        m = Wide_ResNet(28, 10, 0.3, num_classes)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return m.to(device), num_classes


# ═══════════════════════════════════════════════════════════════════════════
#  Attacks — pixel space [0,1], autograd.grad, proper clamping
# ═══════════════════════════════════════════════════════════════════════════

def _fgsm(norm_model: nn.Module, X: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    delta = torch.zeros_like(X, requires_grad=True)
    loss = F.cross_entropy(norm_model(X + delta), y)
    grad = torch.autograd.grad(loss, delta)[0]
    delta = (eps * grad.sign())
    delta = (X + delta).clamp(0.0, 1.0) - X
    return delta.detach()


def _pgd_linf(norm_model: nn.Module, X: torch.Tensor, y: torch.Tensor,
              eps: float, alpha: float, num_iter: int, random_start: bool = True) -> torch.Tensor:
    delta = torch.zeros_like(X)
    if random_start:
        delta.uniform_(-eps, eps)
        delta = (X + delta).clamp(0.0, 1.0) - X

    for _ in range(num_iter):
        delta.requires_grad_(True)
        loss = F.cross_entropy(norm_model(X + delta), y)
        grad = torch.autograd.grad(loss, delta)[0]
        delta = (delta.detach() + alpha * grad.sign()).clamp(-eps, eps)
        delta = (X + delta).clamp(0.0, 1.0) - X

    return delta.detach()


def _pgd_with_restarts(norm_model, X, y, eps, alpha, num_iter, restarts, random_start=True):
    best_delta = torch.zeros_like(X)
    best_loss = torch.full((X.shape[0],), -float("inf"), device=X.device)

    for r in range(restarts):
        rs = random_start or r > 0
        delta = _pgd_linf(norm_model, X, y, eps, alpha, num_iter, random_start=rs)
        with torch.no_grad():
            loss_i = F.cross_entropy(norm_model(X + delta), y, reduction="none")
        improved = loss_i > best_loss
        best_loss[improved] = loss_i[improved]
        best_delta[improved] = delta[improved]

    return best_delta


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluation functions
# ═══════════════════════════════════════════════════════════════════════════

def eval_standard(norm_model, loader, device):
    norm_model.eval()
    total_err, total_loss, n = 0.0, 0.0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = norm_model(X)
            total_err += (out.argmax(1) != y).sum().item()
            total_loss += F.cross_entropy(out, y, reduction="sum").item()
            n += len(y)
    return {"STD/Error": total_err / n, "STD/Acc": 1.0 - total_err / n, "STD/Loss": total_loss / n}


def eval_adversarial(norm_model, loader, device, eps_01, alpha_01, num_iter,
                     attack_type="pgd_linf", restarts=1):
    """Run adversarial attack in pixel space. eps_01/alpha_01 are in [0,1] scale."""
    norm_model.eval()
    total_err, total_loss, n = 0.0, 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        if attack_type == "fgsm":
            delta = _fgsm(norm_model, X, y, eps_01)
        elif attack_type == "pgd_linf_rs":
            delta = _pgd_with_restarts(norm_model, X, y, eps_01, alpha_01, num_iter,
                                       max(restarts, 1), random_start=True)
        else:
            delta = _pgd_with_restarts(norm_model, X, y, eps_01, alpha_01, num_iter,
                                       restarts, random_start=False)
        with torch.no_grad():
            out = norm_model((X + delta).clamp(0.0, 1.0))
            total_err += (out.argmax(1) != y).sum().item()
            total_loss += F.cross_entropy(out, y, reduction="sum").item()
        n += len(y)
    err = total_err / n
    return {"Error": err, "Acc": 1.0 - err, "Loss": total_loss / n}


def eval_autoattack(norm_model, loader, device, eps_01, norm="Linf", version="standard", batch_size=256):
    if not HAS_AUTOATTACK:
        print("  [SKIP] autoattack not installed (pip install autoattack)")
        return None
    norm_model.eval()
    all_x, all_y = [], []
    for x, y in loader:
        all_x.append(x)
        all_y.append(y)
    all_x = torch.cat(all_x).to(device)
    all_y = torch.cat(all_y).to(device)

    adversary = AutoAttack(norm_model, norm=norm, eps=eps_01, version=version, verbose=True)
    x_adv = adversary.run_standard_evaluation(all_x, all_y, bs=batch_size)

    with torch.no_grad():
        total_err, n = 0, 0
        for i in range(0, len(x_adv), batch_size):
            xb, yb = x_adv[i:i + batch_size], all_y[i:i + batch_size]
            total_err += (norm_model(xb).argmax(1) != yb).sum().item()
            n += len(yb)
    err = total_err / n
    return {"Error": err, "Acc": 1.0 - err}


def eval_corruption(norm_model, device, dataset_name, corruption, severity, batch_size=64):
    """Evaluate on CIFAR-10-C / CIFAR-100-C (unnormalized loader, NormalizeModel handles norm)."""
    data_root = _resolve_data_root()
    tf = transforms.ToTensor()
    try:
        if dataset_name == "cifar10":
            ds = CIFAR10C(data_root, name=corruption, severity=severity, transform=tf)
        elif dataset_name == "cifar100":
            ds = CIFAR100C(data_root, name=corruption, severity=severity, transform=tf)
        else:
            return None
    except Exception as e:
        print(f"  [C] Could not load {corruption} s{severity}: {e}")
        return None

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    norm_model.eval()
    total_err, total_loss, n = 0.0, 0.0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = norm_model(X)
            total_err += (out.argmax(1) != y).sum().item()
            total_loss += F.cross_entropy(out, y, reduction="sum").item()
            n += len(y)
    if n == 0:
        return None
    err = total_err / n
    return {"Error": err, "Acc": 1.0 - err, "Loss": total_loss / n}


# ── Calibration ──────────────────────────────────────────────────────────

def eval_calibration(norm_model, loader, device, n_bins=15):
    """ECE, NLL (= cross-entropy), Brier score on clean test data."""
    norm_model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            all_logits.append(norm_model(X).cpu())
            all_labels.append(y.cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = F.softmax(logits, dim=1)
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels).float()

    nll = F.cross_entropy(logits, labels).item()

    one_hot = F.one_hot(labels, probs.shape[1]).float()
    brier = ((probs - one_hot) ** 2).sum(dim=1).mean().item()

    ece = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        mask = (confidences > lo) & (confidences <= hi)
        if mask.any():
            ece += mask.sum().item() * abs(accuracies[mask].mean().item() - confidences[mask].mean().item())
    ece /= len(labels)

    return {"CAL/ECE": ece, "CAL/NLL": nll, "CAL/Brier": brier}


# ── Long-tail per-class analysis ─────────────────────────────────────────

def _expected_class_counts(num_classes, imb_factor=100, imb_seed=0,
                           max_per_class=5000):
    """Replicate train.py _longtail_train_indices counts without loading data."""
    counts = np.array([
        max(1, int(max_per_class * (imb_factor ** (-k / max(1, num_classes - 1)))))
        for k in range(num_classes)
    ])
    if imb_seed and imb_seed != 0:
        rng = np.random.default_rng(imb_seed)
        perm = rng.permutation(num_classes)
        counts = counts[perm]
    return counts


def eval_longtail(norm_model, loader, device, num_classes,
                  imbalance="none", imb_factor=100, imb_seed=0):
    """Per-class accuracy with training-frequency-based grouping."""
    norm_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            all_preds.append(norm_model(X).argmax(1).cpu().numpy())
            all_labels.append(y.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    per_class_correct = np.zeros(num_classes)
    per_class_total = np.zeros(num_classes)
    for c in range(num_classes):
        mask = labels == c
        per_class_total[c] = mask.sum()
        if per_class_total[c] > 0:
            per_class_correct[c] = (preds[mask] == c).sum()
    per_class_acc = np.where(per_class_total > 0, per_class_correct / per_class_total, 0.0)
    valid = per_class_total > 0

    result = {
        "LT/BalancedAcc": float(np.mean(per_class_acc[valid])) if valid.any() else 0.0,
        "LT/PerClassAcc_std": float(np.std(per_class_acc[valid])) if valid.any() else 0.0,
        "LT/PerClassAcc_min": float(np.min(per_class_acc[valid])) if valid.any() else 0.0,
        "LT/PerClassAcc_max": float(np.max(per_class_acc[valid])) if valid.any() else 0.0,
    }

    if imbalance.strip().lower() in ("exp", "longtail") and num_classes >= 3:
        base = {"cifar10": 5000, "binaryCifar10": 5000, "cifar100": 500,
                "svhn": 7325, "imageNet": 1300}
        max_pc = base.get("cifar10", 5000)
        for ds, v in base.items():
            if ds in str(loader.dataset.__class__):
                max_pc = v
                break
        counts = _expected_class_counts(num_classes, float(imb_factor), int(imb_seed), max_pc)
        order = np.argsort(-counts)
        sorted_acc = per_class_acc[order]
        third = max(1, num_classes // 3)
        result["LT/ManyAcc"] = float(np.mean(sorted_acc[:third]))
        result["LT/MediumAcc"] = float(np.mean(sorted_acc[third:2 * third]))
        result["LT/FewAcc"] = float(np.mean(sorted_acc[2 * third:]))
        result["_LT_note"] = "diagnostic_only_unless_trained_on_LT"

    return result


# ── ECGS mechanism diagnostics ───────────────────────────────────────────

def eval_ecgs_diagnostics(norm_model, loader, device):
    """Confidence-signal diagnostics from model outputs (no ECGS params needed)."""
    norm_model.eval()
    all_pmax, all_margin, all_lgap, all_ent, all_correct = [], [], [], [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = norm_model(X)
            probs = F.softmax(logits, dim=1)

            pmax, pred = probs.max(dim=1)
            all_pmax.append(pmax.cpu())
            all_correct.append((pred == y).cpu())

            top2p = torch.topk(probs, min(2, probs.shape[1]), dim=1).values
            all_margin.append((top2p[:, 0] - top2p[:, -1]).cpu())

            top2l = torch.topk(logits, min(2, logits.shape[1]), dim=1).values
            all_lgap.append((top2l[:, 0] - top2l[:, -1]).cpu())

            ent = -(probs * (probs + 1e-12).log()).sum(1)
            all_ent.append(ent.cpu())

    pmax = torch.cat(all_pmax)
    margin = torch.cat(all_margin)
    lgap = torch.cat(all_lgap)
    ent = torch.cat(all_ent)
    correct = torch.cat(all_correct)

    r = {
        "DIAG/pmax_mean": pmax.mean().item(),
        "DIAG/pmax_std": pmax.std().item(),
        "DIAG/pmax_median": pmax.median().item(),
        "DIAG/pmax_p90": pmax.quantile(0.9).item(),
        "DIAG/pmax_p95": pmax.quantile(0.95).item(),
        "DIAG/margin_mean": margin.mean().item(),
        "DIAG/margin_std": margin.std().item(),
        "DIAG/logit_gap_mean": lgap.mean().item(),
        "DIAG/logit_gap_std": lgap.std().item(),
        "DIAG/entropy_mean": ent.mean().item(),
        "DIAG/entropy_std": ent.std().item(),
    }
    if correct.any():
        r["DIAG/pmax_correct_mean"] = pmax[correct].mean().item()
    if (~correct).any():
        r["DIAG/pmax_wrong_mean"] = pmax[~correct].mean().item()
    return r


# ═══════════════════════════════════════════════════════════════════════════
#  Checkpoint discovery & SLURM resolution
# ═══════════════════════════════════════════════════════════════════════════

def extract_epoch(path):
    m = re.search(r"_epoch(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else None


def detect_dataset(path):
    name = os.path.basename(path).lower()
    for key, ds in [("binarycifar10", "binaryCifar10"), ("cifar100", "cifar100"),
                    ("cifar10", "cifar10"), ("svhn", "svhn"),
                    ("imagenet", "imageNet"), ("imnet", "imageNet")]:
        if key in name:
            return ds
    return None


def find_checkpoints(ckpt_dir, pattern="*_epoch*.pt"):
    paths = sorted(glob.glob(os.path.join(ckpt_dir, pattern)))
    result = [(extract_epoch(p), p) for p in paths if extract_epoch(p) is not None]
    result.sort(key=lambda x: x[0])
    return result


def resolve_via_slurm_log(slurm_logs_dir, array_job, task):
    log_path = os.path.join(slurm_logs_dir, f"slurm_cegs_{array_job}_{task}.out")
    if not os.path.isfile(log_path):
        return None
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("RUN="):
                    return line.strip().split("=", 1)[1]
    except Exception:
        pass
    return None


def resolve_job_models_dir(runs_dir, array_job, task, slurm_logs_dir=None):
    if slurm_logs_dir:
        run_dir = resolve_via_slurm_log(slurm_logs_dir, array_job, task)
        if run_dir:
            models_dir = os.path.join(run_dir, "src", "models")
            if os.path.isdir(models_dir):
                return models_dir

    direct = os.path.join(runs_dir, f"cegs_{array_job}_{task}", "src", "models")
    if os.path.isdir(direct):
        return direct

    for pat in [f"*_{array_job}_{task}", f"*{array_job}*_{task}"]:
        matches = glob.glob(os.path.join(runs_dir, pat, "src", "models"))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"No run dir for job={array_job} task={task} under {runs_dir}")


# ═══════════════════════════════════════════════════════════════════════════
#  TSV batch mode
# ═══════════════════════════════════════════════════════════════════════════

def _parse_eval_tsv(conf_path):
    raw = Path(conf_path).read_text(encoding="utf-8").splitlines()
    meta, header, data = {}, None, []
    for ln in raw:
        s = ln.rstrip()
        if not s.strip():
            continue
        if s.lstrip().startswith("#"):
            t = s.lstrip()[1:].strip()
            for key in ("slurm_logs_dir", "runs_dir", "wandb_project", "data_dir"):
                if t.lower().startswith(f"{key}="):
                    meta[key] = t.split("=", 1)[1].strip()
            if header is None and t.lower().startswith("dataset\t"):
                header = t
            continue
        if header is None:
            header = s
        else:
            data.append(s)
    if header is None:
        return [], [], meta
    return header.split("\t"), data, meta


def _read_eval_row(conf_path, idx):
    hdr, data, meta = _parse_eval_tsv(conf_path)
    if idx < 0 or idx >= len(data):
        raise IndexError(f"idx {idx} out of range ({len(data)} rows)")
    row = data[idx].split("\t")
    if len(row) < len(hdr):
        row += [""] * (len(hdr) - len(row))
    return dict(zip(hdr, row[:len(hdr)])), meta


def _g(hp, key, default=""):
    return (hp.get(key, "") or "").strip() or default


def args_from_tsv(conf_path, idx):
    hp, meta = _read_eval_row(conf_path, idx)
    for k, v in hp.items():
        if k.startswith("env_") and v.strip():
            os.environ[k[4:]] = v.strip()

    slurm_logs_dir = _g(hp, "slurm_logs_dir", meta.get("slurm_logs_dir", ""))
    runs_dir = _g(hp, "runs_dir", meta.get("runs_dir", DEFAULT_RUNS_DIR))
    data_dir = _g(hp, "data_dir", meta.get("data_dir", ""))
    if data_dir:
        os.environ["CEGS_DATA_DIR"] = data_dir

    wandb_project = _g(hp, "wandb_project", meta.get("wandb_project", ""))
    aj = os.environ.get("SLURM_ARRAY_JOB_ID", "0")
    ti = os.environ.get("SLURM_ARRAY_TASK_ID", str(idx))
    suffix = f"_j{aj}_t{ti}"
    wn = _g(hp, "wandb_name")
    if wn:
        wn = wn[:200 - len(suffix)] + suffix
    else:
        wn = f"eval_{_g(hp, 'dataset', 'unk')}_from{_g(hp, 'source_job')}_{_g(hp, 'source_task', '0')}{suffix}"

    os.environ.setdefault("WANDB_MODE", "online")

    eps_raw = float(_g(hp, "adv_eps", "8"))
    steps_raw = int(_g(hp, "adv_steps", "20"))
    alpha_raw = _g(hp, "adv_alpha", "")
    alpha_01 = float(alpha_raw) / 255.0 if alpha_raw else 2.5 * (eps_raw / 255.0) / max(steps_raw, 1)

    return argparse.Namespace(
        job=_g(hp, "source_job") or None,
        task=_g(hp, "source_task", "0"),
        ckpt=None, ckpt_dir=_g(hp, "ckpt_dir") or None,
        runs_dir=runs_dir, slurm_logs_dir=slurm_logs_dir,
        pattern=_g(hp, "pattern", "*_epoch*.pt"),
        dataset=_g(hp, "dataset") or None,
        device="cuda", batch_size=int(_g(hp, "batch_size", "64")),
        attacks=_g(hp, "attacks", "fgsm,pgd_linf,pgd_linf_rs"),
        adv_eps=eps_raw, adv_steps=steps_raw, adv_alpha_01=alpha_01,
        adv_restarts=int(_g(hp, "adv_restarts", "1")),
        autoattack=_g(hp, "autoattack", ""),
        autoattack_norm=_g(hp, "autoattack_norm", "Linf"),
        autoattack_eps_l2=float(_g(hp, "autoattack_eps_l2", "0")) or None,
        c_corruptions=_g(hp, "c_corruptions"),
        c_severity=int(_g(hp, "c_severity", "5")),
        imbalance=_g(hp, "imbalance", "none"),
        imb_factor=float(_g(hp, "imb_factor", "100")),
        imb_seed=int(_g(hp, "imb_seed", "0")),
        eval_calibration=_g(hp, "eval_calibration", "True").lower() in ("1", "true", "yes"),
        eval_ecgs_diag=_g(hp, "eval_ecgs_diag", "True").lower() in ("1", "true", "yes"),
        output_csv=_g(hp, "output_csv") or None,
        wandb_project=wandb_project or None,
        wandb_name=wn,
        wandb_group=_g(hp, "wandb_group", f"eval_j{aj}"),
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Main evaluation loop
# ═══════════════════════════════════════════════════════════════════════════

def run_eval(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # -- resolve checkpoints --
    if getattr(args, "job", None):
        ckpt_dir = resolve_job_models_dir(
            args.runs_dir, args.job, args.task,
            slurm_logs_dir=getattr(args, "slurm_logs_dir", None) or None)
        print(f"Resolved job={args.job} task={args.task} -> {ckpt_dir}")
        ckpts = find_checkpoints(ckpt_dir, getattr(args, "pattern", "*_epoch*.pt"))
    elif getattr(args, "ckpt", None):
        ckpts = [(extract_epoch(args.ckpt) or 0, args.ckpt)]
    elif getattr(args, "ckpt_dir", None):
        ckpts = find_checkpoints(args.ckpt_dir, args.pattern)
    else:
        print("ERROR: no checkpoint source"); return

    if not ckpts:
        print("No checkpoints found."); return

    ds = args.dataset or detect_dataset(ckpts[0][1])
    if not ds:
        print("Cannot detect dataset."); return

    # -- build model + NormalizeModel wrapper --
    raw_model, num_classes = build_model(ds, device)
    mean, std = _get_norm_params(ds)
    norm_model = NormalizeModel(raw_model, mean, std).to(device)

    test_loader, _ = build_test_loader(ds, batch_size=args.batch_size)
    print(f"Dataset: {ds}  |  Checkpoints: {len(ckpts)}  |  Device: {device}")
    print(f"Normalization: mean={mean}  std={std}")

    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()] if args.attacks else []
    eps_01 = args.adv_eps / 255.0
    alpha_01 = getattr(args, "adv_alpha_01", 2.5 * eps_01 / max(args.adv_steps, 1))
    restarts = getattr(args, "adv_restarts", 1)
    corruptions = [c.strip() for c in args.c_corruptions.split(",") if c.strip()] if args.c_corruptions else []

    # -- W&B --
    use_wandb = bool(getattr(args, "wandb_project", None))
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project,
                   entity=os.environ.get("WANDB_ENTITY", None),
                   name=getattr(args, "wandb_name", None) or f"eval_{ds}_{time.strftime('%m%d_%H%M')}",
                   group=getattr(args, "wandb_group", None) or "Eval",
                   job_type="eval",
                   config={k: v for k, v in vars(args).items() if not k.startswith("_")})
        try:
            wandb.define_metric("epoch")
            for pfx in ("STD/*", "ADV/*", "C/*", "CAL/*", "LT/*", "DIAG/*", "AA/*"):
                wandb.define_metric(pfx, step_metric="epoch")
        except Exception:
            pass

    csv_rows = []

    for epoch, ckpt_path in ckpts:
        print(f"\n{'=' * 60}\n  Epoch {epoch}  |  {os.path.basename(ckpt_path)}\n{'=' * 60}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        raw_model.load_state_dict(ckpt["state_dict"])
        norm_model.eval()
        row = {"epoch": epoch}

        # ── STD ──
        row.update(eval_standard(norm_model, test_loader, device))
        print(f"  STD  Error={row['STD/Error']:.4f}  Acc={row['STD/Acc']:.4f}")

        # ── ADV suite ──
        for atk in attacks:
            t0 = time.time()
            res = eval_adversarial(norm_model, test_loader, device,
                                   eps_01, alpha_01, args.adv_steps, atk, restarts)
            dt = time.time() - t0
            pfx = f"ADV/{atk}/eps{int(args.adv_eps)}"
            if "pgd" in atk:
                pfx += f"/steps{args.adv_steps}"
            for k, v in res.items():
                row[f"{pfx}/{k}"] = v
            print(f"  {atk:15s}  Err={res['Error']:.4f}  Acc={res['Acc']:.4f}  ({dt:.1f}s)")

        # ── AutoAttack (optional) ──
        aa_flag = getattr(args, "autoattack", "").strip().lower()
        if aa_flag in ("1", "true", "yes", "linf", "l2", "both"):
            norms = []
            if aa_flag in ("1", "true", "yes", "linf", "both"):
                norms.append("Linf")
            if aa_flag in ("l2", "both"):
                norms.append("L2")
            aa_eps_l2 = getattr(args, "autoattack_eps_l2", None)
            for aa_norm in norms:
                if aa_norm == "L2":
                    if aa_eps_l2 is None or aa_eps_l2 <= 0:
                        print(f"  [SKIP] AA-L2: set autoattack_eps_l2 (no default)")
                        continue
                    aa_eps = aa_eps_l2
                else:
                    aa_eps = eps_01
                t0 = time.time()
                aa_res = eval_autoattack(norm_model, test_loader, device, aa_eps, norm=aa_norm)
                dt = time.time() - t0
                if aa_res:
                    for k, v in aa_res.items():
                        row[f"AA/{aa_norm}/{k}"] = v
                    print(f"  AA-{aa_norm:4s}  Err={aa_res['Error']:.4f}  Acc={aa_res['Acc']:.4f}  ({dt:.1f}s)")

        # ── C-suite ──
        for corr in corruptions:
            res = eval_corruption(norm_model, device, ds, corr, args.c_severity, args.batch_size)
            if res:
                for k, v in res.items():
                    row[f"C/{corr}/s{args.c_severity}/{k}"] = v
                print(f"  C/{corr}/s{args.c_severity}  Err={res['Error']:.4f}")

        # ── Calibration ──
        if getattr(args, "eval_calibration", True):
            cal = eval_calibration(norm_model, test_loader, device)
            row.update(cal)
            print(f"  CAL  ECE={cal['CAL/ECE']:.4f}  NLL={cal['CAL/NLL']:.4f}  Brier={cal['CAL/Brier']:.4f}")

        # ── LT ──
        imbalance = getattr(args, "imbalance", "none") or "none"
        if imbalance.strip().lower() != "none" or getattr(args, "eval_calibration", True):
            lt = eval_longtail(norm_model, test_loader, device, num_classes,
                               imbalance, getattr(args, "imb_factor", 100),
                               getattr(args, "imb_seed", 0))
            row.update({k: v for k, v in lt.items() if not k.startswith("_")})
            print(f"  LT  BalAcc={lt['LT/BalancedAcc']:.4f}  std={lt['LT/PerClassAcc_std']:.4f}"
                  f"  min={lt['LT/PerClassAcc_min']:.4f}  max={lt['LT/PerClassAcc_max']:.4f}")
            if "LT/ManyAcc" in lt:
                print(f"      Many={lt['LT/ManyAcc']:.4f}  Med={lt['LT/MediumAcc']:.4f}  Few={lt['LT/FewAcc']:.4f}"
                      f"  (diagnostic only)")

        # ── ECGS diagnostics ──
        if getattr(args, "eval_ecgs_diag", True):
            diag = eval_ecgs_diagnostics(norm_model, test_loader, device)
            row.update(diag)
            print(f"  DIAG pmax={diag['DIAG/pmax_mean']:.4f}±{diag['DIAG/pmax_std']:.4f}"
                  f"  lgap={diag['DIAG/logit_gap_mean']:.2f}±{diag['DIAG/logit_gap_std']:.2f}"
                  f"  ent={diag['DIAG/entropy_mean']:.4f}")

        csv_rows.append(row)
        if use_wandb:
            import wandb
            wandb.log(row, step=epoch)

    # ── CSV output ──
    output_csv = getattr(args, "output_csv", None)
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        all_keys = []
        for r in csv_rows:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)
        with open(output_csv, "w") as f:
            f.write("\t".join(all_keys) + "\n")
            for r in csv_rows:
                f.write("\t".join(str(r.get(k, "")) for k in all_keys) + "\n")
        print(f"\nSaved to {output_csv}")

    if use_wandb:
        import wandb
        wandb.finish()

    print(f"\nDone. {len(ckpts)} checkpoints evaluated.")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Offline checkpoint evaluator v2")
    p.add_argument("--conf", type=str, help="Eval TSV (batch mode)")
    p.add_argument("--idx", type=int, help="Row index in TSV")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--ckpt", type=str)
    g.add_argument("--ckpt_dir", type=str)
    g.add_argument("--job", type=str)
    p.add_argument("--task", type=str, default="0")
    p.add_argument("--runs_dir", type=str, default=DEFAULT_RUNS_DIR)
    p.add_argument("--slurm_logs_dir", type=str, default="")
    p.add_argument("--pattern", type=str, default="*_epoch*.pt")
    p.add_argument("--dataset", type=str, choices=list(DATASET_STATS.keys()))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--attacks", type=str, default="fgsm,pgd_linf,pgd_linf_rs")
    p.add_argument("--adv_eps", type=float, default=8, help="Epsilon in pixel units [0,255]")
    p.add_argument("--adv_steps", type=int, default=20)
    p.add_argument("--adv_alpha", type=float, default=0, help="Step size in pixel units (0=auto)")
    p.add_argument("--adv_restarts", type=int, default=1)
    p.add_argument("--autoattack", type=str, default="", help="Linf|L2|both|True to run AutoAttack")
    p.add_argument("--autoattack_norm", type=str, default="Linf")
    p.add_argument("--autoattack_eps_l2", type=float, default=0, help="L2 eps for AutoAttack (required if L2)")
    p.add_argument("--c_corruptions", type=str, default="")
    p.add_argument("--c_severity", type=int, default=5)
    p.add_argument("--imbalance", type=str, default="none")
    p.add_argument("--imb_factor", type=float, default=100)
    p.add_argument("--imb_seed", type=int, default=0)
    p.add_argument("--eval_calibration", type=str, default="True")
    p.add_argument("--eval_ecgs_diag", type=str, default="True")
    p.add_argument("--output_csv", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default=None)

    args = p.parse_args()

    if args.conf is not None:
        if args.idx is None:
            p.error("--idx required with --conf")
        args = args_from_tsv(args.conf, args.idx)
    else:
        eps_01 = args.adv_eps / 255.0
        args.adv_alpha_01 = (args.adv_alpha / 255.0) if args.adv_alpha > 0 else (2.5 * eps_01 / max(args.adv_steps, 1))
        args.eval_calibration = args.eval_calibration.lower() in ("1", "true", "yes")
        args.eval_ecgs_diag = args.eval_ecgs_diag.lower() in ("1", "true", "yes")
        args.autoattack_eps_l2 = args.autoattack_eps_l2 if args.autoattack_eps_l2 > 0 else None

    if not any([getattr(args, "job", None), getattr(args, "ckpt", None),
                getattr(args, "ckpt_dir", None)]):
        p.error("Specify: --conf+--idx, --job, --ckpt, or --ckpt_dir")

    run_eval(args)


if __name__ == "__main__":
    main()
