#!/usr/bin/env python3
"""
Offline checkpoint evaluator: ADV / C-suite / LT metrics.

Loads saved .pt checkpoints and runs evaluations without retraining.

Usage:

  # TSV-driven batch mode (for RunA/RunB eval sweeps via SLURM)
  python tools/eval_checkpoints.py --conf sweeps/RunA_eval.tsv --idx 0

  # Evaluate by SLURM array-job/task ID (resolves real dir via slurm logs)
  python tools/eval_checkpoints.py --job 37988109 --task 0 --dataset binaryCifar10 \
      --slurm_logs_dir slurm_logs --attacks fgsm,pgd_linf,pgd_linf_rs \
      --wandb_project ecg_binary_pmax

  # Evaluate all checkpoints in a directory
  python tools/eval_checkpoints.py --ckpt_dir /path/to/models/ \
      --dataset cifar10 --attacks fgsm,pgd_linf,pgd_linf_rs

  # Single checkpoint
  python tools/eval_checkpoints.py --ckpt /path/to/model_epoch60.pt --dataset cifar10
"""

import os
import sys
import glob
import argparse
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models import ResNet, BasicBlock, Bottleneck, Wide_ResNet
import torchvision.models as tv_models
from torchvision import datasets, transforms
from torchvision.datasets.vision import VisionDataset

DEFAULT_RUNS_DIR = "/ocean/projects/cis260049p/zliu49/cegs_runs"

# ---------------------------------------------------------------------------
#  Dataset helpers (mirrors train.py logic, test-only)
# ---------------------------------------------------------------------------

class BinaryCIFAR10(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, positive_class="cat", negative_class="dog"):
        cifar = datasets.CIFAR10(root, train=train, download=True, transform=transform)
        class_names = cifar.classes
        pos_idx = class_names.index(positive_class)
        neg_idx = class_names.index(negative_class)
        mask = [(t == pos_idx or t == neg_idx) for t in cifar.targets]
        indices = [i for i, m in enumerate(mask) if m]
        self.data = torch.utils.data.Subset(cifar, indices)
        self._label_map = {pos_idx: 1, neg_idx: 0}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        return img, self._label_map[label]


class CIFAR10C(VisionDataset):
    def __init__(self, root, name="gaussian_noise", severity=5, transform=None):
        super().__init__(root, transform=transform)
        data_path = os.path.join(root, "CIFAR-10-C", f"{name}.npy")
        labels_path = os.path.join(root, "CIFAR-10-C", "labels.npy")
        self.data = np.load(data_path)
        self.targets = np.load(labels_path).astype(np.int64)
        n_per_sev = 10000
        start = (severity - 1) * n_per_sev
        end = severity * n_per_sev
        if len(self.data) >= end:
            self.data = self.data[start:end]
            self.targets = self.targets[start:end]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.fromarray(self.data[idx])
        if self.transform:
            img = self.transform(img)
        return img, int(self.targets[idx])


class CIFAR100C(VisionDataset):
    def __init__(self, root, name="gaussian_noise", severity=5, transform=None):
        super().__init__(root, transform=transform)
        data_path = os.path.join(root, "CIFAR-100-C", f"{name}.npy")
        labels_path = os.path.join(root, "CIFAR-100-C", "labels.npy")
        self.data = np.load(data_path)
        self.targets = np.load(labels_path).astype(np.int64)
        n_per_sev = 10000
        start = (severity - 1) * n_per_sev
        end = severity * n_per_sev
        if len(self.data) >= end:
            self.data = self.data[start:end]
            self.targets = self.targets[start:end]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.fromarray(self.data[idx])
        if self.transform:
            img = self.transform(img)
        return img, int(self.targets[idx])


def _get_small_imagenet_class():
    from train import SmallImagenet
    return SmallImagenet


def build_test_loader(dataset_name, batch_size=64, data_root=None):
    """Build test DataLoader. Returns (test_loader, num_classes, test_transform)."""
    if data_root is None:
        data_root = os.environ.get("CEGS_DATA_DIR", os.path.join(ROOT_DIR, "..", "data"))
    if not os.path.isdir(data_root):
        data_root = "../data"

    if dataset_name == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
        test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        data_test = datasets.CIFAR10(data_root, train=False, download=True, transform=test_tf)
        num_classes = 10
    elif dataset_name == "binaryCifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
        test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        data_test = BinaryCIFAR10(data_root, train=False, transform=test_tf)
        num_classes = 2
    elif dataset_name == "cifar100":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        data_test = datasets.CIFAR100(data_root, train=False, download=True, transform=test_tf)
        num_classes = 100
    elif dataset_name == "svhn":
        test_tf = transforms.Compose([transforms.ToTensor()])
        data_test = datasets.SVHN(data_root, split="test", download=True, transform=test_tf)
        num_classes = 10
    elif dataset_name == "imageNet":
        imagenet_original = os.environ.get("IMAGENET_ORIGINAL", "0").lower() in ("1", "true")
        if not imagenet_original:
            root_dir = os.environ.get("IMAGENET_DS_ROOT", os.path.join(data_root, "smallimagenet_32"))
            resolution = int(os.environ.get("IMAGENET_RES", "32"))
            classes = int(os.environ.get("IMAGENET_CLASSES", "1000"))
            normalize = transforms.Normalize(mean=[0.4810, 0.4574, 0.4078], std=[0.2146, 0.2104, 0.2138])
            test_tf = transforms.Compose([transforms.ToTensor(), normalize])
            SmallImagenet = _get_small_imagenet_class()
            data_test = SmallImagenet(root=root_dir, size=resolution, train=False, transform=test_tf, classes=range(classes))
        else:
            imagenet_root = os.environ.get("IMAGENET_ROOT", os.path.join(data_root, "imageNet"))
            valdir = os.path.join(imagenet_root, "val")
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            test_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
            data_test = datasets.ImageFolder(valdir, test_tf)
        num_classes = 1000
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return test_loader, num_classes, test_tf


def build_model(dataset_name, device="cuda"):
    """Build model architecture (same as train.py resetModel)."""
    dropout_rate = 0.5
    if dataset_name == "imageNet":
        num_classes = 1000
        depth = 50
        dropout_rate = 0.3
        use_tv = os.environ.get("IMAGENET_TORCHVISION", "1").lower() in ("1", "true", "yes")
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


# ---------------------------------------------------------------------------
#  Attack implementations
# ---------------------------------------------------------------------------

def fgsm_attack(model, X, y, epsilon):
    delta = torch.zeros_like(X, requires_grad=True)
    loss = F.cross_entropy(model(X + delta), y)
    loss.backward()
    return (epsilon * delta.grad.detach().sign()).clamp(-epsilon, epsilon)


def pgd_linf_attack(model, X, y, epsilon, alpha, num_iter, random_start=False):
    delta = torch.zeros_like(X).uniform_(-epsilon, epsilon) if random_start else torch.zeros_like(X)
    for _ in range(num_iter):
        delta.requires_grad = True
        loss = F.cross_entropy(model(X + delta), y)
        loss.backward()
        delta = (delta.detach() + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad = None
    return delta.detach()


# ---------------------------------------------------------------------------
#  Evaluation functions
# ---------------------------------------------------------------------------

def eval_standard(model, loader, device):
    model.eval()
    total_err, total_loss, n = 0.0, 0.0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            total_err += (out.argmax(1) != y).sum().item()
            total_loss += F.cross_entropy(out, y, reduction="sum").item()
            n += len(y)
    return total_err / n, total_loss / n


def eval_adversarial(model, loader, device, epsilon, alpha, num_iter, attack_type="pgd_linf"):
    model.eval()
    total_err, total_loss, n = 0.0, 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        if attack_type == "fgsm":
            delta = fgsm_attack(model, X, y, epsilon)
        elif attack_type == "pgd_linf_rs":
            delta = pgd_linf_attack(model, X, y, epsilon, alpha, num_iter, random_start=True)
        else:
            delta = pgd_linf_attack(model, X, y, epsilon, alpha, num_iter, random_start=False)
        with torch.no_grad():
            out = model(X + delta)
            total_err += (out.argmax(1) != y).sum().item()
            total_loss += F.cross_entropy(out, y, reduction="sum").item()
        n += len(y)
    return total_err / n, total_loss / n


def eval_corruption(model, device, dataset_name, corruption, severity, batch_size=64):
    """Evaluate on CIFAR-10-C or CIFAR-100-C. Returns (err, loss) or None."""
    data_root = os.environ.get("CEGS_DATA_DIR",
                               os.environ.get("DATA_DIR", os.path.join(ROOT_DIR, "..", "data")))
    if not os.path.isdir(data_root):
        data_root = "../data"
    if dataset_name == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
    elif dataset_name == "cifar100":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    else:
        return None
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    try:
        CClass = CIFAR10C if dataset_name == "cifar10" else CIFAR100C
        c_data = CClass(data_root, name=corruption, severity=severity, transform=tf)
    except Exception as e:
        print(f"  [C] Could not load {corruption} s{severity}: {e}")
        return None
    c_loader = DataLoader(c_data, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    total_err, total_loss, n = 0.0, 0.0, 0
    with torch.no_grad():
        for X, y in c_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            total_err += (out.argmax(1) != y).sum().item()
            total_loss += F.cross_entropy(out, y, reduction="sum").item()
            n += len(y)
    if n == 0:
        return None
    return total_err / n, total_loss / n


def eval_longtail(model, loader, device, num_classes):
    """Per-class accuracy -> balanced acc, many/medium/few splits."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            all_preds.append(model(X).argmax(1).cpu().numpy())
            all_labels.append(y.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    per_class_acc = np.zeros(num_classes)
    per_class_n = np.zeros(num_classes)
    for c in range(num_classes):
        mask = labels == c
        per_class_n[c] = mask.sum()
        if per_class_n[c] > 0:
            per_class_acc[c] = (preds[mask] == labels[mask]).sum() / per_class_n[c]
    valid = per_class_n > 0
    balanced = float(np.mean(per_class_acc[valid])) if valid.any() else 0.0
    third = max(1, num_classes // 3)
    many = float(np.mean(per_class_acc[:third])) if third else 0.0
    medium = float(np.mean(per_class_acc[third:2 * third])) if 2 * third > third else 0.0
    few = float(np.mean(per_class_acc[2 * third:])) if num_classes > 2 * third else 0.0
    return {"BalancedAcc": balanced, "ManyAcc": many, "MediumAcc": medium, "FewAcc": few}


# ---------------------------------------------------------------------------
#  Checkpoint discovery & SLURM resolution
# ---------------------------------------------------------------------------

def extract_epoch(path):
    m = re.search(r"_epoch(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else None


def detect_dataset(path):
    name = os.path.basename(path).lower()
    for ds_key, ds_name in [("binarycifar10", "binaryCifar10"), ("cifar100", "cifar100"),
                            ("cifar10", "cifar10"), ("svhn", "svhn"),
                            ("imagenet", "imageNet"), ("imnet", "imageNet")]:
        if ds_key in name:
            return ds_name
    return None


def find_checkpoints(ckpt_dir, pattern="*_epoch*.pt"):
    paths = sorted(glob.glob(os.path.join(ckpt_dir, pattern)))
    result = [(extract_epoch(p), p) for p in paths if extract_epoch(p) is not None]
    result.sort(key=lambda x: x[0])
    return result


def resolve_via_slurm_log(slurm_logs_dir: str, array_job: str, task: str) -> Optional[str]:
    """Read slurm .out log to find the actual RUN= directory.

    SLURM array jobs: SLURM_ARRAY_JOB_ID (in log filename) != SLURM_JOB_ID (in dir name).
    The .out log contains RUN=/path/to/cegs_{WRAPPER_ID}_{task}.
    """
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


def resolve_job_models_dir(runs_dir: str, array_job: str, task: str,
                           slurm_logs_dir: Optional[str] = None) -> str:
    """Resolve checkpoint models/ directory from SLURM array-job/task IDs.

    1. Parse SLURM .out log for the real RUN= path (most reliable)
    2. Fallback: direct match cegs_{array_job}_{task}/src/models
    3. Fallback: glob in runs_dir
    """
    if slurm_logs_dir:
        run_dir = resolve_via_slurm_log(slurm_logs_dir, array_job, task)
        if run_dir:
            models_dir = os.path.join(run_dir, "src", "models")
            if os.path.isdir(models_dir):
                return models_dir
            print(f"  [WARN] SLURM log -> {run_dir} but src/models/ missing, trying glob")

    direct = os.path.join(runs_dir, f"cegs_{array_job}_{task}", "src", "models")
    if os.path.isdir(direct):
        return direct

    for pat in [f"*_{array_job}_{task}", f"*{array_job}*_{task}"]:
        matches = glob.glob(os.path.join(runs_dir, pat, "src", "models"))
        if matches:
            if len(matches) > 1:
                print(f"  [WARN] Multiple matches, using first: {matches[0]}")
            return matches[0]

    raise FileNotFoundError(
        f"No run directory for job={array_job} task={task} under {runs_dir}.\n"
        f"  slurm_logs_dir={'(not set)' if not slurm_logs_dir else slurm_logs_dir}"
    )


# ---------------------------------------------------------------------------
#  TSV batch mode
# ---------------------------------------------------------------------------

def _parse_eval_tsv(conf_path: str) -> Tuple[List[str], List[str], Dict[str, str]]:
    """Parse eval TSV. Returns (header_fields, data_lines, meta).

    Meta directives (in # comments):
      #slurm_logs_dir=...    #runs_dir=...    #wandb_project=...    #data_dir=...
    """
    raw_lines = Path(conf_path).read_text(encoding="utf-8").splitlines()
    meta: Dict[str, str] = {}
    header_line: Optional[str] = None
    data_lines: List[str] = []

    for ln in raw_lines:
        s = ln.rstrip()
        if not s.strip():
            continue
        if s.lstrip().startswith("#"):
            t = s.lstrip()[1:].strip()
            for key in ("slurm_logs_dir", "runs_dir", "wandb_project", "data_dir"):
                if t.lower().startswith(f"{key}="):
                    meta[key] = t.split("=", 1)[1].strip()
            if header_line is None and t.lower().startswith("dataset\t"):
                header_line = t
            continue
        if header_line is None:
            header_line = s
        else:
            data_lines.append(s)
    if header_line is None:
        return [], [], meta
    return header_line.split("\t"), data_lines, meta


def _read_eval_row(conf_path: str, idx: int) -> Tuple[Dict[str, str], Dict[str, str]]:
    header, data_lines, meta = _parse_eval_tsv(conf_path)
    if idx < 0 or idx >= len(data_lines):
        raise IndexError(f"Task index {idx} out of range (have {len(data_lines)} rows)")
    row = data_lines[idx].split("\t")
    if len(row) < len(header):
        row += [""] * (len(header) - len(row))
    elif len(row) > len(header):
        row = row[:len(header)]
    return dict(zip(header, row)), meta


def args_from_tsv(conf_path: str, idx: int) -> argparse.Namespace:
    """Build argparse.Namespace from an eval TSV row."""
    hp, meta = _read_eval_row(conf_path, idx)

    for k, v in hp.items():
        if k.startswith("env_") and v.strip():
            os.environ[k[4:]] = v.strip()

    slurm_logs_dir = hp.get("slurm_logs_dir", "").strip() or meta.get("slurm_logs_dir", "")
    runs_dir = hp.get("runs_dir", "").strip() or meta.get("runs_dir", "") or DEFAULT_RUNS_DIR
    data_dir = hp.get("data_dir", "").strip() or meta.get("data_dir", "")
    if data_dir:
        os.environ["CEGS_DATA_DIR"] = data_dir

    wandb_project = hp.get("wandb_project", "").strip() or meta.get("wandb_project", "")

    array_job = os.environ.get("SLURM_ARRAY_JOB_ID", "0")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", str(idx))
    job_task_suffix = f"_j{array_job}_t{task_id}"

    wandb_name = hp.get("wandb_name", "").strip()
    if wandb_name:
        wandb_name = wandb_name[:200 - len(job_task_suffix)] + job_task_suffix
    else:
        src = hp.get("source_job", "").strip()
        stk = hp.get("source_task", "0").strip()
        wandb_name = f"eval_{hp.get('dataset', 'unk')}_from{src}_{stk}{job_task_suffix}"

    wandb_group = hp.get("wandb_group", "").strip() or f"eval_j{array_job}"

    os.environ.setdefault("WANDB_MODE", "online")

    return argparse.Namespace(
        job=hp.get("source_job", "").strip() or None,
        task=hp.get("source_task", "0").strip(),
        ckpt=None,
        ckpt_dir=hp.get("ckpt_dir", "").strip() or None,
        runs_dir=runs_dir,
        slurm_logs_dir=slurm_logs_dir,
        pattern=hp.get("pattern", "").strip() or "*_epoch*.pt",
        dataset=hp.get("dataset", "").strip() or None,
        device="cuda",
        batch_size=int(hp.get("batch_size", "64").strip() or "64"),
        attacks=hp.get("attacks", "fgsm,pgd_linf,pgd_linf_rs").strip(),
        adv_eps=float(hp.get("adv_eps", "8").strip() or "8"),
        adv_steps=int(hp.get("adv_steps", "20").strip() or "20"),
        adv_pixel=True,
        c_corruptions=hp.get("c_corruptions", "").strip(),
        c_severity=int(hp.get("c_severity", "5").strip() or "5"),
        imbalance=hp.get("imbalance", "none").strip() or "none",
        output_csv=hp.get("output_csv", "").strip() or None,
        wandb_project=wandb_project or None,
        wandb_name=wandb_name,
        wandb_group=wandb_group,
    )


# ---------------------------------------------------------------------------
#  Main evaluation loop
# ---------------------------------------------------------------------------

def run_eval(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    slurm_logs_dir = getattr(args, "slurm_logs_dir", None) or None

    # --- Resolve checkpoints ---
    if getattr(args, "job", None):
        ckpt_dir = resolve_job_models_dir(
            args.runs_dir, args.job, args.task, slurm_logs_dir=slurm_logs_dir
        )
        print(f"Resolved job={args.job} task={args.task} -> {ckpt_dir}")
        ckpts = find_checkpoints(ckpt_dir, getattr(args, "pattern", "*_epoch*.pt"))
    elif getattr(args, "ckpt", None):
        ckpts = [(extract_epoch(args.ckpt) or 0, args.ckpt)]
    elif getattr(args, "ckpt_dir", None):
        ckpts = find_checkpoints(args.ckpt_dir, args.pattern)
    else:
        print("ERROR: No checkpoint source specified.")
        return

    if not ckpts:
        print("No checkpoints found.")
        return

    ds = args.dataset or detect_dataset(ckpts[0][1])
    if not ds:
        print("Cannot detect dataset. Use --dataset.")
        return
    print(f"Dataset: {ds}  |  Checkpoints: {len(ckpts)}  |  Device: {device}")

    model, num_classes = build_model(ds, device)
    test_loader, _, _ = build_test_loader(ds, batch_size=args.batch_size)

    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()] if args.attacks else []
    eps_01 = (args.adv_eps / 255.0) if args.adv_pixel else args.adv_eps
    alpha = eps_01 * 2.5 / max(args.adv_steps, 1)

    corruptions = [c.strip() for c in args.c_corruptions.split(",") if c.strip()] if args.c_corruptions else []

    # --- W&B ---
    use_wandb = bool(getattr(args, "wandb_project", None))
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=os.environ.get("WANDB_ENTITY", None),
            name=getattr(args, "wandb_name", None) or f"eval_{ds}_{time.strftime('%m%d_%H%M')}",
            group=getattr(args, "wandb_group", None) or "Eval",
            job_type="eval",
            config={k: v for k, v in vars(args).items() if not k.startswith("_")},
        )
        try:
            wandb.define_metric("epoch")
            for pfx in ("STD/*", "PGD/*", "ADV/*", "C/*", "LT/*"):
                wandb.define_metric(pfx, step_metric="epoch")
        except Exception:
            pass

    csv_rows = []

    for epoch, ckpt_path in ckpts:
        print(f"\n{'=' * 60}")
        print(f"  Epoch {epoch}  |  {os.path.basename(ckpt_path)}")
        print(f"{'=' * 60}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        row = {"epoch": epoch, "ckpt": os.path.basename(ckpt_path)}

        std_err, std_loss = eval_standard(model, test_loader, device)
        row["STD/Error"] = std_err
        row["STD/Loss"] = std_loss
        print(f"  STD  Error={std_err:.4f}  Loss={std_loss:.4f}")

        for atk in attacks:
            t0 = time.time()
            adv_err, adv_loss = eval_adversarial(
                model, test_loader, device, eps_01, alpha, args.adv_steps, attack_type=atk
            )
            dt = time.time() - t0
            prefix = f"ADV/{atk}/eps{int(args.adv_eps)}"
            if "pgd" in atk:
                prefix += f"/steps{args.adv_steps}"
            row[f"{prefix}/Error"] = adv_err
            row[f"{prefix}/Acc"] = 1.0 - adv_err
            row[f"{prefix}/Loss"] = adv_loss
            print(f"  {atk:15s}  Error={adv_err:.4f}  Acc={1.0 - adv_err:.4f}  ({dt:.1f}s)")

        for corr in corruptions:
            result = eval_corruption(model, device, ds, corr, args.c_severity, args.batch_size)
            if result is not None:
                c_err, c_loss = result
                row[f"C/{corr}/s{args.c_severity}/Error"] = c_err
                row[f"C/{corr}/s{args.c_severity}/Acc"] = 1.0 - c_err
                row[f"C/{corr}/s{args.c_severity}/Loss"] = c_loss
                print(f"  C/{corr}/s{args.c_severity}  Error={c_err:.4f}")

        imbalance = getattr(args, "imbalance", "none") or "none"
        if imbalance.strip().lower() != "none":
            lt = eval_longtail(model, test_loader, device, num_classes)
            for k, v in lt.items():
                row[f"LT/{k}"] = v
            print(f"  LT  BalAcc={lt['BalancedAcc']:.4f}  Many={lt['ManyAcc']:.4f}"
                  f"  Med={lt['MediumAcc']:.4f}  Few={lt['FewAcc']:.4f}")

        csv_rows.append(row)

        if use_wandb:
            import wandb
            wandb.log({k: v for k, v in row.items() if k != "ckpt"}, step=epoch)

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
        print(f"\nResults saved to {output_csv}")

    if use_wandb:
        import wandb
        wandb.finish()
        print("W&B run finished.")

    print(f"\nDone. Evaluated {len(ckpts)} checkpoints.")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Offline checkpoint evaluator: ADV / C / LT")

    parser.add_argument("--conf", type=str, default=None, help="Eval TSV (batch mode).")
    parser.add_argument("--idx", type=int, default=None, help="Row index in TSV (0-based).")

    g = parser.add_mutually_exclusive_group()
    g.add_argument("--ckpt", type=str, help="Single checkpoint path.")
    g.add_argument("--ckpt_dir", type=str, help="Directory containing checkpoints.")
    g.add_argument("--job", type=str, help="SLURM array-job ID.")

    parser.add_argument("--task", type=str, default="0")
    parser.add_argument("--runs_dir", type=str, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--slurm_logs_dir", type=str, default="")

    parser.add_argument("--pattern", type=str, default="*_epoch*.pt")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["cifar10", "binaryCifar10", "cifar100", "svhn", "imageNet"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--attacks", type=str, default="fgsm,pgd_linf,pgd_linf_rs")
    parser.add_argument("--adv_eps", type=float, default=8)
    parser.add_argument("--adv_steps", type=int, default=20)
    parser.add_argument("--adv_pixel", action="store_true", default=True)

    parser.add_argument("--c_corruptions", type=str, default="")
    parser.add_argument("--c_severity", type=int, default=5)

    parser.add_argument("--imbalance", type=str, default="none")

    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)

    args = parser.parse_args()

    if args.conf is not None:
        if args.idx is None:
            parser.error("--idx required with --conf")
        args = args_from_tsv(args.conf, args.idx)
    elif not any([args.job, args.ckpt, args.ckpt_dir]):
        parser.error("Specify: --conf+--idx, --job, --ckpt, or --ckpt_dir")

    run_eval(args)


if __name__ == "__main__":
    main()
