#!/usr/bin/env python3
"""
Offline checkpoint evaluator: ADV / C-suite / LT metrics.

Decoupled from training -- loads saved checkpoints and runs evaluations
without any risk of polluting training (BN stats, optimizer, etc.).

Usage examples:

  # Evaluate by SLURM job/task ID (resolves cegs_runs path automatically)
  python tools/eval_checkpoints.py --job 37976254 --task 0 --dataset cifar10 \
      --attacks fgsm,pgd_linf,pgd_linf_rs --wandb_project ecg-cifar10

  # Evaluate all checkpoints in a directory
  python tools/eval_checkpoints.py --ckpt_dir models/ --pattern "*cifar10*_epoch*.pt" \
      --attacks fgsm,pgd_linf,pgd_linf_rs --wandb_project cegs-eval

  # Evaluate a single checkpoint
  python tools/eval_checkpoints.py --ckpt models/some_model_epoch60.pt --dataset cifar10

  # Output to CSV instead of W&B
  python tools/eval_checkpoints.py --ckpt_dir models/ --dataset cifar10 \
      --output_csv results/cifar10_eval.csv
"""

import os
import sys
import glob
import argparse
import time
import re

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


def build_test_loader(dataset_name, batch_size=32, **_kw):
    """Build test DataLoader. Returns (test_loader, num_classes, test_transform)."""
    if dataset_name == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
        test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        data_test = datasets.CIFAR10("../data", train=False, download=True, transform=test_tf)
        num_classes = 10
    elif dataset_name == "binaryCifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
        test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        data_test = BinaryCIFAR10("../data", train=False, transform=test_tf)
        num_classes = 2
    elif dataset_name == "cifar100":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        data_test = datasets.CIFAR100("../data", train=False, download=True, transform=test_tf)
        num_classes = 100
    elif dataset_name == "svhn":
        test_tf = transforms.Compose([transforms.ToTensor()])
        data_test = datasets.SVHN("../data", split="test", download=True, transform=test_tf)
        num_classes = 10
    elif dataset_name == "imageNet":
        imagenet_original = os.environ.get("IMAGENET_ORIGINAL", "0").lower() in ("1", "true")
        if not imagenet_original:
            root_dir = os.environ.get("IMAGENET_DS_ROOT", "../data/imageNet/")
            resolution = int(os.environ.get("IMAGENET_RES", "32"))
            classes = int(os.environ.get("IMAGENET_CLASSES", "1000"))
            normalize = transforms.Normalize(mean=[0.4810, 0.4574, 0.4078], std=[0.2146, 0.2104, 0.2138])
            test_tf = transforms.Compose([transforms.ToTensor(), normalize])
            SmallImagenet = _get_small_imagenet_class()
            data_test = SmallImagenet(root=root_dir, size=resolution, train=False, transform=test_tf, classes=range(classes))
        else:
            imagenet_root = os.environ.get("IMAGENET_ROOT", "../data/imageNet")
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
    """Build model architecture (same as train.py resetModel) and return (model, num_classes)."""
    dropout_rate = 0.5
    if dataset_name == "imageNet":
        num_classes = 1000
        depth = 50
        dropout_rate = 0.3
        use_tv = os.environ.get("IMAGENET_TORCHVISION", "1").lower() in ("1", "true", "yes")
        if use_tv:
            ctor = {18: tv_models.resnet18, 34: tv_models.resnet34, 50: tv_models.resnet50, 101: tv_models.resnet101}.get(depth, tv_models.resnet50)
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
#  Attack implementations (standalone, no class dependency)
# ---------------------------------------------------------------------------

def fgsm_attack(model, X, y, epsilon, half_prec=False):
    delta = torch.zeros_like(X, requires_grad=True)
    loss = F.cross_entropy(model(X + delta), y)
    loss.backward()
    return (epsilon * delta.grad.detach().sign()).clamp(-epsilon, epsilon)


def pgd_linf_attack(model, X, y, epsilon, alpha, num_iter, random_start=False, half_prec=False):
    if random_start:
        delta = torch.zeros_like(X).uniform_(-epsilon, epsilon)
    else:
        delta = torch.zeros_like(X)
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


def eval_corruption(model, device, dataset_name, corruption, severity, batch_size=32):
    """Evaluate on CIFAR-10-C or CIFAR-100-C. Returns (err, loss) or None."""
    data_root = os.environ.get("DATA_DIR", os.path.join(ROOT_DIR, "..", "data"))
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
    medium = float(np.mean(per_class_acc[third:2*third])) if 2*third > third else 0.0
    few = float(np.mean(per_class_acc[2*third:])) if num_classes > 2*third else 0.0
    return {"BalancedAcc": balanced, "ManyAcc": many, "MediumAcc": medium, "FewAcc": few}


# ---------------------------------------------------------------------------
#  Checkpoint discovery
# ---------------------------------------------------------------------------

def extract_epoch(path):
    """Extract epoch number from filename like *_epoch30.pt."""
    m = re.search(r"_epoch(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else None


def detect_dataset(path):
    """Guess dataset from checkpoint filename."""
    name = os.path.basename(path).lower()
    if "binarycifar10" in name:
        return "binaryCifar10"
    if "cifar100" in name:
        return "cifar100"
    if "cifar10" in name:
        return "cifar10"
    if "svhn" in name:
        return "svhn"
    if "imagenet" in name or "imnet" in name:
        return "imageNet"
    return None


def find_checkpoints(ckpt_dir, pattern="*_epoch*.pt"):
    """Find and sort checkpoints by epoch."""
    paths = sorted(glob.glob(os.path.join(ckpt_dir, pattern)))
    result = []
    for p in paths:
        ep = extract_epoch(p)
        if ep is not None:
            result.append((ep, p))
    result.sort(key=lambda x: x[0])
    return result


def resolve_job_models_dir(runs_dir, job, task):
    """Resolve checkpoint directory from SLURM job/task IDs.

    Path pattern: {runs_dir}/*{job}*_{task}/src/models/
    """
    pattern = os.path.join(runs_dir, f"*{job}*_{task}", "src", "models")
    matches = glob.glob(pattern)
    if not matches:
        pattern_alt = os.path.join(runs_dir, f"*{job}_{task}", "src", "models")
        matches = glob.glob(pattern_alt)
    if not matches:
        raise FileNotFoundError(
            f"No run directory found for job={job} task={task} under {runs_dir}\n"
            f"  Tried: {pattern}"
        )
    if len(matches) > 1:
        print(f"  [WARN] Multiple matches for job={job} task={task}, using first: {matches[0]}")
    return matches[0]


# ---------------------------------------------------------------------------
#  Main evaluation loop
# ---------------------------------------------------------------------------

def run_eval(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- Resolve checkpoint directory from job/task or ckpt/ckpt_dir ---
    if getattr(args, "job", None) is not None:
        runs_dir = getattr(args, "runs_dir", None) or DEFAULT_RUNS_DIR
        ckpt_dir = resolve_job_models_dir(runs_dir, args.job, args.task)
        print(f"Resolved job={args.job} task={args.task} -> {ckpt_dir}")
        ckpts = find_checkpoints(ckpt_dir, getattr(args, "pattern", "*_epoch*.pt"))
    elif getattr(args, "ckpt", None):
        ckpts = [(extract_epoch(args.ckpt) or 0, args.ckpt)]
    else:
        ckpts = find_checkpoints(args.ckpt_dir, args.pattern)

    if not ckpts:
        print("No checkpoints found.")
        return

    ds = args.dataset or detect_dataset(ckpts[0][1])
    if not ds:
        print("Cannot detect dataset from filenames. Use --dataset.")
        return
    print(f"Dataset: {ds}  |  Checkpoints: {len(ckpts)}  |  Device: {device}")

    model, num_classes = build_model(ds, device)
    test_loader, _, test_tf = build_test_loader(ds, batch_size=args.batch_size)

    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()] if args.attacks else []
    eps_01 = (args.adv_eps / 255.0) if args.adv_pixel else args.adv_eps
    alpha = (eps_01 * 2.5 / max(args.adv_steps, 1))

    corruptions = [c.strip() for c in args.c_corruptions.split(",") if c.strip()] if args.c_corruptions else []

    # --- W&B init ---
    use_wandb = (getattr(args, "wandb_project", None) is not None)
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=os.environ.get("WANDB_ENTITY", None),
            name=getattr(args, "wandb_name", None) or f"eval_{ds}_{time.strftime('%m%d_%H%M')}",
            group=getattr(args, "wandb_group", None) or "Eval",
            config={k: v for k, v in vars(args).items() if not k.startswith("_")},
        )
        try:
            wandb.define_metric("epoch")
            for prefix in ("train/*", "STD/*", "PGD/*", "ECG/*", "ADV/*", "C/*", "LT/*", "TIME/*", "config/*"):
                wandb.define_metric(prefix, step_metric="epoch")
        except Exception:
            pass

    csv_rows = []

    for epoch, ckpt_path in ckpts:
        print(f"\n{'='*60}")
        print(f"  Epoch {epoch}  |  {os.path.basename(ckpt_path)}")
        print(f"{'='*60}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        row = {"epoch": epoch, "ckpt": os.path.basename(ckpt_path)}

        # -- STD --
        std_err, std_loss = eval_standard(model, test_loader, device)
        row["STD/Error"] = std_err
        row["STD/Loss"] = std_loss
        print(f"  STD  Error={std_err:.4f}  Loss={std_loss:.4f}")

        # -- ADV --
        for atk in attacks:
            t0 = time.time()
            adv_err, adv_loss = eval_adversarial(model, test_loader, device, eps_01, alpha, args.adv_steps, attack_type=atk)
            dt = time.time() - t0
            prefix = f"ADV/{atk}/eps{int(args.adv_eps)}"
            if "pgd" in atk:
                prefix += f"/steps{args.adv_steps}"
            row[f"{prefix}/Error"] = adv_err
            row[f"{prefix}/Acc"] = 1.0 - adv_err
            row[f"{prefix}/Loss"] = adv_loss
            print(f"  {atk:15s}  Error={adv_err:.4f}  Acc={1.0 - adv_err:.4f}  Loss={adv_loss:.4f}  ({dt:.1f}s)")

            if atk == "pgd_linf":
                row["PGD/Error"] = adv_err
                row["PGD/Loss"] = adv_loss

        # -- C-suite --
        for corr in corruptions:
            result = eval_corruption(model, device, ds, corr, args.c_severity, args.batch_size)
            if result is not None:
                c_err, c_loss = result
                row[f"C/{corr}/s{args.c_severity}/Error"] = c_err
                row[f"C/{corr}/s{args.c_severity}/Acc"] = 1.0 - c_err
                row[f"C/{corr}/s{args.c_severity}/Loss"] = c_loss
                print(f"  C/{corr}/s{args.c_severity}  Error={c_err:.4f}  Acc={1.0 - c_err:.4f}")

        # -- LT --
        imbalance = getattr(args, "imbalance", "none") or "none"
        if imbalance != "none":
            lt = eval_longtail(model, test_loader, device, num_classes)
            for k, v in lt.items():
                row[f"LT/{k}"] = v
            print(f"  LT  BalAcc={lt['BalancedAcc']:.4f}  Many={lt['ManyAcc']:.4f}  Med={lt['MediumAcc']:.4f}  Few={lt['FewAcc']:.4f}")

        csv_rows.append(row)

        if use_wandb:
            import wandb
            wandb.log({k: v for k, v in row.items() if k != "ckpt"}, step=epoch)

    # --- Save CSV ---
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


def main():
    parser = argparse.ArgumentParser(description="Offline checkpoint evaluator: ADV / C / LT")

    # Checkpoint source (mutually exclusive)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt", type=str, help="Single checkpoint path.")
    g.add_argument("--ckpt_dir", type=str, help="Directory containing checkpoints.")
    g.add_argument("--job", type=str, help="SLURM job ID (resolves cegs_runs path with --task).")

    parser.add_argument("--task", type=str, default="0", help="SLURM array task ID (used with --job).")
    parser.add_argument("--runs_dir", type=str, default=DEFAULT_RUNS_DIR,
                        help="Base directory for SLURM run outputs.")

    parser.add_argument("--pattern", type=str, default="*_epoch*.pt", help="Glob pattern for checkpoints.")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["cifar10", "binaryCifar10", "cifar100", "svhn", "imageNet"],
                        help="Dataset name (auto-detected from filename if not set).")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)

    # ADV
    parser.add_argument("--attacks", type=str, default="fgsm,pgd_linf,pgd_linf_rs", help="Comma-separated attacks.")
    parser.add_argument("--adv_eps", type=float, default=8, help="Epsilon (pixel scale by default).")
    parser.add_argument("--adv_steps", type=int, default=20, help="PGD steps.")
    parser.add_argument("--adv_pixel", action="store_true", default=True, help="Epsilon in pixel scale (divide by 255).")

    # C-suite
    parser.add_argument("--c_corruptions", type=str, default="", help="Comma-separated corruptions (empty = skip).")
    parser.add_argument("--c_severity", type=int, default=5)

    # LT
    parser.add_argument("--imbalance", type=str, default="none")
    parser.add_argument("--imb_factor", type=float, default=None)
    parser.add_argument("--imb_seed", type=int, default=None)

    # Output
    parser.add_argument("--output_csv", type=str, default=None, help="Save results to TSV file.")
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project (enables W&B logging).")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)

    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
