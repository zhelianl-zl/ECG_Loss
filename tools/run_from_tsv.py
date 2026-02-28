import argparse
import os
import re
import subprocess
from pathlib import Path
import fcntl
from typing import List, Dict, Optional, Tuple


# -----------------------------
# TSV parsing with meta support
# -----------------------------
# Supported meta directives (full-line comments):
#   #wandb_project=ecg-loss
#   #wandb_project: ecg-loss
#
# Header line:
#   dataset\tseed\t...
# Header can be commented as:
#   #dataset\tseed\t...
#
# Data rows: non-empty, non-comment lines after header.
# Full-line comments anywhere are ignored.


def _parse_tsv_with_meta(conf_path: Path) -> Tuple[List[str], List[str], Dict[str, str]]:
    raw_lines = conf_path.read_text(encoding="utf-8").splitlines()

    meta: Dict[str, str] = {}
    header: Optional[str] = None
    data: List[str] = []

    for ln in raw_lines:
        s = ln.strip()
        if not s:
            continue

        # comment line
        if s.lstrip().startswith("#"):
            t = s.lstrip()[1:].strip()  # after '#'
            tl = t.lower()

            # meta: wandb_project
            if tl.startswith("wandb_project=") or tl.startswith("wandb_project:"):
                if "=" in t:
                    meta["wandb_project"] = t.split("=", 1)[1].strip()
                else:
                    meta["wandb_project"] = t.split(":", 1)[1].strip()
                continue

            # allow commented header like "#dataset\tseed\t..."
            if header is None and t.lower().startswith("dataset\t"):
                header = t
            # ignore other comments
            continue

        # non-comment line
        if header is None:
            header = s.lstrip("#").strip()
        else:
            data.append(s)

    if header is None:
        return [], [], meta
    return [header], data, meta


def read_tsv_row(conf_path: Path, idx: int) -> Tuple[Dict[str, str], Dict[str, str]]:
    header_lines, data_lines, meta = _parse_tsv_with_meta(conf_path)
    if len(header_lines) < 1 or len(data_lines) < 1:
        raise RuntimeError(f"{conf_path} must have a header + at least one data row")

    header = header_lines[0].split("\t")

    if idx < 0 or idx >= len(data_lines):
        raise RuntimeError(f"idx {idx} out of range for {conf_path} (rows={len(data_lines)})")

    row = data_lines[idx].split("\t")
    if len(row) != len(header):
        raise RuntimeError(f"Row field count != header field count (idx={idx})")
    return dict(zip(header, row)), meta


# -----------------------------
# Dataset helpers
# -----------------------------
def normalize_dataset_name(ds: str) -> str:
    d = (ds or "").strip().lower()
    mapping = {
        "binary": "binaryCifar10",
        "binarycifar10": "binaryCifar10",
        "binary-cifar10": "binaryCifar10",
        "imagenet": "imageNet",
        "image1k": "imageNet",
        "image-net": "imageNet",
        "cifar-10": "cifar10",
        "cifar10": "cifar10",
        "cifar-100": "cifar100",
        "cifar100": "cifar100",
        "svhn": "svhn",
        "mnist": "mnist",
    }
    return mapping.get(d, ds)


def ensure_dataset(data_root: Path, dataset: str, auto_download: bool) -> None:
    ds = (dataset or "").strip().lower()
    if not auto_download:
        return
    supported = {"cifar10", "cifar100", "svhn", "mnist"}
    if ds not in supported:
        return

    data_root.mkdir(parents=True, exist_ok=True)
    lock_path = data_root / ".download.lock"
    with open(str(lock_path), "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            from torchvision import datasets as tvds
            if ds == "cifar10":
                tvds.CIFAR10(root=str(data_root), train=True, download=True)
                tvds.CIFAR10(root=str(data_root), train=False, download=True)
            elif ds == "cifar100":
                tvds.CIFAR100(root=str(data_root), train=True, download=True)
                tvds.CIFAR100(root=str(data_root), train=False, download=True)
            elif ds == "svhn":
                tvds.SVHN(root=str(data_root), split="train", download=True)
                tvds.SVHN(root=str(data_root), split="test", download=True)
            elif ds == "mnist":
                tvds.MNIST(root=str(data_root), train=True, download=True)
                tvds.MNIST(root=str(data_root), train=False, download=True)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _add_arg(cmd: List[str], flag: str, hp: Dict[str, str], key: str, default: Optional[str] = None) -> None:
    v = hp.get(key, default)
    if v is None:
        return
    v = str(v).strip()
    if v == "":
        return
    cmd.extend([flag, v])


def _add_flag(cmd: List[str], hp: Dict[str, str], key: str) -> None:
    """
    Add a boolean flag (e.g. --force_run) if TSV column value is truthy.
    Truthy: 1/true/True/yes/y
    """
    v = str(hp.get(key, "")).strip().lower()
    if v in {"1", "true", "yes", "y"}:
        cmd.append(f"--{key}")


def choose_wandb_project(dataset: str) -> str:
    ds = (dataset or "").strip().lower()
    ds_key = re.sub(r"[^a-z0-9]+", "_", ds).strip("_")  # cifar100, svhn, cifar10...
    # allow either lower or upper key in env
    for k in (f"WANDB_PROJECT_{ds_key}", f"WANDB_PROJECT_{ds_key.upper()}"):
        if k in os.environ and os.environ[k].strip():
            return os.environ[k].strip()
    return os.environ.get("WANDB_PROJECT_DEFAULT", "CEGS").strip()


def _merge_tags(existing: str, new_tags: List[str]) -> str:
    ex = [t.strip() for t in (existing or "").split(",") if t.strip()]
    merged = ex[:]
    for t in new_tags:
        if t and t not in merged:
            merged.append(t)
    return ",".join(merged)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", required=True)
    ap.add_argument("--idx", type=int, required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--commit", default="unknown")
    args = ap.parse_args()

    conf_path = Path(args.conf).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    hp, meta = read_tsv_row(conf_path, args.idx)

    dataset_raw = hp.get("dataset", "").strip()
    dataset = normalize_dataset_name(dataset_raw)
    seed = hp.get("seed", "0").strip()

    stop_val = hp.get("stop_val", "").strip()
    stage1 = hp.get("stage1_epochs", "").strip()
    stage2 = hp.get("stage2_epochs", "").strip()

    # method/run_kind (optional columns; otherwise inferred)
    method_name = (hp.get("method_name") or hp.get("loss_stage2") or "exp").strip()
    run_kind = (hp.get("run_kind") or os.environ.get("WANDB_JOB_TYPE_DEFAULT", "official")).strip()

    # pick W&B project:
    # priority: per-row column -> meta directive -> old mapping by dataset
    project = (hp.get("wandb_project") or meta.get("wandb_project") or "").strip()
    if not project:
        project = choose_wandb_project(dataset)

    # notebook-style naming (used only if sbatch didn't already set WANDB_NAME/GROUP)
    total_epochs = stop_val or "T?"
    run_name_default = f"{dataset}_{method_name}_s{seed}_T{total_epochs}"
    group_default = f"{dataset}_{run_kind}_s{seed}_T{total_epochs}"

    tags = [
        run_kind, dataset, f"s{seed}", method_name,
        f"T{total_epochs}",
    ]
    if stage1:
        tags.append(f"s1{stage1}")
    if stage2:
        tags.append(f"s2{stage2}")

    # W&B env
    os.environ["WANDB_PROJECT"] = project
    if os.environ.get("WANDB_ENTITY", "").strip():
        os.environ["WANDB_ENTITY"] = os.environ["WANDB_ENTITY"].strip()

    # IMPORTANT: if sbatch already set WANDB_NAME/GROUP for Slurm mapping, keep them.
    if not os.environ.get("WANDB_NAME", "").strip():
        # allow TSV override
        wn = (hp.get("wandb_name") or "").strip()
        os.environ["WANDB_NAME"] = wn if wn else run_name_default

    if not os.environ.get("WANDB_GROUP", "").strip():
        wg = (hp.get("wandb_group") or "").strip()
        os.environ["WANDB_GROUP"] = wg if wg else group_default

    os.environ["WANDB_JOB_TYPE"] = os.environ.get("WANDB_JOB_TYPE", "").strip() or run_kind

    # merge tags with any existing tags from sbatch
    os.environ["WANDB_TAGS"] = _merge_tags(os.environ.get("WANDB_TAGS", ""), tags)
    os.environ["PYTHONUNBUFFERED"] = "1"

    # data root
    data_root = Path(os.environ.get("CEGS_DATA_DIR", str(run_dir / "data"))).expanduser().resolve()
    auto_download = os.environ.get("CEGS_AUTO_DOWNLOAD", "1") not in ("0", "false", "False")
    ensure_dataset(data_root, dataset, auto_download)

    # build train command
    cmd = ["python", "train.py"]

    # core training args
    _add_arg(cmd, "--type", hp, "type")                  # std / robust
    _add_arg(cmd, "--alg", hp, "alg")                    # pgd / fgsm ...
    _add_arg(cmd, "--ratio_adv", hp, "ratio_adv")
    _add_arg(cmd, "--ratio", hp, "ratio")
    _add_arg(cmd, "--epsilon", hp, "epsilon")
    _add_arg(cmd, "--num_iter", hp, "num_iter")
    _add_arg(cmd, "--alpha", hp, "alpha")

    _add_arg(cmd, "--dataset", {"dataset": dataset}, "dataset")
    _add_arg(cmd, "--stop", hp, "stop")
    _add_arg(cmd, "--stop_val", hp, "stop_val")
    _add_arg(cmd, "--lr", hp, "lr")
    _add_arg(cmd, "--momentum", hp, "momentum")          # NOTE: notebook uses 0.9
    _add_arg(cmd, "--batch", hp, "batch")
    _add_arg(cmd, "--workers", hp, "workers")
    _add_arg(cmd, "--half_prec", hp, "half_prec")
    _add_arg(cmd, "--variants", hp, "variants")
    _add_arg(cmd, "--pe_mode", hp, "pe_mode")

    _add_arg(cmd, "--seed", hp, "seed")

    # 2-stage
    _add_arg(cmd, "--stage1_epochs", hp, "stage1_epochs")
    _add_arg(cmd, "--stage2_epochs", hp, "stage2_epochs")
    _add_arg(cmd, "--loss_stage1", hp, "loss_stage1")
    _add_arg(cmd, "--loss_stage2", hp, "loss_stage2")
    _add_flag(cmd, hp, "full_ecg")       # column full_ecg True/1 to enable
    _add_flag(cmd, hp, "force_run")      # column force_run True/1 to enable

    # ECG/CEGS params (constants)
    _add_arg(cmd, "--ecg_lam", hp, "ecg_lam")
    _add_arg(cmd, "--ecg_tau", hp, "ecg_tau")
    _add_arg(cmd, "--ecg_k", hp, "ecg_k")
    _add_arg(cmd, "--ecg_conf_type", hp, "ecg_conf_type")
    _add_arg(cmd, "--ecg_detach_gates", hp, "ecg_detach_gates")
    _add_arg(cmd, "--ecg_schedule", hp, "ecg_schedule")

    # ECG schedules (start/end)
    _add_arg(cmd, "--ecg_lam_start", hp, "ecg_lam_start")
    _add_arg(cmd, "--ecg_lam_end", hp, "ecg_lam_end")
    _add_arg(cmd, "--ecg_tau_start", hp, "ecg_tau_start")
    _add_arg(cmd, "--ecg_tau_end", hp, "ecg_tau_end")
    _add_arg(cmd, "--ecg_k_start", hp, "ecg_k_start")
    _add_arg(cmd, "--ecg_k_end", hp, "ecg_k_end")

    # adaptive / tau_target knobs (optional)
    _add_arg(cmd, "--ecg_adapt_warmup", hp, "ecg_adapt_warmup")
    _add_arg(cmd, "--ecg_adapt_window", hp, "ecg_adapt_window")
    _add_arg(cmd, "--ecg_tau_target", hp, "ecg_tau_target")
    _add_arg(cmd, "--ecg_tau_lr", hp, "ecg_tau_lr")
    _add_arg(cmd, "--ecg_tau_ema", hp, "ecg_tau_ema")
    _add_arg(cmd, "--ecg_tau_deadzone", hp, "ecg_tau_deadzone")
    _add_arg(cmd, "--ecg_tau_min", hp, "ecg_tau_min")
    _add_arg(cmd, "--ecg_tau_max", hp, "ecg_tau_max")

    # record
    (run_dir / "cmd.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    (run_dir / "wandb_meta.txt").write_text(
        f"project={project}\nentity={os.environ.get('WANDB_ENTITY','')}\nname={os.environ.get('WANDB_NAME','')}\n"
        f"group={os.environ.get('WANDB_GROUP','')}\njob_type={os.environ.get('WANDB_JOB_TYPE','')}\n"
        f"tags={os.environ.get('WANDB_TAGS','')}\n",
        encoding="utf-8",
    )

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
