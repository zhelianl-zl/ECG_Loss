#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_from_tsv.py (patched)

What you get:
- TSV supports setting W&B project like your notebook:
    1) Top-of-file directive (applies to all rows):
         #wandb_project=ecg-loss
       or  #wandb_project: ecg-loss
    2) Per-row column "wandb_project" (overrides the directive)

  If the project exists in W&B -> the run goes into it.
  If it doesn't exist -> W&B will create it automatically.

- Run name is clear + cancellable (contains job/task id):
    <dataset>_s1-<e1>-<loss1>_s2-<e2>-<loss2>_pe-<pe>_j<arrayjob>_t<task>

- Adds notebook-base args support via TSV columns:
  momentum, workers, half_prec, variants, type, pe_mode, loss_stage1,
  and all ecg_* schedule start/end fields.

Expected CLI (kept compatible with your sbatch):
  python run_from_tsv.py --conf sweeps/universal.tsv --idx $SLURM_ARRAY_TASK_ID --run_dir ... --commit ...
"""

import argparse
import os
import re
import subprocess
from pathlib import Path
import fcntl
from typing import List, Dict, Optional, Tuple


# -----------------------------
# TSV parsing (with meta support)
# -----------------------------
def _parse_tsv_with_meta(conf_path: Path) -> Tuple[List[str], List[str], Dict[str, str]]:
    """
    Returns (header_fields, data_lines, meta)

    Meta directives (full-line comments only):
      #wandb_project=xxx
      #wandb_project: xxx

    Header:
      dataset\tseed\t...
    Header can be commented:
      #dataset\tseed\t...

    All other full-line comments are ignored.
    Inline comments (end-of-line #...) are NOT supported.
    """
    raw_lines = conf_path.read_text(encoding="utf-8").splitlines()

    meta: Dict[str, str] = {}
    header_line: Optional[str] = None
    data_lines: List[str] = []

    for ln in raw_lines:
        s = ln.strip()
        if not s:
            continue

        # full-line comment
        if s.lstrip().startswith("#"):
            t = s.lstrip()[1:].strip()
            tl = t.lower()

            # meta directive
            if tl.startswith("wandb_project=") or tl.startswith("wandb_project:"):
                if "=" in t:
                    meta["wandb_project"] = t.split("=", 1)[1].strip()
                else:
                    meta["wandb_project"] = t.split(":", 1)[1].strip()
                continue

            # allow commented header like "#dataset\tseed\t..."
            if header_line is None and t.lower().startswith("dataset\t"):
                header_line = t
            continue

        # non-comment line
        if header_line is None:
            header_line = s.lstrip("#").strip()
        else:
            data_lines.append(s)

    if header_line is None:
        return [], [], meta
    header_fields = header_line.split("\t")
    return header_fields, data_lines, meta


def read_tsv_row(conf_path: Path, idx: int) -> Tuple[Dict[str, str], Dict[str, str]]:
    header, data_lines, meta = _parse_tsv_with_meta(conf_path)
    if len(header) == 0 or len(data_lines) == 0:
        raise RuntimeError(f"{conf_path} must have a header + at least one data row")

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
    Truthy: 1/true/yes/y
    """
    v = str(hp.get(key, "")).strip().lower()
    if v in {"1", "true", "yes", "y"}:
        cmd.append(f"--{key}")


def _merge_tags(existing: str, new_tags: List[str]) -> str:
    ex = [t.strip() for t in (existing or "").split(",") if t.strip()]
    out = ex[:]
    for t in new_tags:
        if t and t not in out:
            out.append(t)
    return ",".join(out)


def choose_wandb_project(dataset: str) -> str:
    ds = (dataset or "").strip().lower()
    ds_key = re.sub(r"[^a-z0-9]+", "_", ds).strip("_")
    for k in (f"WANDB_PROJECT_{ds_key}", f"WANDB_PROJECT_{ds_key.upper()}"):
        if k in os.environ and os.environ[k].strip():
            return os.environ[k].strip()
    return os.environ.get("WANDB_PROJECT_DEFAULT", "CEGS").strip()


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

    # method/run_kind (optional columns)
    method_name = (hp.get("method_name") or hp.get("loss_stage2") or "exp").strip()
    run_kind = (hp.get("run_kind") or os.environ.get("WANDB_JOB_TYPE_DEFAULT", "official")).strip()

    # --- W&B project selection ---
    # Priority: per-row -> TSV meta directive -> env mapping by dataset
    project = (hp.get("wandb_project") or meta.get("wandb_project") or "").strip()
    if not project:
        project = choose_wandb_project(dataset)

    total_epochs = stop_val or "T?"
    default_group = f"{dataset}_{run_kind}_s{seed}_T{total_epochs}"

    tags = [run_kind, dataset, f"s{seed}", method_name, f"T{total_epochs}"]
    if stage1:
        tags.append(f"s1{stage1}")
    if stage2:
        tags.append(f"s2{stage2}")

    # W&B env: always set project (exists -> use; not exists -> W&B creates)
    os.environ["WANDB_PROJECT"] = project

    # ---- Run name (clear + cancellable) ----
    # Priority:
    # 1) TSV per-row column "wandb_name" (explicit override)
    # 2) Auto name:
    #    <dataset>_s1-<e1>-<loss1>_s2-<e2>-<loss2>_pe-<pe>_j<arrayjob>_t<task>
    tsv_wandb_name = (hp.get("wandb_name") or "").strip()
    if tsv_wandb_name:
        os.environ["WANDB_NAME"] = tsv_wandb_name
    else:
        array_job = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or "noj"
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID") or "0"
        e1 = (hp.get("stage1_epochs") or "").strip()
        e2 = (hp.get("stage2_epochs") or "").strip()
        l1 = (hp.get("loss_stage1") or "").strip() or "none"
        l2 = (hp.get("loss_stage2") or "").strip() or "none"
        pe = (hp.get("pe_mode") or "").strip() or "none"
        raw = f"{dataset}_s1-{e1}-{l1}_s2-{e2}-{l2}_pe-{pe}_j{array_job}_t{task_id}"
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
        os.environ["WANDB_NAME"] = safe[:180]

    # Group: keep array grouping by default unless TSV provides one
    if not os.environ.get("WANDB_GROUP", "").strip():
        wg = (hp.get("wandb_group") or "").strip()
        if wg:
            os.environ["WANDB_GROUP"] = wg
        else:
            array_job = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or "noj"
            os.environ["WANDB_GROUP"] = f"j{array_job}"

    # Job type: run_kind (e.g., official/ablation)
    os.environ["WANDB_JOB_TYPE"] = run_kind

    # merge tags with any existing tags from sbatch
    os.environ["WANDB_TAGS"] = _merge_tags(os.environ.get("WANDB_TAGS", ""), tags)
    os.environ["PYTHONUNBUFFERED"] = "1"

    # data root
    data_root = Path(os.environ.get("CEGS_DATA_DIR", str(run_dir / "data"))).expanduser().resolve()
    auto_download = os.environ.get("CEGS_AUTO_DOWNLOAD", "1") not in ("0", "false", "False")
    ensure_dataset(data_root, dataset, auto_download=auto_download)

    # build train.py cmd
    cmd: List[str] = ["python", "-u", "train.py"]

    # core training args (notebook base_args compatible)
    _add_arg(cmd, "--type", hp, "type")
    _add_arg(cmd, "--dataset", {"dataset": dataset}, "dataset")
    _add_arg(cmd, "--seed", hp, "seed")
    _add_arg(cmd, "--stop", hp, "stop", "epochs")
    _add_arg(cmd, "--stop_val", hp, "stop_val")
    _add_arg(cmd, "--lr", hp, "lr")
    _add_arg(cmd, "--momentum", hp, "momentum")
    _add_arg(cmd, "--batch", hp, "batch")
    _add_arg(cmd, "--workers", hp, "workers")
    _add_arg(cmd, "--half_prec", hp, "half_prec")
    _add_arg(cmd, "--variants", hp, "variants")
    _add_arg(cmd, "--pe_mode", hp, "pe_mode")

    # adversarial / robust knobs (optional)
    _add_arg(cmd, "--alg", hp, "alg")
    _add_arg(cmd, "--ratio_adv", hp, "ratio_adv")
    _add_arg(cmd, "--ratio", hp, "ratio")
    _add_arg(cmd, "--epsilon", hp, "epsilon")
    _add_arg(cmd, "--num_iter", hp, "num_iter")
    _add_arg(cmd, "--alpha", hp, "alpha")

    # 2-stage controls
    _add_arg(cmd, "--stage1_epochs", hp, "stage1_epochs")
    _add_arg(cmd, "--stage2_epochs", hp, "stage2_epochs")
    _add_arg(cmd, "--loss_stage1", hp, "loss_stage1")
    _add_arg(cmd, "--loss_stage2", hp, "loss_stage2")
    _add_flag(cmd, hp, "full_ecg")
    _add_flag(cmd, hp, "force_run")

    # ECG/CEGS params
    _add_arg(cmd, "--ecg_conf_type", hp, "ecg_conf_type")
    _add_arg(cmd, "--ecg_detach_gates", hp, "ecg_detach_gates")
    _add_arg(cmd, "--ecg_schedule", hp, "ecg_schedule")

    # constants (optional)
    _add_arg(cmd, "--ecg_lam", hp, "ecg_lam")
    _add_arg(cmd, "--ecg_tau", hp, "ecg_tau")
    _add_arg(cmd, "--ecg_k", hp, "ecg_k")

    # schedules (start/end)
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

    # record for debugging
    (run_dir / "cmd.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    (run_dir / "wandb_meta.txt").write_text(
        f"project={project}\nname={os.environ.get('WANDB_NAME','')}\n"
        f"group={os.environ.get('WANDB_GROUP','')}\njob_type={os.environ.get('WANDB_JOB_TYPE','')}\n"
        f"tags={os.environ.get('WANDB_TAGS','')}\n",
        encoding="utf-8",
    )

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()