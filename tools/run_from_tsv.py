#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_from_tsv.py (v2)

Purpose
- Run one experiment per TSV row (PSC Slurm array friendly).
- Supports a top-of-file metadata directive:
    #wandb_project=YOUR_PROJECT
  which acts as the default W&B project for all rows.
  A per-row column "wandb_project" overrides the default.

Other features
- Skips blank lines and full-line comments beginning with '#'
- Dataset aliases:
    binary -> binaryCifar10
    image1k -> imageNet
    cifar-10 -> cifar10
    cifar-100 -> cifar100
- Passes schedule start/end params so ecg_schedule=linear/cosine truly varies:
    ecg_lam_start/end, ecg_tau_start/end, ecg_k_start/end

Usage
- Local:   python tools/run_from_tsv.py --conf sweeps/universal.tsv --idx 0
- Slurm:   SLURM_ARRAY_TASK_ID is used when --idx is not provided.
"""

import argparse
import csv
import os
import shlex
import subprocess
import sys
from typing import Any, Dict, List, Tuple


TRAIN_ENTRY = "train.py"  # change if your entry script is different


def _to_str(x: Any) -> str:
    return "" if x is None else str(x).strip()


def _normalize_dataset_name(ds: str) -> str:
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
    }
    return mapping.get(d, ds)


def _add_arg(cmd: List[str], flag: str, row: Dict[str, str], key: str) -> None:
    if key not in row:
        return
    val = _to_str(row.get(key))
    if val == "":
        return
    cmd.extend([flag, val])


def _read_tsv_with_meta(path: str) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    """
    Parses TSV rows + top-of-file meta directives.

    Meta directives (full-line comments only):
        #wandb_project=foo
        #wandb_project: foo
    """
    rows: List[Dict[str, str]] = []
    meta: Dict[str, str] = {}

    data_lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.rstrip("\n")
            if s.strip() == "":
                continue

            # Full-line comments
            if s.lstrip().startswith("#"):
                t = s.lstrip()[1:].strip()  # remove '#'
                # parse wandb_project directive
                lower = t.lower()
                if lower.startswith("wandb_project=") or lower.startswith("wandb_project:"):
                    if "=" in t:
                        meta["wandb_project"] = t.split("=", 1)[1].strip()
                    else:
                        meta["wandb_project"] = t.split(":", 1)[1].strip()
                # ignore all other comment lines
                continue

            data_lines.append(s)

    if not data_lines:
        raise ValueError(f"TSV has no data/header after removing comments: {path}")

    reader = csv.DictReader(data_lines, delimiter="\t")
    if reader.fieldnames is None:
        raise ValueError(f"Cannot parse TSV header: {path}")

    for r in reader:
        rr = {k.strip(): (_to_str(v)) for k, v in r.items() if k is not None}
        rows.append(rr)

    return rows, meta


def _build_cmd(row: Dict[str, str], meta: Dict[str, str], python_bin: str) -> List[str]:
    cmd: List[str] = [python_bin, TRAIN_ENTRY]

    # dataset
    ds_raw = _to_str(row.get("dataset"))
    ds = _normalize_dataset_name(ds_raw)
    if ds:
        cmd.extend(["--dataset", ds])

    # common
    _add_arg(cmd, "--seed", row, "seed")
    _add_arg(cmd, "--lr", row, "lr")
    _add_arg(cmd, "--batch", row, "batch")
    _add_arg(cmd, "--epochs", row, "epochs")
    _add_arg(cmd, "--stage1_epochs", row, "stage1_epochs")
    _add_arg(cmd, "--stage2_epochs", row, "stage2_epochs")
    _add_arg(cmd, "--stop", row, "stop")
    _add_arg(cmd, "--stop_val", row, "stop_val")
    _add_arg(cmd, "--weight_decay", row, "weight_decay")
    _add_arg(cmd, "--model", row, "model")
    _add_arg(cmd, "--optimizer", row, "optimizer")

    # ECG/CEGS knobs
    _add_arg(cmd, "--loss_stage2", row, "loss_stage2")
    _add_arg(cmd, "--ecg_conf_type", row, "ecg_conf_type")
    _add_arg(cmd, "--ecg_schedule", row, "ecg_schedule")

    # legacy constants (still OK)
    _add_arg(cmd, "--ecg_lam", row, "ecg_lam")
    _add_arg(cmd, "--ecg_tau", row, "ecg_tau")
    _add_arg(cmd, "--ecg_k", row, "ecg_k")

    # schedule start/end for true linear/cosine
    _add_arg(cmd, "--ecg_lam_start", row, "ecg_lam_start")
    _add_arg(cmd, "--ecg_lam_end", row, "ecg_lam_end")
    _add_arg(cmd, "--ecg_tau_start", row, "ecg_tau_start")
    _add_arg(cmd, "--ecg_tau_end", row, "ecg_tau_end")
    _add_arg(cmd, "--ecg_k_start", row, "ecg_k_start")
    _add_arg(cmd, "--ecg_k_end", row, "ecg_k_end")

    # tau_target controller knobs (optional)
    _add_arg(cmd, "--ecg_tau_target", row, "ecg_tau_target")
    _add_arg(cmd, "--ecg_tau_lr", row, "ecg_tau_lr")
    _add_arg(cmd, "--ecg_tau_ema", row, "ecg_tau_ema")
    _add_arg(cmd, "--ecg_tau_deadzone", row, "ecg_tau_deadzone")
    _add_arg(cmd, "--ecg_tau_min", row, "ecg_tau_min")
    _add_arg(cmd, "--ecg_tau_max", row, "ecg_tau_max")

    # W&B (row overrides meta)
    if _to_str(row.get("wandb_project")):
        _add_arg(cmd, "--wandb_project", row, "wandb_project")
    elif _to_str(meta.get("wandb_project")):
        cmd.extend(["--wandb_project", meta["wandb_project"]])

    _add_arg(cmd, "--wandb_name", row, "wandb_name")
    _add_arg(cmd, "--wandb_group", row, "wandb_group")

    # misc logging/run dirs (if your train.py supports these)
    _add_arg(cmd, "--run_dir", row, "run_dir")
    _add_arg(cmd, "--tag", row, "tag")

    return cmd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", required=True, help="TSV config path")
    ap.add_argument("--idx", type=int, default=None, help="Row index (0-based). If omitted, uses SLURM_ARRAY_TASK_ID.")
    ap.add_argument("--python", default=sys.executable, help="Python executable to use.")
    ap.add_argument("--dry_run", action="store_true", help="Print the command and exit.")
    args = ap.parse_args()

    rows, meta = _read_tsv_with_meta(args.conf)

    idx = args.idx
    if idx is None:
        sid = os.environ.get("SLURM_ARRAY_TASK_ID")
        if sid and sid.strip():
            idx = int(sid)

    if idx is None:
        print("ERROR: provide --idx or set SLURM_ARRAY_TASK_ID", file=sys.stderr)
        return 2

    if idx < 0 or idx >= len(rows):
        print(f"ERROR: idx {idx} out of range (rows={len(rows)})", file=sys.stderr)
        return 2

    row = rows[idx]
    cmd = _build_cmd(row, meta, args.python)

    print(f"[run_from_tsv] conf={args.conf} idx={idx} dataset={row.get('dataset','')}")
    if meta.get("wandb_project"):
        print(f"[run_from_tsv] default wandb_project={meta['wandb_project']}")
    print("[run_from_tsv] cmd:")
    print("  " + " ".join(shlex.quote(x) for x in cmd))

    if args.dry_run:
        return 0

    proc = subprocess.run(cmd)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
