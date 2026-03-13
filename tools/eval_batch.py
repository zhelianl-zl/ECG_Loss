#!/usr/bin/env python3
"""
Batch evaluation runner: reads a TSV config and runs eval_checkpoints.run_eval()
for each row.

Usage:
  # Run all rows sequentially
  python tools/eval_batch.py --conf sweeps/eval_best_runs.tsv

  # Run a single row (0-indexed, for SLURM array jobs)
  python tools/eval_batch.py --conf sweeps/eval_best_runs.tsv --row 3

  # Auto-detect row from SLURM_ARRAY_TASK_ID
  python tools/eval_batch.py --conf sweeps/eval_best_runs.tsv --row slurm
"""

import os
import sys
import csv
import argparse
import types

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tools.eval_checkpoints import run_eval, DEFAULT_RUNS_DIR


TSV_COLUMNS = [
    "dataset", "job", "task", "wandb_project", "wandb_group", "wandb_name",
    "attacks", "c_corruptions", "c_severity", "imbalance", "imb_factor",
]


def parse_tsv(path):
    """Parse the evaluation TSV, skipping comments and the header."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if parts[0] == "dataset":
                continue
            row = {}
            for i, col in enumerate(TSV_COLUMNS):
                row[col] = parts[i].strip() if i < len(parts) else ""
            rows.append(row)
    return rows


def row_to_args(row, runs_dir, device, batch_size):
    """Convert a TSV row dict into an argparse-like namespace for run_eval()."""
    args = types.SimpleNamespace()

    args.job = row["job"]
    args.task = row.get("task", "0")
    args.runs_dir = runs_dir
    args.ckpt = None
    args.ckpt_dir = None
    args.pattern = "*_epoch*.pt"

    args.dataset = row["dataset"]
    args.device = device
    args.batch_size = batch_size

    args.attacks = row.get("attacks", "fgsm,pgd_linf,pgd_linf_rs")
    args.adv_eps = 8.0
    args.adv_steps = 20
    args.adv_pixel = True

    args.c_corruptions = row.get("c_corruptions", "")
    sev = row.get("c_severity", "5")
    args.c_severity = int(sev) if sev else 5

    args.imbalance = row.get("imbalance", "none") or "none"
    imb_f = row.get("imb_factor", "")
    args.imb_factor = float(imb_f) if imb_f else None
    args.imb_seed = None

    args.wandb_project = row.get("wandb_project") or None
    args.wandb_group = row.get("wandb_group") or "Eval"
    args.wandb_name = row.get("wandb_name") or None
    args.output_csv = None

    return args


def main():
    parser = argparse.ArgumentParser(description="Batch offline checkpoint evaluator")
    parser.add_argument("--conf", type=str, required=True, help="Path to eval TSV config.")
    parser.add_argument("--row", type=str, default=None,
                        help="Run only this row (0-indexed). Use 'slurm' to read SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--runs_dir", type=str, default=DEFAULT_RUNS_DIR,
                        help="Base directory for SLURM run outputs.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    cli = parser.parse_args()

    rows = parse_tsv(cli.conf)
    print(f"Loaded {len(rows)} evaluation rows from {cli.conf}")

    if cli.row is not None:
        if cli.row == "slurm":
            idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
        else:
            idx = int(cli.row)
        if idx >= len(rows):
            print(f"Row index {idx} out of range (only {len(rows)} data rows).")
            sys.exit(1)
        rows = [(idx, rows[idx])]
    else:
        rows = list(enumerate(rows))

    for i, row in rows:
        if "<job_id>" in row.get("job", "") or "<task_id>" in row.get("task", ""):
            print(f"\n[row {i}] SKIP (placeholder IDs): {row.get('wandb_name', '?')}")
            continue
        print(f"\n{'#'*70}")
        print(f"  [row {i}]  {row.get('wandb_name', row['dataset'])}")
        print(f"{'#'*70}")
        args = row_to_args(row, cli.runs_dir, cli.device, cli.batch_size)
        try:
            run_eval(args)
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()

    print(f"\nAll done.")


if __name__ == "__main__":
    main()
