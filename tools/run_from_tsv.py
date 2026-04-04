#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tools/run_from_tsv.py

Key features:
- TSV supports meta directive (full-line comment):
    #wandb_project=ecg-loss
  (or "#wandb_project: ecg-loss")

- Optional per-row column "wandb_project" overrides the meta directive.

- W&B run name is descriptive AND distinguishes ecg_conf_type (e.g., 1-pe vs pmax),
  while keeping Slurm array job/task id so you can scancel quickly.

  Format:
    <dataset>_s1-<e1>-<loss1>_s2-<e2>-<loss2>_pe-<pe>_conf-<conf>_sch-<sch>_lam<...>_tau<...>_k<...>_j<arrayjob>_t<task>

  Examples:
    binaryCifar10_s1-60-ecg_s2-40-ecg_pe-logk_rms_conf-1pe_sch-linear_lam0.05-0.2_tau0.75-0.98_k8-15_j37772932_t1
    binaryCifar10_s1-60-ecg_s2-40-ecg_pe-logk_rms_conf-pmax_sch-linear_lam0.05-0.2_tau0.65-0.93_k8-15_j37772932_t4

- Builds the train.py command from TSV columns. Missing/empty fields are not passed.

- Tau quantile: ecg_tau_start="quantile" or "q" + ecg_tau_end=q (e.g. 0.8) => fixed tau=quantile(pmax).
  ecg_tau_start="auto_q" + ecg_tau_end=q_start => q scheduled linearly from q_start to 0.9 over epochs.

- Auto-lambda: set ecg_lam_start to "auto" (or "auto_w" for 5-epoch delta warmup) and ecg_lam_end
  to delta (e.g. 0.05). For reference-based auto-delta, use ecg_lam_start=auto_d or auto_dw in TSV
  (ecg_lam_end = initial_delta, e.g. 0.05); auto_dw adds 5-epoch warmup. Scale normalized to mean 1.

CLI (compatible with your sbatch):
  python -u tools/run_from_tsv.py --conf <tsv> --idx <row_idx> --run_dir <dir> --commit <sha>
"""

import argparse
import fcntl
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# -----------------------------
# TSV parsing with meta support
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
        s = ln.rstrip("\n")
        if not s.strip():
            continue

        if s.lstrip().startswith("#"):
            t = s.lstrip()[1:].strip()
            tl = t.lower()

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
            header_line = s.rstrip('\n').rstrip('\r')
        else:
            data_lines.append(s.rstrip('\n').rstrip('\r'))

    if header_line is None:
        return [], [], meta
    return header_line.split("\t"), data_lines, meta


def read_tsv_row(conf_path: Path, idx: int) -> Tuple[Dict[str, str], Dict[str, str]]:
    header, data_lines, meta = _parse_tsv_with_meta(conf_path)
    if len(header) == 0 or len(data_lines) == 0:
        raise RuntimeError(f"{conf_path} must have a header + at least one data row")

    if idx < 0 or idx >= len(data_lines):
        raise RuntimeError(f"idx {idx} out of range for {conf_path} (rows={len(data_lines)})")

    row = data_lines[idx].split("	")
    # tolerate missing trailing empty columns (common when editors trim tabs)
    if len(row) < len(header):
        row = row + [""] * (len(header) - len(row))
    elif len(row) > len(header):
        row = row[:len(header)]
    return dict(zip(header, row)), meta


# -----------------------------
# Helpers
# -----------------------------
def normalize_dataset_name(ds: str) -> str:
    d = (ds or "").strip().lower()
    mapping = {
        "binary": "binaryCifar10",
        "binarycifar10": "binaryCifar10",
        "binary-cifar10": "binaryCifar10",
        "cifar-10": "cifar10",
        "cifar10": "cifar10",
        "cifar-100": "cifar100",
        "cifar100": "cifar100",
        "svhn": "svhn",
        "mnist": "mnist",
        "imagenet": "imageNet",
        "image1k": "imageNet",
        "image-net": "imageNet",
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
    v = str(hp.get(key, "")).strip().lower()
    if v in {"1", "true", "yes", "y"}:
        cmd.append(f"--{key}")


def choose_wandb_project(dataset: str) -> str:
    ds = (dataset or "").strip().lower()
    ds_key = re.sub(r"[^a-z0-9]+", "_", ds).strip("_")
    for k in (f"WANDB_PROJECT_{ds_key}", f"WANDB_PROJECT_{ds_key.upper()}"):
        if k in os.environ and os.environ[k].strip():
            return os.environ[k].strip()
    return os.environ.get("WANDB_PROJECT_DEFAULT", "CEGS").strip()


def _safe_token(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


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

    dataset = normalize_dataset_name(hp.get("dataset", "").strip())
    seed = (hp.get("seed") or "0").strip()
    stop_val = (hp.get("stop_val") or "").strip()
    e1 = (hp.get("stage1_epochs") or "").strip() or "0"
    e2 = (hp.get("stage2_epochs") or "").strip() or "0"

    loss1 = (hp.get("loss_stage1") or "").strip() or "none"
    loss2 = (hp.get("loss_stage2") or "").strip() or "none"
    pe = (hp.get("pe_mode") or "").strip() or "none"

    # ecg knobs (for naming only; train args are passed later)
    sch = (hp.get("ecg_schedule") or "").strip() or "none"
    conf = (hp.get("ecg_conf_type") or "").strip() or "none"
    conf_safe = re.sub(r"[^A-Za-z0-9]+", "", conf.lower())  # "1-pe" -> "1pe"

    lam_s = (hp.get("ecg_lam_start") or "").strip()
    lam_e = (hp.get("ecg_lam_end") or "").strip()
    lam_c = (hp.get("ecg_lam") or "").strip()
    if lam_s and lam_s.lower() in ("auto", "auto_w", "auto_d", "auto_dw"):
        lam_part = f"{lam_s.lower()}{lam_e}" if lam_e else (lam_s.lower() + "0.05")
    else:
        lam_part = f"{lam_s}-{lam_e}" if (lam_s and lam_e) else (lam_c if lam_c else "na")

    tau_s = (hp.get("ecg_tau_start") or "").strip()
    tau_e = (hp.get("ecg_tau_end") or "").strip()
    tau_c = (hp.get("ecg_tau") or "").strip()
    if tau_s and tau_s.lower() in ("quantile", "q"):
        tau_part = f"q{tau_e}" if tau_e else "q0.8"
    elif tau_s and tau_s.lower() == "auto_q":
        tau_part = f"autoq{tau_e}" if tau_e else "autoq0.6"
    else:
        tau_part = f"{tau_s}-{tau_e}" if (tau_s and tau_e) else (tau_c if tau_c else "na")

    k_s = (hp.get("ecg_k_start") or "").strip()
    k_e = (hp.get("ecg_k_end") or "").strip()
    k_c = (hp.get("ecg_k") or "").strip()
    k_part = f"{k_s}-{k_e}" if (k_s and k_e) else (k_c if k_c else "na")

    method_name = (hp.get("method_name") or loss2 or "exp").strip()
    run_kind = (hp.get("run_kind") or os.environ.get("WANDB_JOB_TYPE_DEFAULT", "official")).strip()

    # Project selection: per-row -> TSV meta -> env mapping
    project = (hp.get("wandb_project") or meta.get("wandb_project") or "").strip()
    if not project:
        project = choose_wandb_project(dataset)

    # Slurm ids (for cancel and wandb name suffix)
    array_job = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or "noj"
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID") or str(args.idx)
    job_task_suffix = f"_j{array_job}_t{task_id}"

    # Run name (TSV override supported). RunA/RunB: always append j<job>_t<task> for tracking.
    tsv_wandb_name = (hp.get("wandb_name") or "").strip()
    if tsv_wandb_name:
        base = _safe_token(tsv_wandb_name)[: 200 - len(job_task_suffix)]
        run_name = base + job_task_suffix
    else:
        _tm = (hp.get("train_mode") or "").strip()
        _tm_tag = ""
        if _tm and _tm != "standard":
            _tm_tag = f"_tm-{_tm}"
            _r_eps = (hp.get("robust_eps") or "").strip()
            _r_st = (hp.get("robust_steps") or "").strip()
            _r_beta = (hp.get("robust_beta") or "").strip()
            if _r_eps:
                _tm_tag += f"_eps{_r_eps}"
            if _r_st:
                _tm_tag += f"_st{_r_st}"
            if _r_beta and _tm in ("trades", "mart"):
                _tm_tag += f"_b{_r_beta}"
        elif loss2 == "focal":
            _fg = (hp.get("focal_gamma") or "").strip()
            _tm_tag = f"_focal" + (f"_g{_fg}" if _fg else "")
        elif loss2 == "clue_lite":
            _cl = (hp.get("clue_lambda") or "").strip()
            _tm_tag = f"_cluelite" + (f"_l{_cl}" if _cl else "")
        elif loss2 == "clue":
            _ca = (hp.get("clue_alpha") or "").strip()
            _cm = (hp.get("clue_mc_passes") or "").strip()
            _tm_tag = f"_clue" + (f"_a{_ca}" if _ca else "") + (f"_mc{_cm}" if _cm else "")
        raw = (
            f"{dataset}"
            f"_s1-{e1}-{loss1}"
            f"_s2-{e2}-{loss2}"
            f"_pe-{pe}"
            f"_conf-{conf_safe}"
            f"_sch-{sch}"
            f"_lam{lam_part}"
            f"_tau{tau_part}"
            f"_k{k_part}"
            f"{_tm_tag}"
            f"{job_task_suffix}"
        )
        run_name = _safe_token(raw)[:200]

    # Group (TSV override supported; default group by array)
    group = (hp.get("wandb_group") or "").strip()
    if not group:
        group = f"j{array_job}"

    # tags (optional); wandb allows max 64 chars per tag
    WANDB_MAX_TAG_LEN = 64
    tags = [
        run_kind, dataset, f"s{seed}", method_name,
        f"T{stop_val or 'T?'}", f"s1{e1}", f"s2{e2}",
        f"conf-{conf_safe}", f"sch-{sch}",
    ]
    tags = [str(t)[:WANDB_MAX_TAG_LEN] for t in tags if t]

    # W&B env
    os.environ["WANDB_PROJECT"] = project
    if os.environ.get("WANDB_ENTITY", "").strip():
        os.environ["WANDB_ENTITY"] = os.environ["WANDB_ENTITY"].strip()
    os.environ["WANDB_NAME"] = run_name
    os.environ["WANDB_GROUP"] = group
    os.environ["WANDB_JOB_TYPE"] = run_kind
    os.environ["WANDB_TAGS"] = ",".join([t for t in tags if t])
    os.environ["PYTHONUNBUFFERED"] = "1"

    
    # per-row env overrides: any TSV column named env_<VARNAME> will set environment variable <VARNAME>
    # Example columns: env_IMAGENET_ORIGINAL, env_IMAGENET_RES, env_IMAGENET_DS_ROOT, env_DL_WORKERS.
    # For 224x224 (e.g. PSC): env_IMAGENET_ORIGINAL=1, env_IMAGENET_ROOT=/path/to/dir/with/train/and/val
    for k, v in hp.items():
        if not k.startswith("env_"):
            continue
        vv = str(v).strip()
        if vv == "":
            continue
        os.environ[k[len("env_"):]] = vv

    # ImageNet: defaults for 32x32 SmallImageNet; TSV can override via env_IMAGENET_* (e.g. 64x64).
    if dataset.lower() == "imagenet":
        if not os.environ.get("IMAGENET_ORIGINAL", "").strip():
            os.environ["IMAGENET_ORIGINAL"] = "0"
        if not os.environ.get("IMAGENET_RES", "").strip():
            os.environ["IMAGENET_RES"] = "32"
        if not os.environ.get("IMAGENET_DS_ROOT", "").strip():
            data_root_for_ds = Path(os.environ.get("CEGS_DATA_DIR", str(run_dir / "data"))).expanduser().resolve()
            default_32 = data_root_for_ds / "smallimagenet_32"
            alt_32 = data_root_for_ds.parent / "cegs_data" / "smallimagenet_32"
            if default_32.exists():
                os.environ["IMAGENET_DS_ROOT"] = str(default_32)
            elif alt_32.exists():
                os.environ["IMAGENET_DS_ROOT"] = str(alt_32)
            else:
                os.environ["IMAGENET_DS_ROOT"] = str(default_32)
        # Use multiple DataLoader workers for 32x32 so GPU isn't starved (0 = single-thread = very slow)
        if os.environ.get("DL_WORKERS", "").strip() == "":
            os.environ["DL_WORKERS"] = "10"

# data root
    data_root = Path(os.environ.get("CEGS_DATA_DIR", str(run_dir / "data"))).expanduser().resolve()
    auto_download = os.environ.get("CEGS_AUTO_DOWNLOAD", "1") not in ("0", "false", "False")
    ensure_dataset(data_root, dataset, auto_download=auto_download)

    # build train.py cmd
    cmd: List[str] = [sys.executable, "-u", "train.py"]

    # common training args
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

    # 2-stage
    _add_arg(cmd, "--stage1_epochs", hp, "stage1_epochs")
    _add_arg(cmd, "--stage2_epochs", hp, "stage2_epochs")
    _add_arg(cmd, "--loss_stage1", hp, "loss_stage1")
    _add_arg(cmd, "--loss_stage2", hp, "loss_stage2")

    # flags
    _add_flag(cmd, hp, "force_run")
    _add_flag(cmd, hp, "full_ecg")

    # ECG/CEGS params (constants)
    _add_arg(cmd, "--ecg_lam", hp, "ecg_lam")
    _add_arg(cmd, "--ecg_tau", hp, "ecg_tau")
    _add_arg(cmd, "--ecg_k", hp, "ecg_k")
    _add_arg(cmd, "--ecg_conf_type", hp, "ecg_conf_type")
    _add_arg(cmd, "--ecg_gate_temp", hp, "ecg_gate_temp")
    _add_arg(cmd, "--ecg_detach_gates", hp, "ecg_detach_gates")
    _add_arg(cmd, "--ecg_schedule", hp, "ecg_schedule")

    # schedules (start/end)
    _add_arg(cmd, "--ecg_lam_start", hp, "ecg_lam_start")
    _add_arg(cmd, "--ecg_lam_end", hp, "ecg_lam_end")
    _add_arg(cmd, "--ecg_tau_start", hp, "ecg_tau_start")
    _add_arg(cmd, "--ecg_tau_end", hp, "ecg_tau_end")
    _add_arg(cmd, "--ecg_k_start", hp, "ecg_k_start")
    _add_arg(cmd, "--ecg_k_end", hp, "ecg_k_end")

    # adaptive tau_target schedule (optional)
    _add_arg(cmd, "--ecg_tau_target", hp, "ecg_tau_target")
    _add_arg(cmd, "--ecg_tau_lr", hp, "ecg_tau_lr")
    _add_arg(cmd, "--ecg_tau_ema", hp, "ecg_tau_ema")
    _add_arg(cmd, "--ecg_tau_deadzone", hp, "ecg_tau_deadzone")
    _add_arg(cmd, "--ecg_tau_min", hp, "ecg_tau_min")
    _add_arg(cmd, "--ecg_tau_max", hp, "ecg_tau_max")
    # auto_q_valley params (optional)
    _add_arg(cmd, "--ecg_tau_valley_warmup", hp, "ecg_tau_valley_warmup")
    _add_arg(cmd, "--ecg_tau_valley_smooth", hp, "ecg_tau_valley_smooth")

    # auto-lambda (optional overrides when ecg_lam_start=auto)
    _add_arg(cmd, "--ecg_lam_max", hp, "ecg_lam_max")
    _add_arg(cmd, "--ecg_lam_beta", hp, "ecg_lam_beta")
    _add_arg(cmd, "--ecg_lam_eps", hp, "ecg_lam_eps")
    # tail-ratio auto_tr (optional when ecg_lam_start=auto_tr)
    _add_arg(cmd, "--ecg_tail_ratio_target", hp, "ecg_tail_ratio_target")
    _add_arg(cmd, "--ecg_tail_ratio_beta", hp, "ecg_tail_ratio_beta")
    _add_arg(cmd, "--ecg_active_frac_floor", hp, "ecg_active_frac_floor")
    _add_arg(cmd, "--ecg_sparse_lam_decay", hp, "ecg_sparse_lam_decay")
    _add_arg(cmd, "--ecg_sparse_lam_zero", hp, "ecg_sparse_lam_zero")
    _add_arg(cmd, "--ecg_tail_lam_ema", hp, "ecg_tail_lam_ema")
    _add_arg(cmd, "--ecg_tail_invalid_decay", hp, "ecg_tail_invalid_decay")

    # ---- suites / long-tail / demo dump (optional) ----
    _add_arg(cmd, "--eval_extra_every", hp, "eval_extra_every")

    # ADV suite
    _add_arg(cmd, "--eval_adv_suite", hp, "eval_adv_suite")
    _add_arg(cmd, "--adv_attacks", hp, "adv_attacks")
    _add_arg(cmd, "--adv_eps", hp, "adv_eps")
    _add_arg(cmd, "--adv_steps", hp, "adv_steps")
    _add_arg(cmd, "--adv_restarts", hp, "adv_restarts")
    _add_arg(cmd, "--adv_alpha", hp, "adv_alpha")
    _add_arg(cmd, "--adv_pixel", hp, "adv_pixel")

    # C-suite
    _add_arg(cmd, "--eval_c_suite", hp, "eval_c_suite")
    _add_arg(cmd, "--c_corruptions", hp, "c_corruptions")
    _add_arg(cmd, "--c_severities", hp, "c_severities")
    _add_arg(cmd, "--c_name", hp, "c_name")
    _add_arg(cmd, "--c_severity", hp, "c_severity")

    # Long-tail
    _add_arg(cmd, "--imbalance", hp, "imbalance")
    _add_arg(cmd, "--imb_factor", hp, "imb_factor")
    _add_arg(cmd, "--imb_seed", hp, "imb_seed")

    # Demo dump
    _add_arg(cmd, "--dump_gates", hp, "dump_gates")
    _add_arg(cmd, "--dump_gates_n", hp, "dump_gates_n")

    # stage2 speed (for large datasets e.g. ImageNet32: fewer full-train passes)
    _add_arg(cmd, "--stage2_fast", hp, "stage2_fast")
    _add_arg(cmd, "--stage2_find_every", hp, "stage2_find_every")
    _add_arg(cmd, "--stage2_ce_log_every", hp, "stage2_ce_log_every")
    _add_arg(cmd, "--stage2_lr_scale", hp, "stage2_lr_scale")

    # ---- baseline comparison: focal / clue_lite / robust training ----
    _add_arg(cmd, "--train_mode", hp, "train_mode")
    _add_arg(cmd, "--focal_gamma", hp, "focal_gamma")
    _add_arg(cmd, "--focal_alpha", hp, "focal_alpha")
    _add_arg(cmd, "--clue_lambda", hp, "clue_lambda")
    _add_arg(cmd, "--clue_detach_proxy", hp, "clue_detach_proxy")
    _add_arg(cmd, "--clue_dropout_p", hp, "clue_dropout_p")
    _add_arg(cmd, "--clue_mc_passes", hp, "clue_mc_passes")
    _add_arg(cmd, "--clue_alpha", hp, "clue_alpha")
    _add_arg(cmd, "--clue_enable_mcdo", hp, "clue_enable_mcdo")
    _add_arg(cmd, "--robust_eps", hp, "robust_eps")
    _add_arg(cmd, "--robust_alpha", hp, "robust_alpha")
    _add_arg(cmd, "--robust_steps", hp, "robust_steps")
    _add_arg(cmd, "--robust_beta", hp, "robust_beta")
    _add_arg(cmd, "--robust_random_start", hp, "robust_random_start")
    _add_arg(cmd, "--robust_pixel", hp, "robust_pixel")

    # ---- runtime timing options ----
    _add_arg(cmd, "--rt_step_sample_every", hp, "rt_step_sample_every")
    _add_arg(cmd, "--rt_minimal_mode", hp, "rt_minimal_mode")

    # record
    (run_dir / "cmd.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    (run_dir / "wandb_meta.txt").write_text(
        f"project={project}\n"
        f"entity={os.environ.get('WANDB_ENTITY','')}\n"
        f"name={run_name}\n"
        f"group={group}\n"
        f"job_type={run_kind}\n",
        encoding="utf-8",
    )

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()