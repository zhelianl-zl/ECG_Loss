# tools/run_from_tsv.py
import argparse
import os
import subprocess
from pathlib import Path
import fcntl


def _read_lines_keep_header(conf_path: Path) -> list[str]:
    """
    Returns non-empty lines.
    - The first non-empty line is treated as header (even if it starts with '#dataset').
    - Subsequent lines starting with '#' are treated as comments and skipped.
    """
    raw_lines = [ln.rstrip("\n") for ln in conf_path.read_text(encoding="utf-8").splitlines()]
    # drop empty lines
    nonempty = [ln.strip() for ln in raw_lines if ln.strip()]
    if not nonempty:
        return []

    header = nonempty[0]
    data = []
    for ln in nonempty[1:]:
        if ln.lstrip().startswith("#"):
            continue
        data.append(ln)
    return [header] + data


def read_tsv_row(conf_path: Path, idx: int) -> dict:
    lines = _read_lines_keep_header(conf_path)
    if len(lines) < 2:
        raise RuntimeError(
            f"{conf_path} must have a header + at least one data row (comments allowed)."
        )

    header_line = lines[0].strip()
    # allow header like "#dataset\tseed\t..."
    if header_line.startswith("#"):
        header_line = header_line.lstrip("#").strip()

    header = header_line.split("\t")
    data_rows = lines[1:]

    if idx < 0 or idx >= len(data_rows):
        raise IndexError(f"idx={idx} out of range. valid: [0, {len(data_rows)-1}]")

    row = data_rows[idx].split("\t")
    if len(row) != len(header):
        raise RuntimeError(
            f"Row field count ({len(row)}) != header field count ({len(header)}).\n"
            f"Header={header}\nRow={data_rows[idx]}"
        )
    return dict(zip(header, row))


def ensure_dataset(data_root: Path, dataset: str, auto_download: bool = True) -> None:
    """
    Auto-download common torchvision datasets into data_root, with a filesystem lock
    to avoid concurrent downloads corrupting files.

    If dataset is not recognized or auto_download=False, this is a no-op.
    """
    ds = (dataset or "").strip().lower()
    if not auto_download:
        return

    # Only handle common small datasets; ImageNet etc should be staged manually.
    supported = {"cifar10", "cifar100", "svhn", "mnist"}
    if ds not in supported:
        return

    data_root.mkdir(parents=True, exist_ok=True)
    lock_path = data_root / ".download.lock"

    with open(lock_path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            from torchvision import datasets as tvds  # import under lock

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", required=True, help="TSV path with header (comments allowed)")
    ap.add_argument("--idx", type=int, required=True, help="0-based row index excluding header")
    ap.add_argument("--run_dir", required=True, help="Run directory (e.g., $SCRATCH/cegs_runs/...)")
    ap.add_argument("--commit", default="unknown", help="Short git commit hash for naming")
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "CEGS"))
    ap.add_argument(
        "--wandb_mode",
        default=os.environ.get("WANDB_MODE", "offline"),
        choices=["online", "offline", "disabled"],
    )
    ap.add_argument(
        "--auto_download",
        action="store_true",
        default=(os.environ.get("CEGS_AUTO_DOWNLOAD", "1") != "0"),
        help="Auto-download supported datasets (default on; set env CEGS_AUTO_DOWNLOAD=0 to disable).",
    )
    args = ap.parse_args()

    conf_path = Path(args.conf).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    hp = read_tsv_row(conf_path, args.idx)

    # ---------- W&B isolation ----------
    os.environ["WANDB_DIR"] = str(run_dir / "wandb")
    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_GROUP"] = f"{hp.get('dataset','exp')}_{args.commit}"
    os.environ["WANDB_NAME"] = f"i{args.idx}_seed{hp.get('seed','0')}_{args.commit}"
    os.environ["WANDB_MODE"] = args.wandb_mode

    # ---------- Data root (shared) ----------
    # sbatch will export CEGS_DATA_DIR=$SCRATCH/cegs_data
    data_root = Path(os.environ.get("CEGS_DATA_DIR", str(run_dir / "data"))).expanduser().resolve()

    # Auto download with lock (no-op for unsupported datasets)
    ensure_dataset(data_root, hp.get("dataset", ""), auto_download=args.auto_download)

    # ---------- Build train.py command ----------
    def add(flag: str, key: str, default: str | None = None) -> list[str]:
        v = hp.get(key, default)
        if v is None or v == "":
            return []
        return [flag, str(v)]

    cmd: list[str] = ["python", "-u", "train.py"]

    # core
    cmd += add("--dataset", "dataset")
    cmd += add("--seed", "seed")
    cmd += add("--stop", "stop", "epochs")
    cmd += add("--stop_val", "stop_val")
    cmd += add("--lr", "lr")
    cmd += add("--batch", "batch")
    cmd += add("--stage1_epochs", "stage1_epochs")
    cmd += add("--stage2_epochs", "stage2_epochs")
    cmd += add("--loss_stage2", "loss_stage2")

    # ECG/CEGS
    cmd += add("--ecg_lam", "ecg_lam")
    cmd += add("--ecg_tau", "ecg_tau")
    cmd += add("--ecg_k", "ecg_k")
    cmd += add("--ecg_conf_type", "ecg_conf_type")
    cmd += add("--ecg_schedule", "ecg_schedule")

    # record for reproducibility
    (run_dir / "cmd.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    (run_dir / "config_row.tsv").write_text(
        "\t".join([f"{k}={v}" for k, v in hp.items()]) + "\n", encoding="utf-8"
    )
    (run_dir / "data_root.txt").write_text(str(data_root) + "\n", encoding="utf-8")

    # run
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()