# tools/run_from_tsv.py
import argparse
import os
import subprocess
from pathlib import Path


def _load_noncomment_lines(p: Path) -> list[str]:
    lines = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return lines


def read_tsv_row(conf_path: Path, idx: int) -> dict:
    lines = _load_noncomment_lines(conf_path)
    if len(lines) < 2:
        raise RuntimeError(
            f"{conf_path} must have a header line + at least one data row (comments allowed)."
        )
    header = lines[0].split("\t")
    data_rows = lines[1:]
    if idx < 0 or idx >= len(data_rows):
        raise IndexError(f"idx={idx} out of range, valid: [0, {len(data_rows)-1}]")
    row = data_rows[idx].split("\t")
    if len(row) != len(header):
        raise RuntimeError(
            f"Row field count ({len(row)}) != header field count ({len(header)}). "
            f"Row={data_rows[idx]}"
        )
    return dict(zip(header, row))


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

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()