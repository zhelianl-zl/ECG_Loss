import argparse
import os
import re
import subprocess
from pathlib import Path
import fcntl
from typing import List, Dict, Optional


def _read_lines_keep_header(conf_path: Path) -> List[str]:
    raw_lines = conf_path.read_text(encoding="utf-8").splitlines()
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


def read_tsv_row(conf_path: Path, idx: int) -> Dict[str, str]:
    lines = _read_lines_keep_header(conf_path)
    if len(lines) < 2:
        raise RuntimeError(f"{conf_path} must have a header + at least one data row")

    header_line = lines[0].strip()
    if header_line.startswith("#"):
        header_line = header_line.lstrip("#").strip()
    header = header_line.split("\t")

    row = lines[idx + 1].split("\t")
    if len(row) != len(header):
        raise RuntimeError("Row field count != header field count")
    return dict(zip(header, row))


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
    if v is None or v == "":
        return
    cmd.extend([flag, str(v)])


def choose_wandb_project(dataset: str) -> str:
    ds = (dataset or "").strip().lower()
    ds_key = re.sub(r"[^a-z0-9]+", "_", ds).strip("_")  # cifar100, svhn, cifar10...
    # allow either lower or upper key in env
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

    hp = read_tsv_row(conf_path, args.idx)

    dataset = hp.get("dataset", "").strip()
    seed = hp.get("seed", "0").strip()
    stop_val = hp.get("stop_val", "").strip()
    stage1 = hp.get("stage1_epochs", "").strip()
    stage2 = hp.get("stage2_epochs", "").strip()

    # method/run_kind (optional columns; otherwise inferred)
    method_name = (hp.get("method_name") or hp.get("loss_stage2") or "exp").strip()
    run_kind = (hp.get("run_kind") or os.environ.get("WANDB_JOB_TYPE_DEFAULT", "official")).strip()

    # choose project by dataset (cifar100 -> ecg_Cifar_100)
    project = choose_wandb_project(dataset)

    # notebook-style naming
    total_epochs = stop_val or "T?"
    run_name = f"{dataset}_{method_name}_s{seed}_T{total_epochs}"
    group = f"{dataset}_{run_kind}_s{seed}_T{total_epochs}"

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
    os.environ["WANDB_NAME"] = run_name
    os.environ["WANDB_GROUP"] = group
    os.environ["WANDB_JOB_TYPE"] = run_kind
    os.environ["WANDB_TAGS"] = ",".join(tags)
    os.environ["PYTHONUNBUFFERED"] = "1"

    # data root
    data_root = Path(os.environ.get("CEGS_DATA_DIR", str(run_dir / "data"))).expanduser().resolve()
    auto_download = os.environ.get("CEGS_AUTO_DOWNLOAD", "1") != "0"
    ensure_dataset(data_root, dataset, auto_download=auto_download)

    # build train.py cmd
    cmd: List[str] = ["python", "-u", "train.py"]
    _add_arg(cmd, "--dataset", hp, "dataset")
    _add_arg(cmd, "--seed", hp, "seed")
    _add_arg(cmd, "--stop", hp, "stop", "epochs")
    _add_arg(cmd, "--stop_val", hp, "stop_val")
    _add_arg(cmd, "--lr", hp, "lr")
    _add_arg(cmd, "--batch", hp, "batch")
    _add_arg(cmd, "--stage1_epochs", hp, "stage1_epochs")
    _add_arg(cmd, "--stage2_epochs", hp, "stage2_epochs")
    _add_arg(cmd, "--loss_stage2", hp, "loss_stage2")

    _add_arg(cmd, "--ecg_lam", hp, "ecg_lam")
    _add_arg(cmd, "--ecg_tau", hp, "ecg_tau")
    _add_arg(cmd, "--ecg_k", hp, "ecg_k")
    _add_arg(cmd, "--ecg_conf_type", hp, "ecg_conf_type")
    _add_arg(cmd, "--ecg_schedule", hp, "ecg_schedule")

    # record
    (run_dir / "cmd.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    (run_dir / "wandb_meta.txt").write_text(
        f"project={project}\nentity={os.environ.get('WANDB_ENTITY','')}\nname={run_name}\ngroup={group}\n",
        encoding="utf-8",
    )

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()