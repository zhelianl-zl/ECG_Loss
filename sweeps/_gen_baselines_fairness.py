"""Generate sweeps/baselines_all.tsv — run once; delete after."""
from pathlib import Path

OUT = Path(__file__).with_name("baselines_all.tsv")

HEADER = """#wandb_project=ecgs-baselines
# Fairness protocol (main): unified lr=0.01; EUAT = two-stage CE warmup then EUAT (loss_stage1=ce, loss_stage2=euat).
# Main comparison rows: run_kind=baseline_main AND wandb_group=main_lr001 (lr=0.01 only). Includes focal, pgd_at/trades/mart (CIFAR-like 8/2/10), EUAT, cifar10 CLUE (faithful).
# Row order: within each dataset, main baselines first; clue_lite (proxy) rows last, after a separator comment - not part of the main CLUE comparison.
# Proxy / appendix rows: run_kind=baseline_proxy AND wandb_group=proxy_cluelite - clue_lite only; NOT faithful CLUE; do not mix with main CLUE.
# Supplementary rows: CIFAR-100 lr=0.1 - run_kind=baseline_supp, wandb_group=supp_c100_lr0p1, wandb_project=ecg-cifar100-pmax (not the lr=0.01 main block).
# Robust training: CIFAR-like datasets use eps/alpha/steps = 8/2/10 (pgd_at, trades, mart). ImageNet-32 uses 4/1/10.
# Timing: rt_step_sample_every=20, rt_minimal_mode=True when columns present.
dataset\tseed\tstop\tstop_val\tlr\tmomentum\tbatch\tworkers\thalf_prec\tvariants\ttype\tpe_mode\tstage1_epochs\tstage2_epochs\tloss_stage1\tloss_stage2\tecg_conf_type\tecg_detach_gates\tecg_schedule\tecg_lam_start\tecg_lam_end\tecg_tau_start\tecg_tau_end\tecg_k_start\tecg_k_end\tforce_run\tmethod_name\trun_kind\tenv_IMAGENET_ORIGINAL\tenv_IMAGENET_RES\tenv_IMAGENET_DS_ROOT\tenv_DL_WORKERS\twandb_project\twandb_group\twandb_name\ttrain_mode\tfocal_gamma\tfocal_alpha\tclue_lambda\tclue_detach_proxy\trobust_eps\trobust_alpha\trobust_steps\trobust_beta\trobust_random_start\trobust_pixel\tclue_dropout_p\tclue_mc_passes\tclue_alpha\tclue_enable_mcdo\trt_step_sample_every\trt_minimal_mode"""


def R(
    dataset,
    stop_val,
    lr,
    half_prec,
    pe_mode,
    s1,
    s2,
    l1,
    l2,
    method_name,
    run_kind,
    wandb_project,
    wandb_group,
    wandb_name,
    train_mode="standard",
    focal_gamma="",
    focal_alpha="",
    clue_lambda="",
    clue_detach_proxy="",
    robust_eps="",
    robust_alpha="",
    robust_steps="",
    robust_beta="",
    robust_random_start="True",
    robust_pixel="True",
    clue_dropout_p="",
    clue_mc_passes="",
    clue_alpha="",
    clue_enable_mcdo="",
    rt_every="20",
    rt_minimal="True",
    imagenet_env=("0", "32", "/ocean/projects/cis260049p/zliu49/cegs_data/smallimagenet_32", "0"),
):
    env_o, env_r, env_root, env_dw = ("", "", "", "")
    if dataset == "imageNet":
        env_o, env_r, env_root, env_dw = imagenet_env
    cells = [
        dataset,
        "0",
        "epochs",
        str(stop_val),
        str(lr),
        "0.9",
        "64",
        "0",
        str(half_prec),
        "none",
        "std",
        pe_mode,
        str(s1),
        str(s2),
        l1,
        l2,
        "none",
        "True",
        "none",
        "",
        "",
        "",
        "",
        "",
        "",
        "True",
        method_name,
        run_kind,
        env_o,
        env_r,
        env_root,
        env_dw,
        wandb_project,
        wandb_group,
        wandb_name,
        train_mode,
        focal_gamma,
        focal_alpha,
        clue_lambda,
        clue_detach_proxy,
        robust_eps,
        robust_alpha,
        robust_steps,
        robust_beta,
        robust_random_start,
        robust_pixel,
        clue_dropout_p,
        clue_mc_passes,
        clue_alpha,
        clue_enable_mcdo,
        rt_every,
        rt_minimal,
    ]
    return "\t".join(cells)


rows = []

# ----- binaryCifar10 (150) -----
rows.append("# --- binaryCifar10: 150 epochs; EUAT 75+75 (ce warmup + euat) ---")
rows.append(
    R(
        "binaryCifar10",
        150,
        "0.01",
        "False",
        "logk_rms",
        0,
        150,
        "ce",
        "focal",
        "binary_focal_g2",
        "baseline_main",
        "ecg_binary_pmax",
        "main_lr001",
        "binary_focal_g2",
        focal_gamma="2.0",
        focal_alpha="1.0",
    )
)
rows.append(
    R(
        "binaryCifar10",
        150,
        "0.01",
        "False",
        "logk_rms",
        0,
        150,
        "ce",
        "ce",
        "binary_pgdat_eps8_a2",
        "baseline_main",
        "ecg_binary_pmax",
        "main_lr001",
        "binary_pgdat_eps8_a2",
        train_mode="pgd_at",
        robust_eps="8",
        robust_alpha="2",
        robust_steps="10",
    )
)
rows.append(
    R(
        "binaryCifar10",
        150,
        "0.01",
        "False",
        "logk_rms",
        0,
        150,
        "ce",
        "ce",
        "binary_trades_eps8_a2_b6",
        "baseline_main",
        "ecg_binary_pmax",
        "main_lr001",
        "binary_trades_eps8_a2_b6",
        train_mode="trades",
        robust_eps="8",
        robust_alpha="2",
        robust_steps="10",
        robust_beta="6.0",
    )
)
rows.append(
    R(
        "binaryCifar10",
        150,
        "0.01",
        "False",
        "logk_rms",
        0,
        150,
        "ce",
        "ce",
        "binary_mart_eps8_a2_b6",
        "baseline_main",
        "ecg_binary_pmax",
        "main_lr001",
        "binary_mart_eps8_a2_b6",
        train_mode="mart",
        robust_eps="8",
        robust_alpha="2",
        robust_steps="10",
        robust_beta="6.0",
    )
)
rows.append(
    R(
        "binaryCifar10",
        150,
        "0.01",
        "False",
        "logk_rms",
        75,
        75,
        "ce",
        "euat",
        "binary_euat_s75_ce_s75_euat",
        "baseline_main",
        "ecg_binary_pmax",
        "main_lr001",
        "binary_euat_s75_ce_s75_euat",
    )
)
rows.append("# --- binaryCifar10: clue_lite = proxy only (below; not main CLUE) ---")
rows.append(
    R(
        "binaryCifar10",
        150,
        "0.01",
        "False",
        "logk_rms",
        0,
        150,
        "ce",
        "clue_lite",
        "proxy_binary_cluelite_l02",
        "baseline_proxy",
        "ecg_binary_pmax",
        "proxy_cluelite",
        "proxy_binary_cluelite_l02",
        clue_lambda="0.2",
        clue_detach_proxy="True",
    )
)

# ----- cifar10 -----
rows.append("# --- cifar10: 60 epochs; EUAT 30+30 (ce warmup + euat) ---")
rows.append(
    R(
        "cifar10",
        60,
        "0.01",
        "False",
        "none",
        0,
        60,
        "ce",
        "focal",
        "c10_focal_g2",
        "baseline_main",
        "ecg-cifar10",
        "main_lr001",
        "c10_focal_g2",
        focal_gamma="2.0",
        focal_alpha="1.0",
    )
)
rows.append(
    R(
        "cifar10",
        60,
        "0.01",
        "False",
        "none",
        0,
        60,
        "ce",
        "ce",
        "c10_pgdat_eps8_a2",
        "baseline_main",
        "ecg-cifar10",
        "main_lr001",
        "c10_pgdat_eps8_a2",
        train_mode="pgd_at",
        robust_eps="8",
        robust_alpha="2",
        robust_steps="10",
    )
)
rows.append(
    R(
        "cifar10",
        60,
        "0.01",
        "False",
        "none",
        0,
        60,
        "ce",
        "ce",
        "c10_trades_eps8_a2_b6",
        "baseline_main",
        "ecg-cifar10",
        "main_lr001",
        "c10_trades_eps8_a2_b6",
        train_mode="trades",
        robust_eps="8",
        robust_alpha="2",
        robust_steps="10",
        robust_beta="6.0",
    )
)
rows.append(
    R(
        "cifar10",
        60,
        "0.01",
        "False",
        "none",
        0,
        60,
        "ce",
        "ce",
        "c10_mart_eps8_a2_b6",
        "baseline_main",
        "ecg-cifar10",
        "main_lr001",
        "c10_mart_eps8_a2_b6",
        train_mode="mart",
        robust_eps="8",
        robust_alpha="2",
        robust_steps="10",
        robust_beta="6.0",
    )
)
rows.append(
    R(
        "cifar10",
        60,
        "0.01",
        "False",
        "none",
        30,
        30,
        "ce",
        "euat",
        "c10_euat_s30_ce_s30_euat",
        "baseline_main",
        "ecg-cifar10",
        "main_lr001",
        "c10_euat_s30_ce_s30_euat",
    )
)
rows.append(
    R(
        "cifar10",
        60,
        "0.01",
        "False",
        "none",
        30,
        30,
        "ce",
        "clue",
        "c10_clue_paper",
        "baseline_main",
        "ecg-cifar10",
        "main_lr001",
        "c10_clue_paper",
        clue_dropout_p="0.3",
        clue_mc_passes="5",
        clue_alpha="0.5",
        clue_enable_mcdo="True",
    )
)
rows.append("# --- cifar10: clue_lite = proxy only (below; faithful CLUE is c10_clue_paper above) ---")
rows.append(
    R(
        "cifar10",
        60,
        "0.01",
        "False",
        "none",
        0,
        60,
        "ce",
        "clue_lite",
        "proxy_c10_cluelite_l02",
        "baseline_proxy",
        "ecg-cifar10",
        "proxy_cluelite",
        "proxy_c10_cluelite_l02",
        clue_lambda="0.2",
        clue_detach_proxy="True",
    )
)

# ----- cifar100 lr=0.01 -----
rows.append("# --- cifar100 lr=0.01 (main block): project ecg-cifar100-pmax1 ---")
for spec in [
    ("c100_focal_g2", "ce", "focal", {}, {"focal_gamma": "2.0", "focal_alpha": "1.0"}),
    (
        "c100_pgdat_eps8_a2",
        "ce",
        "ce",
        {"train_mode": "pgd_at", "robust_eps": "8", "robust_alpha": "2", "robust_steps": "10"},
        {},
    ),
    (
        "c100_trades_eps8_a2_b6",
        "ce",
        "ce",
        {
            "train_mode": "trades",
            "robust_eps": "8",
            "robust_alpha": "2",
            "robust_steps": "10",
            "robust_beta": "6.0",
        },
        {},
    ),
    (
        "c100_mart_eps8_a2_b6",
        "ce",
        "ce",
        {
            "train_mode": "mart",
            "robust_eps": "8",
            "robust_alpha": "2",
            "robust_steps": "10",
            "robust_beta": "6.0",
        },
        {},
    ),
    ("c100_euat_s30_ce_s30_euat", "ce", "euat", {"s1": 30, "s2": 30}, {}),
]:
    name, l1, l2, kw, extra = spec
    s1, s2 = kw.pop("s1", 0), kw.pop("s2", 60)
    rk = kw.pop("run_kind", "baseline_main")
    group = kw.pop("group", "main_lr001")
    rows.append(
        R(
            "cifar100",
            60,
            "0.01",
            "True",
            "logk_rms",
            s1,
            s2,
            l1,
            l2,
            name,
            rk,
            "ecg-cifar100-pmax1",
            group,
            name,
            **kw,
            **extra,
        )
    )
rows.append("# --- cifar100 lr=0.01: clue_lite = proxy only (below) ---")
for spec in [
    (
        "proxy_c100_cluelite_l02",
        "ce",
        "clue_lite",
        {"run_kind": "baseline_proxy", "group": "proxy_cluelite"},
        {"clue_lambda": "0.2", "clue_detach_proxy": "True"},
    ),
]:
    name, l1, l2, kw, extra = spec
    s1, s2 = kw.pop("s1", 0), kw.pop("s2", 60)
    rk = kw.pop("run_kind", "baseline_main")
    group = kw.pop("group", "main_lr001")
    rows.append(
        R(
            "cifar100",
            60,
            "0.01",
            "True",
            "logk_rms",
            s1,
            s2,
            l1,
            l2,
            name,
            rk,
            "ecg-cifar100-pmax1",
            group,
            name,
            **kw,
            **extra,
        )
    )

# ----- cifar100 lr=0.1 supplementary -----
rows.append("# --- cifar100 lr=0.1 supplementary: project ecg-cifar100-pmax, group supp_c100_lr0p1 ---")
for spec in [
    ("c100_lr01_focal_g2", "ce", "focal", {}, {"focal_gamma": "2.0", "focal_alpha": "1.0"}),
    (
        "c100_lr01_pgdat_eps8_a2",
        "ce",
        "ce",
        {"train_mode": "pgd_at", "robust_eps": "8", "robust_alpha": "2", "robust_steps": "10"},
        {},
    ),
    (
        "c100_lr01_trades_eps8_a2_b6",
        "ce",
        "ce",
        {
            "train_mode": "trades",
            "robust_eps": "8",
            "robust_alpha": "2",
            "robust_steps": "10",
            "robust_beta": "6.0",
        },
        {},
    ),
    (
        "c100_lr01_mart_eps8_a2_b6",
        "ce",
        "ce",
        {
            "train_mode": "mart",
            "robust_eps": "8",
            "robust_alpha": "2",
            "robust_steps": "10",
            "robust_beta": "6.0",
        },
        {},
    ),
    ("c100_lr01_euat_s30_ce_s30_euat", "ce", "euat", {"s1": 30, "s2": 30}, {}),
]:
    name, l1, l2, kw, extra = spec
    s1, s2 = kw.pop("s1", 0), kw.pop("s2", 60)
    group = kw.pop("group", "supp_c100_lr0p1")
    rows.append(
        R(
            "cifar100",
            60,
            "0.1",
            "True",
            "logk_rms",
            s1,
            s2,
            l1,
            l2,
            name,
            "baseline_supp",
            "ecg-cifar100-pmax",
            group,
            name,
            **kw,
            **extra,
        )
    )
rows.append("# --- cifar100 lr=0.1: clue_lite = proxy only (below) ---")
rows.append(
    R(
        "cifar100",
        60,
        "0.1",
        "True",
        "logk_rms",
        0,
        60,
        "ce",
        "clue_lite",
        "c100_lr01_proxy_cluelite_l02",
        "baseline_supp",
        "ecg-cifar100-pmax",
        "supp_c100_lr0p1",
        "c100_lr01_proxy_cluelite_l02",
        clue_lambda="0.2",
        clue_detach_proxy="True",
    )
)

# ----- SVHN -----
rows.append("# --- svhn: EUAT 30+30 (ce warmup + euat) ---")
rows.append(
    R(
        "svhn",
        60,
        "0.01",
        "True",
        "logk_rms",
        0,
        60,
        "ce",
        "focal",
        "svhn_focal_g2",
        "baseline_main",
        "ecg-svhn-pmax",
        "main_lr001",
        "svhn_focal_g2",
        focal_gamma="2.0",
        focal_alpha="1.0",
    )
)
rows.append(
    R(
        "svhn",
        60,
        "0.01",
        "True",
        "logk_rms",
        0,
        60,
        "ce",
        "ce",
        "svhn_pgdat_eps8_a2",
        "baseline_main",
        "ecg-svhn-pmax",
        "main_lr001",
        "svhn_pgdat_eps8_a2",
        train_mode="pgd_at",
        robust_eps="8",
        robust_alpha="2",
        robust_steps="10",
    )
)
rows.append(
    R(
        "svhn",
        60,
        "0.01",
        "True",
        "logk_rms",
        0,
        60,
        "ce",
        "ce",
        "svhn_trades_eps8_a2_b6",
        "baseline_main",
        "ecg-svhn-pmax",
        "main_lr001",
        "svhn_trades_eps8_a2_b6",
        train_mode="trades",
        robust_eps="8",
        robust_alpha="2",
        robust_steps="10",
        robust_beta="6.0",
    )
)
rows.append(
    R(
        "svhn",
        60,
        "0.01",
        "True",
        "logk_rms",
        0,
        60,
        "ce",
        "ce",
        "svhn_mart_eps8_a2_b6",
        "baseline_main",
        "ecg-svhn-pmax",
        "main_lr001",
        "svhn_mart_eps8_a2_b6",
        train_mode="mart",
        robust_eps="8",
        robust_alpha="2",
        robust_steps="10",
        robust_beta="6.0",
    )
)
rows.append(
    R(
        "svhn",
        60,
        "0.01",
        "True",
        "logk_rms",
        30,
        30,
        "ce",
        "euat",
        "svhn_euat_s30_ce_s30_euat",
        "baseline_main",
        "ecg-svhn-pmax",
        "main_lr001",
        "svhn_euat_s30_ce_s30_euat",
    )
)
rows.append("# --- svhn: clue_lite = proxy only (below) ---")
rows.append(
    R(
        "svhn",
        60,
        "0.01",
        "True",
        "logk_rms",
        0,
        60,
        "ce",
        "clue_lite",
        "proxy_svhn_cluelite_l02",
        "baseline_proxy",
        "ecg-svhn-pmax",
        "proxy_cluelite",
        "proxy_svhn_cluelite_l02",
        clue_lambda="0.2",
        clue_detach_proxy="True",
    )
)

# ----- ImageNet-32: robust 4/1/10 (unchanged) -----
rows.append("# --- imageNet-32: EUAT 30+30; robust pgd_at/trades/mart at 4/1/10 ---")
rows.append(
    R(
        "imageNet",
        60,
        "0.01",
        "True",
        "logk_rms",
        0,
        60,
        "ce",
        "focal",
        "imnet32_focal_g2",
        "baseline_main",
        "cegs-imageNet32-pmax",
        "main_lr001",
        "imnet32_focal_g2",
        focal_gamma="2.0",
        focal_alpha="1.0",
    )
)
rows.append(
    R(
        "imageNet",
        60,
        "0.01",
        "True",
        "logk_rms",
        0,
        60,
        "ce",
        "ce",
        "imnet32_pgdat_eps4",
        "baseline_main",
        "cegs-imageNet32-pmax",
        "main_lr001",
        "imnet32_pgdat_eps4",
        train_mode="pgd_at",
        robust_eps="4",
        robust_alpha="1",
        robust_steps="10",
    )
)
rows.append(
    R(
        "imageNet",
        60,
        "0.01",
        "True",
        "logk_rms",
        0,
        60,
        "ce",
        "ce",
        "imnet32_trades_eps4_b6",
        "baseline_main",
        "cegs-imageNet32-pmax",
        "main_lr001",
        "imnet32_trades_eps4_b6",
        train_mode="trades",
        robust_eps="4",
        robust_alpha="1",
        robust_steps="10",
        robust_beta="6.0",
    )
)
rows.append(
    R(
        "imageNet",
        60,
        "0.01",
        "True",
        "logk_rms",
        0,
        60,
        "ce",
        "ce",
        "imnet32_mart_eps4_b6",
        "baseline_main",
        "cegs-imageNet32-pmax",
        "main_lr001",
        "imnet32_mart_eps4_b6",
        train_mode="mart",
        robust_eps="4",
        robust_alpha="1",
        robust_steps="10",
        robust_beta="6.0",
    )
)
rows.append(
    R(
        "imageNet",
        60,
        "0.01",
        "True",
        "logk_rms",
        30,
        30,
        "ce",
        "euat",
        "imnet32_euat_s30_ce_s30_euat",
        "baseline_main",
        "cegs-imageNet32-pmax",
        "main_lr001",
        "imnet32_euat_s30_ce_s30_euat",
    )
)
rows.append("# --- imageNet-32: clue_lite = proxy only (below) ---")
rows.append(
    R(
        "imageNet",
        60,
        "0.01",
        "True",
        "logk_rms",
        0,
        60,
        "ce",
        "clue_lite",
        "proxy_imnet32_cluelite_l02",
        "baseline_proxy",
        "cegs-imageNet32-pmax",
        "proxy_cluelite",
        "proxy_imnet32_cluelite_l02",
        clue_lambda="0.2",
        clue_detach_proxy="True",
    )
)

# Filter: comment lines for humans only — parser skips lines that don't match data
data_lines = [ln for ln in rows if not ln.startswith("#")]
comment_lines = [ln for ln in rows if ln.startswith("#")]

text = HEADER + "\n" + "\n".join(comment_lines) + "\n" + "\n".join(data_lines) + "\n"
OUT.write_text(text, encoding="utf-8")
print("Wrote", OUT, "data rows", len(data_lines))
