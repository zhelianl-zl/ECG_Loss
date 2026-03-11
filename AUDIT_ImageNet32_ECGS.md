# ImageNet1k-32 ECGS Implementation Audit

## A. Actual ImageNet execution path

### 1. Dataset branch when `dataset=="imageNet"`

**train.py** (around 1091–1105):

```python
elif dataset_name ==  "imageNet":
    self.num_classes = 1000
    print("[DATA] Using ImageNet 32x32 (SmallImageNet)", flush=True)
    root_dir = os.environ.get('IMAGENET_DS_ROOT', '../data/imageNet/')
    resolution = int(os.environ.get('IMAGENET_RES', '32'))
    classes = int(os.environ.get('IMAGENET_CLASSES', '1000'))
    ...
    self.data_train = SmallImagenet(root=root_dir, size=resolution, train=True, ...)
    self.data_test = SmallImagenet(root=root_dir, size=resolution, train=False, ...)
```

- The only active branch for `dataset_name == "imageNet"` is the one above. The 224x224 ImageFolder branch is commented out (lines 1111–1149).
- So for `dataset=="imageNet"` the code **always** uses the SmallImageNet (pickled 32×32 / 64×64) path, not 224×224 ImageFolder.

### 2. SmallImageNet vs 224×224

- **Always** uses `SmallImagenet` with `root_dir`, `resolution`, and `classes` from env.
- 224×224 is not used unless you uncomment that block and set `IMAGENET_ORIGINAL`/paths accordingly.

### 3. Runtime log lines for SmallImageNet path

If the SmallImageNet path is actually used, you should see **exactly** these lines (order may vary with other logs):

1. `[DATA] Using ImageNet 32x32 (SmallImageNet)`
2. `[DATA] SmallImageNet root: <path>  (resolution=<R>, classes=<C>)`  
   e.g. `(resolution=32, classes=1000)`
3. `[DATA] ImageNet train sample shape: (3, 32, 32)  (expect (3,32,32) for 32x32)`  
   (or `(3, R, R)` if `IMAGENET_RES` is not 32)
4. `[DATA] DataLoader: num_workers=... pin_memory=True persistent_workers=...`

If any of these are missing or the shape is not `(3, 32, 32)` (for 32×32), the path or resolution is wrong.

### 4. Path mismatch risk

- **train.py**: Expects `IMAGENET_DS_ROOT` to point to a directory that **contains** `train_data_batch_1`, ..., `train_data_batch_10`, and `val_data` (see `SmallImagenet` in train.py ~577–578: `filename = os.path.join(self.root, filename)`).
- **smallimagenet.sh**: Extracts into `OUT_DIR="$DATA_DIR/SmallImageNet_${RES}x${RES}"` (e.g. `SmallImageNet_32x32`), and that directory contains `train_data_batch_1`, `val_data`, etc. after unzip.
- **run_from_tsv.py** (322–335): When `dataset.lower() == "imagenet"` and `IMAGENET_DS_ROOT` is unset, it sets:
  - `default_32 = data_root_for_ds / "smallimagenet_32"`
  - `alt_32 = ... / "cegs_data" / "smallimagenet_32"`
  So it looks for **`smallimagenet_32`** (lowercase, no `x32` suffix), while the script creates **`SmallImageNet_32x32`**. If you rely on this default and never set `IMAGENET_DS_ROOT`, the loader can point at a non-existent or wrong directory and fail at first pickle open, or silently use another dir if you happened to create `smallimagenet_32` elsewhere.

**Conclusion:** There is a **naming mismatch**: script → `SmallImageNet_32x32`, run_from_tsv default → `smallimagenet_32`. You must set `IMAGENET_DS_ROOT` explicitly (e.g. to `$DATA_DIR/SmallImageNet_32x32`) or align directory names.

### 5. Environment variables controlling the path

| Variable | Role | Default in train.py | Notes |
|----------|------|---------------------|--------|
| `IMAGENET_DS_ROOT` | Root dir for pickle files (must contain `train_data_batch_*`, `val_data`) | `'../data/imageNet/'` | Wrong or missing → FileNotFoundError or wrong data. |
| `IMAGENET_RES` | Size for SmallImagenet (32 or 64) | `'32'` | Must match the pickles (32→32×32). |
| `IMAGENET_CLASSES` | Number of classes (1000 for full) | `'1000'` | |
| `IMAGENET_ORIGINAL` | 1/true → 224 path (commented out); 0 → 32 path | Not read in active branch | Only relevant if you restore 224 code. |
| `DL_WORKERS` | DataLoader num_workers | `10` when set by run_from_tsv for ImageNet | |

If `IMAGENET_DS_ROOT` is wrong, the process can either fail on first `open(filename, 'rb')` or use another directory that happens to contain the same filenames; so the only way to be sure is to check the three log lines above and the reported `root` and shape.

---

## B. Core ECGS math implementation

### 1. ScaleGrad.forward returns logits unchanged

**ecg_loss.py** (5–9):

```python
@staticmethod
def forward(ctx, x, scale):
    ctx.save_for_backward(scale)
    return x
```

- Forward returns `x` (logits) unchanged. **Confirmed.**

### 2. Change only in backward

**ecg_loss.py** (11–14):

```python
@staticmethod
def backward(ctx, grad_output):
    scale, = ctx.saved_tensors
    return grad_output * scale, None
```

- Backward returns `grad_output * scale` (per-element; `scale` is `[B,1]`, so broadcasts over features). **Confirmed:** only gradients are scaled; forward logits unchanged.

### 3. Scale formula and gate

**ecg_loss.py** (47–55):

```python
wrong_gate = 1.0 - py
if detach_gates:
    wrong_gate = wrong_gate.detach()
    conf_gate = conf_gate.detach()
gate = wrong_gate * conf_gate
scale = (1.0 + lam * gate).view(-1, 1)
```

- `gate = wrong_gate * conf_gate`; `scale = 1 + lam * gate`. **Confirmed.** No formula bug here.

### 4. Forward CE vs backward reweighting

- Forward: `loss = F.cross_entropy(scaled_logits, targets)` where `scaled_logits = ScaleGrad.apply(logits, scale)` and `forward` returns logits unchanged, so the **forward** cross-entropy is computed on the **same** logits as CE; the **backward** is the only place where gradients are scaled by `scale`. So forward CE is unchanged; only backward contribution is reweighted. **Confirmed.**

### 5. scale_normalize

**ecg_loss.py** (56–57, 74–83):

```python
if scale_normalize:
    scale = scale / (scale.mean().detach().clamp_min(1e-8))
...
if scale_normalize:
    stats["scale_std_after_norm"] = ...
    s_flat = scale.detach().float().view(-1)
    stats["scale_p99_after_norm"] = torch.quantile(s_flat, 0.99).item()
```

- After normalization, `scale.mean()` is 1 (within `clamp_min`), so global step size is not inflated. **Confirmed.**

### 6. Other potential bugs

- **Forward/backward:** Correct: forward identity, backward scales grad by scale.
- **Scale applied to wrong tensor:** Scale is applied to `logits` in `ScaleGrad.apply(logits, scale)`, and backward multiplies `grad_output` (w.r.t. logits) by scale. Correct.
- **Logits shape:** `scale` is `(B, 1)`, `grad_output` is `(B, C)`; `grad_output * scale` broadcasts. Correct.
- **Detached gates:** `wrong_gate` and `conf_gate` are detached when `detach_gates=True`, so gate doesn’t get gradients; only `lam` (and scale) affect the backward. Intended. No bug.
- **pmax:** `conf = p.max(dim=1).values`; **1-pe:** `pe_norm = pe / math.log(C)`, `conf = 1.0 - pe_norm`. Both are standard. **quantile:** `tau = torch.quantile(conf.detach(), float(tau_quantile))`. Correct.

### 7. Verdict on math

**The ECGS math implementation in ecg_loss.py is correct.** No formula bug, no forward/backward mismatch, no wrong tensor or shape, and scale_normalize behaves as intended.

---

## C. What your “bad” ImageNet run is actually doing

Run name pattern:  
`imageNet_s1-60-ecg_s2-0-ecg_..._lamauto0.12_tauq0.85_k20-20`

From **train.py** naming (schedule none branch, 5710–5722):

- `lam_disp = f"lam_{_ls_lower}{args.ecg_lam_end or '0.05'}"` when `_ls_lower in ("auto", "auto_w", "auto_d", "auto_dw")` → so `lamauto0.12` ⇒ **ecg_lam_start = "auto"**, **ecg_lam_end = 0.12**.
- `tau_disp = f"tauq{args.ecg_tau_end or '0.8'}"` when `_ts_lower in ("quantile", "q")` → so `tauq0.85` ⇒ **ecg_tau_start = "quantile" (or "q")**, **ecg_tau_end = 0.85**.
- `k20-20` ⇒ **ecg_k_start = 20**, **ecg_k_end = 20** (fixed k=20).

Parsing (main(), 5234–5252):

- **Lam:** `_lam_rule in ("auto", "auto_w", "auto_d", "auto_dw")` → `ecg_lam_rule = "auto"`, `ecg_lam_delta = 0.12` (ecg_lam_end becomes delta).
- **Tau:** `_tau_start in ("quantile", "q")` → `ecg_tau_rule = "quantile"`, `ecg_tau_quantile = 0.85`. No "auto_q" branch.
- **Stage:** `s1-60-ecg`, `s2-0-ecg` ⇒ stage1_epochs=60, loss_stage1=ecg, stage2_epochs=0, loss_stage2=ecg.

So:

1. **Full ECG from epoch 1** – Yes (stage1 uses ecg, stage2_epochs=0).
2. **stage2_epochs = 0** – Yes.
3. **Not two-stage “CE then ECG”** – Correct; it’s single-stage full ECG.
4. **Lam is auto-lambda** – Yes (`ecg_lam_rule = "auto"`).
5. **Tau is fixed quantile q=0.85** – Yes (`ecg_tau_rule = "quantile"`, `ecg_tau_quantile = 0.85`).
6. **NOT auto_q** – Correct (tau is fixed quantile, not scheduled).
7. **NOT auto_dw** – Correct (lam is "auto", not "auto_d" or "auto_dw"); so no delta warmup and no auto-delta.
8. **k fixed at 20** – Yes.

**CLI / schedule summary**

- **quantile vs auto_q:** `ecg_tau_start` in `("quantile", "q")` → fixed q (ecg_tau_end = q). `ecg_tau_start == "auto_q"` → scheduled q (ecg_tau_end = q_start, q_end=0.9).
- **auto / auto_w / auto_d / auto_dw:** `ecg_lam_start` in these four → auto-lambda; `ecg_lam_end` is then **delta** (or initial_delta for auto_d/auto_dw), not a literal “end lam” for interpolation.

---

## D. Is the ImageNet run too aggressive by design?

### 1. Full ECG from epoch 1

- Your run uses ECG from epoch 1 with no CE pretrain. For 1000-class 32×32, early epochs are very noisy; full ECG from the start can be more aggressive than a “CE then ECG” setup.

### 2. Auto-lambda first batches: lam_cur = lam_max

**train.py** (3105–3108):

```python
if gate_ema is None:
    lam_cur = lam_max
else:
    lam_cur = min(lam_max, max(0.0, delta_eff / (gate_ema + eps)))
```

- On the **very first batch**, `_ecg_gate_ema` is None, so **lam_cur = lam_max** (default 1.5). So the first batch (and until gate_ema is set) uses the **maximum** lambda. That is very strong scaling and can make the first ECG-active batches too aggressive on ImageNet1k-32, especially with many classes and high initial error.

### 3. Fixed high quantile tau (0.85)

- q=0.85 means tau is the 85th percentile of confidence each batch, so only the top ~15% of samples (by confidence) get low gate; the rest get strong scaling. Early in training, confidence is low and volatile, so this can be harsh. auto_q (e.g. q_start 0.6 → 0.9) would start gentler.

### 4. No delta warmup

- With `ecg_lam_start = "auto"` (not "auto_w" or "auto_dw"), there is no 5-epoch warmup on delta. So from epoch 1 you use full delta=0.12, which can be aggressive for a high-class dataset.

### 5. Gate sparsity / instability

- With k=20 and q=0.85, conf_gate is very steep (sigmoid(20*(conf-tau))). Small changes in conf or tau make the gate flip; early on this can be unstable. Combined with lam_max on the first batch, the first few steps can be extremely aggressive.

### 6. Normalization

- scale_normalize keeps mean(scale)=1, so it doesn’t remove the **redistribution** of gradient: tail samples still get much larger effective learning rate. So control strength (high lam, high q, no warmup) is not “cancelled” by normalization.

### auto_q progress (for reference)

- **q_start:** from `ecg_tau_end` when `ecg_tau_start == "auto_q"` (main(), ~5227).
- **q_end:** fixed 0.9 (main(), ~5228).
- **Progress t:**  
  - Two-stage (ECG only in stage2): `t = (global_epoch - (s1+1)) / max(1, s2-1)` (train.py 1585–1587).  
  - Full-ECG (stage2_epochs=0): `t = (global_epoch - 1) / max(1, total-1)` (1589–1591).  
So “ECG-active-stage-only” scheduling is implemented as intended for auto_q; your current run simply doesn’t use auto_q.

---

## E. Current run vs “final” automated ECGS design

You don’t have Attachment_3 in the repo; inferring “final” design from code and your description:

**Intended (typical “final” design):**

- Fixed k.
- Dynamic tau (e.g. auto_q: wide early, narrow late).
- Automated lam (e.g. auto or auto_d/auto_dw).
- Optional auto_dw for delta warmup.
- More conservative for high-class datasets (warmup, gentler start).

**Your current ImageNet run:**

- Fixed k=20.
- **Fixed** tau q=0.85 (not dynamic, not auto_q).
- Auto lam with **fixed** delta=0.12, **no** warmup (not auto_w / auto_dw).
- Full ECG from epoch 1, **no** CE pretrain.
- First-batch lam = lam_max (1.5).

So the current run **does not** match the final intended design: it is a more aggressive ablation (fixed high q, no warmup, no auto_q, no auto_dw, and first-batch lam_max). If the “final” design is “auto_q + auto_dw + conservative high-class behavior”, this run explicitly deviates from it.

---

## F. W&B metrics to inspect for over-aggressive ECG on ImageNet1k-32

From **train.py** and **ecg_loss.py**, these are the main diagnostics. “Bad” = consistent with overly strong or unstable scaling.

| Metric | Source | “Bad” behavior |
|--------|--------|----------------|
| **ECG/gate_mean** | ecg_loss stats → epoch avg | Very high early (e.g. >0.5) and/or big jumps epoch-to-epoch. |
| **ECG/conf_gate_active_frac** | stats `conf_gate_active_frac` | Near 0 (gate almost never >0.5) or near 1; or large swings. |
| **ECG/scale_mean** | stats `scale_mean` (after norm) | Should be ~1.0 when scale_normalize=True; if not, normalization or logging bug. |
| **ECG/scale_std_after_norm** | stats when scale_normalize | Very large (e.g. >0.5) → heavy tail of scale; some samples scaled up a lot. |
| **ECG/scale_p99_after_norm** | stats when scale_normalize | High (e.g. >1.5–2) → strongest 1% of samples get much larger effective LR. |
| **ECG/tau_q_cur** | Only when using **auto_q** | N/A for your current run (fixed q). |
| **ECG/lam_auto** | Epoch begin log (auto modes) | Stuck at 1.5 early (lam_max) or very high for many epochs. |
| **ECG/gate_ema** | Epoch begin (auto modes) | Very low (e.g. &lt;0.01) while lam_auto is high → very strong effective scaling. |
| **ECG/delta_cur**, **ECG/delta_eff**, **ECG/scale_p99_ema** | Only for **auto_d / auto_dw** | N/A for your run (you use "auto"). |

So for your **current** run, focus on: **ECG/gate_mean**, **ECG/conf_gate_active_frac**, **ECG/scale_std_after_norm**, **ECG/scale_p99_after_norm**, **ECG/lam_auto**, **ECG/gate_ema**. If the first epochs show lam_auto = 1.5, very low gate_ema, and high scale_p99_after_norm / scale_std_after_norm, that confirms an aggressive start.

---

## G. Final verdict

### 1. Is there an implementation bug in the ECG gradient-scaling core?

**No.** The core in **ecg_loss.py** (ScaleGrad, scale = 1 + lam*gate, scale_normalize, pmax/1-pe, tau_quantile) is correct. No forward/backward mismatch, no wrong tensor, no shape bug.

### 2. Is the current ImageNet issue more likely caused by experimental logic / control strength being too aggressive?

**Yes.** The run is configured in an aggressive way: full ECG from epoch 1, auto-lambda with **lam = lam_max on the first batch**, fixed high q=0.85, no delta warmup, no auto_q. That is enough to explain bad or unstable behavior on ImageNet1k-32 without any core bug.

### 3. Top 3 most likely causes (ranked)

1. **First-batch (and early) lam = lam_max (1.5)** when `gate_ema is None`, making the very first ECG steps extremely strong and potentially destabilizing training.
2. **Full ECG from epoch 1** with no CE pretrain and **fixed high tau quantile (0.85)** on a 1000-class, 32×32 setup, leading to harsh and volatile gating early on.
3. **Path / env risk:** `IMAGENET_DS_ROOT` default in run_from_tsv (`smallimagenet_32`) does not match smallimagenet.sh output (`SmallImageNet_32x32`). If that was ever used or mis-set, wrong or missing data could contribute; but the primary explanation for “ECG too strong” is (1) and (2).

---

*Audit based on train.py, ecg_loss.py, run_from_tsv.py, smallimagenet.sh, submit_array.sh. No Attachment_1/2/3 found in repo.*
