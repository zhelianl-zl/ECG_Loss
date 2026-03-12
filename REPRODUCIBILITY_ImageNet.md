# ImageNet (同 seed=0) 曲线不一致的原因与修复

## 原因说明

在 **seed=0** 固定、且 `set_seed(0)` 已正确设置 `random` / `numpy` / `torch` 的前提下，ImageNet baseline 多次运行曲线仍不同，主要来自两处：

### 1. DataLoader 的 shuffle 未与 seed 绑定

- ImageNet 使用 `DataLoader(..., shuffle=True, num_workers=DL_WORKERS)`（例如 10）。
- 未传入 `generator=torch.Generator().manual_seed(seed)` 时，PyTorch 用**默认**的 global generator 做 `RandomSampler`。
- 默认 generator 的状态取决于**在此之前**所有 torch 操作（模型创建、to(device) 等），不同运行/机器上这些操作的顺序或次数可能略有差异，导致**每个 epoch 的 batch 顺序**在不同 run 间不一致。

### 2. Worker 进程里的增强未用 seed

- `num_workers > 0` 时，**数据加载和 transform 在子进程里执行**。
- `RandomHorizontalFlip()` 等增强用的是**子进程**里的 `random` / `numpy`，而 `set_seed(0)` 只在**主进程**里执行。
- 子进程（spawn）不会继承主进程的 RNG 状态，因此**同一条样本在不同 run 里可能得到不同的增强**，曲线就会不同。

## 代码修改摘要

1. **`_seed_worker(worker_id)`**  
   - 在每个 DataLoader worker 里用 `TRAIN_DATALOADER_SEED + worker_id` 对 `random` 和 `np.random` 做 seed，保证同一 run、同一 worker、同一 index 的增强一致，且不同 run 间可复现。

2. **`main(..., seed=None)`**  
   - 增加参数 `seed`，并在创建 dataset 前设置 `os.environ["TRAIN_DATALOADER_SEED"] = str(seed)`，供 worker 读取（spawn 会继承环境变量）。

3. **`dataset(..., seed=None)`**  
   - 增加参数 `seed`；对 **ImageNet** 分支：
     - 若 `seed is not None`，则为 `train_loader` / `trainAvd_loader` 传入：
       - `generator=torch.Generator().manual_seed(seed + offset)`（train 用 0，adv 用 1，避免两个 loader 共用一个 generator 状态）
       - `worker_init_fn=_seed_worker`
     - 这样**每个 epoch 的 batch 顺序**和**每个 worker 内的增强**都由同一 seed 决定，多次运行一致。

4. **`__main__` 里调用 `main()`**  
   - 增加传入 `getattr(args, "seed", 0)`，保证 TSV/CLI 里的 `seed=0` 会传进 `main` 并用于 DataLoader。

## 你这边需要做的

- 使用**当前代码**重新跑同一配置（例如 `imnet32_ce60_baseline_b64_lr0p01`，seed=0）多次，曲线应一致。
- 若仍不一致，可再检查：
  - 是否启用了 `--deterministic`（可选，进一步收紧 CUDA 确定性）；
  - 运行环境是否一致（同一机器、同一 GPU 驱动/CUDA 版本）。

---

## 复现性检查清单（当前实现）

以下为对 train.py / run_from_tsv 的复现性相关检查结论，便于排查「同配置两次曲线不同」的问题。

### 已做对、有利于复现的部分

1. **set_seed(seed)**（约 5565–5603 行）
   - 设置 `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`。
   - `torch.backends.cudnn.deterministic = True`、`torch.backends.cudnn.benchmark = False`，避免 cuDNN 选不同算法导致差异。
   - `--deterministic` 时关闭 TF32、尝试 `torch.use_deterministic_algorithms(True)`。

2. **ImageNet DataLoader**
   - `_dl_kw_repro(offset)` 为 train_loader / trainAvd_loader 传入 `generator=torch.Generator().manual_seed(seed+offset)` 和 `worker_init_fn=_seed_worker`，保证同 seed 下 batch 顺序一致、多 worker 时增强可复现。

3. **TRAIN_DATALOADER_SEED**
   - `main()` 在创建 dataset 前设置 `os.environ["TRAIN_DATALOADER_SEED"]=str(seed)`，`_seed_worker` 据此对每个 worker 的 `random`/`np.random` 打 seed。

4. **SmallImagenet(shuffle=True)**
   - 构造函数内用 `random.shuffle(list_val)` 打乱样本顺序；此时主进程已执行 `set_seed(seed)`，故该 shuffle 与 seed 一致，同 seed 同顺序。

5. **run_from_tsv**
   - 从 TSV 读 `seed` 并传 `--seed` 给 train.py，主流程会收到相同 seed。

6. **调用顺序**
   - `__main__` 里先 `set_seed(args.seed)`，再调 `main(...)`；`main()` 里先设 `TRAIN_DATALOADER_SEED` 再创建 `dataset(..., seed=seed)`，顺序正确。

### 可能影响复现的因素（需注意）

1. **超参或代码版本不一致**
   - 若两次运行间改了超参（例如 lam_max 从固定 1.5 改为线性 1.5→2.0）、或拉取不同 commit，曲线不同是预期行为。

2. **workers=0 时的增强**
   - workers=0 时增强在主进程执行，使用已由 `set_seed` 设置过的 torch/random 状态；DataLoader 仍用 `generator` 决定 batch 顺序，因此同 seed 下应可复现。若仍遇差异，可优先确认两次运行的代码与超参完全一致。

3. **stage2 内部 DataLoader**
   - 部分 stage2 路径会 `DataLoader(..., shuffle=True)` 且未显式传 `generator`（如 wrong/correct subset 的 loader），其 shuffle 依赖当时全局 RNG 状态；在相同代码路径与 seed 下通常一致，但在极端情况下对 RNG 消耗顺序敏感。若将来要极致复现，可考虑为这些 loader 也传入带 seed 的 generator。

4. **环境与硬件**
   - 不同机器/GPU/驱动/CUDA 版本在理论上可能带来极小数值差异；同一环境、同一 commit、同一 TSV 行下，曲线应基本一致。

5. **可选：更强确定性**
   - 若需进一步收紧：运行前设置环境变量 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，并在命令行加 `--deterministic`（可能略慢或触发少数非确定性 op 的报错）。

### 小结

- 在**同一代码版本、同一 TSV 行（含 seed）、未改超参**的前提下，当前实现已保证：  
  - 主进程与 DataLoader 的 shuffle/增强与 seed 绑定；  
  - cuDNN 确定性、SmallImagenet 内 shuffle 与 seed 一致。  
- 若仍出现「同配置两次曲线不同」，优先核对：代码是否同一 commit、TSV 是否同一行、是否改过 lam_max 等超参；再考虑加 `--deterministic` 或固定运行环境。
