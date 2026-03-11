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
