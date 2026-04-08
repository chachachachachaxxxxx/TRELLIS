# latent_replace_union 显存爆炸分析

## 问题现象

在 UniEdit 的 `latent_replace_union` 消融测试中，Stage 0b（SLAT inversion with trajectory caching）在第 2 步（8%）就发生 OOM：

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. 
GPU 0 has a total capacity of 31.36 GiB of which 18.12 MiB is free. 
Including non-PyTorch memory, this process has 31.33 GiB memory in use.
Of the allocated memory 27.89 GiB is allocated by PyTorch, 
and 2.84 GiB is reserved by PyTorch but unallocated.
```

**关键数据**:
- GPU 总容量: 31.36 GiB
- 已使用: 31.33 GiB (99.9%)
- PyTorch 分配: 27.89 GiB
- PyTorch 保留: 2.84 GiB
- 失败步骤: 2/25 (8%)

## 根本原因分析

### 1. 累积的显存占用

在 `latent_replace_union` 之前，已经完成了：

1. **Stage 0 - SS Inversion** (~12s, 25 steps)
   - 输入: source voxels (7125 voxels)
   - 输出: ss_terminal_noise
   - 显存占用: ~2-3 GB

2. **Stage 0 - SLAT Inversion (常规)** (~15s, 25 steps)
   - 输入: source SLAT features
   - 输出: slat_terminal_noise
   - 显存占用: ~3-4 GB

3. **Pipeline 和模型加载**
   - TRELLIS-image-large 模型
   - Sparse structure flow model
   - SLAT flow model
   - 显存占用: ~15-20 GB

**累积显存**: ~20-27 GB (已经接近上限)

### 2. latent_replace_union 的额外开销

#### 问题代码 (editing/inversion/latent_replace_sampler.py:69-103)

```python
def invert_with_cache(self, model, sample, cond_dict, steps, ...):
    latent_cache = {}
    t_pairs = build_rf_t_pairs(steps=steps, rescale_t=rescale_t, inverse=True)
    
    for t_curr, t_next in tqdm(t_pairs, ...):
        # 1. 执行一步 inversion
        sample = self.sample_once(model, sample, t_curr, t_next, ...)
        
        # 2. 缓存到 CPU
        latent_cache[_time_key(t_next)] = sample.detach().cpu()
    
    return sample, latent_cache
```

#### 显存爆炸的原因

**原因 1: sample_once 的中间状态**

`sample_once` 调用链：
```
sample_once()
  └─> _guided_prediction()  # 计算 CFG
       ├─> _run_model(cond)      # 条件预测
       ├─> _run_model(neg_cond)  # 负条件预测
       └─> CFG 融合
```

每次 `sample_once` 会创建：
- 条件预测结果 (SparseTensor)
- 负条件预测结果 (SparseTensor)
- CFG 融合结果 (SparseTensor)
- 中间计算的梯度和激活

**单步显存**: ~2-3 GB

**原因 2: SparseTensor.cpu() 的延迟释放**

```python
latent_cache[_time_key(t_next)] = sample.detach().cpu()
```

问题：
1. `sample.detach()` 创建新的 tensor（GPU 上）
2. `.cpu()` 复制到 CPU，但 GPU tensor 可能未立即释放
3. Python GC 延迟，GPU 内存累积

**原因 3: 没有显式清理**

代码中缺少：
- `torch.cuda.empty_cache()`
- `del` 中间变量
- `gc.collect()`

**原因 4: 模型前向传播的激活缓存**

SLAT flow model 的前向传播会缓存：
- Attention 中间结果
- Transformer block 激活
- Sparse convolution 中间状态

这些在 `sample_once` 中累积，没有及时清理。

### 3. 为什么其他两个方法不会 OOM？

#### preserve_uniedit 和 free_target

```python
# 只做一次常规 SLAT inversion，不缓存轨迹
slat_terminal_noise = invert_slat(
    pipeline=pipeline,
    cond_src={"cond": source_cond, "neg_cond": neg_cond},
    slat_src=source_slat,
    params=slat_params,
    cfg_interval=cfg_interval,
    verbose=True,
)
```

- 每步完成后，中间状态被自动释放
- 不需要额外的缓存空间
- 显存峰值: ~27 GB (可以接受)

#### latent_replace_union

```python
# 需要缓存所有 25 步的中间 latent
slat_terminal_noise, slat_latent_cache = self._invert_slat_with_cache(...)
```

- 每步的 latent 都要保存（即使在 CPU）
- GPU 上的中间状态累积
- 显存峰值: >31 GB (超出容量)

## 显存占用估算

### SLAT SparseTensor 大小

假设 SLAT 有 N 个体素，每个体素 C 个通道：

```python
# 典型值
N = 7125 voxels
C = 8 channels (SLAT feature dimension)
dtype = float32 (4 bytes)

# 单个 SLAT tensor
coords: (N, 4) * 4 bytes = 7125 * 4 * 4 = 114 KB
feats: (N, C) * 4 bytes = 7125 * 8 * 4 = 228 KB
total per tensor: ~342 KB
```

### 缓存 25 步的开销

```python
# CPU 缓存 (可接受)
25 steps * 342 KB = 8.55 MB (CPU)

# 但 GPU 上的问题：
# 每步 sample_once 的峰值显存
model_forward: ~2 GB
CFG (cond + neg_cond): ~2 GB
intermediate activations: ~1 GB
total per step: ~5 GB

# 如果没有及时释放
accumulated: 5 GB * 2 steps = 10 GB (额外)
baseline: 27 GB
total: 37 GB > 31 GB (OOM!)
```

## 解决方案

### 方案 1: 激进的显存清理 (推荐)

```python
def invert_with_cache(self, model, sample, cond_dict, steps, ...):
    import gc
    latent_cache = {}
    t_pairs = build_rf_t_pairs(steps=steps, rescale_t=rescale_t, inverse=True)
    
    for t_curr, t_next in tqdm(t_pairs, ...):
        # 执行一步
        sample = self.sample_once(model, sample, t_curr, t_next, ...)
        
        # 立即缓存并清理
        with torch.no_grad():
            cached = sample.detach()
            latent_cache[_time_key(t_next)] = cached.cpu()
            del cached
        
        # 强制清理 GPU 显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 每 5 步做一次 Python GC
        if (len(latent_cache) % 5) == 0:
            gc.collect()
    
    return sample, latent_cache
```

**优点**: 
- 最小化显存占用
- 不改变算法逻辑

**缺点**: 
- 频繁的 `empty_cache()` 可能影响性能
- 增加 ~10-20% 运行时间

### 方案 2: 梯度检查点 (Gradient Checkpointing)

```python
def sample_once_with_checkpoint(self, model, sample, t_curr, t_next, ...):
    """使用梯度检查点减少显存."""
    from torch.utils.checkpoint import checkpoint
    
    def forward_fn(s):
        return self._guided_prediction(model, s, t_mid, cond_dict, ...)
    
    # 使用 checkpoint 减少中间激活的显存
    pred_mid = checkpoint(forward_fn, sample_mid, use_reentrant=False)
    ...
```

**优点**: 
- 显著减少激活缓存
- 可节省 30-50% 显存

**缺点**: 
- 需要重新计算，增加 20-30% 时间
- 需要修改 model forward

### 方案 3: 分块缓存 (Chunked Caching)

```python
def invert_with_chunked_cache(self, model, sample, steps, chunk_size=5, ...):
    """分块缓存，每次只保留部分轨迹."""
    full_cache = {}
    
    for chunk_start in range(0, steps, chunk_size):
        chunk_end = min(chunk_start + chunk_size, steps)
        chunk_cache = self._invert_chunk(model, sample, chunk_start, chunk_end, ...)
        
        # 将 chunk 缓存保存到磁盘
        torch.save(chunk_cache, f"cache_chunk_{chunk_start}.pt")
        full_cache.update(chunk_cache)
        
        # 清理 GPU
        del chunk_cache
        torch.cuda.empty_cache()
    
    return sample, full_cache
```

**优点**: 
- 可以处理任意长的轨迹
- 显存占用恒定

**缺点**: 
- 需要磁盘 I/O
- 实现复杂

### 方案 4: 降低精度 (Mixed Precision)

```python
def invert_with_cache(self, model, sample, ...):
    latent_cache = {}
    
    # 使用 autocast 减少显存
    with torch.cuda.amp.autocast(dtype=torch.float16):
        for t_curr, t_next in tqdm(t_pairs, ...):
            sample = self.sample_once(model, sample, t_curr, t_next, ...)
            
            # 缓存为 float16
            latent_cache[_time_key(t_next)] = sample.detach().half().cpu()
    
    return sample, latent_cache
```

**优点**: 
- 减少 50% 显存
- 几乎不影响质量

**缺点**: 
- 需要模型支持 mixed precision
- 可能影响数值稳定性

### 方案 5: 使用更大的 GPU

**最简单的方案**: 使用 40GB 或 80GB GPU

- A100 40GB: 足够
- A100 80GB: 绰绰有余
- H100 80GB: 最佳

## 推荐实施顺序

1. **立即实施**: 方案 1 (激进清理) - 最简单，可能解决问题
2. **如果不够**: 方案 4 (混合精度) - 效果好，改动小
3. **如果还不够**: 方案 2 (梯度检查点) - 需要更多工作
4. **终极方案**: 方案 5 (更大 GPU) - 最可靠

## 性能对比

| 方案 | 显存节省 | 时间开销 | 实现难度 | 推荐度 |
|------|---------|---------|---------|--------|
| 激进清理 | ~2-3 GB | +10-20% | 低 | ⭐⭐⭐⭐⭐ |
| 混合精度 | ~50% | +5% | 低 | ⭐⭐⭐⭐ |
| 梯度检查点 | ~40% | +20-30% | 中 | ⭐⭐⭐ |
| 分块缓存 | ~70% | +30-50% | 高 | ⭐⭐ |
| 更大 GPU | 100% | 0% | N/A | ⭐⭐⭐⭐⭐ |

## 总结

**核心问题**: latent_replace_union 需要缓存 25 步的中间 latent，虽然缓存在 CPU，但 GPU 上的中间状态没有及时清理，导致显存累积超过 31GB。

**最佳方案**: 
1. 先尝试激进清理（方案 1）
2. 如果不够，加上混合精度（方案 4）
3. 如果还不够，考虑使用 40GB+ GPU

**预期效果**: 
- 方案 1: 可能降到 ~28-29 GB（边缘）
- 方案 1 + 4: 可能降到 ~20-22 GB（安全）
