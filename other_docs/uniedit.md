# UniEdit-Flow 仓库核心实现说明

本文档面向“读代码的人”，目标是快速回答 3 个问题：

1. 这个仓库到底在做什么？
2. 核心算法在代码里是怎么落地的？
3. 如果要改功能，应该从哪些文件开始看？

## 1. 仓库目标与整体结构

`UniEdit-Flow` 提供的是一种面向 flow-based generative models 的“反演 + 编辑”推理策略，特点是：

- 不训练、不微调，只改采样过程。
- 同时支持图像和视频。
- 同时支持不同后端：
  - `UniEdit-Flow_diffusers/`：基于 Hugging Face `diffusers`，适配 FLUX、SD3、SDXL、Wan。
  - `UniEdit-Flow_FLUX/`：基于官方 FLUX 仓库实现，性能通常略好，也保留了更底层的特征注入能力。

从工程角度看，这个仓库的核心不是“重新实现一个生成模型”，而是“在已有模型外面包一层新的采样器 / 采样循环”。

## 2. 统一的算法主链路

无论是 `diffusers` 版还是 `FLUX` 版，主流程都可以概括成下面 5 步：

1. 把输入图像或视频编码到 latent 空间。
2. 用 `Uni-Inv` 把 latent 反推到更靠近噪声端的状态。
3. 把这个“反演噪声 latent”复制成两份。
4. 一份走 source prompt，一份走 target prompt，同时做一次联合编辑。
5. 用 `Uni-Edit` 在采样过程中融合两路 velocity，并把结果解码回像素空间。

对应到代码里，最核心的思想是：

- `Uni-Inv`：不是直接用原始 scheduler 反演，而是换成更适合 flow 的 predictor-corrector 风格逆过程。
- `Uni-Edit`：不是简单做 prompt 替换，而是同时计算 `source` 和 `target` 两路预测，再从两者差异里构造一个“区域感知 mask”，让编辑更多发生在该变的区域，不该变的区域尽量保留原图。

## 3. diffusers 版本：把方法封装成自定义 scheduler

### 3.1 入口文件

`UniEdit-Flow_diffusers/` 的入口非常直接：

- `demo_inv_rec.py`：只做反演与重建。
- `demo_edit.py`：图像编辑主入口。
- `demo_wan_video_edit.py`：视频编辑主入口。

最值得先看的其实是 `demo_edit.py`，因为它把整条链路串得最完整：

1. 根据模型类型选择 pipeline 和 scheduler。
2. 用 `image2latent()` 把输入图像编码进 latent。
3. 把 scheduler 切到 `UniInv*Scheduler` 做反演。
4. 用 `pack_edit_input()` 把反演得到的 latent 复制成两份，并组成 `[src_prompt, trg_prompt]`。
5. 把 scheduler 切到 `UniEdit*Scheduler` 做编辑生成。
6. 最后再解码保存结果。

也就是说，`diffusers` 版的核心设计是“尽量不改 pipeline 主体，只替换 scheduler 行为”。

### 3.2 工具函数层

`UniEdit-Flow_diffusers/utils.py` 主要做 4 件事：

- `image2latent()`：把图像归一化后送进 VAE 编码；如果是 FLUX 这类 sequence latent，还会调用 `_pack_latents()`。
- `latent2image()`：负责把 latent 还原成图像；如果是 packed latent，会先 `_unpack_latents()`。
- `pack_edit_input()`：把单个 latent 复制成双 batch，并把 prompt 组装成 `[src_prompt, trg_prompt]`。
- `encode_video()` / `decode_video()`：负责 Wan 视频 latent 的编码和解码。

这部分是“模型适配层”：上层编辑逻辑并不关心底层是普通 4D latent、视频 5D latent，还是 FLUX 的序列形式，它只依赖统一的 latent 输入输出。

### 3.3 Uni-Inv 的实现

#### Flow 模型：`UniInvEulerScheduler`

文件：`UniEdit-Flow_diffusers/schedulers/UniInvEulerScheduler.py`

`UniInvEulerScheduler` 继承自 `FlowMatchEulerDiscreteScheduler`，做了两件关键修改：

1. 重写 `set_timesteps()`
   - 重新构造从噪声端到数据端的 `sigmas/timesteps`。
   - 支持 `alpha < 1`，只保留一部分反演步数，相当于“提前停止反演”。
   - 支持 `zero_initial`，允许以 0 速度初始化第一步。

2. 重写 `step()`
   - 第一阶段把当前 sample 存到 `self.sample` 中。
   - 当前 velocity 先做一次更新，再用下一步步长做一次外推。
   - 整体上是一个轻量的 predictor-corrector / extrapolation 风格逆更新，而不是直接照搬普通 Euler 反解。

从代码行为看，它的目标是提高从真实图像 latent 回到噪声 latent 时的可逆性和重建精度。

#### DDIM 模型：`UniInvDDIMScheduler`

同文件里的 `UniInvDDIMScheduler` 则是给 SDXL 这类 DDIM 系模型准备的版本。核心思想相同：

- 支持 `alpha` 做部分反演。
- 支持 `zero_initial`。
- 在 DDIM 的闭式更新公式上做适配，让“反演 latent -> 重建”这条链路可用。

### 3.4 Uni-Edit 的实现

#### Flow 模型：`UniEditEulerScheduler`

文件：`UniEdit-Flow_diffusers/schedulers/UniEditEulerScheduler.py`

这是整个 `diffusers` 版最关键的文件。它的编辑逻辑可以概括成下面几步：

1. 把 batch 按 source / target 一分为二：
   - `v_src, v_trg = model_output.chunk(2, dim=0)`

2. 计算两路预测差值：
   - `guidance = v_trg - v_src`

3. 从差值里构造一个归一化 mask：
   - 图像 latent 时按通道求均值；
   - 视频 latent 时按时空维适配；
   - FLUX packed latent 时按 token 维适配。

4. 用这个 mask 融合两路 velocity：
   - `velocity_fusion = mask * v_trg + (1 - mask) * v_src`

5. 再加一项更强的编辑校正：
   - `stride_corr = omega * dt * (1 + mask) * guidance`

6. 最终一步更新：
   - `sample + stride_corr + dt * velocity_fusion`

这就是仓库里“区域感知编辑”的核心来源：

- `mask` 大的位置，说明 source / target 差异大，应该更偏向 target。
- `mask` 小的位置，说明原图与目标差异不大，更偏向 source，保留原内容。

#### DDIM 模型：`UniEditDDIMScheduler`

同文件里的 `UniEditDDIMScheduler` 做的是同一个想法的 DDIM 版本：

- 仍然先算 `v_src`、`v_trg`、`guidance`、`mask`。
- 先在 velocity 层融合出新的 `model_output`。
- 再把它代回 DDIM 的 `pred_original_sample / pred_epsilon / prev_sample` 更新公式。

### 3.5 `alpha` 和 `omega` 在 diffusers 版中的真实作用

- `alpha`
  - 在 `UniInv*Scheduler` 中：控制反演步数保留比例，`alpha < 1` 表示只做前一部分逆过程。
  - 在 `UniEdit*Scheduler` 中：通过裁剪 `timesteps`，实现 delayed editing，只在更靠后的采样阶段真正开始编辑。

- `omega`
  - 控制编辑校正项强度。
  - 越大，`target` 对结果的拉动越强；但同时也更容易破坏非编辑区域。

### 3.6 视频编辑为什么单独做了内存管理

视频版本在 `demo_wan_video_edit.py` 里额外拆开了 `vae` 和 `pipe`，并通过 `memory_management.py` 做 GPU/CPU 来回搬运。

原因很简单：Wan 的视频 latent 更大，直接把所有模块同时常驻 GPU 更容易 OOM。

所以视频链路会在：

- 编码时把 `vae` 放上 GPU。
- 编解码结束后再把 `pipe` 放回 GPU。

这部分不改变算法，只是为了让视频编辑在普通显存环境下更容易跑起来。

## 4. FLUX 原生版本：把方法直接写进采样循环

`UniEdit-Flow_FLUX/` 没有走 `scheduler` 封装，而是直接改官方 FLUX 的采样实现。

### 4.1 总入口：`src/edit.py`

这个文件是 FLUX 版的总调度器，主要做 6 件事：

1. 加载 T5、CLIP、FLUX 主模型和 AE。
2. 读取输入图像，并裁剪到 16 的倍数。
3. 用 `encode()` 把图像编码到 AE latent。
4. 用 `prepare()` 构造图像 token、位置 id、文本 token 和 pooled text vector。
5. 根据 `sampling_strategy` 选择不同采样函数：
   - `reflow`
   - `rf_solver`
   - `fireflow`
   - `rf_midpoint`
   - `uniinv`
   - `uniedit`
6. 先反演得到 `z`，再把 `z` 当成编辑阶段的初始噪声，最后解码保存。

这里和 `diffusers` 版最大的差异是：采样循环完全由仓库自己控制，所以实现会更直白，也更容易插入额外逻辑。

### 4.2 FLUX 输入准备：`prepare()`

文件：`UniEdit-Flow_FLUX/src/flux/sampling.py`

`prepare()` 做的是 FLUX 特有的数据整理：

- 把 `[B, C, H, W]` latent 重排成 token 序列。
- 构造 `img_ids` 作为图像 token 的二维位置编码索引。
- 用 T5 编码文本 token。
- 用 CLIP 编码 pooled text vector。

所以 FLUX 版的本质是“在 token 序列空间里做 flow matching 采样”。

### 4.3 时间步调度：`get_schedule()`

`get_schedule()` 会生成从 `1 -> 0` 的时间序列，并根据序列长度做 shift。

这一步非常重要，因为 FLUX/flow 模型不是传统 DDPM 的离散噪声步，而是连续时间上的 flow 轨迹。仓库通过：

- 额外保留一个到 `0` 的终点；
- 按图像 token 数量调整 shift；

让不同分辨率下的采样稳定性更好。

### 4.4 FLUX 版 Uni-Inv：`denoise_uniinv()`

文件：`UniEdit-Flow_FLUX/src/flux/sampling.py`

`denoise_uniinv()` 的行为非常清晰：

- 如果是正向生成，直接退化成普通 `denoise()`。
- 如果是逆过程：
  - 反转时间步；
  - 先用当前 velocity 做一步预测；
  - 再在下一时刻重新估计 velocity；
  - 用这个“下一时刻 velocity”真正更新图像。

这对应的也是一种 predictor-corrector 风格的逆积分，比单纯的 Euler 反推更稳。

### 4.5 FLUX 版 Uni-Edit：`edit_uniedit()`

这是 FLUX 分支里最核心的函数，逻辑分成两个阶段。

#### 阶段 A：反演阶段

- `step_threshold = round(alpha * total_steps)`
- 逆过程只执行前 `alpha` 比例的步数。
- 第一轮如果开启 `zero_init`，可以直接用全零 velocity 启动。
- 同样采用“预测下一步 velocity 再更新”的二阶式逆过程。

#### 阶段 B：编辑阶段

- 正向过程会跳过前半段，只在后 `alpha` 比例的步数里真正编辑。
- 对同一个 latent，同步跑两次模型：
  - 一次用 `target prompt`
  - 一次用 `source prompt`
- 计算：
  - `cfg_component = pred_trg - pred_src`
  - `save_sub_map = normalize(mean(abs(cfg_component)))`
- 再得到最终 velocity：
  - `fused_v = save_sub_map * pred_trg + (1 - save_sub_map) * pred_src`
  - `pred = fused_v + (save_sub_map + 1) * omega * cfg_component`

可以看到，这和 `diffusers` 版的核心思想完全一致，只是这里直接写在采样循环里，而不是塞进 scheduler。

### 4.6 FLUX 模型主体并不是重点改造对象

`UniEdit-Flow_FLUX/src/flux/model.py` 基本仍是标准 FLUX：

- 图像 token、文本 token、时间嵌入、guidance 嵌入先投影到统一 hidden size。
- 经过 `double_blocks` 做图文双流交互。
- 再经过 `single_blocks` 做单流融合。
- 最后由 `LastLayer` 输出每个图像 token 的 velocity / residual。

也就是说，仓库的创新点主要不在 backbone，而在“怎么组织采样、怎么组织 source/target 双分支的 velocity 融合”。

## 5. FLUX 版额外保留的能力：特征注入

文件：`UniEdit-Flow_FLUX/src/flux/modules/layers.py`

`SingleStreamBlock.forward()` 中保留了一套基于 `info` 字典的特征缓存与复用机制：

- 在逆过程里，可以按层、按时间步保存 `Q/K/V` 或仅保存 `V`。
- 在正向过程里，可以把保存的特征重新注入：
  - `replace_v`
  - `add_v`
  - `replace_k`
  - `add_k`
  - `replace_q`
  - `add_q`

这部分更像是和 RF-Solver / FireFlow 一脉相承的“特征共享编辑”能力。

需要注意的是：

- 默认的 UniEdit-Flow 主逻辑并不依赖它。
- 当 `inject=0` 时，这条路径基本不会真正启用。
- 但它让 FLUX 版保留了更强的研究型可扩展性。

## 6. 关键参数的工程语义

仓库里最值得理解的参数只有 3 个：

- `alpha`
  - 控制“延迟编辑”的比例。
  - 值越小，真正发生编辑的时间越靠后，保真通常更强，但编辑幅度可能变弱。

- `omega`
  - 控制 source/target 差分引导强度。
  - 值越大，编辑更明显，但更容易带来结构漂移或非编辑区域污染。

- `zero_init`
  - 只影响反演起点。
  - 在 FLUX 版 `edit_uniedit()` 和 `diffusers` 版 `UniInv*Scheduler` 中，它决定首步是否用零速度作为初始化。

## 7. 如果要继续读代码，建议顺序

推荐按下面顺序阅读：

1. `UniEdit-Flow_diffusers/demo_edit.py`
2. `UniEdit-Flow_diffusers/utils.py`
3. `UniEdit-Flow_diffusers/schedulers/UniInvEulerScheduler.py`
4. `UniEdit-Flow_diffusers/schedulers/UniEditEulerScheduler.py`
5. `UniEdit-Flow_FLUX/src/edit.py`
6. `UniEdit-Flow_FLUX/src/flux/sampling.py`
7. `UniEdit-Flow_FLUX/src/flux/modules/layers.py`

如果你的目标是：

- 看懂“方法论文如何落地”：先看 `diffusers` 分支。
- 做研究复现或改采样策略：重点看 `FLUX` 分支。
- 做视频编辑：额外看 `demo_wan_video_edit.py` 和 `memory_management.py`。

## 8. 一句话总结

这个仓库的核心实现可以压缩成一句话：

> 它把“反演”和“编辑”都改写成了适用于 flow model 的采样问题，其中 `Uni-Inv` 负责高质量地把真实输入送回噪声轨道，`Uni-Edit` 负责在 source/target 双分支 velocity 的差异上做区域感知融合，从而实现保真且可控的训练外编辑。
