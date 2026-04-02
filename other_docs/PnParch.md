# PnPInversion 架构笔记：Inversion 与 Attention 的抽象

> 用途：给以后自己看代码时快速回忆这个仓库的主结构，尤其是 `inversion` 抽象为什么做得好，以及 attention 控制这一侧是怎么和 inversion 解耦组合的。
>
> 说明：你提到的“爱丽丝方法”，仓库里没有一个明确叫 `Alice/ALICE` 的模块。这里按“attention 控制这一侧的方法”来理解，也就是代码里对应的 Prompt-to-Prompt controller、MasaCtrl、PnP 这类 attention/feature 注入逻辑。

## 1. 先说结论

这个项目最值得记住的，不是“它支持很多编辑方法”，而是它把编辑问题拆成了两个相对独立的维度：

1. `inversion` 负责把真实图像放回扩散轨迹里，并产出一个“前向去噪时可消费的校正物”。
2. `attention/editing` 负责在去噪过程中控制“目标 prompt 到底怎么改、改多少、改哪些区域”。

这两个维度被拆开以后，组合关系就很清楚：

```text
image + source prompt
    -> inversion
    -> noisy latent x_T + per-step auxiliary signals
    -> editing forward loop
    -> attention/controller hooks 决定编辑行为
    -> edited image
```

也就是说：

- `inversion` 决定“能不能稳地回到原图附近”
- `attention controller` 决定“如何在保结构的同时改语义”

这正是这个仓库里最好的抽象点。

## 2. 项目整体结构

### 2.1 顶层职责

- `run_editing_*.py`
  - 每个脚本对应一种论文方法或一类 pipeline。
  - 负责 benchmark 数据遍历、参数组织、方法名到实现的映射。
- `models/`
  - 真正的方法实现都在这里。
  - 其中 `models/p2p_*` 这一组是全仓库里抽象最清楚的一块。
- `utils/utils.py`
  - 图像和 latent 的常用转换、加载等工具。

### 2.2 最清楚的一条主线

如果要理解仓库设计，优先看这条链：

```text
run_editing_p2p.py
    -> models/p2p_editor.py
        -> models/p2p/inversion.py
        -> models/p2p/p2p_guidance_forward.py
        -> models/p2p/attention_control.py
```

原因是：

- `run_editing_p2p.py` 是最清晰的入口
- `P2PEditor` 像一个 facade，把不同 inversion 和不同 forward/controller 拼起来
- `inversion.py` 把“反演”统一成一类对象
- `attention_control.py` 把“attention 控制”统一成一类 controller
- `p2p_guidance_forward.py` 则是消费这两者的执行层

## 3. 从入口看调用链

### 3.1 runner 层

`run_editing_p2p.py` 做的事很简单：

- 解析 `edit_method_list`
- 读取 benchmark 的每条编辑数据
- 构造 `P2PEditor`
- 对每张图调用 `p2p_editor(edit_method=..., prompt_src=..., prompt_tar=...)`

这里的重要点是：`edit_method` 只是一个字符串分发键，不直接参与算法细节。

### 3.2 editor 层

`models/p2p_editor.py` 是项目里非常关键的一层。它做了两件事：

1. 把“不同 inversion 方法”统一封到一个编辑器类里。
2. 把“重建”和“编辑”统一成相同的执行骨架。

典型流程几乎都是：

```python
inversion = SomeInversion(...)
_, _, x_stars, aux = inversion.invert(...)
x_t = x_stars[-1]

controller = AttentionStore()   # 先做 reconstruction
reconstruct_latent, _ = some_forward(..., latent=x_t, aux=aux)

controller = make_controller(...)   # 再做 editing
latents, _ = some_forward(..., latent=x_t, aux=aux)
```

这个模式说明：

- `x_stars[-1]` 是统一的“噪声起点”
- `aux` 是 inversion 输出给 forward 的“额外校正信息”
- controller 是独立注入的，不和 inversion 写死耦合

## 4. inversion 抽象为什么好

## 4.1 统一骨架

`models/p2p/inversion.py` 里虽然有多个类：

- `NegativePromptInversion`
- `NullInversion`
- `DirectInversion`

但它们共享一套非常稳定的骨架：

- `init_prompt`
- `ddim_loop`
- `ddim_inversion`
- `invert`

也就是说，它们都在做同一件大事：

```text
image -> latent z_0 -> DDIM inversion -> latent trajectory [z_0 ... z_T]
```

差别不在“是否反演”，而在“反演之后给前向去噪留下什么补偿信号”。

这就是抽象做得好的地方。

## 4.2 统一产物：轨迹 + 辅助量

这几个 inversion 类本质上都输出两种东西：

1. `x_stars` / `ddim_latents`
   - 表示整条反演轨迹
   - 最后一个元素 `x_stars[-1]` 是 forward 的起点
2. 一个“每步可消费的辅助量”
   - NPI / NTI：`uncond_embeddings`
   - DirectInversion：`noise_loss_list`

可以把它理解成下面这个统一接口：

```text
invert(image, prompt, ...)
    -> start latent
    -> inversion trajectory
    -> per-step correction package
```

于是 forward 层根本不需要知道“你是怎么反演出来的”，它只需要知道：

- 从哪里开始采样
- 每一步要额外喂什么

这就是典型的“生产者-消费者”式边界。

## 4.3 三种 inversion 的区别，不在主干，在“校正物”

### 4.3.1 Negative Prompt Inversion

`NegativePromptInversion` 不做优化，只是把 `cond embedding` 伪装成 `uncond embedding`。

它输出的校正物是：

- `uncond_embeddings`

forward 阶段直接把这组 embedding 当作 CFG 的负分支输入。

所以它的本质是：

```text
不改 latent 轨迹
改的是 CFG 里的 unconditional branch
```

### 4.3.2 Null-Text Inversion

`NullInversion` 也是围绕 CFG 做文章，但它更进一步：

- 对每个时间步优化一份 `uncond embedding`
- 目标是让 forward 轨迹尽可能贴合 inversion 轨迹

它的校正物仍然是：

- `uncond_embeddings`

只是这次是“逐步优化得到的”。

所以它和 NPI 的 forward 接口几乎一样，只是产生辅助量的方法更贵、更精确。

### 4.3.3 Direct Inversion

`DirectInversion` 是这个仓库里最漂亮的抽象点。

它没有继续优化 text embedding，而是直接承认一件事：

- CFG 前向去噪会和 DDIM inversion 轨迹出现偏差

于是它把问题改写成：

```text
不要去改条件
直接把这个“偏差”算出来，然后在 forward 时补回去
```

这就是 `noise_loss_list` 的含义：

- 每个时间步一个 offset
- 代表“CFG 预测出来的 x_{t-1}”和“反演轨迹里的真 x_{t-1}”之间的差

所以 DirectInversion 输出的不是 embedding schedule，而是 latent offset schedule。

这件事非常关键，因为它把反演的抽象从“条件空间修正”提升成了“状态空间修正”。

## 4.4 DirectInversion 为什么特别适合作为通用抽象

DirectInversion 的好处有三层。

### 4.4.1 它只负责提供 offset，不干涉编辑策略

`offset_calculate` 负责预先算好每一步的补偿量，forward 只要在采样后加上去：

```python
latents = model.scheduler.step(... )["prev_sample"]
latents = torch.concat((latents[:1] + noise_loss[:1], latents[1:]))
```

注意这里的设计重点：

- offset 是 per-step 的
- offset 作用在 latent 上
- 它不关心当前用的是 P2P、MasaCtrl 还是别的 attention 方法

也就是说，DirectInversion 输出的是一种“与编辑器无关”的修正信号。

### 4.4.2 它把 source branch 和 target branch 明确拆开

这是论文思想在代码里的核心落点。

在 `DirectInversion.ddim_loop` 里，虽然 `prompt` 可能传进来的是 `[prompt_src, prompt_tar]`，但真正做 inversion 时只拿 source 条件：

- `cond_embeddings = cond_embeddings[[0]]`

也就是说：

- inversion 只对 source branch 负责
- target branch 不应该反过来污染 inversion

随后在 forward 里，默认 offset 也只加到 source branch：

- `latents[:1] + noise_loss[:1]`
- `latents[1:]` 保持不变

这就把职责切得非常干净：

- source branch：负责内容保真、结构回归
- target branch：负责按新 prompt 编辑

这个边界不是“代码上碰巧这样写”，而是整个方法成立的核心。

### 4.4.3 仓库里的 ablation 正好证明这条边界是对的

`models/p2p_editor.py` 里专门保留了几种 ablation：

- `ablation_directinversion_add-target+p2p`
- `ablation_directinversion_add-source+p2p`
- `ablation_null-text-inversion_single_branch+p2p`

这几项的存在很有说明性：

- 作者不是只把方法“写出来”
- 而是把“边界为什么要这样切”也做成了显式实验开关

所以以后看这个仓库，要记住一句话：

> DirectInversion 最核心的抽象不是 `noise_loss_list` 本身，而是“这个 offset 默认只修 source，不修 target”。

## 5. attention 这一侧为什么也抽象得好

## 5.1 attention 不是写死在采样循环里的

`models/p2p/attention_control.py` 里最重要的入口是：

- `register_attention_control(model, controller)`

它做的事情是 monkey patch UNet 的 attention forward，把 attention map 交给 controller：

```python
attn = sim.softmax(dim=-1)
attn = controller(attn, is_cross, place_in_unet)
out = ...
```

这意味着：

- 采样循环本身不需要知道 controller 的具体策略
- controller 只需要遵守统一调用协议

这就是一个很标准的 hook 式抽象。

## 5.2 controller 层次也分得很清楚

### 第一层：空实现和记录器

- `EmptyControl`
  - 什么都不改
- `AttentionStore`
  - 只记录 attention，不改动

这层主要服务于：

- reconstruction 阶段统计注意力
- 后续 local blend 或分析

### 第二层：可编辑 controller 的统一父类

- `AttentionControlEdit`

这个类非常关键，因为它把编辑 controller 的公共逻辑抽出来了：

- 什么时候替换 self-attention
- 什么时候替换 cross-attention
- 如何按 batch 切出 `base` 分支和 `replace` 分支
- 如何在 step 之间累计状态

换句话说：

- 子类只需要回答“怎么替换”
- 父类负责“何时替换、替换哪个分支、替换流程怎么走”

这就是好的面向对象边界。

### 第三层：具体策略类

- `AttentionReplace`
  - 直接替换 cross-attention
- `AttentionRefine`
  - 按 token 对齐结果做渐进式融合
- `AttentionReweight`
  - 对某些词的注意力做 reweight
- `LocalBlend`
  - 用 cross-attention 生成空间 mask，把编辑约束在局部区域

这些类的关系很清楚：

- Replace / Refine / Reweight 决定“token 级别怎么改”
- LocalBlend 决定“空间上改哪里”

于是 token 控制和空间控制也被拆开了。

## 5.3 `make_controller` 是一个很实用的装配层

`make_controller(...)` 把各种组合规则封装起来：

- 是否用 replace 还是 refine
- 是否叠加 local blend
- 是否叠加 equalizer reweight

所以外层编辑器只要说：

```python
controller = make_controller(...)
```

而不用自己关心 controller 之间怎么嵌套。

这也是这个仓库 attention 侧做得很好的地方：

- 不是只有类
- 还有一层明确的装配工厂

## 6. inversion 和 attention 是怎么优雅组合起来的

这是全仓库最值得记住的一点。

## 6.1 在 P2P 这条链里，两者是正交的

以 `edit_image_directinversion` 为例，流程是：

1. `DirectInversion.invert(...)`
   - 产出 `x_stars`
   - 产出 `noise_loss_list`
2. `direct_inversion_p2p_guidance_forward(...)`
   - 消费 `x_t = x_stars[-1]`
   - 每一步消费 `noise_loss_list[i]`
3. `make_controller(...)`
   - 决定 attention 怎么改
4. `register_attention_control(...)`
   - 在 UNet attention 上挂 controller

所以：

- inversion 提供“轨迹校正”
- attention controller 提供“语义编辑控制”

两者互不覆盖职责。

## 6.2 forward 函数本身就是组合点

`models/p2p/p2p_guidance_forward.py` 系列函数其实就是组合层：

- `p2p_guidance_forward`
- `p2p_guidance_forward_single_branch`
- `direct_inversion_p2p_guidance_forward`

它们共同接收的东西基本是：

- `model`
- `prompt`
- `controller`
- `latent`
- 以及 inversion 提供的 auxiliary package

所以 forward 函数并不属于 inversion，也不属于 attention。

它是这两个抽象的“接缝”。

这条边界很重要，因为它让以后扩展新方法时有明确落点：

- 新的 inversion：改 auxiliary package 的生产方式
- 新的 attention method：改 controller/hook
- forward 只负责把两者拼起来

## 6.3 MasaCtrl 进一步说明了这个抽象是可迁移的

`run_editing_masactrl.py` 不是 P2P，但它依然复用了 DirectInversion 的核心产物：

- `x_stars[-1]`
- `noise_loss_list`

同时 attention 这一侧换成了：

- `regiter_attention_editor_diffusers(...)`
- `MutualSelfAttentionControl`

也就是说，换了一个 attention 编辑器之后，inversion 这边并没有推倒重来。

这说明 DirectInversion 的输出确实具有“跨编辑器可消费性”。

## 6.4 Pix2Pix-Zero 也在复用同一个思想

`models/pix2pix_zero/edit_pipeline.py` 里虽然没直接调用 `DirectInversion` 这个类，但做的是同一种事：

- 先跑一遍 reconstruction
- 记录每一步 `latent_list[-2-i] - latents`
- 再在 reconstruction/editing 过程中把这个 per-step 差值补回去

换句话说，代码形式不同，但抽象思想一致：

```text
先得到 reference trajectory
再得到 per-step offset
最后在 forward 过程中补偿回去
```

这进一步说明：“per-step latent correction” 才是项目中更通用的思想资产。

## 7. 为什么说这个仓库的核心不是“某个方法”，而是“两个正交轴”

把整个项目压缩成一句话，就是：

```text
编辑效果 = inversion 的可回归性 + attention/editing 的可控性
```

对应到代码里，就是两个正交轴：

### 轴 1：如何回到原图附近

- DDIM inversion
- Negative Prompt Inversion
- Null-Text Inversion
- DirectInversion

### 轴 2：如何在回到原图附近后施加编辑

- P2P attention replace/refine/reweight
- Proximal Guidance
- MasaCtrl mutual self-attention
- PnP 的 feature/attention injection
- Pix2Pix-Zero 的 cross-attention alignment

如果以后要继续扩展这个仓库，最自然的做法不是加一个“大一统方法类”，而是沿这两个轴各自扩展。

## 8. 以后改代码时的建议心智模型

### 8.1 如果要加一个新的 inversion

优先问自己两件事：

1. 它输出的统一起点是什么？
   - 一般是 `x_stars[-1]`
2. 它输出的 per-step 校正物是什么？
   - embedding schedule
   - latent offset schedule
   - 或者别的 schedule

只要这两个东西清楚，forward 层通常就能复用。

### 8.2 如果要加一个新的 attention 方法

优先问自己三件事：

1. 它改的是 cross-attention，还是 self-attention，还是卷积特征？
2. 它需要在每个 step 改，还是只在部分 step 改？
3. 它能不能被包装成一个 hook/controller/editor？

只要答案是“能包装成 hook”，它就可以和现有 inversion 解耦组合。

### 8.3 最值得保护的边界

以后重构时最不该破坏的是下面这条边界：

```text
inversion 负责生成 trajectory 和 correction package
forward 负责消费这些 package
attention controller 负责在 UNet 内部改注意力/特征
```

一旦把三者重新搅在一起，项目会很快失去现在这种“可组合、可做 ablation、可跨方法迁移”的优点。

## 9. 一句话记忆版

### 对 inversion 的一句话

这个项目把 inversion 抽象成了：

> “从真实图像恢复扩散轨迹，并输出一个可被 forward 每步消费的校正包。”

### 对 attention 的一句话

这个项目把 attention 编辑抽象成了：

> “通过 hook/controller 在 UNet 内部改注意力，但不接管 inversion 本身。”

### 对两者关系的一句话

最关键的设计不是“把它们合成一个大方法”，而是：

> inversion 负责回到原图，attention 负责如何编辑；两者在 forward loop 处组合，而不是互相侵入。
