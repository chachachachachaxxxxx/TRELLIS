这个文件可以把流程理解成 5 层：启动配置、prompt 对齐、运行时 patch、两阶段采样、结果导出。

**1. 启动层**
文件一开始先做一件很关键的事：在导入 TRELLIS 之前决定 attention backend，因为 TRELLIS 会在 import 时读取环境变量。[example_text_prompt_to_prompt.py#L41](/home/wangxinxing/code/TRELLIS/example_text_prompt_to_prompt.py#L41) 先偷看 `--attn-backend`，然后 [example_text_prompt_to_prompt.py#L69](/home/wangxinxing/code/TRELLIS/example_text_prompt_to_prompt.py#L69) 的 `configure_attention_backend()` 会检查 `flash_attn` / `xformers` 是否存在，并写入 `ATTN_BACKEND` / `SPARSE_ATTN_BACKEND`。  
这样做的目的很直接：避免你还没开始跑，就在 TRELLIS 内部 import 时炸掉。

**2. Prompt 预处理层**
这部分是在把“源 prompt”和“编辑 prompt”变成可注入的 token 映射。

- [example_text_prompt_to_prompt.py#L124](/home/wangxinxing/code/TRELLIS/example_text_prompt_to_prompt.py#L124) 的 `build_token_metadata()` 会做 tokenizer 编码，拿到：
  - `input_ids`
  - `raw_tokens`
  - `display_tokens`
  - `valid_token_count`
  - `special_mask`
- [example_text_prompt_to_prompt.py#L158](/home/wangxinxing/code/TRELLIS/example_text_prompt_to_prompt.py#L158) 的 `build_prompt_alignment()` 会只看“内容 token”，然后用 `SequenceMatcher` 找两边完全相同的 token 片段。
- 它最后产出两组最重要的索引：
  - `source_keep_indices`
  - `edit_keep_indices`

这两组索引其实就是你公式里 `M` 的离散实现版本：  
“哪些 edit token 的 attention 列，要被 source token 的 attention 列替换”。

注意这里的对齐是“token 级相等对齐”，不是语义相似对齐。所以它对 `cat -> tiger` 这种替换很好用，对大幅改写、重排、同义改写就会保守一些。

**3. Patch 控制层**
核心类是 [example_text_prompt_to_prompt.py#L272](/home/wangxinxing/code/TRELLIS/example_text_prompt_to_prompt.py#L272) 的 `PromptToPromptEditor`。

它做三件事：

1. `prepare_stage()` 给指定 stage 挂 patch。  
   入口在 [example_text_prompt_to_prompt.py#L293](/home/wangxinxing/code/TRELLIS/example_text_prompt_to_prompt.py#L293)

2. `_patch_model_forward()` 先 patch 每个 flow model 的 `forward`。  
   位置在 [example_text_prompt_to_prompt.py#L305](/home/wangxinxing/code/TRELLIS/example_text_prompt_to_prompt.py#L305)  
   这里会记录当前：
   - 是 `sparse_structure` 还是 `slat`
   - 当前时间步 `t`
   - 归一化后的 `t_norm = t / 1000`
   - 当前 CFG 分支是 `cond` 还是 `neg`

3. `_infer_pass_kind()` 用 cond 张量签名去区分 CFG 的正分支和负分支。  
   位置在 [example_text_prompt_to_prompt.py#L326](/home/wangxinxing/code/TRELLIS/example_text_prompt_to_prompt.py#L326)  
   这很重要，因为脚本只在 `cond` 分支注入，`neg` 分支保持原样，不然 CFG 会把 source 语义也混进负分支里。

**4. 真正的 Prompt-to-Prompt 注入**
这是文件最核心的部分。

- dense 的 `cross_attn` patch 在 [example_text_prompt_to_prompt.py#L339](/home/wangxinxing/code/TRELLIS/example_text_prompt_to_prompt.py#L339)
- sparse 的 `cross_attn` patch 在 [example_text_prompt_to_prompt.py#L376](/home/wangxinxing/code/TRELLIS/example_text_prompt_to_prompt.py#L376)

它们的共同流程是：

1. 先算当前 edit 分支的 `Q`
2. 用 edit prompt 的 context 算 `K_edit, V_edit`
3. 用 source prompt 的 context 再算一份 `K_src`
4. 如果模块启用了 `qk_rms_norm`，就和原模型保持一致地做归一化
5. 调用 chunked attention 逻辑，算：
   - `A_src = softmax(Q K_src^T / sqrt(d))`
   - `A_edit = softmax(Q K_edit^T / sqrt(d))`
6. 对 keep token 对应的列做替换/混合
7. 最后输出 `A_tilde V_edit`

所以它和你写的公式是一致的：  
保留 source 的布局约束，但值仍然来自 edit prompt。  
这里特意没有用 `V_src`，只用 `V_edit`，因为这正是 Prompt-to-Prompt 的关键。

真正做列替换的是 [example_text_prompt_to_prompt.py#L452](/home/wangxinxing/code/TRELLIS/example_text_prompt_to_prompt.py#L452) 的 `_mix_attention_maps()`：
- `strength = 1` 时，keep token 直接完全替换
- `strength < 1` 时，做线性混合，方便消融

真正做 `softmax(QK^T)` 和 `A V` 的是 [example_text_prompt_to_prompt.py#L472](/home/wangxinxing/code/TRELLIS/example_text_prompt_to_prompt.py#L472)：
- 它按 query chunk 分块，减少显存峰值
- sparse 版本在 [example_text_prompt_to_prompt.py#L504](/home/wangxinxing/code/TRELLIS/example_text_prompt_to_prompt.py#L504)，本质上是对每个 batch 的稀疏 query 单独复用同一个 dense kernel

**5. Stage / t 消融怎么接进来的**
阶段控制靠 [example_text_prompt_to_prompt.py#L258](/home/wangxinxing/code/TRELLIS/example_text_prompt_to_prompt.py#L258) 的 `StageConfig`，它定义：
- 是否启用
- `t_start`
- `t_end`
- `strength`

真正判断“这一层当前要不要注入”的逻辑在 [example_text_prompt_to_prompt.py#L414](/home/wangxinxing/code/TRELLIS/example_text_prompt_to_prompt.py#L414) 的 `_active_strength()`：
- 当前不是目标 stage，不注入
- 当前是 `neg` 分支，不注入
- 当前 `t_norm` 不在区间内，不注入
- 否则返回对应 stage 的 `strength`

所以你要的消融基本都能直接做：
- 只注入 `st/ss`：`--inject-stages st`
- 只注入 `slat`：`--inject-stages slat`
- 两阶段都注入：`--inject-stages st,slat`
- 完全关闭：`--inject-stages none`
- 时间区间分别控：
  - `--ss-t-start --ss-t-end`
  - `--slat-t-start --slat-t-end`
- 强度分别控：
  - `--ss-strength`
  - `--slat-strength`

**6. 一次完整运行的时序**
入口在 [example_text_prompt_to_prompt.py#L609](/home/wangxinxing/code/TRELLIS/example_text_prompt_to_prompt.py#L609)。

它按这个顺序走：

1. 解析参数，配置 backend
2. 这时候才导入 TRELLIS 模块
3. 解析 `--inject-stages`，构造两个 `StageConfig`
4. 加载 `TrellisTextTo3DPipeline`
5. 对 source/edit prompt 做 token metadata 和 token alignment
6. 编码出 `source_cond`、`edit_cond`、`neg_cond`
7. 创建 `PromptToPromptEditor`
8. 整理采样参数 `ss_params` / `slat_params`
9. 在输出目录保存：
   - `config.json`
   - `source_tokens.json`
   - `edit_tokens.json`
   - `token_alignment.json`
10. `try/finally` 里先挂 patch，再执行：
   - `sample_sparse_structure()`
   - `sample_slat()`
   - `decode_slat()`
11. 无论中间是否报错，`finally` 都会 `restore()`，把 monkey patch 全恢复
12. 最后导出 mp4 / glb / ply  
   导出逻辑在 [example_text_prompt_to_prompt.py#L561](/home/wangxinxing/code/TRELLIS/example_text_prompt_to_prompt.py#L561)

**7. 这个文件最重要的设计点**
- 不改 TRELLIS 核心源码，只 monkey patch
- 只 patch `cross_attn`，不碰主模型结构
- 只在 CFG 的 `cond` 分支注入
- `source` 只提供 `K_src`，输出仍然走 `V_edit`
- `st` 和 `slat` 独立开关、独立时间区间、独立 strength
- patch 都放在 `try/finally` 里，结束后模型会恢复干净

如果你愿意，我下一条可以继续把这个文件画成一张“函数调用流程图”，或者直接按“从 `main()` 往下”逐函数解释每个参数和 tensor shape。