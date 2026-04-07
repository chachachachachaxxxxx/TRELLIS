# 使用代理运行测试

## 测试命令（带代理）

```bash
# 使用 7890 端口代理
HTTP_PROXY=http://127.0.0.1:7890 \
HTTPS_PROXY=http://127.0.0.1:7890 \
CUDA_VISIBLE_DEVICES=3 \
python run_edit_experiment.py \
  --method image_prompt_to_prompt \
  --source-image assets/edit_example/images/2d_render.png \
  --edit-image assets/edit_example/images/2d_edit.png \
  --mask-image assets/edit_example/images/2d_mask.png \
  --case-name test_with_proxy \
  --seed 1 \
  --preprocess
```

## 预期流程

1. ✅ 检测方法类
2. ✅ 加载 pipeline
3. 🔄 通过代理下载 DINOv2 模型
4. 🔄 预处理图像
5. 🔄 运行 Prompt-to-Prompt 编辑
6. 🔄 保存结果

## 输出位置

```
outputs/image_prompt_to_prompt/test_with_proxy/
├── edit/
│   ├── config.json
│   ├── input_preprocess.json
│   ├── source_preprocessed.png
│   ├── edit_preprocessed.png
│   ├── mask_preprocessed.png
│   ├── mask_token_overlay.png
│   ├── edited_patch_grid.png
│   ├── sample.glb
│   └── ...
└── source_original/
    └── sample.glb
```

测试正在运行中...
