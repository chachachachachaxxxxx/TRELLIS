# configs/ 目录说明

这个目录包含 TRELLIS 模型的配置文件（VAE 和生成模型）。

## 目录结构

```
configs/
├── vae/           # VAE 模型配置
│   ├── slat_vae_dec_mesh_swin8_B_64l8_fp16.json
│   ├── slat_vae_dec_rf_swin8_B_64l8_fp16.json
│   ├── slat_vae_enc_dec_gs_swin8_B_64l8_fp16.json
│   └── ss_vae_conv3d_16l8_fp16.json
└── generation/    # 生成模型配置
    ├── slat_flow_img_dit_L_64l8p2_fp16.json
    ├── slat_flow_txt_dit_B_64l8p2_fp16.json
    ├── slat_flow_txt_dit_L_64l8p2_fp16.json
    ├── slat_flow_txt_dit_XL_64l8p2_fp16.json
    ├── ss_flow_img_dit_L_16l8_fp16.json
    ├── ss_flow_txt_dit_B_16l8_fp16.json
    ├── ss_flow_txt_dit_L_16l8_fp16.json
    └── ss_flow_txt_dit_XL_16l8_fp16.json
```

## 说明

- **vae/**: VAE 编码器/解码器配置
  - `ss_vae_*`: Sparse Structure VAE
  - `slat_vae_*`: SLAT VAE

- **generation/**: Flow Transformer 生成模型配置
  - `ss_flow_*`: Sparse Structure Flow
  - `slat_flow_*`: SLAT Flow
  - `*_img_*`: Image-to-3D 模型
  - `*_txt_*`: Text-to-3D 模型

## 编辑实验配置

编辑实验的配置文件已迁移到 `edit_configs/` 目录。

详见: [../edit_configs/README.md](../edit_configs/README.md)
