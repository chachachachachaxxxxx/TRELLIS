# Edit3D-Bench Evaluation System

An integrated evaluation system for computing multiple evaluation metrics on the Edit3D-Bench dataset.

## Features

Supported Evaluation Metrics:
1. **Image Metrics**: PSNR, SSIM, LPIPS, DINO-IF
2. **Distribution Metrics**: FID
3. **Video Metrics**: FVD
4. **3D Geometry Metrics**: Chamfer Distance
5. **Text Alignment Metrics**: CLIP-T

## System Architecture

```
evaluation/
├── eval_main.py              # Main evaluation program
├── eval_modules/             # Evaluation modules
│   ├── __init__.py
│   ├── data_loader.py        # Data loader
│   ├── image_metrics.py      # Image quality metrics
│   ├── distribution_metrics.py # Distribution metrics
│   ├── video_metrics.py      # Video metrics
│   ├── geometry_metrics.py   # Geometry metrics
│   └── text_alignment_metrics.py # Text alignment metrics
└── README.md                 # Documentation
```

## Data Organization

### Ground Truth Data

Download the ground truth data from [Edit3D-Bench dataset on Hugging Face](https://huggingface.co/datasets/huanngzh/Edit3D-Bench):

```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

# Clone the dataset
git clone https://huggingface.co/datasets/huanngzh/Edit3D-Bench

# If you want to clone without large files - just their pointers
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/huanngzh/Edit3D-Bench
```

The expected data structure is:

```
data/
├── metadata.json
├── GSO/
│   └── {object_name}/
│       ├── source_model/
│       │   └── model.glb
│       └── prompt_{1,2,3}/
│           ├── render/
│           │   ├── visual_*.png
│           │   ├── mask_*.png
│           │   └── visual3d.mp4
│           ├── 2d_edit.png
│           ├── 3d_edit_region.glb
│           └── prompt.txt (optional)
└── PartObjaverse-Tiny/
    └── ...
```

### Prediction Results Data

The prediction results should be organized as follows:

```
exp_comparison/
└── {method_name}/
    ├── GSO/
    │   └── {object_name}/
    │       └── prompt_{1,2,3}/
    │           ├── images/
    │           │   └── render_*.png
    │           ├── video_rgb.mp4
    │           └── edit.glb
    └── PartObjaverse-Tiny/
        └── ...
```

**Important Notes:**
- The `{method_name}` should be a unique identifier for your method
- The directory structure must match exactly with the ground truth data
- Image files should be named as `render_XXXX.png` where XXXX is a 4-digit number (e.g., `render_0000.png`, `render_0001.png`, etc.)
- The number of rendered images should match the ground truth data

## Installation

```Bash
pip install -r requirements.txt
```

## Usage

### Prepare Your Results

Organize your predicted results in the following structures. Make sure your results are aligned with the original model in Edit3D-Bench in terms of position, orientation and scale.

```
exp_comparison/
└── {method_name}/
    ├── GSO/
    │   └── {object_name}/
    │       └── prompt_{1,2,3}/
    │           └── edit.glb
    └── PartObjaverse-Tiny/
        └── ...
```

Then you can render multi-view images and videos for evaluation using:

```Bash
python render.py --base_dir /path/to/exp_comparison/your_method_name
```

The rendering may take some time.

### Method 1: Command Line

```bash
python eval_main.py \
    --gt_root /path/to/Edit3D-Bench/data \
    --pred_root /path/to/exp_comparison/your_method_name \
    --metrics psnr ssim lpips fid dino_if_max dino_if_mean chamfer clip_t \
    --device cuda:0 \
    --image_size 512 512 \
    --output_dir evaluation_results
```

### Method 2: Python Script

```python
from eval_main import EvaluationConfig, EvaluationManager

# Configure evaluation parameters
config = EvaluationConfig(
    gt_root="/path/to/ground/truth",
    pred_root="/path/to/predictions",
    metrics=["psnr", "ssim", "lpips", "fid"],
    device="cuda:0",
    image_size=(512, 512),
    output_dir="evaluation_results"
)

# Run evaluation
evaluator = EvaluationManager(config)
results = evaluator.run_evaluation()
```

## Output Results

After evaluation completion, the following files will be generated in the output directory:

```
evaluation_results/
├── detailed_results.json  # Detailed results (including scores for each sample)
└── summary.json          # Summary results (including mean and standard deviation)
```

### Result Format Example

```json
{
  "config": {
    "gt_root": "/path/to/ground/truth",
    "pred_root": "/path/to/predictions",
    "metrics": ["psnr", "ssim", "lpips", "fid"],
    "device": "cuda:0",
    "image_size": [512, 512]
  },
  "results": {
    "psnr": {
      "mean": 25.1234,
      "std": 2.3456,
      "count": 1000
    },
    "ssim": {
      "mean": 0.8765,
      "std": 0.0456,
      "count": 1000
    },
    "lpips": {
      "mean": 0.1234,
      "std": 0.0234,
      "count": 1000
    },
    "fid": 45.6789
  }
}
```

## Troubleshooting

### Common Issues

1. **CUDA Memory Insufficient**
   - Reduce the `batch_size` parameter
   - Use CPU device: `--device cpu`

2. **File Path Errors**
   - Check if paths exist
   - Use absolute paths

3. **Model Loading Failures**
   - Ensure all dependency packages are installed
   - Check if model files exist

4. **Data Pairing Failures**
   - Check if data organization structure is correct
   - Ensure file names match between GT and prediction results

## Extension Development

To add new evaluation metrics:

1. Create a new evaluator class in the `eval_modules/` directory
2. Implement the corresponding computation methods
3. Register the new evaluator in `eval_main.py`
4. Update the `__init__.py` file
