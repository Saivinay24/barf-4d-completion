# Diffusion Experiments — Novel View Generation

This folder contains novel view generation experiments, results, and the packaged generator for the BARF pipeline.

## Structure

```
diffusion_experiments/
├── generator.py            # Callable NovelViewGenerator class for pipeline
├── methods_comparison.md   # SV3D vs Zero123++ vs SVD comparison
└── README.md               # This file
```

## Quick Start

### Generate a back-view from a single image:
```bash
python diffusion_experiments/generator.py \
    --input front_view.jpg \
    --output back_view.png \
    --angle 180 \
    --model sv3d
```

### Batch process a video's frames:
```bash
python diffusion_experiments/generator.py \
    --input-dir datasets/custom/v01_person_walk/frames/ \
    --output-dir diffusion_experiments/backview_results/ \
    --angle 180 \
    --measure-flicker
```

### Generate full orbital sequence:
```bash
python diffusion_experiments/generator.py \
    --input front_view.jpg \
    --output-dir orbital/ \
    --full-orbit
```

## API Usage

```python
from diffusion_experiments.generator import NovelViewGenerator

gen = NovelViewGenerator(model="sv3d", device="cuda")
back_view = gen.generate_backview("front.jpg", angle=180)
back_view.save("back.png")
```

## Requirements (GPU needed)
```bash
pip install torch diffusers transformers accelerate Pillow numpy
```

## Key Findings

See `methods_comparison.md` for detailed analysis:
- **SV3D** — Best quality (4/5), 15s/image, 16GB VRAM
- **Zero123++** — Faster (5s/image), 8GB VRAM, lower resolution
- **Temporal flicker** — 3-4x worse than real video without smoothing
- **BARF's temporal smoothing** reduces flicker by ~40-60%
