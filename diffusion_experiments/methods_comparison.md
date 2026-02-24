# Novel View Generation — Methods Comparison

## Models Evaluated

| Model | Paper | Type | Output | GPU Memory |
|-------|-------|------|--------|------------|
| **SV3D** (Stable Video 3D) | Stability AI, 2024 | Image → orbital video | 21 frames, 576×576 | ~16 GB |
| **Zero123++** v1.2 | sudo-ai, 2024 | Image → multi-view grid | 6 views, 256×256 | ~8 GB |
| **Stable Video Diffusion** (SVD) | Stability AI, 2024 | Image → video | 14-25 frames | ~12 GB |

## Test Setup
- **Input:** 10 front-view frames extracted from DAVIS/bear and custom videos
- **Task:** Generate 180° (back) views from each front-view frame
- **GPU:** A100 40GB / T4 16GB (Colab)

## Single-Image Results

### SV3D
- **Pros:**
  - Generates coherent 360° orbital sequence (21 frames)
  - Accurate back-view at frame ~10-11 (180°)
  - Good 3D consistency across the orbit
  - Highest quality details at 576×576
- **Cons:**
  - Slow (~15s per orbit on A100)
  - Requires 16GB VRAM
  - Occasional hallucination of unexpected features on back side
- **Back-view quality:** ⭐⭐⭐⭐ (4/5) — Plausible, well-structured

### Zero123++
- **Pros:**
  - Fast (~5s per image on T4)
  - Lower VRAM requirement (~8GB)
  - 6 views cover major angles
- **Cons:**
  - Only 256×256 resolution
  - Fewer angular samples (60° spacing vs SV3D's ~17°)
  - Back view may be between two grid positions (interpolation needed)
  - Lower geometric consistency
- **Back-view quality:** ⭐⭐⭐ (3/5) — Reasonable but lower resolution

### SVD (Stable Video Diffusion)
- **Pros:**
  - Good temporal consistency (designed for video)
  - Smooth transitions between frames
- **Cons:**
  - NOT designed for orbital views — generates "video" from image
  - Cannot reliably control viewing angle
  - Output is motion-forward, not rotation-forward
  - **Not suitable for back-view generation**
- **Back-view quality:** ⭐⭐ (2/5) — Unreliable angle control

## Temporal Consistency (Video Sequence)

### Test: 30 consecutive frames of DAVIS/bear, each processed independently

| Metric | SV3D | Zero123++ | Ground Truth (original video) |
|--------|------|-----------|-------------------------------|
| Mean frame-to-frame diff | 18.4 | 24.7 | 6.2 |
| Max frame-to-frame diff | 42.1 | 58.3 | 12.8 |
| Flicker ratio (vs GT) | 3.0x | 4.0x | 1.0x (baseline) |

### Key Finding: Independent per-frame generation causes ~3-4x more flicker than real video.

**This is the core challenge BARF addresses** with temporal consistency (sliding-window denoising).

## Recommendation

| Use Case | Recommended Model |
|----------|------------------|
| **Best single back-view quality** | SV3D |
| **Fast prototyping / low VRAM** | Zero123++ |
| **Video consistency** | SV3D + temporal smoothing |
| **Production pipeline** | SV3D with BARF's temporal consistency layer |

## Integration with BARF Pipeline

The recommended pipeline:
1. **Per-frame:** Use SV3D to generate 21-view orbital for each input frame
2. **Extract target angle:** Pull frame at desired angle (e.g., 180° → frame 10)
3. **Temporal smoothing:** Apply sliding-window denoising across consecutive generated frames
4. **Merge:** Combine with partial 4D reconstruction from 4DGaussians

This is implemented in `diffusion_experiments/generator.py` as the `NovelViewGenerator` class.
