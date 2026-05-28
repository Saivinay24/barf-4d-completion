# BARF 4D — Ablation Table
### Paper Table 1: VRC-Score comparison across methods

Last updated: 2026-05-28
Status: **TEMPLATE** — real numbers require GPU benchmark runs (see instructions below)

---

## Main Ablation Results

| Method | VRC-Coverage C(S) | VRC-Coherence H(S) | VRC-Quality Q(S) | **VRC-Score** | Angular Coverage % | Runtime (s/frame) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| NeoVerse only (baseline) | [RESULT: run GPU benchmark] | [RESULT: run GPU benchmark] | [RESULT: run GPU benchmark] | [RESULT: run GPU benchmark] | [RESULT: run GPU benchmark] | [RESULT: run GPU benchmark] |
| NeoVerse + Vivid4D (prior work) | [RESULT: run GPU benchmark] | [RESULT: run GPU benchmark] | [RESULT: run GPU benchmark] | [RESULT: run GPU benchmark] | [RESULT: run GPU benchmark] | [RESULT: run GPU benchmark] |
| BARF w/o temporal window | [RESULT: run GPU benchmark] | [RESULT: run GPU benchmark] | [RESULT: run GPU benchmark] | [RESULT: run GPU benchmark] | [RESULT: run GPU benchmark] | [RESULT: run GPU benchmark] |
| **BARF (full, ours)** | [RESULT: run GPU benchmark] | [RESULT: run GPU benchmark] | [RESULT: run GPU benchmark] | [RESULT: run GPU benchmark] | [RESULT: run GPU benchmark] | [RESULT: run GPU benchmark] |

**Target values** (from BARF_VRC_SCORE.md):

| Method | C(S) | H(S) | Q(S) | VRC-Score |
|---|:---:|:---:|:---:|:---:|
| NeoVerse only | ~0.45 | ~0.92 | ~0.50 | ~0.21 |
| + Vivid4D | ~0.68 | ~0.79 | ~0.60 | ~0.32 |
| BARF w/o temp | ~0.88 | ~0.71 | ~0.70 | ~0.44 |
| **BARF (full)** | **~0.91** | **~0.88** | **~0.78** | **~0.62** |

---

## Per-Angle Coverage Breakdown

Shows angular coverage at each of the 8 test viewpoints (0°=front, 180°=back):

| Method | 0° | 45° | 90° | 135° | 180° | 225° | 270° | 315° |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| NeoVerse only | [GPU] | [GPU] | [GPU] | [GPU] | [GPU] | [GPU] | [GPU] | [GPU] |
| + Vivid4D | [GPU] | [GPU] | [GPU] | [GPU] | [GPU] | [GPU] | [GPU] | [GPU] |
| **BARF (full)** | [GPU] | [GPU] | [GPU] | [GPU] | [GPU] | [GPU] | [GPU] | [GPU] |

Key metric: **180° (back view)** is the hardest angle — this is where BARF's
generative completion provides the largest improvement over baselines.

---

## Synthetic Data (Local Baseline — No GPU Required)

Synthetic front-facing PLY (`data/test_plys/synthetic_front_facing.ply`):

| Method | VRC-Coverage | Covered Angles | Empty Angles |
|---|:---:|:---:|:---:|
| NeoVerse placeholder (synthetic) | 0.50 | [0, 45, 90, 315] | [135, 180, 225, 270] |
| BARF target (after GPU completion) | ~0.91 | [0,45,90,135,180,225,270,315] | [] |

These numbers confirm the gap detection and VRC-Coverage metric work correctly
on synthetic data. Real GPU numbers will replace the placeholders above.

---

## VR Performance (Quest 3)

| Scene | Gaussians | Mode | FPS | Notes |
|---|:---:|:---:|:---:|:---:|
| Synthetic placeholder | [GPU] | Standalone | [GPU] | After LOD to <500K |
| NeoVerse output | [GPU] | Tethered | [GPU] | Pre-LOD |
| BARF completed | [GPU] | Standalone | [GPU] | After LOD |
| **Target** | <500K | Standalone | **72** | Quest 3 comfort threshold |

---

## How to Fill This Table

### Step 1: Run GPU benchmarks (Vast.ai H100 or Colab A100)

```bash
# Benchmark NeoVerse baseline
bash scripts/run_pipeline.sh data/test_video.mp4 outputs/neoverse_baseline/
python3 -m src.metrics.vrc_score \
    --scene_ply outputs/neoverse_baseline/neoverse/scene.ply \
    --output results/neoverse_vrc.json

# Benchmark Vivid4D
bash scripts/run_vivid4d_baseline.sh data/test_video.mp4 outputs/vivid4d_baseline/
python3 -m src.metrics.vrc_score \
    --scene_ply outputs/vivid4d_baseline/vivid4d_scene.ply \
    --output results/vivid4d_vrc.json

# Benchmark BARF (full pipeline)
python3 -m src.completion.spherical_completion \
    --scene_ply outputs/neoverse_baseline/neoverse/scene.ply \
    --gaps_json outputs/neoverse_baseline/gaps/gaps.json \
    --output_ply outputs/barf_complete/scene.ply \
    --device cuda
python3 -m src.metrics.vrc_score \
    --scene_ply outputs/barf_complete/scene.ply \
    --output results/barf_vrc.json
```

### Step 2: Copy numbers from JSON outputs into the table above

The JSON files output by `vrc_score.py` have structure:
```json
{
  "vrc_coverage": 0.91,
  "vrc_coherence": 0.88,
  "vrc_quality": 0.78,
  "vrc_score": 0.62,
  "coverage_detail": {
    "per_angle": { "0": ..., "180": ... }
  }
}
```

### Step 3: Run VR performance benchmark on Quest 3

```bash
# Export to Quest .splat
python3 -m src.vr.export_splat \
    --input outputs/barf_complete/scene.ply \
    --output outputs/splat/scene.splat \
    --max_gaussians 500000

# Load in Meta Spatial SDK viewer and measure FPS
# (see src/vr/export_splat.py for Quest 3 deployment instructions)
```

---

## Test Datasets

| Dataset | Type | Scenes | Available |
|---|---|---|:---:|
| `data/test_plys/synthetic_front_facing.ply` | Synthetic, front-facing | 1 | ✅ Local |
| DyNeRF | Real multi-view | 6 | Download required |
| Dynamic Scene | Synthetic multi-view | 9 | Download required |
| Custom two-camera captures | Real with GT back-view | TBD | Record needed |
