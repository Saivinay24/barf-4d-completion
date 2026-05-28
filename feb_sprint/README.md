# Feb 2026 Sprint Archive

> **These files are from the original February 14-28, 2026 sprint.**
> They are preserved for reference but are **no longer part of the active codebase**.

The Feb sprint was a research/reconnaissance sprint that produced:
- SOTA benchmark artifacts (demo screenshots/videos from CAT4D, Vivid4D, NeoVerse, Vidu4D)
- An old pipeline design using 4DGaussians + SV3D/Zero123++ (superseded by NeoVerse + D4RT)
- Team task files for the 6-person sprint team
- An old gap detection module using open3d (superseded by `src/gap_detection/detect_gaps.py`)
- Diffusion experiment artifacts

## What replaced these files

| Old (Feb Sprint) | New (May 2026) |
|---|---|
| `core/pipeline.py` (4DGaussians + SV3D) | `scripts/run_pipeline.sh` (NeoVerse + D4RT) |
| `reconstructions/gap_detection/detect.py` (open3d) | `src/gap_detection/detect_gaps.py` (pure numpy) |
| `datasets/process_video.py` | Frame extraction embedded in pipeline script |
| `diffusion_experiments/generator.py` | `src/completion/spherical_completion.py` |
| `core/temporal_smooth.py` | Temporal consistency module in spherical_completion.py |

## Contents

```
feb_sprint/
├── core/                    # Old pipeline + temporal smoothing
├── datasets/                # Old video processing scripts
├── reconstructions/         # Old gap detection (open3d-based)
├── diffusion_experiments/   # Old diffusion experiment results
├── tasks/                   # Team task assignments (R1-R6)
├── research/                # Old benchmark comparisons
├── presentation/            # Old demo outline
├── benchmark_outputs/       # Demo artifacts from SOTA methods
├── gap_evidence/            # Screenshots showing reconstruction gaps
├── gap_evidence.md          # Gap evidence documentation
├── benchmark_comparison.md  # Old comparison tables
└── MASTER_PLAN.pdf          # PDF version of the master plan
```
