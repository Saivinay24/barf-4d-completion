# 4D Reconstruction Benchmark Comparison

Last updated: 2026-03-24

This file consolidates current benchmark evidence from local run attempts and official demo artifacts saved under `research/benchmark_outputs/`.

| Method | Speed (sec/frame) | Quality (1-5) | Input Requirements | Output Format | Gaps Present? | Notes |
|---|---:|---:|---|---|---|---|
| CAT4D / CAP4D | N/A (local run blocked) | 4.0 | Monocular reference images/video + FLAME assets + Pixel3DMM | `.ply`, rendered `.mp4` | YES | Rear/head contour appears thinner and less reliable than front in sampled back view |
| Vidu4D | N/A (local run blocked) | 3.5 | Monocular video (`database/raw/<seq>/0.mp4`), CUDA build toolchain | rendered `.mp4` (+ internal checkpoints/meshes) | YES | Unseen/rear geometry confidence is limited from sampled views |
| NeoVerse | N/A (local run not executed in this pass) | 4.0 | Monocular video/image + large model checkpoints | output `.mp4` | YES (tentative) | Strong visual quality in official demo; strict gap judgment still needs same-input local A/B |
| 4DGaussians (R3 baseline) | TBD | TBD | Frames + COLMAP | `.ply` | YES | Pending import of R3 runtime/output measurements |

## Measurement Notes

- `Speed (sec/frame)` is intentionally `N/A` where local end-to-end inference did not complete.
- Quality scores are subjective visual ratings from currently saved front/side/back snapshots.
- To finalize this table, run the same 3 real videos through each runnable method and append:
  - total runtime
  - frame count
  - computed sec/frame
  - exact command used

## Artifact Locations

- CAT4D/CAP4D: `research/benchmark_outputs/cat4d/`, `research/benchmark_outputs/cap4d/`
- Vidu4D: `research/benchmark_outputs/Vidu4D/`, `research/benchmark_outputs/vivid4d/`
- NeoVerse: `research/benchmark_outputs/neoverse/`
