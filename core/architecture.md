# BARF Architecture — Technical Design Document

## System Overview

BARF (Binarily Augmented Reality Footage) is a generative 4D completion pipeline that transforms monocular video into explorable 360° 4D scenes.

```
                    ┌─────────────────────────────────────────┐
                    │        BARF Pipeline Architecture        │
                    └─────────────────────────────────────────┘

    Input: Monocular Video (MP4, 1080p, 30fps)
                        │
                        ▼
    ┌──────────────────────────────────┐
    │  [Stage 1] Data Processing (R2)  │
    │  - Frame extraction (ffmpeg)     │
    │  - Depth maps (Depth Anything V2)│
    │  - Camera poses (COLMAP)         │
    │  - Segmentation masks (SAM2)     │
    └──────────────────────────────────┘
                        │
                        ▼
    ┌──────────────────────────────────┐
    │ [Stage 2] 4D Reconstruction (R3) │
    │  - 4DGaussians training          │
    │  - Point cloud sequence (.ply)   │
    │  - Partial 360° geometry         │
    └──────────────────────────────────┘
                        │
                        ▼
    ┌──────────────────────────────────┐
    │   [Stage 3] Gap Detection (R3)   │
    │  - Voxel-based analysis          │
    │  - DBSCAN clustering             │
    │  - Gap regions → JSON            │
    │  - Coverage statistics            │
    └──────────────────────────────────┘
                        │
                        ▼
    ┌──────────────────────────────────┐
    │ [Stage 4] Novel View Gen (R4)    │
    │  - SV3D orbital generation       │
    │  - Back-view extraction (180°)   │
    │  - Per-frame diffusion           │
    └──────────────────────────────────┘
                        │
                        ▼
    ┌───────────────────────────────────┐
    │ [Stage 5] Temporal Consistency    │  ← OUR INNOVATION
    │  - Sliding-window denoising      │
    │  - Optical flow warping (RAFT)   │
    │  - Cross-frame attention          │
    │  - Flicker reduction             │
    └───────────────────────────────────┘
                        │
                        ▼
    ┌──────────────────────────────────┐
    │     [Stage 6] Merge + Output     │
    │  - Combine partial + generated   │
    │  - Resolve overlap regions       │
    │  - Export complete .ply sequence  │
    └──────────────────────────────────┘
                        │
                        ▼
    ┌──────────────────────────────────┐
    │    [Stage 7] Visualization       │
    │  - Web viewer (R5)               │
    │  - VR viewer (R6)                │
    │  - Before/after comparison       │
    │  - 4D timeline playback          │
    └──────────────────────────────────┘
```

## Key Technical Decisions

### Why 4DGaussians for Reconstruction?
- Best speed/quality tradeoff for monocular input
- Produces per-frame point clouds (easy to analyze for gaps)
- Active community, well-documented

### Why SV3D for Novel View Generation?
- Best single-image-to-orbit quality available
- 21-frame orbital provides fine angular resolution (~17° per frame)
- 576×576 output resolution sufficient for initial prototype

### Why Voxel-Based Gap Detection?
- Simple, interpretable, fast
- Directly maps to 3D space (not image-space heuristics)
- DBSCAN clustering produces clean gap regions
- Easy to feed into downstream generation

### Our Innovation: Temporal Consistency
- **Problem:** Independent per-frame generation causes 3-4x more flicker than real video
- **Solution:** Sliding-window denoising with cross-frame attention
- Treat consecutive generated frames as a batch
- Share attention keys/values across the window
- Use optical flow (RAFT) to warp previous generated frame → initialize next

## Data Flow Formats

| Stage | Input Format | Output Format |
|-------|-------------|---------------|
| Data Processing | .mp4 video | Frames (.jpg) + depth (.png) + COLMAP (.txt) |
| 4D Reconstruction | Frames + COLMAP | .ply point clouds (per frame) |
| Gap Detection | .ply | gaps.json (cluster centers, sizes, bounding boxes) |
| Novel View Gen | Frame (.jpg) + angle | Generated view (.png) |
| Temporal Smooth | Sequence of generated views | Temporally consistent sequence |
| Merge | Partial .ply + generated views + depth | Complete .ply sequence |
| Visualization | .ply + gaps.json | Web/VR interactive viewer |

## Quality Metrics

1. **Gap Coverage %** — What fraction of 360° sphere is filled?
2. **Temporal Consistency** — Mean frame-to-frame pixel difference (lower = smoother)
3. **LPIPS** — Perceptual similarity to ground truth (if available)
4. **Processing Speed** — Seconds per frame for complete pipeline
5. **Point Density** — Points per unit area in generated vs. original regions
