# Core Pipeline — BARF Integration

This folder contains the main integration pipeline that connects all components, plus the temporal consistency module (BARF's key innovation).

## Structure

```
core/
├── pipeline.py          # End-to-end BARF pipeline
├── temporal_smooth.py   # Temporal consistency (EMA + optical flow)
├── architecture.md      # System design document
└── README.md            # This file
```

## Running the Pipeline

### Full pipeline:
```bash
python core/pipeline.py --input video.mp4 --output output_dir/
```

### With pre-computed reconstruction:
```bash
python core/pipeline.py --input video.mp4 --output output_dir/ --skip-depth --skip-colmap
```

### Temporal smoothing only:
```bash
python core/temporal_smooth.py \
    --input generated_views/ \
    --output smoothed_views/ \
    --method ema \
    --strength 0.7
```

## Architecture

```
video → data processing → 4D reconstruction → gap detection →
novel view generation → temporal consistency → merge → output
```

See `architecture.md` for full technical design.

## Pipeline Stages

1. **Data Processing** — Frame extraction, depth maps, COLMAP poses
2. **4D Reconstruction** — 4DGaussians training → partial point clouds
3. **Gap Detection** — Voxel analysis → gap region JSON
4. **Novel View Generation** — SV3D back-view synthesis
5. **Temporal Consistency** — Sliding-window smoothing (EMA or optical flow)
6. **Merge** — Combine partial + generated → complete 360° scene

## Temporal Smoothing Methods

| Method | Quality | Speed | Requirements |
|--------|---------|-------|-------------|
| `ema` | Good | Fast | numpy only |
| `flow` | Better | Slower | torchvision (RAFT) |
| `combined` | Best | Slowest | torchvision (RAFT) |
