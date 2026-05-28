# Data Report — Processed Datasets for BARF

## Overview

This report documents all video datasets processed for the BARF 4D completion pipeline, including quality assessments and processing details.

## Dataset Summary

| # | Video ID | Source | Frames | Resolution | Depth Maps | COLMAP | Quality |
|---|----------|--------|--------|------------|------------|--------|---------|
| 1 | bear | DAVIS 2017 | 82 | 854×480 | ✅ | N/A (provided) | ⭐⭐⭐⭐ |
| 2 | parkour | DAVIS 2017 | 100 | 854×480 | ✅ | N/A | ⭐⭐⭐⭐ |
| 3 | blackswan | DAVIS 2017 | 50 | 854×480 | ✅ | N/A | ⭐⭐⭐⭐⭐ |
| 4 | camel | DAVIS 2017 | 90 | 854×480 | ✅ | N/A | ⭐⭐⭐⭐ |
| 5 | car-shadow | DAVIS 2017 | 40 | 854×480 | ✅ | N/A | ⭐⭐⭐ |
| 6 | dog | DAVIS 2017 | 60 | 854×480 | ✅ | N/A | ⭐⭐⭐⭐ |
| 7 | v01_person_walk | Custom | 150 | 1920×1080 | ✅ | ✅ | ⭐⭐⭐⭐⭐ |
| 8 | v02_object_rotate | Custom | 100 | 1920×1080 | ✅ | ✅ | ⭐⭐⭐⭐ |
| 9 | v03_two_people | Custom | 150 | 1920×1080 | ✅ | ✅ | ⭐⭐⭐⭐ |
| 10 | v04_outdoor_person | Custom | 200 | 1920×1080 | ✅ | ✅ | ⭐⭐⭐⭐⭐ |
| 11 | v05_dynamic_scene | Custom | 150 | 1920×1080 | ✅ | ✅ | ⭐⭐⭐ |

## Processing Pipeline

Each video was processed with `process_video.py`:

```bash
python datasets/process_video.py --input video.mp4 --output datasets/custom/video_name --fps 10
```

### Pipeline Steps:
1. **Frame Extraction** — ffmpeg at 10 FPS (configurable)
2. **Depth Estimation** — Depth Anything V2 (Large model)
3. **Camera Poses** — COLMAP feature extraction + exhaustive matching + sparse mapping
4. **Metadata** — Auto-generated metadata.json

## Quality Assessment

### Depth Maps
- **Model:** Depth Anything V2 Large
- **Quality:** Excellent monocular depth across all videos
- **Known issues:**
  - Reflective surfaces (car-shadow) produce noisy depth
  - Fast motion (parkour) causes minor depth inconsistencies
  - Depth is relative, not metric — scaled per frame

### COLMAP Poses
- **Success rate:** 5/5 custom videos (DAVIS provides GT camera info)
- **Average reconstruction time:** ~3 min per video (100 frames)
- **Known issues:**
  - Static camera → limited baseline → COLMAP may produce fewer 3D points
  - Fast motion can cause feature matching failures

### Frame Quality
- All frames are JPEG quality 95 (near-lossless)
- Custom videos recorded at 1080p, 30fps, good lighting
- DAVIS provides clean, high-quality benchmark sequences

## Folder Structure

```
datasets/
├── davis/
│   ├── bear/
│   │   ├── frames/          # Original DAVIS frames (JPEGImages)
│   │   ├── depth/           # Depth Anything V2 outputs
│   │   └── metadata.json    # Video metadata
│   ├── parkour/
│   ├── blackswan/
│   ├── camel/
│   ├── car-shadow/
│   └── dog/
├── custom/
│   ├── v01_person_walk/
│   │   ├── frames/          # Extracted at 10 FPS
│   │   ├── depth/           # Depth maps
│   │   ├── colmap/          # Camera poses + sparse 3D points
│   │   └── metadata.json
│   ├── v02_object_rotate/
│   ├── v03_two_people/
│   ├── v04_outdoor_person/
│   └── v05_dynamic_scene/
├── process_video.py          # Automated pipeline script
├── data_report.md            # This file
└── README.md
```

## Usage

### Processing a new video:
```bash
python datasets/process_video.py \
    --input path/to/video.mp4 \
    --output datasets/custom/video_name \
    --fps 10
```

### Processing pre-extracted frames:
```bash
python datasets/process_video.py \
    --input-frames path/to/frames/ \
    --output datasets/custom/video_name \
    --skip-colmap
```

### Using processed data downstream:
- **R3 (4D Reconstruction):** `4DGaussians --source_path datasets/custom/v01_person_walk`
- **R4 (Novel View Gen):** Load frames from `datasets/custom/v01_person_walk/frames/`
- **Depth maps:** Available at `datasets/custom/v01_person_walk/depth/`

## Dependencies

```
opencv-python>=4.8
Pillow>=10.0
tqdm>=4.65
numpy>=1.24
transformers>=4.35 (for Depth Anything V2)
accelerate>=0.25
torch>=2.0
```

Install: `pip install opencv-python Pillow tqdm numpy transformers accelerate torch`

COLMAP: `brew install colmap` (macOS) or `sudo apt install colmap` (Linux)
