# Datasets — Processed Video Data for BARF

This folder contains processed video datasets with depth maps and camera poses.

## Structure

```
datasets/
├── davis/              # DAVIS 2017 benchmark videos
│   ├── bear/
│   │   ├── frames/     # Extracted frames (JPEG)
│   │   ├── depth/      # Depth Anything V2 outputs (PNG)
│   │   └── metadata.json
│   ├── parkour/
│   ├── blackswan/
│   ├── camel/
│   ├── car-shadow/
│   └── dog/
├── custom/             # Our recorded videos
│   ├── v01_person_walk/
│   │   ├── frames/     # Extracted at 10 FPS
│   │   ├── depth/      # Depth maps
│   │   ├── colmap/     # Camera poses + sparse 3D
│   │   └── metadata.json
│   └── ...
├── process_video.py    # Automated processing pipeline
├── data_report.md      # Quality assessment report
└── README.md           # This file
```

## Quick Start

### Process a new video:
```bash
python datasets/process_video.py --input video.mp4 --output datasets/custom/my_video --fps 10
```

### Process pre-extracted frames:
```bash
python datasets/process_video.py --input-frames path/to/frames/ --output datasets/custom/my_video
```

## Dependencies
```bash
pip install opencv-python Pillow tqdm numpy transformers accelerate torch
# For COLMAP: brew install colmap (macOS) or sudo apt install colmap (Linux)
```

## Note on Large Files

Raw videos, frames, and depth maps are NOT committed to Git (too large).  
Use `process_video.py` to regenerate from source videos.  
Download DAVIS 2017: https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
