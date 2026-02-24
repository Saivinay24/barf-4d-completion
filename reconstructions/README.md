# 4D Reconstructions & Gap Detection

This folder contains partial 4D reconstructions and gap detection outputs.

## Structure

```
reconstructions/
├── gap_detection/
│   ├── detect.py           # Gap detection script (voxel-based + DBSCAN)
│   └── gap_analysis.md     # 360° rotation analysis with annotated findings
├── [video_name]/           # Per-video reconstruction outputs
│   ├── frames_0000.ply     # Point cloud for frame 0
│   ├── frames_0050.ply     # Point cloud for frame 50
│   └── ...
└── README.md               # This file
```

## Gap Detection

### Run on a single .ply file:
```bash
python reconstructions/gap_detection/detect.py \
    --input reconstruction.ply \
    --output gaps.json \
    --voxel-size 0.02 \
    --coverage
```

### Run on a directory:
```bash
python reconstructions/gap_detection/detect.py \
    --input-dir reconstructions/v01_person_walk/ \
    --output-dir gaps/v01/
```

### Output format (gaps.json):
```json
{
  "gaps": [
    {
      "id": 0,
      "center": [1.2, 0.5, -0.8],
      "size": 0.0042,
      "voxel_count": 156,
      "severity": "high",
      "bounding_box": {"min": [...], "max": [...]}
    }
  ],
  "coverage": {
    "coverage_360_percent": 55.2,
    "front_coverage_percent": 92.0,
    "back_coverage_percent": 18.5
  }
}
```

## Dependencies
```bash
pip install open3d numpy scipy
```

## Note on Large Files
`.ply` and `.splat` files are gitignored (too large).  
Use Git LFS or external storage for actual point cloud data.
