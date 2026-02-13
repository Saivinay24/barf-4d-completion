# Datasets — Aditya (R2)

This folder contains processed video datasets with depth maps and camera poses.

## Structure

```
datasets/
├── davis/              # DAVIS 2017 benchmark
│   ├── bear/
│   │   ├── frames/
│   │   ├── depth/
│   │   └── metadata.json
│   └── ...
├── sintel/             # MPI Sintel benchmark
├── custom/             # Our recorded videos
│   ├── v01_person_walk/
│   │   ├── frames/
│   │   ├── depth/
│   │   ├── colmap/
│   │   └── metadata.json
│   └── ...
└── process_video.py    # Automated pipeline script
```

## Current Status

Coming soon. See `tasks/R2_data_engineer.md` for your task breakdown.

## Note on Large Files

Raw videos and datasets are NOT committed to Git (too large).  
Download links and processing scripts will be provided.
