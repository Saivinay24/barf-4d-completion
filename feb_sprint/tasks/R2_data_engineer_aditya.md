# R2 — Data + Depth Pipeline

**Read `00_README.md` first for project overview.**

## Your Role

Build professional benchmark dataset with depth maps, camera poses, and segmentation masks.  
**Day 1: Download existing datasets, not record videos.**

## Tools You Need
```bash
pip install torch torchvision transformers opencv-python pillow
# Depth Anything V2
pip install transformers accelerate

# COLMAP (for camera poses)
# macOS: brew install colmap
# Linux: sudo apt install colmap
# OR use Docker: docker pull colmap/colmap

# SAM2 for segmentation
pip install segment-anything-2
```

## Week 1: Download + Process Datasets

### Day 1 (Feb 14): Download DAVIS + Sintel
```bash
# DAVIS2017 dataset (standard dynamic video benchmark)
mkdir -p datasets/davis
cd datasets/davis
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
unzip DAVIS-2017-trainval-480p.zip

# Sintel dataset (optical flow benchmark with dynamic scenes)
mkdir -p datasets/sintel
cd datasets/sintel
wget http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip
unzip MPI-Sintel-complete.zip
```

Also record 5 custom videos on your phone:
- Person walking (15s)
- Object on table rotating it by hand (10s)
- Two people talking (15s)
- Outdoor scene with person (20s)
- Any interesting dynamic scene (15s)

Requirements: 1080p, 30fps, good lighting, stable hands

**Output:** DAVIS + Sintel downloaded, 5 custom videos recorded

###Day 2-3 (Feb 15-16): Install + Test Depth Anything V2
```python
# test_depth.py
from transformers import pipeline
from PIL import Image
import numpy as np

# Load Depth Anything V2
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large")

# Test on a single image
image = Image.open("datasets/davis/JPEGImages/480p/bear/00000.jpg")
depth = pipe(image)

# Save depth map
depth_map = depth["depth"]
depth_map.save("test_depth.png")
print("Depth map saved!")
```

Run this script. Verify output looks correct.

**Output:** Depth Anything V2 working, test depth map generated

### Day 3-4 (Feb 16-17): Batch Process All Frames
Create `datasets/process_depth.py`:

```python
import os
from pathlib import Path
from transformers import pipeline
from PIL import Image
from tqdm import tqdm

def process_video_depth(frames_dir, output_dir):
    """Process all frames in a directory with Depth Anything V2"""
    os.makedirs(output_dir, exist_ok=True)
    
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large")
    
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))])
    
    for frame_file in tqdm(frame_files, desc="Processing depth"):
        frame_path = os.path.join(frames_dir, frame_file)
        image = Image.open(frame_path)
        
        depth = pipe(image)
        depth_map = depth["depth"]
        
        output_path = os.path.join(output_dir, frame_file.replace('.jpg', '_depth.png'))
        depth_map.save(output_path)

if __name__ == "__main__":
    # Process DAVIS bear sequence
    process_video_depth(
        "datasets/davis/JPEGImages/480p/bear/",
        "datasets/davis/Depth/bear/"
    )
```

Run on at least 3 DAVIS videos + your custom videos.

**Output:** Depth maps for multiple videos

### Day 5-6 (Feb 18-19): COLMAP for Camera Poses
For your custom videos, run COLMAP to get camera poses:

```bash
# For each custom video:
mkdir -p datasets/custom/v01_person_walk/colmap

# Extract frames first
ffmpeg -i v01_person_walk.mp4 -qscale:v 2 datasets/custom/v01_person_walk/frames/frame_%04d.jpg

# Run COLMAP
cd datasets/custom/v01_person_walk
colmap feature_extractor --database_path database.db --image_path frames/
colmap exhaustive_matcher --database_path database.db
mkdir sparse
colmap mapper --database_path database.db --image_path frames/ --output_path sparse/

# Convert to text format for easier reading
colmap model_converter --input_path sparse/0 --output_path colmap/ --output_type TXT
```

This gives you:
- `cameras.txt` — camera intrinsics
- `images.txt` — camera poses per frame
- `points3D.txt` — sparse 3D points

**Output:** COLMAP camera poses for all custom videos

### Day 7 (Feb 20): Organize Dataset Structure
Create clean structure:

```
datasets/
├── davis/
│   ├── bear/
│   │   ├── frames/          ← JPEGImages from DAVIS
│   │   ├── depth/           ← Your Depth Anything outputs
│   │   └── metadata.json
│   ├── parkour/
│   └── ...
├── custom/
│   ├── v01_person_walk/
│   │   ├── frames/
│   │   ├── depth/
│   │   ├── colmap/          ← camera poses + sparse points
│   │   └── metadata.json
│   └── ...
└── README.md
```

Create `metadata.json` for each video:
```json
{
  "video_id": "v01_person_walk",
  "source": "custom",
  "duration_frames": 450,
  "fps": 30,
  "resolution": "1920x1080",
  "has_depth": true,
  "has_colmap": true,
  "has_segmentation": false,
  "description": "Person walking outdoors, filmed from stationary camera"
}
```

**Output:** Clean organized dataset, all metadata documented

---

## Week 2: Segmentation + Automation + Quality Check

### Day 8-9 (Feb 22-23): SAM2 Segmentation
For videos with clear foreground subjects (person, object), run SAM2:

```python
# segment_video.py
from segment_anything_2 import SAM2VideoPredictor
import cv2
import os

def segment_video(video_frames_dir, output_masks_dir):
    """Segment foreground object using SAM2"""
    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")
    
    # Load first frame, manually mark a point on the object
    # (or use automatic prompting)
    frames = sorted(os.listdir(video_frames_dir))
    first_frame = cv2.imread(os.path.join(video_frames_dir, frames[0]))
    
    # TODO: Add point prompt on object
    # For now, use automatic mode
    masks = predictor.predict_video(video_frames_dir)
    
    # Save masks
    for i, mask in enumerate(masks):
        cv2.imwrite(
            os.path.join(output_masks_dir, f"mask_{i:04d}.png"),
            mask * 255
        )

# Run on select videos
segment_video("datasets/custom/v01_person_walk/frames/", "datasets/custom/v01_person_walk/masks/")
```

**Output:** Segmentation masks for key videos

### Day 10-11 (Feb 24-25): Build Automated Pipeline
Create `datasets/process_video.py` — one command to do everything:

```python
#!/usr/bin/env python3
"""
Automated video processing pipeline
Usage: python process_video.py --input video.mp4 --output datasets/processed/video_name
"""
import argparse
import os
import subprocess
from pathlib import Path
from transformers import pipeline
from PIL import Image
from tqdm import tqdm

def extract_frames(video_path, output_dir, fps=10):
    """Extract frames at specified FPS"""
    os.makedirs(output_dir, exist_ok=True)
    cmd = f"ffmpeg -i {video_path} -vf fps={fps} {output_dir}/frame_%04d.jpg"
    subprocess.run(cmd, shell=True, check=True)

def process_depth(frames_dir, depth_dir):
    """Run Depth Anything V2 on all frames"""
    # (Use code from earlier)
    pass

def run_colmap(frames_dir, colmap_dir):
    """Run COLMAP for camera pose estimation"""
    # (Use code from earlier)
    pass

def create_metadata(output_dir, video_path):
    """Generate metadata.json"""
    # Count frames, get resolution, etc.
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input video file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--skip-colmap", action="store_true")
    args = parser.parse_args()
    
    output = Path(args.output)
    frames_dir = output / "frames"
    depth_dir = output / "depth"
    colmap_dir = output / "colmap"
    
    print(f"Processing {args.input}...")
    extract_frames(args.input, frames_dir, args.fps)
    process_depth(frames_dir, depth_dir)
    
    if not args.skip_colmap:
        run_colmap(frames_dir, colmap_dir)
    
    create_metadata(output, args.input)
    print(f"Done! Output in {output}")

if __name__ == "__main__":
    main()
```

**Output:** One-command processing script

### Day 12-14 (Feb 26-28): Process More Videos + Quality Check
- Run automated pipeline on 5 more videos
- **Quality check** everything:
  - Are depth maps reasonable?
  - Did COLMAP succeed?
  - Any corrupted frames?
- Create `datasets/README.md`:
  - Explain folder structure
  - List all videos
  - Explain how to use the data
- Create `datasets/data_report.md`:
  - Table of all videos
  - Quality assessment
  - Sample visualizations (depth maps, COLMAP point clouds)

**Output:** At least 10 fully processed videos, quality report

---

## Your Deliverables by Feb 28

```
✅ 10+ processed videos (DAVIS + custom)
✅ Depth maps for every frame (Depth Anything V2)
✅ COLMAP camera poses for custom videos
✅ Segmentation masks for 3+ videos
✅ Automated process_video.py script
✅ datasets/README.md explaining structure
✅ datasets/data_report.md with quality assessment
```

---

## Demo Day — Your Part (2 min)

Show:
1. "We have 10+ benchmark videos" (show list)
2. "Each has depth maps + camera poses" (show visualization)
3. "Here's our automated pipeline" (run process_video.py on new video)
4. "This data feeds into R3's 4D reconstruction"

**Message:** "Professional dataset, production-ready pipeline."
