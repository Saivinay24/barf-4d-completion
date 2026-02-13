# R3 — 4D Reconstruction Runner

**Read `00_README.md` first for project overview.**

## Your Role

Produce ACTUAL partial 4D reconstructions — the raw material BARF works on.  
**This needs GPU access** —use Colab Pro or team GPU server.

## Tools You Need
```bash
# 4DGaussians (recommended)
git clone https://github.com/hustvl/4DGaussians
cd 4DGaussians
pip install -r requirements.txt

# OR Shape-of-Motion (backup)
git clone https://github.com/vye16/shape-of-motion
cd shape-of-motion
pip install -e .

# For gap detection
pip install open3d numpy scipy
```

## Week 1: Get 4DGS Running + Produce Reconstructions

### Day 1 (Feb 14): Clone + Install
```bash
git clone https://github.com/hustvl/4DGaussians
cd 4DGaussians

# Read their README carefully
# Note: GPU requirements, data format, etc.

pip install torch torchvision torchaudio
pip install -r requirements.txt

# Common issues:
# - Need CUDA 11.8 or 12.1
# - Need ninja: pip install ninja
# - May need to compile CUDA kernels
```

**Output:** Repo cloned, dependencies installing

### Day 2-3 (Feb 15-16): Run on Sample Data
Most repos include sample data. Run that first:

```bash
# Example (check their README for exact command):
python train.py --config configs/demo.yaml

# OR
bash scripts/run_demo.sh
```

**Goal:** Get ONE successful run producing a `.ply` or `.splat` file.

**If 4DGaussians fails, try Shape-of-Motion:**
```bash
cd shape-of-motion
python demo.py --input data/sample_video.mp4
```

**Output:** Sample reconstruction working

### Day 4-5 (Feb 17-18): Run on Real Data (R2's Videos)
Get processed data from R2:
- Frames
- Depth maps
- COLMAP camera poses (for custom videos)

```bash
# Example workflow for 4DGaussians:
python train.py \
  --source_path ../datasets/custom/v01_person_walk \
  --model_path outputs/v01_person_walk \
  --images frames \
  --colmap_dir colmap

# This trains 4D Gaussians on your video
# Output: .ply point cloud sequence or .splat file
```

**Run on 3 different videos** — mix of DAVIS + custom.

**Output:** 3 actual 4D reconstructions saved in `reconstructions/`

### Day 6-7 (Feb 19-20): Visualize + Save Outputs
- Load the `.ply` files in Open3D to verify they look reasonable
- Rotate 360° around each reconstruction
- Take screenshots of different angles
- Save all outputs with clear naming:
  ```
  reconstructions/
  ├── v01_person_walk/
  │   ├── frames_0000.ply
  │   ├── frames_0050.ply
  │   ├── frames_0100.ply
  │   └── ...
  ├── davis_bear/
  └── davis_parkour/
  ```

```python
# visualize.py
import open3d as o3d

pcd = o3d.io.read_point_cloud("reconstructions/v01_person_walk/frames_0000.ply")
o3d.visualization.draw_geometries([pcd])
```

**Output:** 3 full 4D reconstruction sequences

---

## Week 2: Gap Analysis + Detection + Integration

### Day 8-9 (Feb 22-23): 360° Gap Analysis
For each reconstruction:
1. Load in Open3D
2. Rotate view from 0° → 360° in 45° increments
3. Screenshot each angle
4. **Annotate where gaps are**

Create `gap_detection/gap_analysis.md`:

```markdown
# Gap Analysis — Partial Reconstructions

## Video: v01_person_walk

### 0° (Front)
![front](gap_detection/screenshots/v01_front.png)
✅ Dense reconstruction

### 45°
![45](gap_detection/screenshots/v01_45.png)
✅ Still good

### 90° (Side)
![90](gap_detection/screenshots/v01_90.png)
⚠️ Thinning out

### 135°
![135](gap_detection/screenshots/v01_135.png)
❌ Major gaps visible

### 180° (Back)
![180](gap_detection/screenshots/v01_180.png)
❌ Almost empty — ~70% missing

**Gap Coverage Estimate:** ~60% of full 360° has geometry.  
**Missing regions:** Back 120° arc, some sides.
```

**Output:** `gap_detection/gap_analysis.md` with screenshots

### Day 10-11 (Feb 24-25): Gap Detection Script
Create `gap_detection/detect.py`:

```python
#!/usr/bin/env python3
"""
Gap detector for partial 4D reconstructions
"""
import open3d as o3d
import numpy as np
import json
from pathlib import Path

def detect_gaps(pcd_path, voxel_size=0.02):
    """
    Detect gaps in a partial point cloud
    Returns: list of gap regions as dicts
    """
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    
    # Create voxel grid
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
    
    # Get occupied voxels
    voxels = voxel_grid.get_voxels()
    occupied = set()
    for v in voxels:
        occupied.add(tuple(v.grid_index))
    
    # Find empty voxels within bounding box
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    
    empty_voxels = []
    grid_size = ((max_bound - min_bound) / voxel_size).astype(int) + 1
    
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            for z in range(grid_size[2]):
                if (x, y, z) not in occupied:
                    # Check if surrounded by occupied (real gap,not exterior)
                    neighbors = 0
                    for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
                        if (x+dx, y+dy, z+dz) in occupied:
                            neighbors += 1
                    
                    if neighbors >= 2:  # Likely a gap
                        center = min_bound + np.array([x, y, z]) * voxel_size
                        empty_voxels.append({
                            "center": center.tolist(),
                            "neighbors": neighbors
                        })
    
    # Cluster empty voxels into gap regions
    # (Simple spatial clustering)
    gaps = []
    if empty_voxels:
        centers = np.array([v["center"] for v in empty_voxels])
        
        # Use DBSCAN-like clustering via Open3D
        gap_pcd = o3d.geometry.PointCloud()
        gap_pcd.points = o3d.utility.Vector3dVector(centers)
        labels = np.array(gap_pcd.cluster_dbscan(eps=voxel_size * 3, min_points=5))
        
        for label in set(labels):
            if label == -1:
                continue
            mask = labels == label
            cluster_centers = centers[mask]
            
            gap = {
                "id": len(gaps),
                "center": cluster_centers.mean(axis=0).tolist(),
                "size": len(cluster_centers) * (voxel_size ** 3),
                "bounding_box": {
                    "min": cluster_centers.min(axis=0).tolist(),
                    "max": cluster_centers.max(axis=0).tolist()
                }
            }
            gaps.append(gap)
    
    return sorted(gaps, key=lambda g: g["size"], reverse=True)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input .ply file")
    parser.add_argument("--output", default="gaps.json", help="Output JSON")
    parser.add_argument("--voxel-size", type=float, default=0.02)
    args = parser.parse_args()
    
    gaps = detect_gaps(args.input, args.voxel_size)
    
    with open(args.output, 'w') as f:
        json.dump(gaps, f, indent=2)
    
    print(f"Found {len(gaps)} gap regions")
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
```

**Test it:**
```bash
python gap_detection/detect.py --input reconstructions/v01_person_walk/frames_0000.ply --output gaps_v01_0000.json
```

**Output:** `gap_detection/detect.py` producing JSON gap coordinates

### Day 12-14 (Feb 26-28): Integration + Help Team
- Run gap detection on all your reconstructions
- Feed results to R5 (viewer should show gaps as red highlights)
- Feed partial reconstructions to R6 (VR viewer)
- Help R0 integrate your outputs into the pipeline
- Create `reconstructions/README.md` explaining file formats

**Output:** Everything integrated and documented

---

## Your Deliverables by Feb 28

```
✅ 4DGaussians (or Shape-of-Motion) running
✅ 3+ videos reconstructed as 4D point clouds
✅ 360° gap analysis with annotated screenshots
✅ Gap detection script (detect.py) outputting JSON
✅ All reconstruction files in repo
✅ Integration with R0/R5/R6 complete
```

---

## What to Do If Stuck

**If 4DGaussians won't install (Day 3):**
- Try Shape-of-Motion
- Try gaussian-splatting (original, static version)
- Ask R0 for help debugging CUDA issues

**If training fails / crashes:**
- Reduce resolution
- Use fewer frames  
- Try on simpler videos first (static object vs moving person)

**If no method works by Day 5:**
- Download pre-trained outputs from papers' project pages
- Use R1's benchmark outputs
- Focus on gap detection with existing point clouds

---

## Demo Day — Your Part (3 min)

Show:
1. "We ran 4DGaussians on our videos" (show command)
2. "Here's the reconstruction" (load in Open3D, rotate)
3. "Notice the gaps at 180°" (show rear view screenshot)
4. "Our gap detector finds them automatically" (show gaps.json)
5. "This feeds into R4's generation step"

**Message:** "4D reconstruction works, but has gaps. Here's proof."
