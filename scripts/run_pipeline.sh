#!/usr/bin/env bash
# =============================================================================
# BARF 4D — End-to-End Pipeline
# Usage: bash scripts/run_pipeline.sh <video_path> [output_dir]
#
# Pipeline: Video → D4RT (poses+tracks) → NeoVerse (4DGS) → Gap Detection → PLY
#
# Steps marked [GPU REQUIRED] need a CUDA GPU (A100/H100 recommended).
# Run those steps on Vast.ai H100 (~$2/hr) or Colab Pro A100.
# Steps marked [LOCAL OK] run on CPU.
# =============================================================================

set -e

VIDEO_PATH="${1:-}"
OUTPUT_DIR="${2:-outputs/$(basename ${VIDEO_PATH%.*})}"

if [ -z "$VIDEO_PATH" ]; then
  echo "Usage: bash scripts/run_pipeline.sh <video_path> [output_dir]"
  echo "Example: bash scripts/run_pipeline.sh data/test_video.mp4"
  exit 1
fi

if [ ! -f "$VIDEO_PATH" ]; then
  echo "ERROR: Video file not found: $VIDEO_PATH"
  exit 1
fi

echo "============================================"
echo "  BARF 4D Pipeline"
echo "  Input:  $VIDEO_PATH"
echo "  Output: $OUTPUT_DIR"
echo "============================================"

mkdir -p "$OUTPUT_DIR/frames"
mkdir -p "$OUTPUT_DIR/d4rt"
mkdir -p "$OUTPUT_DIR/neoverse"
mkdir -p "$OUTPUT_DIR/gaps"
mkdir -p "$OUTPUT_DIR/completion"
mkdir -p "$OUTPUT_DIR/splat"

# -----------------------------------------------------------------------------
# STEP 1: Extract frames from video [LOCAL OK]
# -----------------------------------------------------------------------------
echo ""
echo "[STEP 1/5] Extracting frames..."
python - <<PYEOF
import cv2, os, sys
video_path = "$VIDEO_PATH"
out_dir = "$OUTPUT_DIR/frames"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"  Video: {total} frames @ {fps:.1f} FPS")
count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(f"{out_dir}/frame_{count:04d}.jpg", frame)
    count += 1
cap.release()
print(f"  Saved {count} frames to {out_dir}")
PYEOF

# -----------------------------------------------------------------------------
# STEP 2: D4RT — Camera pose estimation + point tracking [GPU REQUIRED]
# -----------------------------------------------------------------------------
echo ""
echo "[STEP 2/5] D4RT: Camera pose estimation + 4D point tracking"
echo "  [GPU REQUIRED] — Skip on CPU, run on Vast.ai H100 or Colab A100"
echo ""

# TODO: GPU EXECUTION REQUIRED
# Uncomment and run on a GPU machine:
#
# Setup:
#   git clone https://github.com/google-deepmind/d4rt
#   cd d4rt && pip install -e .
#
# Run:
#   python d4rt/run_d4rt.py \
#     --input_dir "$OUTPUT_DIR/frames" \
#     --output_dir "$OUTPUT_DIR/d4rt" \
#     --output_poses "$OUTPUT_DIR/d4rt/poses.json" \
#     --output_tracks "$OUTPUT_DIR/d4rt/tracks.npz" \
#     --output_depth "$OUTPUT_DIR/d4rt/depth_maps"
#
# Outputs expected:
#   $OUTPUT_DIR/d4rt/poses.json        — camera poses per frame
#   $OUTPUT_DIR/d4rt/tracks.npz        — 3D point tracks across frames
#   $OUTPUT_DIR/d4rt/depth_maps/       — per-frame depth estimates
#
# D4RT paper: https://arxiv.org/pdf/2512.08924
# Runtime: ~5 min for 100-frame video on H100

echo "  [PLACEHOLDER] D4RT outputs expected at: $OUTPUT_DIR/d4rt/"
echo "  Creating placeholder pose file for pipeline testing..."
python - <<PYEOF
import json, os, glob
frames = sorted(glob.glob("$OUTPUT_DIR/frames/*.jpg"))
n = len(frames)
# Placeholder: identity poses (will be replaced by real D4RT output)
poses = {
    "frames": [
        {"frame": i, "R": [[1,0,0],[0,1,0],[0,0,1]], "t": [0,0,i*0.01],
         "note": "PLACEHOLDER — replace with D4RT output"}
        for i in range(n)
    ],
    "status": "PLACEHOLDER — GPU run required"
}
with open("$OUTPUT_DIR/d4rt/poses.json", "w") as f:
    json.dump(poses, f, indent=2)
print(f"  Placeholder poses written ({n} frames)")
PYEOF

# -----------------------------------------------------------------------------
# STEP 3: NeoVerse — 4D Gaussian Splatting reconstruction [GPU REQUIRED]
# -----------------------------------------------------------------------------
echo ""
echo "[STEP 3/5] NeoVerse: 4D Gaussian Splatting reconstruction"
echo "  [GPU REQUIRED] — Skip on CPU, run on Vast.ai H100 or Colab A100"
echo ""

# TODO: GPU EXECUTION REQUIRED
# Uncomment and run on a GPU machine (needs ~40GB VRAM for full model):
#
# Setup:
#   git clone https://github.com/IamCreateAI/NeoVerse
#   cd NeoVerse && pip install -r requirements.txt
#   # Download checkpoints (see NeoVerse README for links)
#
# Run (pose-free, feed-forward):
#   python inference.py \
#     --video "$VIDEO_PATH" \
#     --output_dir "$OUTPUT_DIR/neoverse" \
#     --num_frames 49 \
#     --resolution 512
#
# OR with D4RT poses:
#   python inference.py \
#     --video "$VIDEO_PATH" \
#     --poses "$OUTPUT_DIR/d4rt/poses.json" \
#     --output_dir "$OUTPUT_DIR/neoverse" \
#     --num_frames 49
#
# Outputs expected:
#   $OUTPUT_DIR/neoverse/scene.ply         — 4D Gaussian Splat point cloud
#   $OUTPUT_DIR/neoverse/gaussians_t*.ply  — per-timestep PLY files
#   $OUTPUT_DIR/neoverse/render_*.png      — rendered views
#
# NeoVerse paper: https://arxiv.org/abs/2601.00393
# GitHub: https://github.com/IamCreateAI/NeoVerse
# Runtime: ~10-20 min for 49-frame video on H100

echo "  [PLACEHOLDER] NeoVerse output expected at: $OUTPUT_DIR/neoverse/scene.ply"
echo "  Creating synthetic placeholder PLY for pipeline testing..."
python - <<PYEOF
import numpy as np, struct, os

# Create a synthetic partial PLY (simulates front-facing reconstruction with gaps)
# Points form a half-sphere (front-facing only) to represent the reconstruction gap
rng = np.random.default_rng(42)
n_points = 5000

# Front-facing hemisphere: theta in [-pi/2, pi/2], phi in [0, pi]
theta = rng.uniform(-np.pi/2, np.pi/2, n_points)   # azimuth (left/right)
phi   = rng.uniform(0, np.pi, n_points)              # elevation
r     = rng.uniform(0.8, 1.2, n_points)

x = r * np.sin(phi) * np.cos(theta)
y = r * np.cos(phi)
z = r * np.sin(phi) * np.sin(theta)

# Only keep points in front half (z > -0.1) — simulates monocular gap
mask = z > -0.1
x, y, z = x[mask], y[mask], z[mask]
n = len(x)

# Colors: warm tones for visible surface
r_col = (np.clip(x/2 + 0.5, 0, 1) * 255).astype(np.uint8)
g_col = (np.clip(y/2 + 0.5, 0, 1) * 255).astype(np.uint8)
b_col = np.full(n, 128, dtype=np.uint8)

ply_path = "$OUTPUT_DIR/neoverse/scene.ply"
header = f"""ply
format ascii 1.0
comment BARF synthetic placeholder PLY (front-facing only, simulates NeoVerse output with gaps)
comment Replace with real NeoVerse output after GPU run
element vertex {n}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
with open(ply_path, "w") as f:
    f.write(header)
    for i in range(n):
        f.write(f"{x[i]:.6f} {y[i]:.6f} {z[i]:.6f} {r_col[i]} {g_col[i]} {b_col[i]}\n")
print(f"  Synthetic placeholder PLY: {n} points -> {ply_path}")
PYEOF

# -----------------------------------------------------------------------------
# STEP 4: Gap Detection [LOCAL OK]
# -----------------------------------------------------------------------------
echo ""
echo "[STEP 4/5] Gap detection — finding empty angular regions..."

PLY_PATH="$OUTPUT_DIR/neoverse/scene.ply"

python -m src.gap_detection.detect_gaps \
  --input "$PLY_PATH" \
  --output_json "$OUTPUT_DIR/gaps/gaps.json" \
  --output_heatmap_dir "$OUTPUT_DIR/gaps/heatmaps" \
  --voxel_size 0.05

echo "  Gap detection complete. Results in $OUTPUT_DIR/gaps/"

# -----------------------------------------------------------------------------
# STEP 5: BARF Completion Module [GPU REQUIRED]
# -----------------------------------------------------------------------------
echo ""
echo "[STEP 5/5] BARF Spherical Completion Module"
echo "  [GPU REQUIRED] — Skip on CPU, run on Vast.ai H100 or Colab A100"
echo ""

# TODO: GPU EXECUTION REQUIRED
# Uncomment and run on a GPU machine:
#
# python -m src.completion.spherical_completion \
#   --scene_ply "$OUTPUT_DIR/neoverse/scene.ply" \
#   --gaps_json "$OUTPUT_DIR/gaps/gaps.json" \
#   --output_ply "$OUTPUT_DIR/completion/scene_complete.ply" \
#   --device cuda
#
# Then run VR export:
#   python -m src.vr.export_splat \
#     --input "$OUTPUT_DIR/completion/scene_complete.ply" \
#     --output "$OUTPUT_DIR/splat/scene.splat" \
#     --max_gaussians 500000
#
# Runtime: ~30-60 min on H100 for full completion

echo "  [PLACEHOLDER] After GPU run, completed PLY will be at:"
echo "    $OUTPUT_DIR/completion/scene_complete.ply"
echo "  Quest-ready splat will be at:"
echo "    $OUTPUT_DIR/splat/scene.splat"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "============================================"
echo "  Pipeline complete (CPU-runnable steps done)"
echo "  Output directory: $OUTPUT_DIR"
echo ""
echo "  Files produced:"
ls -lh "$OUTPUT_DIR/neoverse/scene.ply" 2>/dev/null && echo "    ✅ Placeholder NeoVerse PLY"
ls -lh "$OUTPUT_DIR/d4rt/poses.json"    2>/dev/null && echo "    ✅ Placeholder D4RT poses"
ls -lh "$OUTPUT_DIR/gaps/gaps.json"     2>/dev/null && echo "    ✅ Gap detection results"
echo ""
echo "  Next steps (require GPU):"
echo "    1. Run D4RT on Vast.ai: see STEP 2 comments above"
echo "    2. Run NeoVerse on Vast.ai: see STEP 3 comments above"
echo "    3. Run completion module: see STEP 5 comments above"
echo "    4. See scripts/run_vivid4d_baseline.sh for Vivid4D baseline"
echo "============================================"
