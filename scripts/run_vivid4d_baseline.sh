#!/usr/bin/env bash
# =============================================================================
# BARF 4D — Vivid4D Baseline Runner
#
# Runs Vivid4D on the same test videos used for BARF evaluation.
# Vivid4D is the closest academic prior work (ICCV 2025).
# Its output is the "prior work baseline" in the paper ablation table.
#
# Usage: bash scripts/run_vivid4d_baseline.sh <video_path> [output_dir]
#
# [GPU REQUIRED] — Needs CUDA GPU with ~24GB VRAM (A100 or H100)
# Recommended: Vast.ai H100 at ~$2/hr, or Colab Pro A100
#
# Paper: https://arxiv.org/abs/2504.11092
# =============================================================================

set -e

VIDEO_PATH="${1:-}"
OUTPUT_DIR="${2:-outputs/vivid4d_baseline/$(basename ${VIDEO_PATH%.*})}"

if [ -z "$VIDEO_PATH" ]; then
  echo "Usage: bash scripts/run_vivid4d_baseline.sh <video_path> [output_dir]"
  echo "Example: bash scripts/run_vivid4d_baseline.sh data/test_video.mp4"
  exit 1
fi

echo "============================================"
echo "  Vivid4D Baseline Runner"
echo "  Input:  $VIDEO_PATH"
echo "  Output: $OUTPUT_DIR"
echo "============================================"

mkdir -p "$OUTPUT_DIR"

# =============================================================================
# SETUP (run once per machine)
# =============================================================================
# TODO: GPU EXECUTION REQUIRED
#
# 1. Clone Vivid4D:
#    git clone https://github.com/NVlabs/Vivid4D      # check actual repo URL
#    # Alternatively, Vivid4D was published at ICCV 2025:
#    # See project page: look for official code release linked from arxiv 2504.11092
#    cd Vivid4D
#
# 2. Install dependencies:
#    conda create -n vivid4d python=3.10
#    conda activate vivid4d
#    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
#    pip install -r requirements.txt
#
# 3. Download checkpoints:
#    # Follow the Vivid4D README — checkpoints are typically on HuggingFace or Google Drive
#    # bash scripts/download_checkpoints.sh
#
# =============================================================================
# RUN (after setup)
# =============================================================================
# TODO: GPU EXECUTION REQUIRED
#
# Option A — if Vivid4D has a video inference script:
#   python inference.py \
#     --video "$VIDEO_PATH" \
#     --output_dir "$OUTPUT_DIR" \
#     --num_views 8 \
#     --inpainting_steps 20 \
#     --save_ply
#
# Option B — if Vivid4D requires pre-extracted frames:
#   # Extract frames first
#   python tools/extract_frames.py --video "$VIDEO_PATH" --output "$OUTPUT_DIR/frames"
#
#   # Run monocular depth estimation
#   python tools/run_depth.py --frames "$OUTPUT_DIR/frames" --output "$OUTPUT_DIR/depth"
#
#   # Run view augmentation + inpainting
#   python run_view_augmentation.py \
#     --frames "$OUTPUT_DIR/frames" \
#     --depth "$OUTPUT_DIR/depth" \
#     --output "$OUTPUT_DIR/augmented_views" \
#     --target_angles 0 45 90 135 180 225 270 315
#
#   # Reconstruct 4DGS from augmented multi-view
#   python reconstruct.py \
#     --input_dir "$OUTPUT_DIR/augmented_views" \
#     --output_ply "$OUTPUT_DIR/vivid4d_scene.ply"
#
# =============================================================================
# EXPECTED OUTPUTS
# =============================================================================
# After successful GPU run, collect:
#   $OUTPUT_DIR/vivid4d_scene.ply           — 4DGS reconstruction
#   $OUTPUT_DIR/renders/angle_000.png       — rendered front view
#   $OUTPUT_DIR/renders/angle_045.png
#   $OUTPUT_DIR/renders/angle_090.png
#   $OUTPUT_DIR/renders/angle_135.png
#   $OUTPUT_DIR/renders/angle_180.png       — rendered back view (key metric!)
#   $OUTPUT_DIR/renders/angle_225.png
#   $OUTPUT_DIR/renders/angle_270.png
#   $OUTPUT_DIR/renders/angle_315.png
#
# Then run gap detection on the Vivid4D output:
#   python -m src.gap_detection.detect_gaps \
#     --input "$OUTPUT_DIR/vivid4d_scene.ply" \
#     --output_json "$OUTPUT_DIR/gaps.json" \
#     --output_heatmap_dir "$OUTPUT_DIR/heatmaps"
#
# Then compute VRC-Score for comparison:
#   python -m src.metrics.vrc_score \
#     --scene_ply "$OUTPUT_DIR/vivid4d_scene.ply" \
#     --output "$OUTPUT_DIR/vrc_score.json"
#
# =============================================================================
# WHAT TO RECORD FOR THE PAPER ABLATION TABLE
# =============================================================================
# Fill in results/ablation_table.md after running:
#   1. VRC-Score (Coverage, Coherence, Quality, Composite)
#   2. Angular coverage % at each of 8 test angles
#   3. Runtime (seconds/frame on H100)
#   4. LPIPS at back view (180°) vs ground truth if available
#
# Target (based on Vivid4D paper results): ~68% angular coverage, VRC ~0.57
# Our BARF target: ~91% angular coverage, VRC ~0.82

echo ""
echo "[STATUS] Vivid4D baseline script ready."
echo "  This script contains full setup and run instructions (all marked TODO)."
echo "  Run on Vast.ai H100 or Colab Pro A100."
echo ""
echo "  After GPU run, collect outputs and run:"
echo "    python -m src.gap_detection.detect_gaps --input <vivid4d_scene.ply> ..."
echo "    python -m src.metrics.vrc_score --scene_ply <vivid4d_scene.ply> ..."
echo "    Then update results/ablation_table.md"
