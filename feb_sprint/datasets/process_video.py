#!/usr/bin/env python3
"""
Automated video processing pipeline for BARF.
Extracts frames, generates depth maps, and runs COLMAP for camera poses.

Usage:
    python process_video.py --input video.mp4 --output datasets/processed/video_name
    python process_video.py --input video.mp4 --output datasets/processed/video_name --skip-colmap
    python process_video.py --input-frames path/to/frames/ --output datasets/processed/video_name
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def extract_frames(video_path: str, output_dir: str, fps: int = 10) -> int:
    """Extract frames from video at specified FPS using ffmpeg or OpenCV."""
    os.makedirs(output_dir, exist_ok=True)

    # Try ffmpeg first (faster, better quality)
    try:
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", f"fps={fps}",
            "-q:v", "2",
            os.path.join(output_dir, "frame_%04d.jpg")
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        frame_count = len([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
        print(f"  Extracted {frame_count} frames at {fps} FPS using ffmpeg")
        return frame_count
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("  ffmpeg not found, falling back to OpenCV...")

    # Fallback: OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(video_fps / fps))
    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            out_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1
        frame_idx += 1

    cap.release()
    print(f"  Extracted {saved_count} frames at {fps} FPS using OpenCV")
    return saved_count


def process_depth(frames_dir: str, depth_dir: str, model_name: str = "depth-anything/Depth-Anything-V2-Large") -> None:
    """Run Depth Anything V2 on all frames."""
    os.makedirs(depth_dir, exist_ok=True)

    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        print("  ⚠️  transformers not installed. Install with: pip install transformers accelerate")
        print("  Skipping depth estimation.")
        return

    print(f"  Loading depth model: {model_name}")
    try:
        pipe = hf_pipeline(task="depth-estimation", model=model_name)
    except Exception as e:
        print(f"  ⚠️  Could not load depth model: {e}")
        print("  Trying smaller model: depth-anything/Depth-Anything-V2-Small")
        try:
            pipe = hf_pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small")
        except Exception as e2:
            print(f"  ⚠️  Could not load any depth model: {e2}")
            print("  Skipping depth estimation. Install with: pip install transformers torch")
            return

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))])

    for frame_file in tqdm(frame_files, desc="  Depth estimation"):
        frame_path = os.path.join(frames_dir, frame_file)
        output_path = os.path.join(depth_dir, frame_file.replace('.jpg', '_depth.png').replace('.png', '_depth.png'))

        if os.path.exists(output_path):
            continue  # Skip already processed

        image = Image.open(frame_path)
        result = pipe(image)
        depth_map = result["depth"]
        depth_map.save(output_path)

    print(f"  Saved {len(frame_files)} depth maps to {depth_dir}")


def run_colmap(frames_dir: str, colmap_dir: str) -> bool:
    """Run COLMAP for camera pose estimation."""
    os.makedirs(colmap_dir, exist_ok=True)

    # Check if COLMAP is installed
    try:
        subprocess.run(["colmap", "--help"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("  ⚠️  COLMAP not found. Install with:")
        print("       macOS: brew install colmap")
        print("       Linux: sudo apt install colmap")
        print("  Skipping camera pose estimation.")
        return False

    db_path = os.path.join(colmap_dir, "database.db")
    sparse_dir = os.path.join(colmap_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)

    print("  Running COLMAP feature extraction...")
    try:
        subprocess.run([
            "colmap", "feature_extractor",
            "--database_path", db_path,
            "--image_path", frames_dir,
            "--ImageReader.single_camera", "1",
        ], check=True, capture_output=True)

        print("  Running COLMAP exhaustive matching...")
        subprocess.run([
            "colmap", "exhaustive_matcher",
            "--database_path", db_path,
        ], check=True, capture_output=True)

        print("  Running COLMAP mapper...")
        subprocess.run([
            "colmap", "mapper",
            "--database_path", db_path,
            "--image_path", frames_dir,
            "--output_path", sparse_dir,
        ], check=True, capture_output=True)

        # Convert to text format
        model_path = os.path.join(sparse_dir, "0")
        if os.path.exists(model_path):
            subprocess.run([
                "colmap", "model_converter",
                "--input_path", model_path,
                "--output_path", colmap_dir,
                "--output_type", "TXT",
            ], check=True, capture_output=True)
            print("  COLMAP completed successfully!")
            return True
        else:
            print("  ⚠️  COLMAP mapper produced no output (reconstruction failed)")
            return False

    except subprocess.CalledProcessError as e:
        print(f"  ⚠️  COLMAP failed: {e}")
        return False


def get_video_info(video_path: str) -> dict:
    """Get video metadata using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration_seconds": cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(1, cap.get(cv2.CAP_PROP_FPS)),
    }
    cap.release()
    return info


def create_metadata(output_dir: str, video_path: str = None, frames_dir: str = None,
                    has_depth: bool = False, has_colmap: bool = False) -> dict:
    """Generate metadata.json for the processed video."""
    video_id = os.path.basename(output_dir)

    if video_path and os.path.exists(video_path):
        info = get_video_info(video_path)
        resolution = f"{info.get('width', '?')}x{info.get('height', '?')}"
        fps = info.get('fps', 0)
        duration_frames = info.get('frame_count', 0)
    else:
        resolution = "unknown"
        fps = 0
        duration_frames = 0

    # Count extracted frames
    check_dir = frames_dir or os.path.join(output_dir, "frames")
    if os.path.exists(check_dir):
        extracted_frames = len([f for f in os.listdir(check_dir) if f.endswith(('.jpg', '.png'))])
    else:
        extracted_frames = 0

    metadata = {
        "video_id": video_id,
        "source": "davis" if "davis" in video_id.lower() else "custom",
        "original_video": os.path.basename(video_path) if video_path else None,
        "resolution": resolution,
        "original_fps": fps,
        "duration_frames": duration_frames,
        "extracted_frames": extracted_frames,
        "has_depth": has_depth,
        "has_colmap": has_colmap,
        "has_segmentation": False,
        "processed_date": datetime.now().isoformat(),
        "description": f"Processed video: {video_id}",
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Metadata saved to {metadata_path}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="BARF video processing pipeline")
    parser.add_argument("--input", help="Input video file (.mp4, .avi, etc.)")
    parser.add_argument("--input-frames", help="Input directory of pre-extracted frames")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--fps", type=int, default=10, help="Frame extraction FPS (default: 10)")
    parser.add_argument("--skip-depth", action="store_true", help="Skip depth estimation")
    parser.add_argument("--skip-colmap", action="store_true", help="Skip COLMAP pose estimation")
    parser.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Large",
                        help="Depth estimation model name")
    args = parser.parse_args()

    if not args.input and not args.input_frames:
        parser.error("Either --input (video) or --input-frames (directory) is required")

    output = Path(args.output)
    frames_dir = str(output / "frames")
    depth_dir = str(output / "depth")
    colmap_dir = str(output / "colmap")

    print(f"{'='*60}")
    print(f"BARF Video Processing Pipeline")
    print(f"{'='*60}")

    # Step 1: Extract frames
    if args.input_frames:
        print(f"\n[1/4] Using pre-extracted frames from {args.input_frames}")
        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir, exist_ok=True)
            # Copy or symlink frames
            import shutil
            for f in sorted(os.listdir(args.input_frames)):
                if f.endswith(('.jpg', '.png')):
                    shutil.copy2(os.path.join(args.input_frames, f), os.path.join(frames_dir, f))
    elif args.input:
        print(f"\n[1/4] Extracting frames from {args.input}")
        extract_frames(args.input, frames_dir, args.fps)
    else:
        print("Error: No input specified")
        sys.exit(1)

    # Step 2: Depth estimation
    has_depth = False
    if not args.skip_depth:
        print(f"\n[2/4] Running depth estimation")
        process_depth(frames_dir, depth_dir, args.depth_model)
        has_depth = os.path.exists(depth_dir) and len(os.listdir(depth_dir)) > 0
    else:
        print(f"\n[2/4] Skipping depth estimation (--skip-depth)")

    # Step 3: COLMAP
    has_colmap = False
    if not args.skip_colmap:
        print(f"\n[3/4] Running COLMAP for camera poses")
        has_colmap = run_colmap(frames_dir, colmap_dir)
    else:
        print(f"\n[3/4] Skipping COLMAP (--skip-colmap)")

    # Step 4: Metadata
    print(f"\n[4/4] Generating metadata")
    create_metadata(str(output), args.input, frames_dir, has_depth, has_colmap)

    print(f"\n{'='*60}")
    print(f"✅ Processing complete!")
    print(f"   Output: {output}")
    print(f"   Frames: {frames_dir}")
    print(f"   Depth:  {'✅' if has_depth else '❌ skipped/failed'}")
    print(f"   COLMAP: {'✅' if has_colmap else '❌ skipped/failed'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
