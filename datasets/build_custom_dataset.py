import subprocess
from pathlib import Path
import json
import cv2

# ---------------------------------------------------
# AUTO PATHS
# ---------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = PROJECT_ROOT / "datasets"
CUSTOM = DATASETS / "custom"
VIDEOS = CUSTOM / "videos"

print("Project:", PROJECT_ROOT)

if not VIDEOS.exists():
    raise FileNotFoundError("videos folder not found")

# ---------------------------------------------------
# UTIL
# ---------------------------------------------------

def run(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, frames, f"{w}x{h}"


# ---------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------

for video in VIDEOS.glob("*.mp4"):

    video_id = video.stem
    print(f"\n========== {video_id} ==========")

    seq_dir = CUSTOM / video_id
    frames_dir = seq_dir / "frames"
    depth_dir = seq_dir / "depth"
    colmap_dir = seq_dir / "colmap"
    sparse_dir = seq_dir / "sparse"

    frames_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(exist_ok=True)
    colmap_dir.mkdir(exist_ok=True)
    sparse_dir.mkdir(exist_ok=True)

    # ---------------------------------------------------
    # 1️⃣ Extract Frames (FFMPEG)
    # ---------------------------------------------------

    if not any(frames_dir.glob("*.jpg")):
        run([
            "ffmpeg",
            "-i", str(video),
            "-qscale:v", "2",
            str(frames_dir / "frame_%04d.jpg")
        ])

    # ---------------------------------------------------
    # 2️⃣ COLMAP FEATURE EXTRACTION
    # ---------------------------------------------------

    database = seq_dir / "database.db"

    run([
        "colmap", "feature_extractor",
        "--database_path", str(database),
        "--image_path", str(frames_dir),
        "--ImageReader.single_camera", "1"
    ])

    # ---------------------------------------------------
    # 3️⃣ MATCHING
    # ---------------------------------------------------

    run([
        "colmap", "exhaustive_matcher",
        "--database_path", str(database)
    ])

    # ---------------------------------------------------
    # 4️⃣ MAPPER (POSES)
    # ---------------------------------------------------

    run([
        "colmap", "mapper",
        "--database_path", str(database),
        "--image_path", str(frames_dir),
        "--output_path", str(sparse_dir)
    ])

    # ---------------------------------------------------
    # 5️⃣ CONVERT TO TXT
    # ---------------------------------------------------

    run([
        "colmap", "model_converter",
        "--input_path", str(sparse_dir / "0"),
        "--output_path", str(colmap_dir),
        "--output_type", "TXT"
    ])

    # ---------------------------------------------------
    # 6️⃣ METADATA.JSON
    # ---------------------------------------------------

    fps, total_frames, resolution = get_video_info(video)

    metadata = {
        "video_id": video_id,
        "source": "custom",
        "duration_frames": total_frames,
        "fps": fps,
        "resolution": resolution,
        "has_depth": False,
        "has_colmap": True,
        "has_segmentation": False,
        "description": f"Custom capture: {video_id}"
    }

    with open(seq_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

print("\n✅ CUSTOM DATASET BUILD COMPLETE")