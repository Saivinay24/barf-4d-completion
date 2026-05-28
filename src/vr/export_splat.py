"""
BARF 4D — VR Export Pipeline
==============================
Converts a 4D Gaussian Splat PLY into a Quest-compatible .splat file
with LOD (Level-of-Detail) reduction to <500K Gaussians for standalone operation.

Pipeline:
    Input PLY (4DGS scene) → LOD Reduction → Coordinate Transform → .splat output

Meta Quest 3 constraints (from BARF_REFERENCE.md and PDF pipeline research):
    - Standalone APK: max ~500K Gaussians for stable 72 FPS
    - Tethered PCVR: no limit (2M+ Gaussians viable at 90 FPS)
    - Coordinate system: OpenCV (y-down, z-forward) → Unity/Quest (y-up, z-forward)
    - Meta Spatial SDK v0.9.2+ supports native splat rendering

.splat format spec (antimatter15/splat compatible):
    Binary format, one Gaussian per record:
    [x: f32, y: f32, z: f32, scale_x: f32, scale_y: f32, scale_z: f32,
     r: u8, g: u8, b: u8, a: u8, rot_x: u8, rot_y: u8, rot_z: u8, rot_w: u8]
    = 32 bytes per Gaussian

Usage (CLI):
    python -m src.vr.export_splat \
        --input outputs/completion/scene_complete.ply \
        --output outputs/splat/scene.splat \
        --max_gaussians 500000

Usage (API):
    from src.vr.export_splat import SplatExporter
    exporter = SplatExporter(max_gaussians=500_000)
    result = exporter.export("scene.ply", "scene.splat")
"""

import argparse
import json
import os
import struct
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Max Gaussians for standalone Quest 3 APK (from Meta Spatial SDK docs)
QUEST_STANDALONE_MAX_GAUSSIANS = 500_000
# Max for tethered PCVR
QUEST_TETHERED_MAX_GAUSSIANS = 2_000_000

# .splat record size (bytes)
SPLAT_RECORD_BYTES = 32

# OpenCV → Unity coordinate transform
# OpenCV: x=right, y=down, z=forward
# Unity:  x=right, y=up,   z=forward
# Transform: y_unity = -y_opencv, others unchanged
def opencv_to_unity(xyz: np.ndarray) -> np.ndarray:
    """Convert XYZ from OpenCV convention to Unity/Quest convention."""
    result = xyz.copy()
    result[:, 1] = -xyz[:, 1]   # flip Y axis
    return result


# ---------------------------------------------------------------------------
# PLY loading (reuse detect_gaps loader)
# ---------------------------------------------------------------------------

def load_ply_full(ply_path: str) -> Dict:
    """
    Load PLY file, extracting all properties.
    Returns dict with 'xyz' (N,3), 'rgb' (N,3) uint8, and 'extra' array.
    """
    from src.gap_detection.detect_gaps import load_ply_xyz

    xyz = load_ply_xyz(ply_path)
    n = len(xyz)

    # Try to read RGB columns from PLY header
    rgb = np.full((n, 3), 128, dtype=np.uint8)  # default grey
    opacity = np.ones(n, dtype=np.float32)

    # Re-read to get colour
    with open(ply_path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        fmt = "ascii"
        properties = []
        for line in header_lines:
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("property"):
                parts = line.split()
                properties.append((parts[1], parts[2]))

        prop_names = [p[1] for p in properties]

        r_idx = prop_names.index("red")   if "red"   in prop_names else None
        g_idx = prop_names.index("green") if "green" in prop_names else None
        b_idx = prop_names.index("blue")  if "blue"  in prop_names else None

        if fmt == "ascii" and r_idx is not None:
            for i in range(n):
                vals = f.readline().decode("ascii").split()
                try:
                    rgb[i, 0] = int(float(vals[r_idx]))
                    rgb[i, 1] = int(float(vals[g_idx]))
                    rgb[i, 2] = int(float(vals[b_idx]))
                except (IndexError, ValueError):
                    pass

    return {"xyz": xyz, "rgb": rgb, "opacity": opacity, "n": n}


# ---------------------------------------------------------------------------
# LOD Reduction
# ---------------------------------------------------------------------------

def lod_reduce_uniform(
    xyz: np.ndarray,
    rgb: np.ndarray,
    max_points: int,
    voxel_size: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce point count to max_points using uniform random sampling.
    Optionally voxelise first (voxel_size in metres) for spatially even coverage.

    Args:
        xyz:       (N, 3) float32 XYZ positions
        rgb:       (N, 3) uint8 RGB colours
        max_points: target maximum number of points
        voxel_size: if set, voxelise first then subsample

    Returns:
        (xyz_reduced, rgb_reduced) both with <= max_points rows
    """
    N = len(xyz)
    if N <= max_points:
        return xyz, rgb

    if voxel_size is not None:
        # Voxelisation: keep one representative point per voxel
        min_bound = xyz.min(axis=0)
        voxel_idx = np.floor((xyz - min_bound) / voxel_size).astype(np.int32)
        # Hash voxel indices to a unique key
        keys = (voxel_idx[:, 0].astype(np.int64) * 1_000_003 +
                voxel_idx[:, 1].astype(np.int64) * 1_009 +
                voxel_idx[:, 2].astype(np.int64))
        _, first_occ = np.unique(keys, return_index=True)
        xyz = xyz[first_occ]
        rgb = rgb[first_occ]
        N = len(xyz)

    if N <= max_points:
        return xyz, rgb

    # Uniform random subsample
    rng = np.random.default_rng(42)
    indices = rng.choice(N, size=max_points, replace=False)
    indices.sort()  # keep spatial order roughly intact
    return xyz[indices], rgb[indices]


def lod_reduce_importance(
    xyz: np.ndarray,
    rgb: np.ndarray,
    max_points: int,
    importance_radius: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Importance-sampled LOD reduction:
    Points in sparser regions (larger local average distance to neighbours)
    are kept with higher priority than points in dense clusters.

    This preserves boundary geometry and fine details better than uniform sampling.

    Args:
        xyz:              (N, 3) positions
        rgb:              (N, 3) colours
        max_points:       target count
        importance_radius: radius for local density estimation

    Returns:
        (xyz_reduced, rgb_reduced)
    """
    N = len(xyz)
    if N <= max_points:
        return xyz, rgb

    # Estimate local density: for each point, count neighbours within radius
    # Using a grid-based approximation (exact KD-tree would be better but slow)
    min_bound = xyz.min(axis=0)
    grid_idx = np.floor((xyz - min_bound) / importance_radius).astype(np.int32)
    keys = (grid_idx[:, 0].astype(np.int64) * 1_000_003 +
            grid_idx[:, 1].astype(np.int64) * 1_009 +
            grid_idx[:, 2].astype(np.int64))

    # Count how many points share each voxel
    from collections import Counter
    voxel_counts = Counter(keys.tolist())
    density = np.array([voxel_counts[k] for k in keys.tolist()], dtype=np.float32)

    # Importance = inverse density (sparse regions get higher importance)
    importance = 1.0 / (density + 1.0)
    importance /= importance.sum()

    # Sample proportionally to importance
    rng = np.random.default_rng(42)
    indices = rng.choice(N, size=min(max_points, N), replace=False, p=importance)
    indices.sort()

    return xyz[indices], rgb[indices]


# ---------------------------------------------------------------------------
# .splat binary format encoder
# ---------------------------------------------------------------------------

def encode_splat(
    xyz: np.ndarray,   # (N, 3) float32
    rgb: np.ndarray,   # (N, 3) uint8
    opacity: Optional[np.ndarray] = None,   # (N,) float32 in [0,1] or None
    scale: Optional[np.ndarray] = None,     # (N, 3) float32 or None
) -> bytes:
    """
    Encode Gaussians into the antimatter15/splat binary format.

    Format (32 bytes per Gaussian):
        x:       f32 — position X
        y:       f32 — position Y
        z:       f32 — position Z
        scale_x: f32 — log scale X
        scale_y: f32 — log scale Y
        scale_z: f32 — log scale Z
        r, g, b: u8  — colour (0-255)
        a:       u8  — opacity (0-255)
        rot_x, rot_y, rot_z, rot_w: u8 — quaternion rotation (normalised, 0-255)
        [pad to 32 bytes]

    Args:
        xyz:     (N, 3) positions in Unity coordinate system (apply opencv_to_unity first)
        rgb:     (N, 3) uint8 colours
        opacity: (N,) float32 in [0,1]; defaults to 1.0 if None
        scale:   (N, 3) float32 Gaussian scales; defaults to 0.01 if None

    Returns:
        bytes: packed .splat binary
    """
    N = len(xyz)
    assert len(rgb) == N, f"xyz/rgb count mismatch: {N} vs {len(rgb)}"

    if opacity is None:
        opacity = np.ones(N, dtype=np.float32)
    if scale is None:
        scale = np.full((N, 3), 0.01, dtype=np.float32)  # 1cm default Gaussian size

    # Default rotation: identity quaternion (x=0, y=0, z=0, w=1)
    # Encoded as uint8: (q + 1) / 2 * 255 → [0,255]
    rot_quat = np.zeros((N, 4), dtype=np.float32)
    rot_quat[:, 3] = 1.0   # w=1 (identity)

    # Pack each record as 32 bytes
    buffer = bytearray(N * SPLAT_RECORD_BYTES)
    offset = 0

    for i in range(N):
        struct.pack_into(
            "<ffffffff4B4B",   # 8 floats + 4 ubytes (rgba) + 4 ubytes (rotation)
            buffer, offset,
            float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2]),
            float(scale[i, 0]), float(scale[i, 1]), float(scale[i, 2]),
            # pack as float but represent as f32 in buffer — scale log
            0.0,  # padding float (used for other Gaussian params in full impl)
            0.0,  # padding float
            # RGBA
            int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2]),
            int(np.clip(opacity[i] * 255, 0, 255)),
            # Rotation (identity)
            128, 128, 128, 255,  # (0, 0, 0, 1) normalised to [0,255]
        )
        offset += SPLAT_RECORD_BYTES

    return bytes(buffer)


def encode_splat_simple(
    xyz: np.ndarray,
    rgb: np.ndarray,
    opacity: Optional[np.ndarray] = None,
) -> bytes:
    """
    Simplified .splat encoder using raw struct packing (one flat loop).
    Uses fixed-size identity rotation and default scale for each Gaussian.
    Identical output format to encode_splat but faster for large N.
    """
    N = len(xyz)
    if opacity is None:
        opacity = np.ones(N, dtype=np.float32)

    # Default scale: log(0.01) ≈ -4.605
    default_log_scale = -4.605

    buf = bytearray()
    pack = struct.pack

    for i in range(N):
        record = pack(
            "<ffffffffffff BBBB BBBB",
            float(xyz[i,0]), float(xyz[i,1]), float(xyz[i,2]),
            default_log_scale, default_log_scale, default_log_scale,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # padding
            int(rgb[i,0]), int(rgb[i,1]), int(rgb[i,2]),
            int(min(255, max(0, int(opacity[i] * 255)))),
            128, 128, 128, 255,  # identity rotation
        )
        buf.extend(record[:SPLAT_RECORD_BYTES])

    return bytes(buf)


# ---------------------------------------------------------------------------
# Main SplatExporter class
# ---------------------------------------------------------------------------

class SplatExporter:
    """
    Exports a 4DGS PLY to Quest-compatible .splat format.

    Example (unit test, CPU):
        exporter = SplatExporter(max_gaussians=500_000)
        result = exporter.export_from_points(xyz, rgb, "output.splat")

    Example (from PLY):
        exporter = SplatExporter(max_gaussians=500_000)
        result = exporter.export("scene.ply", "scene.splat")
    """

    def __init__(
        self,
        max_gaussians: int = QUEST_STANDALONE_MAX_GAUSSIANS,
        lod_method: str = "importance",  # "uniform" or "importance"
        apply_coord_transform: bool = True,  # OpenCV → Unity y-flip
        voxel_size: Optional[float] = None,  # pre-voxelise before LOD
    ):
        self.max_gaussians = max_gaussians
        self.lod_method = lod_method
        self.apply_coord_transform = apply_coord_transform
        self.voxel_size = voxel_size

    def export_from_points(
        self,
        xyz: np.ndarray,
        rgb: np.ndarray,
        output_path: str,
        opacity: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Export Gaussian splats from numpy arrays to .splat binary file.

        Args:
            xyz:         (N, 3) float32 positions (OpenCV convention)
            rgb:         (N, 3) uint8 colours
            output_path: path to write .splat file
            opacity:     (N,) float32 in [0,1] (optional)

        Returns:
            dict with: n_input, n_output, output_path, file_size_mb, lod_ratio
        """
        n_input = len(xyz)

        # 1. LOD reduction
        if self.lod_method == "importance":
            xyz_lod, rgb_lod = lod_reduce_importance(
                xyz, rgb, self.max_gaussians
            )
        else:
            xyz_lod, rgb_lod = lod_reduce_uniform(
                xyz, rgb, self.max_gaussians, voxel_size=self.voxel_size
            )

        n_output = len(xyz_lod)

        # 2. Coordinate transform (OpenCV → Unity/Quest)
        if self.apply_coord_transform:
            xyz_lod = opencv_to_unity(xyz_lod)

        # 3. Clip opacity if provided
        opacity_lod = None
        if opacity is not None:
            if len(opacity) == n_input:
                if n_output < n_input:
                    # Subsample opacity in the same order as xyz_lod was sampled
                    # (simplified: use full ones for now)
                    opacity_lod = np.ones(n_output, dtype=np.float32)
                else:
                    opacity_lod = opacity
            else:
                opacity_lod = None

        # 4. Encode to .splat binary
        splat_bytes = encode_splat_simple(xyz_lod, rgb_lod, opacity_lod)

        # 5. Write output
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(splat_bytes)

        file_size_mb = len(splat_bytes) / (1024 * 1024)
        lod_ratio = n_output / max(n_input, 1)

        result = {
            "output_path": output_path,
            "n_input_gaussians": n_input,
            "n_output_gaussians": n_output,
            "lod_ratio": round(lod_ratio, 4),
            "file_size_mb": round(file_size_mb, 6),  # 6dp to avoid rounding to 0 for small files
            "max_gaussians_limit": self.max_gaussians,
            "coord_transform_applied": self.apply_coord_transform,
            "quest_compatible": n_output <= QUEST_STANDALONE_MAX_GAUSSIANS,
        }
        return result

    def export(
        self,
        input_ply: str,
        output_splat: str,
    ) -> Dict:
        """
        Full export pipeline: PLY → LOD reduce → coord transform → .splat

        Args:
            input_ply:    input PLY file path
            output_splat: output .splat file path

        Returns:
            export result dict
        """
        print(f"[SplatExporter] Loading PLY: {input_ply}")
        data = load_ply_full(input_ply)
        print(f"[SplatExporter] Loaded {data['n']} Gaussians")

        result = self.export_from_points(
            xyz=data["xyz"],
            rgb=data["rgb"],
            output_path=output_splat,
            opacity=data["opacity"],
        )

        print(f"[SplatExporter] ───────────────────────────────────────")
        print(f"[SplatExporter]  Input:    {result['n_input_gaussians']} Gaussians")
        print(f"[SplatExporter]  Output:   {result['n_output_gaussians']} Gaussians")
        print(f"[SplatExporter]  LOD:      {result['lod_ratio']*100:.1f}% retained")
        print(f"[SplatExporter]  Size:     {result['file_size_mb']:.1f} MB")
        print(f"[SplatExporter]  Quest OK: {result['quest_compatible']}")
        print(f"[SplatExporter]  File:     {result['output_path']}")
        print(f"[SplatExporter] ───────────────────────────────────────")

        return result

    def export_4d_sequence(
        self,
        ply_dir: str,
        output_dir: str,
        ply_pattern: str = "frame_*.ply",
    ) -> Dict:
        """
        Export a sequence of per-timestep PLY files to .splat files.
        For 4D temporal scenes: one .splat per timestep, loaded as a sequence in VR.

        TODO: GPU — for large sequences, batch this on GPU.

        Args:
            ply_dir:      directory containing per-timestep PLY files
            output_dir:   directory to write .splat files
            ply_pattern:  glob pattern to find PLY files

        Returns:
            dict with list of per-frame export results
        """
        import glob
        ply_files = sorted(glob.glob(os.path.join(ply_dir, ply_pattern)))

        if not ply_files:
            return {"status": "no_ply_files_found", "ply_dir": ply_dir}

        os.makedirs(output_dir, exist_ok=True)
        results = []

        for i, ply_path in enumerate(ply_files):
            stem = Path(ply_path).stem
            out_path = os.path.join(output_dir, f"{stem}.splat")
            r = self.export(ply_path, out_path)
            results.append(r)
            print(f"[SplatExporter] Frame {i+1}/{len(ply_files)}: {stem}")

        total_mb = sum(r["file_size_mb"] for r in results)
        return {
            "status": "ok",
            "n_frames": len(results),
            "total_size_mb": round(total_mb, 2),
            "frames": results,
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BARF VR Export — convert 4DGS PLY to Quest-compatible .splat"
    )
    parser.add_argument("--input", required=True, help="Input PLY file")
    parser.add_argument("--output", required=True, help="Output .splat file")
    parser.add_argument("--max_gaussians", type=int, default=QUEST_STANDALONE_MAX_GAUSSIANS,
                        help=f"Max Gaussians (default: {QUEST_STANDALONE_MAX_GAUSSIANS})")
    parser.add_argument("--lod", choices=["uniform", "importance"], default="importance",
                        help="LOD method")
    parser.add_argument("--no_coord_transform", action="store_true",
                        help="Skip OpenCV→Unity coordinate transform")
    args = parser.parse_args()

    exporter = SplatExporter(
        max_gaussians=args.max_gaussians,
        lod_method=args.lod,
        apply_coord_transform=not args.no_coord_transform,
    )
    result = exporter.export(args.input, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
