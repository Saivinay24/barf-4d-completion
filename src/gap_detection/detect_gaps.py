"""
BARF 4D — Gap Detection Module
================================
Detects angular coverage gaps in a 4D Gaussian Splat PLY reconstruction.

For each of 8 test viewpoints (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°),
measures the point cloud density in that angular sector and identifies
empty/sparse regions (the "gaps" BARF needs to fill).

Outputs:
  - gaps.json: list of gap regions with center, size, angular position
  - heatmaps/: occupancy images at each of the 8 test angles
  - coverage_summary.json: per-angle coverage % and overall VRC-Coverage score

Usage (CLI):
    python -m src.gap_detection.detect_gaps \
        --input path/to/scene.ply \
        --output_json path/to/gaps.json \
        --output_heatmap_dir path/to/heatmaps/ \
        --voxel_size 0.05

Usage (API):
    from src.gap_detection.detect_gaps import GapDetector
    detector = GapDetector(voxel_size=0.05)
    result = detector.detect(pcd_path="scene.ply", output_dir="outputs/")
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# PLY I/O (pure numpy — no open3d required for basic reading)
# ---------------------------------------------------------------------------

def load_ply_xyz(ply_path: str) -> np.ndarray:
    """
    Load x, y, z coordinates from a PLY file.
    Returns: np.ndarray of shape (N, 3), float32
    Supports ASCII and binary_little_endian PLY formats.
    """
    path = Path(ply_path)
    if not path.exists():
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    with open(ply_path, "rb") as f:
        # --- Parse header ---
        header_lines = []
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        # Extract format and vertex count
        fmt = "ascii"
        n_vertices = 0
        properties = []
        for line in header_lines:
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
            elif line.startswith("property"):
                parts = line.split()
                properties.append((parts[1], parts[2]))  # (type, name)

        if n_vertices == 0:
            return np.zeros((0, 3), dtype=np.float32)

        # Find x, y, z indices
        prop_names = [p[1] for p in properties]
        prop_types = [p[0] for p in properties]
        try:
            xi = prop_names.index("x")
            yi = prop_names.index("y")
            zi = prop_names.index("z")
        except ValueError:
            raise ValueError(f"PLY file missing x/y/z properties. Found: {prop_names}")

        # --- Read data ---
        if fmt == "ascii":
            coords = np.zeros((n_vertices, 3), dtype=np.float32)
            for i in range(n_vertices):
                values = f.readline().decode("ascii").split()
                coords[i, 0] = float(values[xi])
                coords[i, 1] = float(values[yi])
                coords[i, 2] = float(values[zi])
            return coords

        else:  # binary_little_endian or binary_big_endian
            # Build numpy dtype from PLY property types
            ply_to_np = {
                "float": "f4", "float32": "f4",
                "double": "f8", "float64": "f8",
                "int": "i4", "int32": "i4",
                "uint": "u4", "uint32": "u4",
                "short": "i2", "int16": "i2",
                "ushort": "u2", "uint16": "u2",
                "char": "i1", "int8": "i1",
                "uchar": "u1", "uint8": "u1",
            }
            dt_list = []
            for ptype, pname in properties:
                np_type = ply_to_np.get(ptype, "f4")
                dt_list.append((pname, np_type))

            dtype = np.dtype(dt_list)
            byteorder = "<" if "little" in fmt else ">"
            dtype = dtype.newbyteorder(byteorder)

            data = np.frombuffer(f.read(n_vertices * dtype.itemsize), dtype=dtype)
            coords = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
            return coords


# ---------------------------------------------------------------------------
# Core gap detection logic
# ---------------------------------------------------------------------------

# 8 standard test azimuth angles (degrees) for VRC evaluation
TEST_ANGLES_DEG = [0, 45, 90, 135, 180, 225, 270, 315]

# Minimum point density to consider an angular sector "covered"
DEFAULT_DENSITY_THRESHOLD = 5  # points per voxel sector


def xyz_to_spherical(points: np.ndarray) -> np.ndarray:
    """
    Convert Cartesian (x, y, z) to spherical (r, theta_deg, phi_deg).
    theta: azimuth [-180, 180] measured in XZ plane from +X axis
    phi:   elevation [-90, 90] from XZ plane upward
    Returns: (N, 3) array of [r, theta_deg, phi_deg]
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2) + 1e-8
    theta = np.degrees(np.arctan2(z, x))   # azimuth
    phi = np.degrees(np.arcsin(np.clip(y / r, -1, 1)))  # elevation
    return np.stack([r, theta, phi], axis=1)


def compute_angular_coverage(
    points: np.ndarray,
    sector_width_deg: float = 45.0,
    density_threshold: int = DEFAULT_DENSITY_THRESHOLD,
) -> Dict:
    """
    For each of the 8 test angles, count how many points fall in that angular sector.

    Returns dict with:
        per_angle: {angle_deg: {"count": int, "covered": bool, "density": float}}
        coverage_fraction: float in [0, 1] — fraction of angles that are covered
        covered_angles: list of covered angle degrees
        empty_angles: list of empty angle degrees
    """
    if len(points) == 0:
        return {
            "per_angle": {a: {"count": 0, "covered": False, "density": 0.0}
                          for a in TEST_ANGLES_DEG},
            "coverage_fraction": 0.0,
            "covered_angles": [],
            "empty_angles": TEST_ANGLES_DEG[:],
        }

    spherical = xyz_to_spherical(points)
    theta = spherical[:, 1]  # azimuth degrees
    half_w = sector_width_deg / 2.0

    per_angle = {}
    covered_angles = []
    empty_angles = []

    for angle in TEST_ANGLES_DEG:
        # Angular distance from test angle (handle wraparound)
        diff = np.abs(((theta - angle + 180) % 360) - 180)
        in_sector = diff <= half_w
        count = int(in_sector.sum())
        covered = count >= density_threshold
        per_angle[angle] = {
            "count": count,
            "covered": covered,
            "density": float(count / max(len(points), 1)),
        }
        if covered:
            covered_angles.append(angle)
        else:
            empty_angles.append(angle)

    coverage_fraction = len(covered_angles) / len(TEST_ANGLES_DEG)
    return {
        "per_angle": per_angle,
        "coverage_fraction": coverage_fraction,
        "covered_angles": covered_angles,
        "empty_angles": empty_angles,
    }


def detect_gap_clusters(
    points: np.ndarray,
    voxel_size: float = 0.05,
    min_cluster_voxels: int = 5,
) -> List[Dict]:
    """
    Voxelise the point cloud, find empty voxels that are adjacent to occupied ones
    (true interior gaps, not exterior empty space), and cluster them into gap regions.

    Returns list of gap dicts with: id, center, size_m3, bounding_box, empty_angles
    """
    if len(points) == 0:
        return []

    # 1. Build voxel occupancy set
    min_bound = points.min(axis=0) - voxel_size
    voxel_indices = np.floor((points - min_bound) / voxel_size).astype(np.int32)
    occupied = set(map(tuple, voxel_indices))

    # 2. Find empty voxels that border occupied ones (true gaps)
    max_idx = voxel_indices.max(axis=0) + 2
    neighbor_offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ]

    candidate_empty = set()
    for vox in occupied:
        for dx, dy, dz in neighbor_offsets:
            nb = (vox[0]+dx, vox[1]+dy, vox[2]+dz)
            if nb not in occupied:
                candidate_empty.add(nb)

    if not candidate_empty:
        return []

    # 3. Keep only empty voxels inside the bounding box of occupied space
    #    (filter out vast exterior empty space)
    occ_arr = np.array(list(occupied))
    box_min = occ_arr.min(axis=0)
    box_max = occ_arr.max(axis=0)

    interior_empty = np.array([
        v for v in candidate_empty
        if all(box_min[i] <= v[i] <= box_max[i] for i in range(3))
    ])

    if len(interior_empty) == 0:
        return []

    # 4. Simple grid-based clustering (DBSCAN-lite)
    #    Group voxels that are within eps voxels of each other
    eps_voxels = 3
    labels = np.full(len(interior_empty), -1, dtype=np.int32)
    cluster_id = 0
    vox_set = {tuple(v): i for i, v in enumerate(interior_empty)}

    for i, vox in enumerate(interior_empty):
        if labels[i] != -1:
            continue
        # BFS
        queue = [i]
        labels[i] = cluster_id
        head = 0
        while head < len(queue):
            curr = interior_empty[queue[head]]
            head += 1
            for dx in range(-eps_voxels, eps_voxels+1):
                for dy in range(-eps_voxels, eps_voxels+1):
                    for dz in range(-eps_voxels, eps_voxels+1):
                        nb = (curr[0]+dx, curr[1]+dy, curr[2]+dz)
                        if nb in vox_set:
                            j = vox_set[nb]
                            if labels[j] == -1:
                                labels[j] = cluster_id
                                queue.append(j)
        cluster_id += 1

    # 5. Build gap descriptors
    gaps = []
    for cid in range(cluster_id):
        mask = labels == cid
        if mask.sum() < min_cluster_voxels:
            continue
        cluster_voxels = interior_empty[mask]
        centers_world = cluster_voxels * voxel_size + min_bound + voxel_size / 2

        # Compute angular position of gap centroid
        centroid = centers_world.mean(axis=0)
        _, theta, phi = xyz_to_spherical(centroid.reshape(1, 3))[0]

        gaps.append({
            "id": len(gaps),
            "center": centroid.tolist(),
            "size_m3": float(mask.sum() * voxel_size**3),
            "voxel_count": int(mask.sum()),
            "azimuth_deg": float(theta),
            "elevation_deg": float(phi),
            "bounding_box": {
                "min": centers_world.min(axis=0).tolist(),
                "max": centers_world.max(axis=0).tolist(),
            },
        })

    # Sort by size descending (biggest gaps first)
    gaps.sort(key=lambda g: g["size_m3"], reverse=True)
    return gaps


# ---------------------------------------------------------------------------
# Heatmap generation (pure numpy/matplotlib — no open3d)
# ---------------------------------------------------------------------------

def save_angular_heatmap(
    points: np.ndarray,
    angle_deg: float,
    output_path: str,
    sector_width_deg: float = 45.0,
    img_size: int = 256,
) -> None:
    """
    Save a top-down density heatmap of points in the angular sector centered
    at angle_deg. Dark = low density (gap), bright = high density (filled).
    Saves as a PNG file using only numpy (no matplotlib required).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    spherical = xyz_to_spherical(points) if len(points) > 0 else np.zeros((0, 3))
    if len(spherical) > 0:
        theta = spherical[:, 1]
        diff = np.abs(((theta - angle_deg + 180) % 360) - 180)
        in_sector = diff <= sector_width_deg / 2.0
        sector_pts = points[in_sector]
    else:
        sector_pts = np.zeros((0, 3))

    # Build 2D density grid (XZ top-down view)
    img = np.zeros((img_size, img_size), dtype=np.float32)

    if len(sector_pts) > 0:
        # Map x,z to image coordinates
        x, z = sector_pts[:, 0], sector_pts[:, 2]
        xmin, xmax = x.min() - 0.1, x.max() + 0.1
        zmin, zmax = z.min() - 0.1, z.max() + 0.1

        xi = np.clip(((x - xmin) / (xmax - xmin) * (img_size - 1)).astype(int), 0, img_size-1)
        zi = np.clip(((z - zmin) / (zmax - zmin) * (img_size - 1)).astype(int), 0, img_size-1)
        np.add.at(img, (zi, xi), 1)

        # Normalize
        if img.max() > 0:
            img = np.log1p(img) / np.log1p(img.max())

    # Convert to RGB PNG (write raw bytes)
    r = (img * 255).astype(np.uint8)
    g = (img * 200).astype(np.uint8)
    b = np.zeros_like(r)  # blue channel = 0 (warm = dense, dark = gap)

    # Encode as PPM (simple, no external library needed)
    ppm_path = output_path.replace(".png", ".ppm")
    with open(ppm_path, "wb") as f:
        f.write(f"P6\n{img_size} {img_size}\n255\n".encode())
        rgb = np.stack([r, g, b], axis=2)
        f.write(rgb.tobytes())

    # Try to convert PPM to PNG if Pillow is available
    try:
        from PIL import Image
        im = Image.open(ppm_path)
        im.save(output_path)
        os.remove(ppm_path)
    except ImportError:
        # Rename PPM to the output path if PIL not available
        os.rename(ppm_path, output_path.replace(".png", ".ppm"))


# ---------------------------------------------------------------------------
# Main GapDetector class
# ---------------------------------------------------------------------------

class GapDetector:
    """
    Main gap detection class for BARF pipeline.

    Example:
        detector = GapDetector(voxel_size=0.05)
        result = detector.detect("scene.ply", output_dir="outputs/gaps/")
    """

    def __init__(
        self,
        voxel_size: float = 0.05,
        density_threshold: int = DEFAULT_DENSITY_THRESHOLD,
        sector_width_deg: float = 45.0,
    ):
        self.voxel_size = voxel_size
        self.density_threshold = density_threshold
        self.sector_width_deg = sector_width_deg

    def detect(
        self,
        pcd_path: str,
        output_dir: Optional[str] = None,
        output_json: Optional[str] = None,
        save_heatmaps: bool = True,
    ) -> Dict:
        """
        Full gap detection pipeline.

        Args:
            pcd_path: path to input PLY file
            output_dir: directory for heatmaps and coverage summary
            output_json: path for gaps.json output
            save_heatmaps: whether to generate per-angle heatmap images

        Returns:
            result dict with: coverage, gaps, summary
        """
        print(f"[GapDetector] Loading PLY: {pcd_path}")
        points = load_ply_xyz(pcd_path)
        print(f"[GapDetector] Loaded {len(points)} points")

        # Angular coverage analysis
        print("[GapDetector] Computing angular coverage...")
        coverage = compute_angular_coverage(
            points,
            sector_width_deg=self.sector_width_deg,
            density_threshold=self.density_threshold,
        )

        # Gap cluster detection
        print("[GapDetector] Detecting gap clusters...")
        gaps = detect_gap_clusters(points, voxel_size=self.voxel_size)
        print(f"[GapDetector] Found {len(gaps)} gap regions")

        # VRC Coverage score (sub-metric)
        vrc_coverage = coverage["coverage_fraction"]

        result = {
            "input_ply": str(pcd_path),
            "n_points": len(points),
            "voxel_size": self.voxel_size,
            "density_threshold": self.density_threshold,
            "vrc_coverage_score": round(vrc_coverage, 4),
            "coverage": coverage,
            "gaps": gaps,
            "summary": {
                "total_gaps": len(gaps),
                "total_gap_volume_m3": sum(g["size_m3"] for g in gaps),
                "covered_angles": coverage["covered_angles"],
                "empty_angles": coverage["empty_angles"],
                "coverage_pct": round(vrc_coverage * 100, 1),
                "largest_gap_azimuth_deg": gaps[0]["azimuth_deg"] if gaps else None,
            },
        }

        # Save outputs
        if output_json:
            os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
            with open(output_json, "w") as f:
                json.dump(result, f, indent=2)
            print(f"[GapDetector] Gaps saved to: {output_json}")

        if save_heatmaps and output_dir:
            heatmap_dir = os.path.join(output_dir, "heatmaps")
            os.makedirs(heatmap_dir, exist_ok=True)
            print(f"[GapDetector] Saving heatmaps to: {heatmap_dir}")
            for angle in TEST_ANGLES_DEG:
                heatmap_path = os.path.join(heatmap_dir, f"angle_{angle:03d}.png")
                save_angular_heatmap(points, angle, heatmap_path,
                                     sector_width_deg=self.sector_width_deg)
            result["heatmap_dir"] = heatmap_dir

        # Print summary
        print("[GapDetector] ─────────────────────────────────────────")
        print(f"[GapDetector]  Total points:       {len(points)}")
        print(f"[GapDetector]  Angular coverage:   {vrc_coverage*100:.1f}%  "
              f"({len(coverage['covered_angles'])}/{len(TEST_ANGLES_DEG)} angles covered)")
        print(f"[GapDetector]  Covered angles:     {coverage['covered_angles']}")
        print(f"[GapDetector]  Empty angles:       {coverage['empty_angles']}")
        print(f"[GapDetector]  Gap clusters found: {len(gaps)}")
        print(f"[GapDetector]  VRC-Coverage score: {vrc_coverage:.4f}")
        print("[GapDetector] ─────────────────────────────────────────")

        return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BARF Gap Detection — detect empty angular regions in a PLY point cloud"
    )
    parser.add_argument("--input", required=True, help="Input PLY file path")
    parser.add_argument("--output_json", default=None, help="Output gaps.json path")
    parser.add_argument("--output_heatmap_dir", default=None, help="Directory for heatmap PNGs")
    parser.add_argument("--voxel_size", type=float, default=0.05, help="Voxel size in meters")
    parser.add_argument("--density_threshold", type=int, default=DEFAULT_DENSITY_THRESHOLD,
                        help="Min points per sector to count as covered")
    args = parser.parse_args()

    detector = GapDetector(
        voxel_size=args.voxel_size,
        density_threshold=args.density_threshold,
    )

    output_dir = args.output_heatmap_dir or (
        os.path.dirname(args.output_json) if args.output_json else "data/gap_heatmaps"
    )

    result = detector.detect(
        pcd_path=args.input,
        output_dir=output_dir,
        output_json=args.output_json,
        save_heatmaps=True,
    )

    print(f"\nDone. Coverage: {result['summary']['coverage_pct']}%")


if __name__ == "__main__":
    main()
