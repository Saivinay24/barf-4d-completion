#!/usr/bin/env python3
"""
Gap detector for partial 4D reconstructions.
Detects empty voxel regions in point clouds that indicate missing geometry.

Usage:
    python detect.py --input reconstruction.ply --output gaps.json
    python detect.py --input reconstruction.ply --output gaps.json --voxel-size 0.02
    python detect.py --input-dir reconstructions/v01/ --output-dir gaps/v01/
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


def detect_gaps(pcd_path: str, voxel_size: float = 0.02, min_neighbors: int = 2,
                cluster_eps: float = None, min_cluster_points: int = 5) -> list:
    """
    Detect gaps in a partial point cloud.

    Algorithm:
    1. Voxelize the point cloud
    2. Find empty voxels within the bounding box
    3. Filter: only keep empty voxels that have occupied neighbors (real gaps, not exterior)
    4. Cluster empty voxels into gap regions using DBSCAN

    Args:
        pcd_path: Path to .ply file
        voxel_size: Size of voxels for analysis
        min_neighbors: Minimum occupied neighbors for a voxel to be considered a gap
        cluster_eps: DBSCAN clustering epsilon (default: voxel_size * 3)
        min_cluster_points: Minimum points per gap cluster

    Returns:
        List of gap dicts with center, size, bounding_box, etc.
    """
    if not HAS_OPEN3D:
        print("Error: open3d is required. Install with: pip install open3d")
        sys.exit(1)

    if cluster_eps is None:
        cluster_eps = voxel_size * 3

    # Load point cloud
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)

    if len(points) == 0:
        print("Warning: Empty point cloud")
        return []

    print(f"  Loaded {len(points)} points from {pcd_path}")

    # Create voxel grid
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
    voxels = voxel_grid.get_voxels()

    # Build set of occupied voxel indices
    occupied = set()
    for v in voxels:
        occupied.add(tuple(v.grid_index))

    print(f"  {len(occupied)} occupied voxels (voxel_size={voxel_size})")

    # Find bounding box in voxel space
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    grid_size = ((max_bound - min_bound) / voxel_size).astype(int) + 1

    print(f"  Grid dimensions: {grid_size[0]}×{grid_size[1]}×{grid_size[2]}")

    # Find empty voxels that are adjacent to occupied ones (interior gaps)
    empty_voxels = []
    neighbor_offsets = [
        (-1, 0, 0), (1, 0, 0),
        (0, -1, 0), (0, 1, 0),
        (0, 0, -1), (0, 0, 1),
    ]

    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            for z in range(grid_size[2]):
                if (x, y, z) in occupied:
                    continue

                # Count occupied neighbors
                neighbors = 0
                for dx, dy, dz in neighbor_offsets:
                    if (x + dx, y + dy, z + dz) in occupied:
                        neighbors += 1

                if neighbors >= min_neighbors:
                    # Convert back to world coordinates
                    center = min_bound + np.array([x, y, z]) * voxel_size + voxel_size / 2
                    empty_voxels.append({
                        "center": center.tolist(),
                        "neighbors": neighbors,
                        "grid_index": [x, y, z],
                    })

    print(f"  Found {len(empty_voxels)} empty voxels with ≥{min_neighbors} occupied neighbors")

    if not empty_voxels:
        return []

    # Cluster empty voxels into gap regions using DBSCAN
    centers = np.array([v["center"] for v in empty_voxels])
    gap_pcd = o3d.geometry.PointCloud()
    gap_pcd.points = o3d.utility.Vector3dVector(centers)
    labels = np.array(gap_pcd.cluster_dbscan(eps=cluster_eps, min_points=min_cluster_points))

    # Build gap region descriptors
    gaps = []
    for label in sorted(set(labels)):
        if label == -1:
            continue  # Noise

        mask = labels == label
        cluster_centers = centers[mask]
        cluster_neighbors = [empty_voxels[i]["neighbors"] for i in range(len(mask)) if mask[i]]

        gap = {
            "id": len(gaps),
            "center": cluster_centers.mean(axis=0).tolist(),
            "size": float(len(cluster_centers) * (voxel_size ** 3)),
            "voxel_count": int(mask.sum()),
            "avg_neighbor_count": float(np.mean(cluster_neighbors)),
            "bounding_box": {
                "min": cluster_centers.min(axis=0).tolist(),
                "max": cluster_centers.max(axis=0).tolist(),
            },
            "severity": "high" if mask.sum() > 100 else ("medium" if mask.sum() > 20 else "low"),
        }
        gaps.append(gap)

    # Sort by size (largest gaps first)
    gaps = sorted(gaps, key=lambda g: g["size"], reverse=True)

    return gaps


def compute_coverage_stats(pcd_path: str, voxel_size: float = 0.05) -> dict:
    """
    Compute spherical coverage statistics for a point cloud.
    Estimates what percentage of the 360° sphere around the object center is covered.
    """
    if not HAS_OPEN3D:
        return {}

    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)

    if len(points) == 0:
        return {"coverage_360": 0.0}

    # Center the point cloud
    centroid = points.mean(axis=0)
    centered = points - centroid

    # Convert to spherical coordinates
    r = np.linalg.norm(centered, axis=1)
    theta = np.arctan2(centered[:, 1], centered[:, 0])  # azimuth [-π, π]
    phi = np.arccos(np.clip(centered[:, 2] / np.maximum(r, 1e-8), -1, 1))  # elevation [0, π]

    # Discretize into angular bins
    n_azimuth = 36   # 10° bins
    n_elevation = 18  # 10° bins

    azimuth_bins = np.linspace(-np.pi, np.pi, n_azimuth + 1)
    elevation_bins = np.linspace(0, np.pi, n_elevation + 1)

    # Count occupied bins
    total_bins = n_azimuth * n_elevation
    occupied_bins = set()

    for t, p in zip(theta, phi):
        a_idx = min(np.searchsorted(azimuth_bins, t) - 1, n_azimuth - 1)
        e_idx = min(np.searchsorted(elevation_bins, p) - 1, n_elevation - 1)
        occupied_bins.add((max(0, a_idx), max(0, e_idx)))

    coverage = len(occupied_bins) / total_bins * 100

    # Analyze which angular regions are empty (approximate front/side/back)
    front_count = 0
    side_count = 0
    back_count = 0

    for (a, e) in occupied_bins:
        angle = (a / n_azimuth) * 360 - 180
        if -45 <= angle <= 45:
            front_count += 1
        elif angle < -135 or angle > 135:
            back_count += 1
        else:
            side_count += 1

    front_total = n_elevation * (n_azimuth // 4)
    side_total = n_elevation * (n_azimuth // 2)
    back_total = n_elevation * (n_azimuth // 4)

    return {
        "coverage_360_percent": round(coverage, 1),
        "occupied_bins": len(occupied_bins),
        "total_bins": total_bins,
        "front_coverage_percent": round(front_count / max(1, front_total) * 100, 1),
        "side_coverage_percent": round(side_count / max(1, side_total) * 100, 1),
        "back_coverage_percent": round(back_count / max(1, back_total) * 100, 1),
    }


def process_directory(input_dir: str, output_dir: str, voxel_size: float = 0.02) -> None:
    """Process all PLY files in a directory."""
    os.makedirs(output_dir, exist_ok=True)

    ply_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.ply')])

    if not ply_files:
        print(f"No .ply files found in {input_dir}")
        return

    print(f"Processing {len(ply_files)} PLY files...")
    all_gaps = {}

    for ply_file in ply_files:
        print(f"\n--- {ply_file} ---")
        input_path = os.path.join(input_dir, ply_file)
        output_file = ply_file.replace('.ply', '_gaps.json')
        output_path = os.path.join(output_dir, output_file)

        gaps = detect_gaps(input_path, voxel_size)
        coverage = compute_coverage_stats(input_path)

        result = {
            "source_file": ply_file,
            "gap_count": len(gaps),
            "coverage": coverage,
            "gaps": gaps,
        }

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        all_gaps[ply_file] = result
        print(f"  → {len(gaps)} gaps detected, {coverage.get('coverage_360_percent', '?')}% coverage")

    # Write summary
    summary_path = os.path.join(output_dir, "gap_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_gaps, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="BARF gap detector for partial 4D reconstructions")
    parser.add_argument("--input", help="Input .ply file")
    parser.add_argument("--input-dir", help="Input directory of .ply files")
    parser.add_argument("--output", default="gaps.json", help="Output JSON file (single file mode)")
    parser.add_argument("--output-dir", help="Output directory (batch mode)")
    parser.add_argument("--voxel-size", type=float, default=0.02, help="Voxel size for analysis (default: 0.02)")
    parser.add_argument("--min-neighbors", type=int, default=2, help="Min occupied neighbors for gap detection")
    parser.add_argument("--coverage", action="store_true", help="Also compute 360° coverage statistics")
    args = parser.parse_args()

    if args.input_dir:
        process_directory(args.input_dir, args.output_dir or "gap_results", args.voxel_size)
    elif args.input:
        print(f"Analyzing: {args.input}")
        gaps = detect_gaps(args.input, args.voxel_size, args.min_neighbors)

        result = {"gaps": gaps, "gap_count": len(gaps)}

        if args.coverage:
            result["coverage"] = compute_coverage_stats(args.input)
            print(f"  Coverage: {result['coverage'].get('coverage_360_percent', '?')}%")

        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"  Found {len(gaps)} gap regions")
        for gap in gaps[:5]:
            print(f"    Gap #{gap['id']}: size={gap['size']:.4f}, severity={gap['severity']}, center={[round(c, 3) for c in gap['center']]}")
        if len(gaps) > 5:
            print(f"    ... and {len(gaps) - 5} more gaps")

        print(f"  Saved to {args.output}")
    else:
        parser.error("Either --input or --input-dir is required")


if __name__ == "__main__":
    main()
