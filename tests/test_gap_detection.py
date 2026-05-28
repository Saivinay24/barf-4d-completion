"""
Tests for src/gap_detection/detect_gaps.py
Runs locally on CPU with synthetic PLY data — no GPU required.
"""

import json
import os
import tempfile

import numpy as np
import pytest

from src.gap_detection.detect_gaps import (
    GapDetector,
    TEST_ANGLES_DEG,
    compute_angular_coverage,
    detect_gap_clusters,
    load_ply_xyz,
    save_angular_heatmap,
    xyz_to_spherical,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_synthetic_ply(path: str, n_points: int = 500, front_only: bool = True) -> str:
    """Write a synthetic ASCII PLY for testing."""
    rng = np.random.default_rng(0)
    if front_only:
        # Front hemisphere only (z > 0) — simulates monocular reconstruction gap
        theta = rng.uniform(-np.pi / 2, np.pi / 2, n_points)
        phi = rng.uniform(0, np.pi, n_points)
    else:
        # Full sphere
        theta = rng.uniform(-np.pi, np.pi, n_points)
        phi = rng.uniform(0, np.pi, n_points)
    r = rng.uniform(0.8, 1.2, n_points)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.cos(phi)
    z = r * np.sin(phi) * np.sin(theta)

    header = (
        f"ply\nformat ascii 1.0\n"
        f"element vertex {n_points}\n"
        f"property float x\nproperty float y\nproperty float z\n"
        f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"end_header\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for i in range(n_points):
            f.write(f"{x[i]:.6f} {y[i]:.6f} {z[i]:.6f} 128 200 128\n")
    return path


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestLoadPly:
    def test_load_returns_ndarray(self, tmp_path):
        ply = make_synthetic_ply(str(tmp_path / "test.ply"), n_points=100)
        pts = load_ply_xyz(ply)
        assert isinstance(pts, np.ndarray)
        assert pts.ndim == 2
        assert pts.shape[1] == 3

    def test_correct_point_count(self, tmp_path):
        ply = make_synthetic_ply(str(tmp_path / "test.ply"), n_points=250)
        pts = load_ply_xyz(ply)
        assert len(pts) == 250

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_ply_xyz("/tmp/does_not_exist_barf.ply")

    def test_coordinates_are_finite(self, tmp_path):
        ply = make_synthetic_ply(str(tmp_path / "test.ply"), n_points=50)
        pts = load_ply_xyz(ply)
        assert np.all(np.isfinite(pts))


class TestSphericalConversion:
    def test_output_shape(self):
        pts = np.random.randn(100, 3).astype(np.float32)
        sph = xyz_to_spherical(pts)
        assert sph.shape == (100, 3)

    def test_azimuth_range(self):
        pts = np.random.randn(200, 3).astype(np.float32)
        sph = xyz_to_spherical(pts)
        theta = sph[:, 1]
        assert np.all(theta >= -180.0) and np.all(theta <= 180.0)

    def test_elevation_range(self):
        pts = np.random.randn(200, 3).astype(np.float32)
        sph = xyz_to_spherical(pts)
        phi = sph[:, 2]
        assert np.all(phi >= -90.0) and np.all(phi <= 90.0)

    def test_radius_positive(self):
        pts = np.random.randn(100, 3).astype(np.float32)
        sph = xyz_to_spherical(pts)
        assert np.all(sph[:, 0] > 0)

    def test_known_point(self):
        # Point at (1, 0, 0): azimuth=0, elevation=0
        pts = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        sph = xyz_to_spherical(pts)
        assert abs(sph[0, 1]) < 1.0   # azimuth ≈ 0°
        assert abs(sph[0, 2]) < 1.0   # elevation ≈ 0°

    def test_back_point(self):
        # Point at (-1, 0, 0): azimuth=180 or -180
        pts = np.array([[-1.0, 0.0, 0.0]], dtype=np.float32)
        sph = xyz_to_spherical(pts)
        assert abs(abs(sph[0, 1]) - 180.0) < 1.0


class TestAngularCoverage:
    def test_full_sphere_coverage(self):
        # Full sphere should give coverage ≈ 1.0
        rng = np.random.default_rng(1)
        theta = rng.uniform(-np.pi, np.pi, 5000)
        phi = rng.uniform(0, np.pi, 5000)
        x = np.cos(theta) * np.sin(phi)
        y = np.cos(phi)
        z = np.sin(theta) * np.sin(phi)
        pts = np.stack([x, y, z], axis=1).astype(np.float32)
        result = compute_angular_coverage(pts, density_threshold=1)
        assert result["coverage_fraction"] == 1.0

    def test_front_only_partial_coverage(self):
        # Front hemisphere (z > 0): only ~4/8 angles should be covered
        rng = np.random.default_rng(2)
        theta = rng.uniform(-np.pi / 2, np.pi / 2, 2000)
        x = np.cos(theta)
        y = np.zeros_like(theta)
        z = np.sin(theta)
        pts = np.stack([x, y, z], axis=1).astype(np.float32)
        result = compute_angular_coverage(pts, density_threshold=1)
        assert result["coverage_fraction"] < 1.0
        # 0° and 315° and 45° should be covered
        assert result["per_angle"][0]["covered"]

    def test_empty_point_cloud(self):
        pts = np.zeros((0, 3), dtype=np.float32)
        result = compute_angular_coverage(pts)
        assert result["coverage_fraction"] == 0.0
        assert len(result["covered_angles"]) == 0

    def test_result_keys(self):
        pts = np.random.randn(100, 3).astype(np.float32)
        result = compute_angular_coverage(pts)
        assert "per_angle" in result
        assert "coverage_fraction" in result
        assert "covered_angles" in result
        assert "empty_angles" in result

    def test_score_range(self):
        pts = np.random.randn(1000, 3).astype(np.float32)
        result = compute_angular_coverage(pts, density_threshold=1)
        assert 0.0 <= result["coverage_fraction"] <= 1.0

    def test_all_test_angles_present(self):
        pts = np.random.randn(100, 3).astype(np.float32)
        result = compute_angular_coverage(pts)
        for angle in TEST_ANGLES_DEG:
            assert angle in result["per_angle"]


class TestGapClusters:
    def test_returns_list(self):
        pts = np.random.randn(200, 3).astype(np.float32) * 0.5
        gaps = detect_gap_clusters(pts, voxel_size=0.3)
        assert isinstance(gaps, list)

    def test_gap_dict_keys(self):
        pts = np.random.randn(200, 3).astype(np.float32) * 0.5
        gaps = detect_gap_clusters(pts, voxel_size=0.3)
        for gap in gaps:
            assert "id" in gap
            assert "center" in gap
            assert "size_m3" in gap
            assert "azimuth_deg" in gap

    def test_empty_input(self):
        pts = np.zeros((0, 3), dtype=np.float32)
        gaps = detect_gap_clusters(pts)
        assert gaps == []

    def test_sorted_by_size(self):
        pts = np.random.randn(500, 3).astype(np.float32) * 0.5
        gaps = detect_gap_clusters(pts, voxel_size=0.2)
        for i in range(len(gaps) - 1):
            assert gaps[i]["size_m3"] >= gaps[i+1]["size_m3"]


class TestGapDetector:
    def test_detect_returns_dict(self, tmp_path):
        ply = make_synthetic_ply(str(tmp_path / "scene.ply"), n_points=300)
        detector = GapDetector(voxel_size=0.1)
        result = detector.detect(ply)
        assert isinstance(result, dict)
        assert "vrc_coverage_score" in result
        assert "gaps" in result
        assert "summary" in result

    def test_vrc_coverage_range(self, tmp_path):
        ply = make_synthetic_ply(str(tmp_path / "scene.ply"), n_points=300)
        detector = GapDetector(voxel_size=0.1)
        result = detector.detect(ply)
        assert 0.0 <= result["vrc_coverage_score"] <= 1.0

    def test_json_output_saved(self, tmp_path):
        ply = make_synthetic_ply(str(tmp_path / "scene.ply"), n_points=300)
        json_out = str(tmp_path / "gaps.json")
        detector = GapDetector(voxel_size=0.1)
        detector.detect(ply, output_json=json_out)
        assert os.path.exists(json_out)
        with open(json_out) as f:
            data = json.load(f)
        assert "vrc_coverage_score" in data

    def test_heatmaps_saved(self, tmp_path):
        ply = make_synthetic_ply(str(tmp_path / "scene.ply"), n_points=500)
        out_dir = str(tmp_path / "gaps")
        detector = GapDetector(voxel_size=0.1)
        result = detector.detect(ply, output_dir=out_dir, save_heatmaps=True)
        # Check heatmaps directory was created
        heatmap_dir = os.path.join(out_dir, "heatmaps")
        assert os.path.isdir(heatmap_dir)
        # Some heatmap files should exist (PPM or PNG)
        heatmap_files = os.listdir(heatmap_dir)
        assert len(heatmap_files) > 0

    def test_front_only_has_gaps(self, tmp_path):
        # Front-only PLY should report empty angles at back
        ply = make_synthetic_ply(str(tmp_path / "front.ply"), n_points=1000, front_only=True)
        detector = GapDetector(voxel_size=0.1, density_threshold=5)
        result = detector.detect(ply)
        # Coverage should be < 1 for front-only scene
        assert result["vrc_coverage_score"] < 1.0
        # 180° (back) should be empty
        assert 180 in result["summary"]["empty_angles"]

    def test_full_sphere_high_coverage(self, tmp_path):
        # Full sphere PLY should have high coverage
        ply = make_synthetic_ply(str(tmp_path / "full.ply"), n_points=5000, front_only=False)
        detector = GapDetector(voxel_size=0.1, density_threshold=1)
        result = detector.detect(ply)
        assert result["vrc_coverage_score"] > 0.5

    def test_data_gap_heatmaps_dir(self, tmp_path):
        """Verify heatmaps save to data/gap_heatmaps/ when using default output."""
        ply = make_synthetic_ply(str(tmp_path / "scene.ply"), n_points=200)
        out_dir = str(tmp_path / "data" / "gap_heatmaps")
        detector = GapDetector(voxel_size=0.2)
        detector.detect(ply, output_dir=out_dir, save_heatmaps=True)
        assert os.path.isdir(out_dir)
