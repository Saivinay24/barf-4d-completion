"""
Tests for src/vr/export_splat.py
Runs locally on CPU with synthetic PLY data — no GPU required.
"""

import os
import struct

import numpy as np
import pytest

from src.vr.export_splat import (
    QUEST_STANDALONE_MAX_GAUSSIANS,
    SPLAT_RECORD_BYTES,
    SplatExporter,
    encode_splat_simple,
    lod_reduce_importance,
    lod_reduce_uniform,
    opencv_to_unity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_random_points(n: int, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
    rgb = rng.integers(0, 255, (n, 3), dtype=np.uint8)
    return xyz, rgb


def make_synthetic_ply(path: str, n: int = 300) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rng = np.random.default_rng(42)
    xyz = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
    header = (
        f"ply\nformat ascii 1.0\n"
        f"element vertex {n}\n"
        f"property float x\nproperty float y\nproperty float z\n"
        f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"end_header\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for i in range(n):
            f.write(f"{xyz[i,0]:.6f} {xyz[i,1]:.6f} {xyz[i,2]:.6f} 128 128 128\n")
    return path


# ---------------------------------------------------------------------------
# Coordinate transform tests
# ---------------------------------------------------------------------------

class TestOpencvToUnity:
    def test_y_axis_flipped(self):
        pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        result = opencv_to_unity(pts)
        assert result[0, 1] == -2.0  # y flipped

    def test_x_z_unchanged(self):
        pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        result = opencv_to_unity(pts)
        assert result[0, 0] == 1.0  # x unchanged
        assert result[0, 2] == 3.0  # z unchanged

    def test_shape_preserved(self):
        pts = np.random.randn(100, 3).astype(np.float32)
        result = opencv_to_unity(pts)
        assert result.shape == (100, 3)

    def test_double_transform_identity(self):
        pts = np.random.randn(50, 3).astype(np.float32)
        result = opencv_to_unity(opencv_to_unity(pts))
        np.testing.assert_allclose(result, pts, atol=1e-6)

    def test_zero_input(self):
        pts = np.zeros((5, 3), dtype=np.float32)
        result = opencv_to_unity(pts)
        np.testing.assert_array_equal(result, pts)


# ---------------------------------------------------------------------------
# LOD reduction tests
# ---------------------------------------------------------------------------

class TestLODReduction:
    def test_uniform_reduces_count(self):
        xyz, rgb = make_random_points(10000)
        xyz_r, rgb_r = lod_reduce_uniform(xyz, rgb, max_points=500)
        assert len(xyz_r) <= 500
        assert len(rgb_r) <= 500

    def test_uniform_preserves_shape(self):
        xyz, rgb = make_random_points(1000)
        xyz_r, rgb_r = lod_reduce_uniform(xyz, rgb, max_points=200)
        assert xyz_r.shape[1] == 3
        assert rgb_r.shape[1] == 3

    def test_uniform_no_reduction_needed(self):
        xyz, rgb = make_random_points(100)
        xyz_r, rgb_r = lod_reduce_uniform(xyz, rgb, max_points=1000)
        assert len(xyz_r) == 100  # no change needed

    def test_importance_reduces_count(self):
        xyz, rgb = make_random_points(10000)
        xyz_r, rgb_r = lod_reduce_importance(xyz, rgb, max_points=500)
        assert len(xyz_r) <= 500

    def test_importance_preserves_shape(self):
        xyz, rgb = make_random_points(2000)
        xyz_r, rgb_r = lod_reduce_importance(xyz, rgb, max_points=300)
        assert xyz_r.shape[1] == 3
        assert rgb_r.shape[1] == 3

    def test_importance_no_reduction_needed(self):
        xyz, rgb = make_random_points(50)
        xyz_r, rgb_r = lod_reduce_importance(xyz, rgb, max_points=1000)
        assert len(xyz_r) == 50

    def test_xyz_rgb_count_matches(self):
        xyz, rgb = make_random_points(5000)
        xyz_r, rgb_r = lod_reduce_importance(xyz, rgb, max_points=200)
        assert len(xyz_r) == len(rgb_r)


# ---------------------------------------------------------------------------
# .splat encoding tests
# ---------------------------------------------------------------------------

class TestEncodeSplat:
    def test_output_is_bytes(self):
        xyz, rgb = make_random_points(10)
        result = encode_splat_simple(xyz, rgb)
        assert isinstance(result, bytes)

    def test_correct_byte_length(self):
        N = 50
        xyz, rgb = make_random_points(N)
        result = encode_splat_simple(xyz, rgb)
        assert len(result) == N * SPLAT_RECORD_BYTES

    def test_empty_input(self):
        xyz = np.zeros((0, 3), dtype=np.float32)
        rgb = np.zeros((0, 3), dtype=np.uint8)
        result = encode_splat_simple(xyz, rgb)
        assert result == b""

    def test_single_point(self):
        xyz = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
        rgb = np.array([[200, 100, 50]], dtype=np.uint8)
        result = encode_splat_simple(xyz, rgb)
        assert len(result) == SPLAT_RECORD_BYTES

    def test_with_opacity(self):
        N = 20
        xyz, rgb = make_random_points(N)
        opacity = np.ones(N, dtype=np.float32) * 0.8
        result = encode_splat_simple(xyz, rgb, opacity)
        assert len(result) == N * SPLAT_RECORD_BYTES

    def test_deterministic(self):
        xyz, rgb = make_random_points(100)
        r1 = encode_splat_simple(xyz, rgb)
        r2 = encode_splat_simple(xyz, rgb)
        assert r1 == r2


# ---------------------------------------------------------------------------
# SplatExporter tests
# ---------------------------------------------------------------------------

class TestSplatExporter:
    def test_export_from_points_returns_dict(self, tmp_path):
        xyz, rgb = make_random_points(1000)
        exporter = SplatExporter(max_gaussians=500)
        result = exporter.export_from_points(xyz, rgb, str(tmp_path / "out.splat"))
        assert isinstance(result, dict)

    def test_export_creates_file(self, tmp_path):
        xyz, rgb = make_random_points(500)
        out = str(tmp_path / "out.splat")
        exporter = SplatExporter(max_gaussians=200)
        exporter.export_from_points(xyz, rgb, out)
        assert os.path.exists(out)

    def test_export_respects_max_gaussians(self, tmp_path):
        xyz, rgb = make_random_points(10000)
        out = str(tmp_path / "out.splat")
        max_g = 1000
        exporter = SplatExporter(max_gaussians=max_g)
        result = exporter.export_from_points(xyz, rgb, out)
        assert result["n_output_gaussians"] <= max_g

    def test_export_file_size_correct(self, tmp_path):
        xyz, rgb = make_random_points(200)
        out = str(tmp_path / "out.splat")
        exporter = SplatExporter(max_gaussians=500)  # no reduction needed
        result = exporter.export_from_points(xyz, rgb, out)
        actual_size = os.path.getsize(out)
        expected_size = result["n_output_gaussians"] * SPLAT_RECORD_BYTES
        assert actual_size == expected_size

    def test_coord_transform_applied(self, tmp_path):
        xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        rgb = np.array([[100, 100, 100]], dtype=np.uint8)
        out = str(tmp_path / "out.splat")
        exporter = SplatExporter(max_gaussians=1, apply_coord_transform=True)
        result = exporter.export_from_points(xyz, rgb, out)
        assert result["coord_transform_applied"] is True

    def test_quest_compatible_flag(self, tmp_path):
        N = 1000
        xyz, rgb = make_random_points(N)
        out = str(tmp_path / "out.splat")
        exporter = SplatExporter(max_gaussians=QUEST_STANDALONE_MAX_GAUSSIANS)
        result = exporter.export_from_points(xyz, rgb, out)
        assert result["quest_compatible"] is True

    def test_over_limit_not_quest_compatible(self, tmp_path):
        # If we set max to over the Quest limit, flag should reflect actual count
        N = 1000
        xyz, rgb = make_random_points(N)
        out = str(tmp_path / "out.splat")
        exporter = SplatExporter(
            max_gaussians=QUEST_STANDALONE_MAX_GAUSSIANS + 100,
            apply_coord_transform=False,
        )
        result = exporter.export_from_points(xyz, rgb, out)
        # 1000 < QUEST_STANDALONE_MAX_GAUSSIANS so still compatible
        assert result["quest_compatible"] is True

    def test_export_from_ply(self, tmp_path):
        ply = make_synthetic_ply(str(tmp_path / "scene.ply"), n=300)
        out = str(tmp_path / "scene.splat")
        exporter = SplatExporter(max_gaussians=200)
        result = exporter.export(ply, out)
        assert os.path.exists(out)
        assert result["n_input_gaussians"] == 300

    def test_lod_ratio_correct(self, tmp_path):
        xyz, rgb = make_random_points(1000)
        out = str(tmp_path / "out.splat")
        exporter = SplatExporter(max_gaussians=500)
        result = exporter.export_from_points(xyz, rgb, out)
        expected_ratio = result["n_output_gaussians"] / result["n_input_gaussians"]
        assert abs(result["lod_ratio"] - round(expected_ratio, 4)) < 1e-4

    def test_uniform_lod_method(self, tmp_path):
        xyz, rgb = make_random_points(2000)
        out = str(tmp_path / "out.splat")
        exporter = SplatExporter(max_gaussians=500, lod_method="uniform")
        result = exporter.export_from_points(xyz, rgb, out)
        assert result["n_output_gaussians"] <= 500

    def test_file_size_reported(self, tmp_path):
        xyz, rgb = make_random_points(100)
        out = str(tmp_path / "out.splat")
        exporter = SplatExporter(max_gaussians=200)
        result = exporter.export_from_points(xyz, rgb, out)
        assert result["file_size_mb"] > 0
