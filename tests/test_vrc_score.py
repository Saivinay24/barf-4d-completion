"""
Tests for src/metrics/vrc_score.py
Runs locally on CPU with synthetic data — no GPU required.
"""

import json
import os

import numpy as np
import pytest

from src.metrics.vrc_score import (
    VRC_TEST_ANGLES,
    VRCScore,
    compute_coherence_from_frames,
    compute_coverage,
    compute_quality,
    composite_vrc_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_synthetic_ply(path: str, n: int = 500, front_only: bool = True) -> str:
    """Write a synthetic ASCII PLY."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rng = np.random.default_rng(42)
    if front_only:
        theta = rng.uniform(-np.pi/2, np.pi/2, n)
    else:
        theta = rng.uniform(-np.pi, np.pi, n)
    x = np.cos(theta).astype(np.float32)
    y = np.zeros(n, dtype=np.float32)
    z = np.sin(theta).astype(np.float32)
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
            f.write(f"{x[i]:.6f} {y[i]:.6f} {z[i]:.6f} 128 128 128\n")
    return path


# ---------------------------------------------------------------------------
# Coverage tests
# ---------------------------------------------------------------------------

class TestComputeCoverage:
    def test_returns_dict(self):
        pts = np.random.randn(200, 3).astype(np.float32)
        result = compute_coverage(pts)
        assert isinstance(result, dict)

    def test_score_in_range(self):
        pts = np.random.randn(500, 3).astype(np.float32)
        result = compute_coverage(pts, density_threshold=1)
        assert 0.0 <= result["score"] <= 1.0

    def test_all_angles_present(self):
        pts = np.random.randn(100, 3).astype(np.float32)
        result = compute_coverage(pts)
        for angle in VRC_TEST_ANGLES:
            assert angle in result["per_angle"]

    def test_empty_input_score_zero(self):
        pts = np.zeros((0, 3), dtype=np.float32)
        result = compute_coverage(pts)
        assert result["score"] == 0.0

    def test_full_sphere_score_one(self):
        rng = np.random.default_rng(0)
        theta = rng.uniform(-np.pi, np.pi, 10000)
        x = np.cos(theta).astype(np.float32)
        z = np.sin(theta).astype(np.float32)
        pts = np.stack([x, np.zeros_like(x), z], axis=1)
        result = compute_coverage(pts, density_threshold=1)
        assert result["score"] == 1.0

    def test_covered_plus_empty_equals_total(self):
        pts = np.random.randn(200, 3).astype(np.float32)
        result = compute_coverage(pts)
        total = len(result["covered_angles"]) + len(result["empty_angles"])
        assert total == len(VRC_TEST_ANGLES)

    def test_score_equals_covered_fraction(self):
        pts = np.random.randn(300, 3).astype(np.float32)
        result = compute_coverage(pts, density_threshold=1)
        expected = len(result["covered_angles"]) / len(VRC_TEST_ANGLES)
        assert abs(result["score"] - expected) < 1e-6


# ---------------------------------------------------------------------------
# Coherence tests
# ---------------------------------------------------------------------------

class TestComputeCoherence:
    def test_identical_frames_score_one(self):
        frames = np.ones((5, 64, 64, 3), dtype=np.uint8) * 128
        result = compute_coherence_from_frames(frames)
        assert result["score"] == 1.0
        assert result["flicker_mae"] == 0.0

    def test_random_frames_lower_score(self):
        rng = np.random.default_rng(5)
        frames = rng.integers(0, 255, (10, 32, 32, 3), dtype=np.uint8)
        result = compute_coherence_from_frames(frames)
        assert result["score"] < 1.0

    def test_score_in_range(self):
        rng = np.random.default_rng(6)
        frames = rng.integers(0, 255, (8, 32, 32, 3), dtype=np.uint8)
        result = compute_coherence_from_frames(frames)
        assert 0.0 <= result["score"] <= 1.0

    def test_single_frame_returns_one(self):
        frames = np.ones((1, 64, 64, 3), dtype=np.uint8) * 200
        result = compute_coherence_from_frames(frames)
        assert result["score"] == 1.0

    def test_returns_method_key(self):
        frames = np.ones((3, 16, 16, 3), dtype=np.uint8)
        result = compute_coherence_from_frames(frames)
        assert "method" in result
        assert result["method"] == "mae"


# ---------------------------------------------------------------------------
# Quality tests
# ---------------------------------------------------------------------------

class TestComputeQuality:
    def test_identical_frames_high_quality(self):
        frames = np.ones((1, 32, 32, 3), dtype=np.uint8) * 128
        result = compute_quality(frames, frames)
        assert result["score"] > 0.8

    def test_different_frames_lower_quality(self):
        pred = np.zeros((1, 32, 32, 3), dtype=np.uint8)
        gt = np.ones((1, 32, 32, 3), dtype=np.uint8) * 255
        result = compute_quality(pred, gt)
        assert result["score"] < 0.8

    def test_score_in_range(self):
        rng = np.random.default_rng(7)
        pred = rng.integers(0, 255, (2, 16, 16, 3), dtype=np.uint8)
        gt = rng.integers(0, 255, (2, 16, 16, 3), dtype=np.uint8)
        result = compute_quality(pred, gt)
        assert 0.0 <= result["score"] <= 1.0

    def test_result_has_psnr(self):
        pred = np.ones((1, 16, 16, 3), dtype=np.uint8) * 100
        gt = np.ones((1, 16, 16, 3), dtype=np.uint8) * 110
        result = compute_quality(pred, gt)
        assert "psnr_db" in result
        assert result["psnr_db"] > 0

    def test_shape_mismatch_raises(self):
        pred = np.ones((1, 16, 16, 3), dtype=np.uint8)
        gt = np.ones((1, 32, 32, 3), dtype=np.uint8)
        with pytest.raises(AssertionError):
            compute_quality(pred, gt)


# ---------------------------------------------------------------------------
# Composite score tests
# ---------------------------------------------------------------------------

class TestCompositeVRCScore:
    def test_all_ones_gives_one(self):
        assert composite_vrc_score(1.0, 1.0, 1.0) == 1.0

    def test_any_zero_gives_zero(self):
        assert composite_vrc_score(0.0, 1.0, 1.0) == 0.0
        assert composite_vrc_score(1.0, 0.0, 1.0) == 0.0
        assert composite_vrc_score(1.0, 1.0, 0.0) == 0.0

    def test_no_gt_uses_harmonic(self):
        # harmonic mean of 0.8 and 0.8 = 0.8
        score = composite_vrc_score(0.8, 0.8, None)
        assert abs(score - 0.8) < 1e-4

    def test_no_gt_both_zero_gives_zero(self):
        score = composite_vrc_score(0.0, 0.0, None)
        assert score == 0.0

    def test_result_in_range(self):
        rng = np.random.default_rng(8)
        for _ in range(20):
            c, h, q = rng.random(3)
            s = composite_vrc_score(float(c), float(h), float(q))
            assert 0.0 <= s <= 1.0

    def test_multiplicative_property(self):
        c, h, q = 0.9, 0.8, 0.7
        score = composite_vrc_score(c, h, q)
        assert abs(score - round(c * h * q, 4)) < 1e-6


# ---------------------------------------------------------------------------
# VRCScore class tests
# ---------------------------------------------------------------------------

class TestVRCScore:
    def test_compute_from_ply_returns_dict(self, tmp_path):
        ply = make_synthetic_ply(str(tmp_path / "scene.ply"), n=300)
        scorer = VRCScore()
        result = scorer.compute_from_ply(ply)
        assert isinstance(result, dict)

    def test_vrc_coverage_in_result(self, tmp_path):
        ply = make_synthetic_ply(str(tmp_path / "scene.ply"), n=300)
        scorer = VRCScore()
        result = scorer.compute_from_ply(ply)
        assert "vrc_coverage" in result
        assert 0.0 <= result["vrc_coverage"] <= 1.0

    def test_output_json_saved(self, tmp_path):
        ply = make_synthetic_ply(str(tmp_path / "scene.ply"), n=300)
        out = str(tmp_path / "vrc.json")
        scorer = VRCScore()
        scorer.compute_from_ply(ply, output_path=out)
        assert os.path.exists(out)
        with open(out) as f:
            data = json.load(f)
        assert "vrc_coverage" in data

    def test_front_only_low_coverage(self, tmp_path):
        ply = make_synthetic_ply(str(tmp_path / "front.ply"), n=2000, front_only=True)
        scorer = VRCScore(density_threshold=5)
        result = scorer.compute_from_ply(ply)
        assert result["vrc_coverage"] < 1.0

    def test_full_sphere_high_coverage(self, tmp_path):
        ply = make_synthetic_ply(str(tmp_path / "full.ply"), n=5000, front_only=False)
        scorer = VRCScore(density_threshold=1)
        result = scorer.compute_from_ply(ply)
        assert result["vrc_coverage"] > 0.5

    def test_compute_from_frames_composite(self):
        """Full VRC-Score from rendered frames with ground truth."""
        rng = np.random.default_rng(10)
        T, H, W = 5, 16, 16

        renders = {
            angle: rng.integers(100, 200, (T, H, W, 3), dtype=np.uint8)
            for angle in VRC_TEST_ANGLES
        }
        gt = {
            angle: rng.integers(100, 200, (T, H, W, 3), dtype=np.uint8)
            for angle in VRC_TEST_ANGLES
        }

        scorer = VRCScore()
        result = scorer.compute_from_frames(renders, ground_truth=gt)

        assert "vrc_score" in result
        assert "vrc_coverage" in result
        assert "vrc_coherence" in result
        assert "vrc_quality" in result
        assert 0.0 <= result["vrc_score"] <= 1.0

    def test_compute_from_frames_no_gt(self):
        """VRC-Score without GT uses harmonic mean fallback."""
        rng = np.random.default_rng(11)
        T, H, W = 3, 8, 8
        renders = {
            0: rng.integers(0, 255, (T, H, W, 3), dtype=np.uint8),
            90: rng.integers(0, 255, (T, H, W, 3), dtype=np.uint8),
            180: rng.integers(0, 255, (T, H, W, 3), dtype=np.uint8),
        }
        scorer = VRCScore()
        result = scorer.compute_from_frames(renders, ground_truth=None)

        assert result["vrc_quality"] is None
        assert result["vrc_score"] is not None
