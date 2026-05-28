"""
Tests for src/completion/spherical_completion.py
Runs locally on CPU with synthetic tensors — no GPU required.
"""

import json
import os
import tempfile

import numpy as np
import pytest

from src.completion.spherical_completion import (
    GaussianFusion,
    SphericalCompletionPipeline,
    TORCH_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_synthetic_ply(path: str, n: int = 200) -> str:
    """Write a minimal synthetic PLY for testing."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rng = np.random.default_rng(99)
    x = rng.uniform(-1, 1, n).astype(np.float32)
    y = rng.uniform(-1, 1, n).astype(np.float32)
    z = rng.uniform(-0.1, 1, n).astype(np.float32)   # front-biased
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


def make_synthetic_gaps_json(path: str) -> str:
    """Write a synthetic gaps.json for testing."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {
        "input_ply": "synthetic.ply",
        "n_points": 200,
        "vrc_coverage_score": 0.45,
        "gaps": [
            {"id": 0, "center": [0.0, 0.0, -0.5], "size_m3": 0.4,
             "azimuth_deg": 180.0, "elevation_deg": 0.0,
             "bounding_box": {"min": [-0.2, -0.2, -0.7], "max": [0.2, 0.2, -0.3]}},
            {"id": 1, "center": [0.5, 0.0, 0.0], "size_m3": 0.2,
             "azimuth_deg": 90.0, "elevation_deg": 5.0,
             "bounding_box": {"min": [0.3, -0.1, -0.2], "max": [0.7, 0.1, 0.2]}},
        ],
        "summary": {
            "total_gaps": 2,
            "empty_angles": [135, 180, 225],
            "covered_angles": [0, 45, 90, 270, 315],
        },
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGaussianFusion:
    """GaussianFusion is pure numpy — always runs."""

    def test_backproject_basic(self):
        fusion = GaussianFusion()
        T, H, W = 2, 8, 8
        rgba = np.random.randint(0, 255, (T, H, W, 4), dtype=np.uint8).astype(np.float32)
        depth = np.ones((T, H, W), dtype=np.float32) * 1.0
        pose = np.eye(4, dtype=np.float32)
        intrinsics = np.array([[100, 0, 4], [0, 100, 4], [0, 0, 1]], dtype=np.float32)
        result = fusion.backproject_rgba_to_gaussians(rgba, depth, pose, intrinsics)
        assert result.ndim == 2
        assert result.shape[1] == 7   # x,y,z,r,g,b,alpha

    def test_backproject_zero_depth_skipped(self):
        fusion = GaussianFusion()
        T, H, W = 1, 4, 4
        rgba = np.ones((T, H, W, 4), dtype=np.float32) * 128
        depth = np.zeros((T, H, W), dtype=np.float32)  # all zero → skip all
        pose = np.eye(4, dtype=np.float32)
        intrinsics = np.array([[50, 0, 2], [0, 50, 2], [0, 0, 1]], dtype=np.float32)
        result = fusion.backproject_rgba_to_gaussians(rgba, depth, pose, intrinsics)
        assert len(result) == 0

    def test_merge_with_scene_basic(self):
        fusion = GaussianFusion()
        existing = np.random.randn(100, 6).astype(np.float32)
        new_g = np.random.randn(50, 7).astype(np.float32)
        new_g[:, 6] = 1.0   # all high confidence
        merged = fusion.merge_with_scene(existing, new_g)
        assert merged.shape[1] == 6
        assert len(merged) >= len(existing)

    def test_merge_filters_low_confidence(self):
        fusion = GaussianFusion()
        existing = np.random.randn(100, 6).astype(np.float32)
        new_g = np.random.randn(50, 7).astype(np.float32)
        new_g[:, 6] = 0.1   # all low confidence → should be filtered
        merged = fusion.merge_with_scene(existing, new_g)
        assert len(merged) == len(existing)  # nothing added

    def test_merge_empty_new(self):
        fusion = GaussianFusion()
        existing = np.random.randn(100, 6).astype(np.float32)
        new_g = np.zeros((0, 7), dtype=np.float32)
        merged = fusion.merge_with_scene(existing, new_g)
        assert len(merged) == len(existing)


class TestSphericalCompletionPipelineSynthetic:
    """
    Tests the synthetic forward pass of SphericalCompletionPipeline.
    All run on CPU — no GPU or real data required.
    """

    def test_pipeline_initialises(self):
        pipeline = SphericalCompletionPipeline(device="cpu")
        assert pipeline is not None

    def test_complete_synthetic_returns_dict(self):
        pipeline = SphericalCompletionPipeline(device="cpu", img_size=16)
        result = pipeline.complete_synthetic()
        assert isinstance(result, dict)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_synthetic_output_status_ok(self):
        pipeline = SphericalCompletionPipeline(device="cpu", img_size=16)
        result = pipeline.complete_synthetic()
        assert result["status"] == "ok"

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_synthetic_scene_latent_shape(self):
        B, T = 2, 10
        pipeline = SphericalCompletionPipeline(device="cpu", feature_dim=512, img_size=16)
        result = pipeline.complete_synthetic()
        # scene_latent should be (B, T, D)
        assert len(result["scene_latent_shape"]) == 3
        assert result["scene_latent_shape"][2] == 512

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_synthetic_denoised_shape(self):
        pipeline = SphericalCompletionPipeline(device="cpu", feature_dim=512, img_size=16)
        result = pipeline.complete_synthetic()
        # denoised: (B, T, 4, H, W)
        assert len(result["denoised_shape"]) == 5
        B, T, C, H, W = result["denoised_shape"]
        assert C == 4   # RGBA
        assert H == 16  # img_size
        assert W == 16

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_synthetic_gap_queries_shape(self):
        pipeline = SphericalCompletionPipeline(device="cpu", feature_dim=512, img_size=16)
        result = pipeline.complete_synthetic()
        # gap_queries: (B, G, D) where G=3 synthetic gaps
        assert len(result["gap_queries_shape"]) == 3
        B, G, D = result["gap_queries_shape"]
        assert G == 3
        assert D == 512

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_synthetic_output_finite(self):
        import torch
        pipeline = SphericalCompletionPipeline(device="cpu", img_size=16)
        result = pipeline.complete_synthetic()
        assert abs(result["denoised_mean"]) < 1000   # sanity check not NaN/inf
        assert result["denoised_std"] >= 0.0

    def test_complete_synthetic_no_torch(self, monkeypatch):
        """When torch unavailable, synthetic returns stub dict with shapes."""
        import src.completion.spherical_completion as mod
        monkeypatch.setattr(mod, "TORCH_AVAILABLE", False)
        # Create pipeline without torch
        pipeline = SphericalCompletionPipeline.__new__(SphericalCompletionPipeline)
        pipeline.feature_dim = 512
        pipeline.img_size = 64
        pipeline.n_diffusion_steps = 20
        pipeline.device = "cpu"
        result = pipeline.complete_synthetic()
        assert result["status"] == "torch_not_available"
        assert "scene_latent_shape" in result


class TestSphericalCompletionFromFiles:
    """
    Tests the pipeline.complete() method that reads from PLY and gaps JSON.
    Creates placeholder outputs — actual completion requires GPU.
    """

    def test_complete_produces_output_ply(self, tmp_path):
        ply = make_synthetic_ply(str(tmp_path / "neoverse" / "scene.ply"))
        gaps = make_synthetic_gaps_json(str(tmp_path / "gaps" / "gaps.json"))
        out_ply = str(tmp_path / "completion" / "scene_complete.ply")

        pipeline = SphericalCompletionPipeline(device="cpu")
        result = pipeline.complete(
            scene_ply=ply,
            gaps_json=gaps,
            output_ply=out_ply,
        )
        assert os.path.exists(out_ply), "Output PLY file should be created"
        assert result["output_ply"] == out_ply

    def test_complete_reports_gap_info(self, tmp_path):
        ply = make_synthetic_ply(str(tmp_path / "scene.ply"))
        gaps = make_synthetic_gaps_json(str(tmp_path / "gaps.json"))
        out_ply = str(tmp_path / "out.ply")

        pipeline = SphericalCompletionPipeline(device="cpu")
        result = pipeline.complete(ply, gaps, out_ply)

        assert result["n_gap_clusters"] == 2
        assert "empty_angles" in result

    def test_complete_missing_ply_raises(self, tmp_path):
        gaps = make_synthetic_gaps_json(str(tmp_path / "gaps.json"))
        pipeline = SphericalCompletionPipeline(device="cpu")
        with pytest.raises(FileNotFoundError):
            pipeline.complete("/nonexistent.ply", str(gaps), str(tmp_path / "out.ply"))

    def test_complete_missing_gaps_raises(self, tmp_path):
        ply = make_synthetic_ply(str(tmp_path / "scene.ply"))
        pipeline = SphericalCompletionPipeline(device="cpu")
        with pytest.raises(FileNotFoundError):
            pipeline.complete(str(ply), "/nonexistent_gaps.json", str(tmp_path / "out.ply"))

    def test_placeholder_output_is_valid_ply(self, tmp_path):
        """The placeholder output should be a readable PLY file."""
        ply = make_synthetic_ply(str(tmp_path / "scene.ply"), n=50)
        gaps = make_synthetic_gaps_json(str(tmp_path / "gaps.json"))
        out_ply = str(tmp_path / "out.ply")

        pipeline = SphericalCompletionPipeline(device="cpu")
        pipeline.complete(ply, gaps, out_ply)

        from src.gap_detection.detect_gaps import load_ply_xyz
        pts = load_ply_xyz(out_ply)
        assert len(pts) == 50   # placeholder preserves input points
