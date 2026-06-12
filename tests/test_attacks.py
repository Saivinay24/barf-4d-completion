"""Tests for src/attacks/degenerate_completions.py"""

import numpy as np
import pytest

from src.attacks.degenerate_completions import (
    dust_attack, chaff_attack, clone_attack, apply_attack,
    sector_counts, empty_sectors, TEST_ANGLES_DEG,
)
from src.metrics.robust_vrc import naive_coverage


def make_front_facing_scene(n_per_frame=2000, n_frames=5, seed=42):
    """Synthetic front-facing sheet: points only in the 90-degree sector.
    Temporally coherent: the same surface points appear in every frame with
    small jitter, as in a real reconstruction."""
    rng = np.random.default_rng(seed)
    theta = np.radians(90 + rng.uniform(-20, 20, n_per_frame))
    r = rng.uniform(0.8, 1.2, n_per_frame)
    y = rng.uniform(-0.5, 0.5, n_per_frame)
    base = np.stack([r * np.cos(theta), y, r * np.sin(theta)], axis=1).astype(np.float32)
    rgb1 = rng.integers(40, 90, (n_per_frame, 3), dtype=np.uint8)  # dark, narrow palette
    xyz_l, rgb_l, frame_l = [], [], []
    for t in range(n_frames):
        jitter = rng.normal(0, 0.002, base.shape).astype(np.float32)
        xyz_l.append(base + jitter)
        rgb_l.append(rgb1)
        frame_l.append(np.full(n_per_frame, t, dtype=np.int32))
    return (np.concatenate(xyz_l), np.concatenate(rgb_l),
            np.concatenate(frame_l))


@pytest.fixture
def scene():
    return make_front_facing_scene()


class TestSectorHelpers:
    def test_sector_counts_sum(self, scene):
        xyz, _, _ = scene
        center = np.zeros(3)
        counts = sector_counts(xyz, center)
        assert sum(counts.values()) == len(xyz)

    def test_empty_sectors_front_facing(self, scene):
        xyz, _, _ = scene
        center = np.zeros(3)
        empties = empty_sectors(xyz, center)
        assert 90 not in empties
        assert 270 in empties


class TestDustAttack:
    def test_flips_naive_coverage_to_full(self, scene):
        xyz, rgb, frame = scene
        center = np.zeros(3)
        assert naive_coverage(xyz, center)["score"] < 1.0
        out = dust_attack(xyz, rgb, frame, center=center)
        assert naive_coverage(out["xyz"], center)["score"] == 1.0

    def test_adds_minimal_points(self, scene):
        xyz, rgb, frame = scene
        center = np.zeros(3)
        n_empty = len(empty_sectors(xyz, center))
        out = dust_attack(xyz, rgb, frame, center=center, density_threshold=5)
        assert out["n_added"] == 5 * n_empty

    def test_deterministic(self, scene):
        xyz, rgb, frame = scene
        a = dust_attack(xyz, rgb, frame, seed=7)
        b = dust_attack(xyz, rgb, frame, seed=7)
        np.testing.assert_array_equal(a["xyz"], b["xyz"])


class TestChaffAttack:
    def test_flips_naive_coverage(self, scene):
        xyz, rgb, frame = scene
        center = np.zeros(3)
        out = chaff_attack(xyz, rgb, frame, center=center)
        assert naive_coverage(out["xyz"], center)["score"] == 1.0

    def test_static_chaff_same_points_every_frame(self, scene):
        xyz, rgb, frame = scene
        out = chaff_attack(xyz, rgb, frame, center=np.zeros(3), temporal_mode="static", seed=3)
        added_xyz = out["xyz"][len(xyz):]
        added_frame = out["frame"][len(frame):]
        f0 = np.sort(added_xyz[added_frame == 0], axis=0)
        f1 = np.sort(added_xyz[added_frame == 1], axis=0)
        np.testing.assert_allclose(f0, f1)

    def test_flicker_chaff_different_points_per_frame(self, scene):
        xyz, rgb, frame = scene
        out = chaff_attack(xyz, rgb, frame, center=np.zeros(3), temporal_mode="flicker", seed=3)
        added_xyz = out["xyz"][len(xyz):]
        added_frame = out["frame"][len(frame):]
        f0 = added_xyz[added_frame == 0]
        f1 = added_xyz[added_frame == 1]
        assert not np.allclose(np.sort(f0, axis=0), np.sort(f1, axis=0))

    def test_invalid_mode_raises(self, scene):
        xyz, rgb, frame = scene
        with pytest.raises(ValueError):
            chaff_attack(xyz, rgb, frame, temporal_mode="bogus")


class TestCloneAttack:
    def test_flips_naive_coverage(self, scene):
        xyz, rgb, frame = scene
        center = np.zeros(3)
        out = clone_attack(xyz, rgb, frame, center=center)
        assert naive_coverage(out["xyz"], center)["score"] == 1.0

    def test_preserves_colours_and_frames(self, scene):
        xyz, rgb, frame = scene
        out = clone_attack(xyz, rgb, frame, center=np.zeros(3))
        added_rgb = out["rgb"][len(rgb):]
        # clone colours must be drawn from the original palette (40..89)
        assert added_rgb.min() >= 40 and added_rgb.max() < 90
        added_frame = out["frame"][len(frame):]
        assert set(np.unique(added_frame)) <= set(np.unique(frame))

    def test_clone_radius_preserved(self, scene):
        xyz, rgb, frame = scene
        center = np.zeros(3)
        out = clone_attack(xyz, rgb, frame, center=center)
        added = out["xyz"][len(xyz):]
        r_added = np.sqrt(added[:, 0] ** 2 + added[:, 2] ** 2)
        r_orig = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 2] ** 2)
        assert abs(r_added.mean() - r_orig.mean()) < 0.05


class TestRegistry:
    def test_all_attacks_runnable(self, scene):
        xyz, rgb, frame = scene
        for name in ["dust", "chaff_static", "chaff_flicker", "clone"]:
            out = apply_attack(name, xyz, rgb, frame, center=np.zeros(3), seed=1)
            assert out["n_added"] > 0
            assert len(out["xyz"]) == len(out["rgb"]) == len(out["frame"])

    def test_unknown_attack_raises(self, scene):
        xyz, rgb, frame = scene
        with pytest.raises(ValueError):
            apply_attack("nope", xyz, rgb, frame)
