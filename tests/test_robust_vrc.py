"""Tests for src/metrics/robust_vrc.py — the adversarially-audited metric suite."""

import numpy as np
import pytest

from src.attacks.degenerate_completions import (
    dust_attack, chaff_attack, clone_attack,
)
from src.metrics.robust_vrc import (
    naive_coverage, relative_density_coverage, appearance_consistency,
    structural_consistency, temporal_coherence, compute_vrc_r,
    render_turntable_frame, reference_quality,
)
from tests.test_attacks import make_front_facing_scene


@pytest.fixture
def scene():
    return make_front_facing_scene()


def make_full_ring_scene(n_per_frame=4000, n_frames=5, seed=0):
    """Complete scene: a full ring (surface sheet) around the centre."""
    rng = np.random.default_rng(seed)
    n = n_per_frame * n_frames
    theta = rng.uniform(-np.pi, np.pi, n)
    r = rng.uniform(0.95, 1.05, n)
    y = rng.uniform(-0.5, 0.5, n)
    xyz = np.stack([r * np.cos(theta), y, r * np.sin(theta)], axis=1).astype(np.float32)
    rgb = rng.integers(40, 90, (n, 3), dtype=np.uint8)
    frame = np.repeat(np.arange(n_frames, dtype=np.int32), n_per_frame)
    return xyz, rgb, frame


class TestNaiveCoverageGameability:
    def test_dust_games_naive(self, scene):
        xyz, rgb, frame = scene
        center = np.zeros(3)
        before = naive_coverage(xyz, center)["score"]
        out = dust_attack(xyz, rgb, frame, center=center)
        after = naive_coverage(out["xyz"], center)["score"]
        assert before < 0.5 and after == 1.0


class TestC1RelativeDensity:
    def test_resists_dust(self, scene):
        xyz, rgb, frame = scene
        center = np.zeros(3)
        before = relative_density_coverage(xyz, center)["score"]
        out = dust_attack(xyz, rgb, frame, center=center)
        after = relative_density_coverage(out["xyz"], center)["score"]
        assert after == pytest.approx(before, abs=1e-9)

    def test_full_ring_scores_one(self):
        xyz, _, _ = make_full_ring_scene()
        assert relative_density_coverage(xyz, np.zeros(3))["score"] == 1.0

    def test_chaff_passes_c1(self, scene):
        # chaff is full-density by construction; C1 alone must NOT catch it
        xyz, rgb, frame = scene
        center = np.zeros(3)
        out = chaff_attack(xyz, rgb, frame, center=center)
        assert relative_density_coverage(out["xyz"], center)["score"] == 1.0


class TestC2Appearance:
    def test_chaff_lowers_appearance(self, scene):
        xyz, rgb, frame = scene
        center = np.zeros(3)
        before = appearance_consistency(xyz, rgb, center)["score"]
        out = chaff_attack(xyz, rgb, frame, center=center, seed=1)
        after = appearance_consistency(out["xyz"], out["rgb"], center)["score"]
        assert after < before - 0.2

    def test_clone_preserves_appearance(self, scene):
        xyz, rgb, frame = scene
        center = np.zeros(3)
        out = clone_attack(xyz, rgb, frame, center=center)
        after = appearance_consistency(out["xyz"], out["rgb"], center)["score"]
        assert after > 0.9


class TestC3Structure:
    def test_chaff_lowers_structure(self, scene):
        xyz, rgb, frame = scene
        center = np.zeros(3)
        before = structural_consistency(xyz, center)["score"]
        out = chaff_attack(xyz, rgb, frame, center=center, seed=1)
        after = structural_consistency(out["xyz"], center)["score"]
        assert after < before

    def test_clone_preserves_structure(self, scene):
        xyz, rgb, frame = scene
        center = np.zeros(3)
        out = clone_attack(xyz, rgb, frame, center=center)
        after = structural_consistency(out["xyz"], center)["score"]
        assert after > 0.85


class TestC4TemporalCoherence:
    def test_flicker_chaff_lowers_coherence(self, scene):
        xyz, rgb, frame = scene
        center = np.zeros(3)
        base = temporal_coherence(xyz, rgb, frame, center)["score"]
        static = chaff_attack(xyz, rgb, frame, center=center, temporal_mode="static", seed=1)
        flicker = chaff_attack(xyz, rgb, frame, center=center, temporal_mode="flicker", seed=1)
        s_static = temporal_coherence(static["xyz"], static["rgb"], static["frame"], center)["score"]
        s_flicker = temporal_coherence(flicker["xyz"], flicker["rgb"], flicker["frame"], center)["score"]
        assert s_flicker < s_static
        assert s_flicker < base

    def test_render_shape_and_dtype(self, scene):
        xyz, rgb, _ = scene
        img = render_turntable_frame(xyz, rgb, 90.0, np.zeros(3), img_size=64)
        assert img.shape == (64, 64, 3) and img.dtype == np.uint8
        assert img.max() > 0  # something rendered at the populated angle

    def test_empty_render_is_black(self):
        img = render_turntable_frame(np.zeros((0, 3), np.float32),
                                     np.zeros((0, 3), np.uint8), 0.0, np.zeros(3))
        assert img.sum() == 0


class TestComposite:
    def test_vrc_r_ranks_attacks_below_clone(self, scene):
        """The headline empirical claim, in miniature: dust/chaff/flicker are
        caught; the clone is the documented reference-free blind spot."""
        xyz, rgb, frame = scene
        center = np.zeros(3)
        base = compute_vrc_r(xyz, rgb, frame, center)
        dust = dust_attack(xyz, rgb, frame, center=center)
        chaff = chaff_attack(xyz, rgb, frame, center=center, seed=1)
        flick = chaff_attack(xyz, rgb, frame, center=center, temporal_mode="flicker", seed=1)
        clone = clone_attack(xyz, rgb, frame, center=center)

        r_dust = compute_vrc_r(dust["xyz"], dust["rgb"], dust["frame"], center)
        r_chaff = compute_vrc_r(chaff["xyz"], chaff["rgb"], chaff["frame"], center)
        r_flick = compute_vrc_r(flick["xyz"], flick["rgb"], flick["frame"], center)
        r_clone = compute_vrc_r(clone["xyz"], clone["rgb"], clone["frame"], center)

        # dust gains nothing on C1 -> composite stays at the honest level
        assert r_dust["c1_relative_density_coverage"]["score"] == \
            base["c1_relative_density_coverage"]["score"]
        # chaff/flicker composite must be well below the clone composite
        assert r_chaff["vrc_r"] < r_clone["vrc_r"]
        assert r_flick["vrc_r"] < r_clone["vrc_r"]
        # flicker is additionally punished on C4 relative to static chaff
        assert r_flick["c4_temporal_coherence"]["score"] < \
            r_chaff["c4_temporal_coherence"]["score"]

    def test_vrc_r_json_roundtrip(self, scene, tmp_path):
        import json
        xyz, rgb, frame = scene
        out = tmp_path / "vrc.json"
        res = compute_vrc_r(xyz, rgb, frame, output_path=str(out))
        loaded = json.loads(out.read_text())
        assert loaded["vrc_r"] == res["vrc_r"]


class TestReferenceQuality:
    def test_identical_renders_perfect(self):
        img = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        q = reference_quality(img, img)
        assert q["score"] == 1.0

    def test_clone_vs_truth_detectable(self):
        """Reference-based quality CAN catch what reference-free cannot:
        a clone differs from the ground-truth content at the cloned angle."""
        rng = np.random.default_rng(0)
        gt = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        wrong = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        assert reference_quality(wrong, gt)["score"] < 0.5


class TestGatedCoverage:
    def test_chaff_gains_nothing_on_gated_composite(self, scene):
        """The leak that motivated gating: chaff saturates C1 but must not
        increase the gated composite above the honest scene's value."""
        xyz, rgb, frame = scene
        center = np.zeros(3)
        base = compute_vrc_r(xyz, rgb, frame, center)
        chaff = chaff_attack(xyz, rgb, frame, center=center, seed=1)
        r_chaff = compute_vrc_r(chaff["xyz"], chaff["rgb"], chaff["frame"], center)
        assert r_chaff["gated_coverage"]["score"] <= \
            base["gated_coverage"]["score"]
        assert r_chaff["vrc_r"] <= base["vrc_r"] + 1e-9

    def test_clone_passes_gates(self, scene):
        xyz, rgb, frame = scene
        center = np.zeros(3)
        clone = clone_attack(xyz, rgb, frame, center=center)
        r = compute_vrc_r(clone["xyz"], clone["rgb"], clone["frame"], center)
        assert r["gated_coverage"]["score"] == 1.0
