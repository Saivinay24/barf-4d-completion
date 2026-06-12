"""
BARF 4D — Robust VRC (VRC-R) Metric Suite
==========================================
Adversarially-audited completeness metrics for generative 4D scene
completion. Companion to `src/attacks/degenerate_completions.py`.

The naive VRC-Coverage in `src/metrics/vrc_score.py` marks an angular
sector "covered" when it contains >= tau points (default tau=5) — on a
2.5M-point cloud this is trivially gameable. VRC-R replaces it with four
reference-free sub-metrics, each designed to defeat a specific attack
class, plus a reference-based quality term for when ground truth exists:

    C1  Relative-density coverage  — sector covered iff it holds at least
        rho * (N/8) points (a fraction of the uniform share). Defeats: dust.
    C2  Appearance consistency     — per-sector RGB histogram divergence
        (Jensen-Shannon) against the densest sector. Defeats: random-colour
        chaff.
    C3  Structural consistency     — per-sector mean nearest-neighbour
        distance ratio against the densest sector. Real surfaces are locally
        dense sheets; volumetric chaff is not. Defeats: chaff (even with
        copied colours).
    C4  Temporal coherence         — mean-absolute-difference between
        consecutive frames of CPU turntable point renders at the 8 test
        angles. Defeats: flicker chaff.

    VRC-R = C1 x C2 x C3 x C4        (reference-free composite)
    VRC-R+Q = VRC-R x Q              (when ground-truth renders exist)

Documented blind spot: a rotational clone of observed content passes all
four reference-free terms (it has real colours, real structure, real
temporal dynamics, full density). Distinguishing plausible-but-wrong from
correct completion requires reference-based Q or human judgement. This is
a fundamental limitation of reference-free evaluation, demonstrated
empirically in the paper.

All computation is CPU-only (numpy + scipy KD-tree).
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from src.attacks.degenerate_completions import (
    TEST_ANGLES_DEG, SECTOR_WIDTH_DEG, sector_azimuth_deg, sector_mask,
)

DEFAULT_RHO = 0.10          # C1: fraction of uniform share (N/8) required
HIST_BINS = 8               # C2: RGB histogram bins per channel
NN_SUBSAMPLE = 5000         # C3: max points per sector for KD-tree NN stat
RENDER_SIZE = 96            # C4: turntable render resolution
COHERENCE_MAE_FULL = 0.2    # C4: MAE at which coherence score reaches 0


# ---------------------------------------------------------------------------
# C1 — relative-density coverage
# ---------------------------------------------------------------------------

def relative_density_coverage(xyz: np.ndarray, center: Optional[np.ndarray] = None,
                              rho: float = DEFAULT_RHO,
                              angles: List[int] = TEST_ANGLES_DEG) -> Dict:
    """Sector covered iff count >= rho * (N / n_sectors)."""
    if center is None:
        center = xyz.mean(axis=0)
    theta = sector_azimuth_deg(xyz, center)
    n = len(xyz)
    share = n / len(angles)
    per_angle, covered = {}, []
    for a in angles:
        count = int(sector_mask(theta, a).sum())
        is_cov = count >= rho * share
        per_angle[a] = {"count": count, "covered": bool(is_cov),
                        "fraction_of_uniform_share": round(count / share, 4) if share else 0.0}
        if is_cov:
            covered.append(a)
    return {"score": round(len(covered) / len(angles), 4), "rho": rho,
            "per_angle": per_angle, "covered_angles": covered,
            "empty_angles": [a for a in angles if a not in covered]}


# ---------------------------------------------------------------------------
# C2 — appearance consistency (RGB histogram Jensen-Shannon divergence)
# ---------------------------------------------------------------------------

def _rgb_histogram(rgb: np.ndarray, bins: int = HIST_BINS) -> np.ndarray:
    idx = (rgb.astype(np.int32) * bins) // 256
    flat = idx[:, 0] * bins * bins + idx[:, 1] * bins + idx[:, 2]
    hist = np.bincount(flat, minlength=bins ** 3).astype(np.float64)
    return hist / max(hist.sum(), 1)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence, base 2, in [0, 1]."""
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _sector_nn_stats(xyz: np.ndarray, masks: Dict[int, np.ndarray],
                     counts: Dict[int, int], min_count: int,
                     seed: int = 0) -> Dict[int, Optional[float]]:
    """
    Mean nearest-neighbour distance per populated sector (subsampled query).
    Exact-duplicate points are removed first: replicating the same point set
    across frames adds no geometry and must not lower the NN statistic.
    """
    rng = np.random.default_rng(seed)
    nn_stat: Dict[int, Optional[float]] = {}
    for a, m in masks.items():
        if counts[a] < min_count:
            nn_stat[a] = None
            continue
        pts = np.unique(xyz[m], axis=0)
        if len(pts) < 2:
            nn_stat[a] = None
            continue
        if len(pts) > NN_SUBSAMPLE:
            query = pts[rng.choice(len(pts), NN_SUBSAMPLE, replace=False)]
        else:
            query = pts
        tree = cKDTree(pts)
        d, _ = tree.query(query, k=2)  # k=2: first hit is the point itself
        nn_stat[a] = float(np.mean(d[:, 1]))
    return nn_stat


def _reference_sector(counts: Dict[int, int],
                      nn_stat: Dict[int, Optional[float]]) -> Optional[int]:
    """
    Reference = the most surface-like (smallest NN distance) sector among
    those holding at least 25% of the maximum sector count. Using density
    alone is gameable when an attack matches the observed density; the
    structure tie-break selects the genuinely observed sheet.
    """
    max_count = max(counts.values()) if counts else 0
    candidates = [a for a, c in counts.items()
                  if c >= 0.25 * max_count and nn_stat.get(a) is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda a: nn_stat[a])


def appearance_consistency(xyz: np.ndarray, rgb: np.ndarray,
                           center: Optional[np.ndarray] = None,
                           angles: List[int] = TEST_ANGLES_DEG,
                           min_count: int = 50, seed: int = 0) -> Dict:
    """
    1 - JS divergence between each populated sector's RGB histogram and the
    reference sector's histogram. Score = mean over populated sectors.
    """
    if center is None:
        center = xyz.mean(axis=0)
    theta = sector_azimuth_deg(xyz, center)
    masks = {a: sector_mask(theta, a) for a in angles}
    counts = {a: int(m.sum()) for a, m in masks.items()}
    nn_stat = _sector_nn_stats(xyz, masks, counts, min_count, seed)
    ref_angle = _reference_sector(counts, nn_stat)
    if ref_angle is None:
        return {"score": 0.0, "per_angle": {}, "ref_angle": None}
    ref_hist = _rgb_histogram(rgb[masks[ref_angle]])

    per_angle, scores = {}, []
    for a in angles:
        if counts[a] < min_count:
            per_angle[a] = None
            continue
        js = _js_divergence(_rgb_histogram(rgb[masks[a]]), ref_hist)
        s = 1.0 - js
        per_angle[a] = round(s, 4)
        scores.append(s)
    return {"score": round(float(np.mean(scores)), 4) if scores else 0.0,
            "per_angle": per_angle, "ref_angle": int(ref_angle)}


# ---------------------------------------------------------------------------
# C3 — structural consistency (nearest-neighbour distance ratio)
# ---------------------------------------------------------------------------

def structural_consistency(xyz: np.ndarray, center: Optional[np.ndarray] = None,
                           angles: List[int] = TEST_ANGLES_DEG,
                           min_count: int = 50, seed: int = 0) -> Dict:
    """
    Per sector: subsample <= NN_SUBSAMPLE points, compute mean distance to the
    nearest neighbour *within the sector's full point set*. Real surface sheets
    have small NN distances; uniform volumetric chaff has large ones.
    Score per sector = nn_ref / max(nn_sector, nn_ref) where nn_ref is the
    reference sector's statistic (most structured among dense sectors).
    Composite = mean over populated sectors.
    """
    if center is None:
        center = xyz.mean(axis=0)
    theta = sector_azimuth_deg(xyz, center)
    masks = {a: sector_mask(theta, a) for a in angles}
    counts = {a: int(m.sum()) for a, m in masks.items()}
    nn_stat = _sector_nn_stats(xyz, masks, counts, min_count, seed)
    ref_angle = _reference_sector(counts, nn_stat)
    if ref_angle is None:
        return {"score": 0.0, "per_angle": {}, "ref_angle": None}

    nn_ref = nn_stat[ref_angle]
    if nn_ref is None or nn_ref <= 0:
        return {"score": 0.0, "per_angle": {}, "ref_angle": int(ref_angle)}

    per_angle, scores = {}, []
    for a in angles:
        if nn_stat[a] is None:
            per_angle[a] = None
            continue
        s = nn_ref / max(nn_stat[a], nn_ref)
        per_angle[a] = {"nn_dist": round(nn_stat[a], 6), "score": round(s, 4)}
        scores.append(s)
    return {"score": round(float(np.mean(scores)), 4) if scores else 0.0,
            "per_angle": per_angle, "ref_angle": int(ref_angle),
            "nn_ref": round(nn_ref, 6)}


# ---------------------------------------------------------------------------
# C4 — temporal coherence via CPU turntable renders
# ---------------------------------------------------------------------------

def render_turntable_frame(xyz: np.ndarray, rgb: np.ndarray,
                           angle_deg: float, center: np.ndarray,
                           img_size: int = RENDER_SIZE,
                           extent: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """
    Orthographic point-splat render from azimuth angle_deg looking at center.
    Z-buffered: nearest point along the view direction wins each pixel.
    Returns (img_size, img_size, 3) uint8.
    """
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    if len(xyz) == 0:
        return img
    rel = xyz - center.reshape(1, 3)
    th = np.radians(angle_deg)
    view = np.array([np.cos(th), 0.0, np.sin(th)])   # camera looks along -view
    right = np.array([-np.sin(th), 0.0, np.cos(th)])
    up = np.array([0.0, 1.0, 0.0])

    u = rel @ right
    v = rel @ up
    depth = rel @ view  # larger = closer to the camera side

    if extent is None:
        lim = max(float(np.percentile(np.abs(u), 98)),
                  float(np.percentile(np.abs(v), 98)), 1e-6)
    else:
        lim = max(extent[0], extent[1], 1e-6)
    ui = np.clip(((u / lim) * 0.5 + 0.5) * (img_size - 1), 0, img_size - 1).astype(np.int32)
    vi = np.clip(((-v / lim) * 0.5 + 0.5) * (img_size - 1), 0, img_size - 1).astype(np.int32)

    order = np.argsort(depth)  # ascending: far first, near last overwrites
    img[vi[order], ui[order]] = rgb[order]
    return img


def temporal_coherence(xyz: np.ndarray, rgb: np.ndarray, frame: np.ndarray,
                       center: Optional[np.ndarray] = None,
                       angles: List[int] = TEST_ANGLES_DEG,
                       img_size: int = RENDER_SIZE,
                       min_count: int = 50) -> Dict:
    """
    Render every frame at each test angle and measure flicker; coherence at
    one angle is 1 - clip(mean MAE between consecutive frames /
    COHERENCE_MAE_FULL, 0, 1). Renders are restricted to the angle's own
    sector content: otherwise static filler in one sector dilutes the
    whole-scene flicker statistic at every other angle (a leak we hit during
    the audit). Composite = mean over angles with sufficient content.
    """
    if center is None:
        center = xyz.mean(axis=0)
    frames_unique = np.sort(np.unique(frame))
    rel = xyz - center.reshape(1, 3)
    lim = max(float(np.percentile(np.abs(rel[:, 0]), 98)),
              float(np.percentile(np.abs(rel[:, 1]), 98)),
              float(np.percentile(np.abs(rel[:, 2]), 98)), 1e-6)
    theta = sector_azimuth_deg(xyz, center)

    per_angle, scores = {}, []
    for a in angles:
        sec = sector_mask(theta, a)
        if int(sec.sum()) < min_count:
            per_angle[a] = None
            continue
        renders = []
        for t in frames_unique:
            m = sec & (frame == t)
            renders.append(render_turntable_frame(
                xyz[m], rgb[m], a, center, img_size, extent=(lim, lim)).astype(np.float32) / 255.0)
        if len(renders) < 2:
            per_angle[a] = None
            continue
        # Coherence = mean Pearson correlation between consecutive frames over
        # pixels lit in either frame. Correlation, not difference magnitude:
        # genuine motion keeps consecutive frames highly correlated even when
        # per-pixel changes are large, while per-frame resampled noise is
        # temporally uncorrelated (~0). The lit-pixel restriction stops black
        # background from inflating the correlation of a small flickering
        # wedge. (An earlier MAE-based variant failed both ways: background
        # diluted flicker, and real scene motion was indistinguishable from
        # noise — see the paper's Section 5.4.)
        corrs, maes = [], []
        for i in range(len(renders) - 1):
            lit = (renders[i].max(axis=2) > 0) | (renders[i + 1].max(axis=2) > 0)
            if int(lit.sum()) < 10:
                continue
            x = renders[i][lit].ravel()
            y = renders[i + 1][lit].ravel()
            maes.append(float(np.abs(y - x).mean()))
            if x.std() < 1e-6 or y.std() < 1e-6:
                corrs.append(1.0 if np.allclose(x, y) else 0.0)
                continue
            corrs.append(float(np.corrcoef(x, y)[0, 1]))
        if not corrs:
            per_angle[a] = None
            continue
        s = float(np.clip(np.mean(corrs), 0.0, 1.0))
        per_angle[a] = {"mae": round(float(np.mean(maes)), 6),
                        "temporal_correlation": round(float(np.mean(corrs)), 4),
                        "score": round(s, 4)}
        scores.append(s)
    return {"score": round(float(np.mean(scores)), 4) if scores else 0.0,
            "per_angle": per_angle, "n_frames": int(len(frames_unique))}


# ---------------------------------------------------------------------------
# Reference-based quality (when ground truth exists)
# ---------------------------------------------------------------------------

def reference_quality(pred_render: np.ndarray, gt_render: np.ndarray) -> Dict:
    """PSNR-mapped quality in [0,1] between two uint8 renders (same shape)."""
    p = pred_render.astype(np.float32) / 255.0
    g = gt_render.astype(np.float32) / 255.0
    mse = float(np.mean((p - g) ** 2))
    psnr = float(10 * np.log10(1.0 / (mse + 1e-8)))
    score = float(np.clip((psnr - 10.0) / 20.0, 0.0, 1.0))
    return {"score": round(score, 4), "psnr_db": round(psnr, 2), "mse": round(mse, 6)}


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

def naive_coverage(xyz: np.ndarray, center: Optional[np.ndarray] = None,
                   tau: int = 5, angles: List[int] = TEST_ANGLES_DEG) -> Dict:
    """The original threshold-based coverage (tau points per sector), centred
    on the scene centroid for comparability with the robust suite."""
    if center is None:
        center = xyz.mean(axis=0)
    theta = sector_azimuth_deg(xyz, center)
    covered = [a for a in angles if int(sector_mask(theta, a).sum()) >= tau]
    return {"score": round(len(covered) / len(angles), 4), "tau": tau,
            "covered_angles": covered}


GATE_APPEARANCE = 0.5   # per-sector appearance score required to count
GATE_STRUCTURE = 0.5    # per-sector structure score required to count


def gated_coverage(c1: Dict, c2: Dict, c3: Dict,
                   angles: List[int] = TEST_ANGLES_DEG) -> Dict:
    """
    A sector counts as covered only if it (a) holds enough points (C1 density
    gate), (b) is chromatically consistent with the reference sector (C2 >=
    GATE_APPEARANCE), and (c) is structurally surface-like (C3 >=
    GATE_STRUCTURE). Partial-credit composites (products of per-metric means)
    leak: an attacker can profit because saturating density coverage outweighs
    the partial appearance/structure penalty. Hard per-sector gating closes
    this — an attacked sector that fails any test contributes exactly nothing.
    """
    per_angle, passing = {}, []
    for a in angles:
        dens_ok = bool(c1["per_angle"][a]["covered"])
        app = c2["per_angle"].get(a)
        struct_entry = c3["per_angle"].get(a)
        struct = struct_entry["score"] if struct_entry else None
        ok = (dens_ok and app is not None and app >= GATE_APPEARANCE
              and struct is not None and struct >= GATE_STRUCTURE)
        per_angle[a] = {"density_ok": dens_ok,
                        "appearance": app, "structure": struct,
                        "passes": bool(ok)}
        if ok:
            passing.append(a)
    return {"score": round(len(passing) / len(angles), 4),
            "passing_angles": passing, "per_angle": per_angle,
            "gates": {"appearance": GATE_APPEARANCE,
                      "structure": GATE_STRUCTURE}}


def compute_vrc_r(xyz: np.ndarray, rgb: np.ndarray, frame: np.ndarray,
                  center: Optional[np.ndarray] = None,
                  rho: float = DEFAULT_RHO,
                  output_path: Optional[str] = None) -> Dict:
    """
    Full reference-free metric suite.

    Primary composite:  vrc_r = GatedCoverage x C4
    Legacy composite:   vrc_r_product = C1 x C2 x C3 x C4 (leaky; reported
                        for the metric-design ablation in the paper)
    """
    if center is None:
        center = xyz.mean(axis=0)
    naive = naive_coverage(xyz, center)
    c1 = relative_density_coverage(xyz, center, rho=rho)
    c2 = appearance_consistency(xyz, rgb, center)
    c3 = structural_consistency(xyz, center)
    c4 = temporal_coherence(xyz, rgb, frame, center)
    gated = gated_coverage(c1, c2, c3)
    # composite coherence over gate-passing sectors only: content that does
    # not count toward coverage must not influence the composite either way
    passing_c4 = [c4["per_angle"][a]["score"] for a in gated["passing_angles"]
                  if c4["per_angle"].get(a)]
    c4_passing = float(np.mean(passing_c4)) if passing_c4 else 0.0
    vrc_r = gated["score"] * c4_passing
    vrc_r_product = c1["score"] * c2["score"] * c3["score"] * c4["score"]
    result = {
        "n_points": int(len(xyz)),
        "center": [round(float(v), 6) for v in center],
        "naive_coverage": naive,
        "c1_relative_density_coverage": c1,
        "c2_appearance_consistency": c2,
        "c3_structural_consistency": c3,
        "c4_temporal_coherence": c4,
        "gated_coverage": gated,
        "c4_over_passing_sectors": round(c4_passing, 4),
        "vrc_r": round(vrc_r, 4),
        "vrc_r_product": round(vrc_r_product, 4),
    }
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
    return result
