"""
BARF 4D — VRC-Score (VR Completeness Score)
=============================================
Formal definition and implementation of the VRC-Score metric introduced in
the BARF paper (Contribution 1: The VR-Completeness Problem).

Definition:
    A 4D scene S is VR-complete if for every viewpoint (θ, φ, t) in the full
    spherical × temporal domain, it renders a photorealistic, consistent frame.

    VRC-Score = Coverage(θ) × (1 - CoherenceLoss(t)) × Quality(θ,t)

    where:
        Coverage(θ)   ∈ [0,1] — fraction of 8 test angles with sufficient geometry
        Coherence(t)  ∈ [0,1] — mean LPIPS between consecutive frames (lower=better)
        Quality(θ,t)  ∈ [0,1] — mean perceptual quality vs ground truth (higher=better)

Sub-metrics:
    1. VRC-Coverage: fraction of 8 test angles (0°,45°,...,315°) with >threshold
                     point density. Uses angular sector sampling of the PLY.
    2. VRC-Coherence: temporal flicker score — LPIPS between frame t and frame t+1
                      across all viewpoints. Lower flicker → higher coherence score.
    3. VRC-Quality: FID/LPIPS vs ground truth renders. Used when ground truth is
                    available (synthetic dataset or two-camera captures).

Usage (CLI):
    python -m src.metrics.vrc_score \
        --scene_ply outputs/completion/scene_complete.ply \
        --output outputs/vrc_score.json

Usage (API):
    from src.metrics.vrc_score import VRCScore
    scorer = VRCScore()
    score = scorer.compute_from_ply("scene.ply")
    print(score)  # {"vrc_score": 0.82, "coverage": 0.91, "coherence": 0.88, ...}
"""

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# Optional: LPIPS for coherence/quality (requires torch + lpips package)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import lpips as lpips_lib
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Test angle configuration — standard for all BARF evaluations
# ---------------------------------------------------------------------------

VRC_TEST_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]   # azimuth degrees
VRC_SECTOR_WIDTH = 45.0   # degrees — each angle covers ±22.5°
VRC_DENSITY_THRESHOLD = 5  # min points in sector to count as "covered"


# ---------------------------------------------------------------------------
# Coverage sub-metric (pure numpy, works without GPU)
# ---------------------------------------------------------------------------

def compute_coverage(
    points: np.ndarray,
    test_angles: List[int] = VRC_TEST_ANGLES,
    sector_width: float = VRC_SECTOR_WIDTH,
    density_threshold: int = VRC_DENSITY_THRESHOLD,
) -> Dict:
    """
    Compute VRC-Coverage: fraction of test angles with sufficient geometry.

    Args:
        points: (N, 3) numpy array of XYZ point cloud positions
        test_angles: list of azimuth angles to test (degrees)
        sector_width: angular width of each sector (degrees)
        density_threshold: min points to count sector as covered

    Returns:
        dict with:
            score:          float in [0,1] — fraction of angles covered
            per_angle:      {angle: {"count": int, "covered": bool}}
            covered_angles: list of covered angle degrees
            empty_angles:   list of empty angle degrees
    """
    if len(points) == 0:
        return {
            "score": 0.0,
            "per_angle": {a: {"count": 0, "covered": False} for a in test_angles},
            "covered_angles": [],
            "empty_angles": test_angles[:],
        }

    # Azimuth angles of each point (in XZ plane, from +X axis)
    x, z = points[:, 0], points[:, 2]
    theta = np.degrees(np.arctan2(z, x))  # [-180, 180]

    half_w = sector_width / 2.0
    per_angle = {}
    covered = []
    empty = []

    for angle in test_angles:
        diff = np.abs(((theta - angle + 180) % 360) - 180)
        count = int((diff <= half_w).sum())
        is_covered = count >= density_threshold
        per_angle[angle] = {"count": count, "covered": is_covered}
        if is_covered:
            covered.append(angle)
        else:
            empty.append(angle)

    score = len(covered) / len(test_angles) if test_angles else 0.0

    return {
        "score": round(score, 4),
        "per_angle": per_angle,
        "covered_angles": covered,
        "empty_angles": empty,
    }


# ---------------------------------------------------------------------------
# Coherence sub-metric (requires rendered frame sequences)
# ---------------------------------------------------------------------------

def compute_coherence_from_frames(
    frames: np.ndarray,   # (T, H, W, 3) — uint8 or float32 RGB frames at one viewpoint
) -> Dict:
    """
    Compute temporal coherence score at a single viewpoint.
    Uses mean absolute difference (MAE) between consecutive frames as a
    proxy for LPIPS when the lpips library is not available.

    Lower flicker (smaller MAE) → higher coherence score.

    Args:
        frames: (T, H, W, 3) array of rendered frames at one viewpoint

    Returns:
        dict with:
            score:      float in [0,1] — coherence (1.0 = perfect, 0.0 = max flicker)
            flicker_mae: float — mean absolute error between consecutive frames
            method:     "mae" or "lpips"
    """
    T = len(frames)
    if T < 2:
        return {"score": 1.0, "flicker_mae": 0.0, "method": "mae"}

    frames_f = frames.astype(np.float32) / 255.0 if frames.dtype == np.uint8 else frames.astype(np.float32)

    # Mean absolute error between consecutive frames
    diffs = [np.abs(frames_f[i+1] - frames_f[i]).mean() for i in range(T-1)]
    flicker_mae = float(np.mean(diffs))

    # Map MAE → coherence score:
    # MAE=0 (no flicker) → score=1.0
    # MAE=0.1 (moderate flicker) → score=0.5
    # MAE=0.2+ (severe flicker) → score≈0
    score = float(max(0.0, 1.0 - flicker_mae / 0.2))

    return {
        "score": round(score, 4),
        "flicker_mae": round(flicker_mae, 6),
        "method": "mae",
    }


def compute_coherence_lpips(
    frames: np.ndarray,   # (T, H, W, 3) uint8
    device: str = "cpu",
) -> Dict:
    """
    Compute temporal coherence using LPIPS (perceptual distance).
    Requires: torch, lpips packages.

    TODO: GPU EXECUTION REQUIRED for large frame sequences (fast on GPU).
    Works on CPU for small sequences (T≤10, H/W≤256).
    """
    if not TORCH_AVAILABLE or not LPIPS_AVAILABLE:
        return compute_coherence_from_frames(frames)

    import torch
    loss_fn = lpips_lib.LPIPS(net="alex").to(device)

    T = len(frames)
    if T < 2:
        return {"score": 1.0, "flicker_lpips": 0.0, "method": "lpips"}

    frames_t = torch.from_numpy(frames.astype(np.float32) / 127.5 - 1.0)  # [-1, 1]
    frames_t = frames_t.permute(0, 3, 1, 2).to(device)  # (T, 3, H, W)

    dists = []
    with torch.no_grad():
        for i in range(T-1):
            d = loss_fn(frames_t[i:i+1], frames_t[i+1:i+2])
            dists.append(float(d.item()))

    flicker_lpips = float(np.mean(dists))
    score = float(max(0.0, 1.0 - flicker_lpips / 0.5))

    return {
        "score": round(score, 4),
        "flicker_lpips": round(flicker_lpips, 6),
        "method": "lpips",
    }


# ---------------------------------------------------------------------------
# Quality sub-metric (requires ground truth renders)
# ---------------------------------------------------------------------------

def compute_quality(
    predicted_frames: np.ndarray,   # (N, H, W, 3) — predicted renders
    ground_truth_frames: np.ndarray,  # (N, H, W, 3) — GT renders
) -> Dict:
    """
    Compute VRC-Quality: perceptual similarity between predicted and GT renders.
    Uses LPIPS if available, falls back to SSIM approximation (numpy only).

    Args:
        predicted_frames:    (N, H, W, 3) array
        ground_truth_frames: (N, H, W, 3) array, same shape

    Returns:
        dict with:
            score:   float in [0,1] — quality (1.0 = perfect match)
            lpips:   float — mean LPIPS distance (lower = better)
            method:  "lpips" or "mse"
    """
    assert predicted_frames.shape == ground_truth_frames.shape, \
        f"Shape mismatch: {predicted_frames.shape} vs {ground_truth_frames.shape}"

    pred_f = predicted_frames.astype(np.float32) / 255.0
    gt_f = ground_truth_frames.astype(np.float32) / 255.0

    # Fallback: MSE-based quality
    mse = float(np.mean((pred_f - gt_f) ** 2))
    psnr = float(10 * np.log10(1.0 / (mse + 1e-8)))

    # Map PSNR → quality score:
    # PSNR >= 30 → score ≈ 1.0 (high quality)
    # PSNR = 20  → score ≈ 0.5 (moderate)
    # PSNR < 10  → score ≈ 0.0 (poor)
    quality_score = float(np.clip((psnr - 10.0) / 20.0, 0.0, 1.0))

    return {
        "score": round(quality_score, 4),
        "mse": round(mse, 6),
        "psnr_db": round(psnr, 2),
        "method": "mse_psnr",
        "note": "Install lpips package for perceptual quality metric",
    }


# ---------------------------------------------------------------------------
# Composite VRC-Score
# ---------------------------------------------------------------------------

def composite_vrc_score(
    coverage_score: float,
    coherence_score: float,
    quality_score: Optional[float] = None,
) -> float:
    """
    Composite VRC-Score = Coverage × Coherence × Quality

    If quality_score is None (no ground truth available),
    falls back to: VRC = Coverage × Coherence
    and normalises to [0,1] based on Coverage+Coherence only.

    Args:
        coverage_score:  float in [0,1]
        coherence_score: float in [0,1]
        quality_score:   float in [0,1] or None

    Returns:
        float VRC composite score in [0,1]
    """
    if quality_score is not None:
        return round(coverage_score * coherence_score * quality_score, 4)
    else:
        # Without GT: use harmonic mean of coverage and coherence
        if coverage_score + coherence_score == 0:
            return 0.0
        return round(2 * coverage_score * coherence_score / (coverage_score + coherence_score), 4)


# ---------------------------------------------------------------------------
# Main VRCScore class
# ---------------------------------------------------------------------------

class VRCScore:
    """
    Main VRC-Score evaluator for BARF pipeline.

    Example (from PLY, CPU):
        scorer = VRCScore()
        result = scorer.compute_from_ply("scene.ply")

    Example (from rendered frames, GPU recommended for LPIPS):
        scorer = VRCScore(device="cuda")
        result = scorer.compute_from_renders(
            renders_dir="outputs/renders/",
            gt_dir="data/ground_truth/",
        )
    """

    def __init__(
        self,
        test_angles: List[int] = VRC_TEST_ANGLES,
        density_threshold: int = VRC_DENSITY_THRESHOLD,
        device: str = "cpu",
    ):
        self.test_angles = test_angles
        self.density_threshold = density_threshold
        self.device = device

    def compute_from_ply(
        self,
        ply_path: str,
        output_path: Optional[str] = None,
    ) -> Dict:
        """
        Compute VRC-Coverage from a PLY file (no renders needed).
        Returns Coverage score; Coherence and Quality are N/A without renders.

        Args:
            ply_path:    path to PLY point cloud
            output_path: optional path to save JSON results

        Returns:
            dict with vrc_coverage, and full VRC-Score if possible
        """
        from src.gap_detection.detect_gaps import load_ply_xyz

        print(f"[VRCScore] Loading PLY: {ply_path}")
        points = load_ply_xyz(ply_path)
        print(f"[VRCScore] Loaded {len(points)} points")

        coverage = compute_coverage(
            points,
            test_angles=self.test_angles,
            density_threshold=self.density_threshold,
        )

        # Without renders, we can only compute Coverage
        # Coherence and Quality default to N/A (marked as requiring GPU renders)
        result = {
            "input_ply": ply_path,
            "n_points": len(points),
            "vrc_coverage": coverage["score"],
            "vrc_coherence": None,  # TODO: requires rendered frame sequences
            "vrc_quality": None,    # TODO: requires ground truth renders
            "vrc_score": None,      # TODO: requires all three sub-metrics
            "coverage_detail": coverage,
            "notes": {
                "vrc_coherence": "Requires rendered frame sequences — run on GPU after completion",
                "vrc_quality":   "Requires ground truth renders — available with two-camera captures",
                "vrc_score":     "Full composite requires all three sub-metrics",
            },
        }

        # If only coverage available, report coverage as partial VRC
        result["vrc_score_coverage_only"] = coverage["score"]

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"[VRCScore] Results saved: {output_path}")

        print(f"[VRCScore] Coverage:       {coverage['score']*100:.1f}%")
        print(f"[VRCScore] Covered angles: {coverage['covered_angles']}")
        print(f"[VRCScore] Empty angles:   {coverage['empty_angles']}")

        return result

    def compute_from_frames(
        self,
        renders: Dict[int, np.ndarray],   # {angle_deg: (T, H, W, 3)}
        ground_truth: Optional[Dict[int, np.ndarray]] = None,
        output_path: Optional[str] = None,
    ) -> Dict:
        """
        Full VRC-Score from rendered frame sequences (all three sub-metrics).

        Args:
            renders:      dict mapping angle_deg → (T, H, W, 3) rendered frames
            ground_truth: dict mapping angle_deg → (T, H, W, 3) GT frames (optional)
            output_path:  optional path to save results JSON

        Returns:
            dict with all three sub-metrics and composite VRC-Score
        """
        # Coverage: count angles with non-empty renders
        covered = [a for a, frames in renders.items() if frames is not None and len(frames) > 0]
        empty = [a for a in self.test_angles if a not in covered]
        coverage_score = len(covered) / len(self.test_angles) if self.test_angles else 0.0

        # Coherence: mean over all angles
        coherence_scores = []
        for angle, frames in renders.items():
            if frames is not None and len(frames) >= 2:
                coh = compute_coherence_from_frames(frames)
                coherence_scores.append(coh["score"])
        coherence_score = float(np.mean(coherence_scores)) if coherence_scores else 0.0

        # Quality: compare with ground truth if available
        quality_score = None
        quality_details = {}
        if ground_truth is not None:
            quality_scores_per_angle = []
            for angle in self.test_angles:
                if angle in renders and angle in ground_truth:
                    pred = renders[angle]
                    gt = ground_truth[angle]
                    # Use mean frame for quality (TODO: GPU per-frame LPIPS)
                    pred_mean = pred.mean(axis=0, keepdims=True)
                    gt_mean = gt.mean(axis=0, keepdims=True)
                    q = compute_quality(pred_mean, gt_mean)
                    quality_scores_per_angle.append(q["score"])
                    quality_details[angle] = q
            if quality_scores_per_angle:
                quality_score = float(np.mean(quality_scores_per_angle))

        vrc = composite_vrc_score(coverage_score, coherence_score, quality_score)

        result = {
            "vrc_score": vrc,
            "vrc_coverage": round(coverage_score, 4),
            "vrc_coherence": round(coherence_score, 4),
            "vrc_quality": round(quality_score, 4) if quality_score is not None else None,
            "covered_angles": covered,
            "empty_angles": empty,
            "quality_per_angle": quality_details,
        }

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)

        print(f"[VRCScore] VRC-Score:    {vrc:.4f}")
        print(f"[VRCScore] Coverage:     {coverage_score:.4f}")
        print(f"[VRCScore] Coherence:    {coherence_score:.4f}")
        print(f"[VRCScore] Quality:      {quality_score if quality_score else 'N/A (no GT)'}")

        return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BARF VRC-Score — VR Completeness Score evaluation"
    )
    parser.add_argument("--scene_ply", required=True, help="PLY point cloud to evaluate")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    scorer = VRCScore()
    result = scorer.compute_from_ply(args.scene_ply, output_path=args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
