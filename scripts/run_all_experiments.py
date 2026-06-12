"""
BARF 4D — Full experiment pipeline (CPU-only, Apple Silicon friendly).

Runs every experiment reported in the paper and writes raw result artifacts
to results/session/. All numbers in paper/barf_paper.html are produced by
this script (see paper/REPRODUCIBILITY.md for the mapping).

Experiments:
    E0  Provenance audit of the Phase-0 scene artifacts
    E1  Diagnostic study: angular coverage of real monocular
        depth-unprojection scenes (capture-centred vs centroid-centred)
    E2  Gameability audit: degenerate completion attacks vs naive coverage
        and the VRC-R metric suite
    E3  Synthetic validation: metric monotonicity under controlled ablation,
        attack saturation curves, and reference-based quality on clones

Usage:
    python3 scripts/run_all_experiments.py [--scenes_dir data/scenes]
                                           [--out results/session]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attacks.degenerate_completions import (
    ATTACK_REGISTRY, apply_attack, sector_counts, TEST_ANGLES_DEG,
)
from src.metrics.robust_vrc import (
    compute_vrc_r, naive_coverage, relative_density_coverage,
    render_turntable_frame, reference_quality,
)

SCENES = ["01_mdn_flower", "02_w3schools_big_buck_bunny", "03_samplelib_5s"]
CAPTURE_CENTER = np.zeros(3, dtype=np.float64)  # camera at origin in unprojected clouds
SEED = 0


def load_scene(scenes_dir: str, name: str):
    d = np.load(Path(scenes_dir) / f"{name}.npz")
    return d["xyz"], d["rgb"], d["frame"], json.loads(str(d["provenance"]))


# ---------------------------------------------------------------------------
# E0 — provenance audit
# ---------------------------------------------------------------------------

def e0_provenance(scenes_dir: str) -> dict:
    rows = {}
    for name in SCENES:
        xyz, rgb, frame, prov = load_scene(scenes_dir, name)
        meta = prov["source_metadata"]
        w = meta["resolution"]["width"]
        h = meta["resolution"]["height"]
        T = meta["num_frames"]
        rows[name] = {
            "n_points": int(len(xyz)),
            "width": w, "height": h, "num_frames": T,
            "w_h_T": w * h * T,
            "missing_vs_w_h_T": w * h * T - int(len(xyz)),
            "frames_present": int(len(np.unique(frame))),
            "artifact_type": prov["artifact_type"],
        }
    return rows


# ---------------------------------------------------------------------------
# E1 — diagnostic study on real scenes
# ---------------------------------------------------------------------------

def e1_diagnostic(scenes_dir: str) -> dict:
    out = {}
    for name in SCENES:
        xyz, rgb, frame, _ = load_scene(scenes_dir, name)
        centroid = xyz.mean(axis=0)
        t0 = time.time()
        res = compute_vrc_r(xyz, rgb, frame, center=CAPTURE_CENTER)
        out[name] = {
            "capture_centered": {
                "sector_counts": sector_counts(xyz, CAPTURE_CENTER),
                "naive_coverage": naive_coverage(xyz, CAPTURE_CENTER),
                "vrc_r_full": res,
            },
            "centroid_centered": {
                "sector_counts": sector_counts(xyz, centroid),
                "naive_coverage": naive_coverage(xyz, centroid),
                "c1_relative_density": relative_density_coverage(xyz, centroid),
            },
            "elapsed_s": round(time.time() - t0, 1),
        }
        print(f"[E1] {name}: naive(capture)="
              f"{out[name]['capture_centered']['naive_coverage']['score']}, "
              f"naive(centroid)="
              f"{out[name]['centroid_centered']['naive_coverage']['score']}, "
              f"VRC-R={res['vrc_r']}  ({out[name]['elapsed_s']}s)")
    return out


# ---------------------------------------------------------------------------
# E2 — gameability audit on real scenes
# ---------------------------------------------------------------------------

def e2_attacks(scenes_dir: str) -> dict:
    out = {}
    for name in SCENES:
        xyz, rgb, frame, _ = load_scene(scenes_dir, name)
        base = compute_vrc_r(xyz, rgb, frame, center=CAPTURE_CENTER)
        rows = {"original": summarize(base, n_added=0)}
        for attack in ["dust", "chaff_static", "chaff_flicker", "clone"]:
            t0 = time.time()
            a = apply_attack(attack, xyz, rgb, frame,
                             center=CAPTURE_CENTER, seed=SEED)
            r = compute_vrc_r(a["xyz"], a["rgb"], a["frame"],
                              center=CAPTURE_CENTER)
            rows[attack] = summarize(r, n_added=int(a["n_added"]))
            print(f"[E2] {name}/{attack}: +{a['n_added']} pts, "
                  f"naive={r['naive_coverage']['score']}, VRC-R={r['vrc_r']} "
                  f"({time.time()-t0:.0f}s)")
        out[name] = rows
    return out


def summarize(r: dict, n_added: int) -> dict:
    return {
        "n_points": r["n_points"],
        "n_added": n_added,
        "naive_coverage": r["naive_coverage"]["score"],
        "c1_relative_density": r["c1_relative_density_coverage"]["score"],
        "c2_appearance": r["c2_appearance_consistency"]["score"],
        "c3_structure": r["c3_structural_consistency"]["score"],
        "c4_temporal_coherence": r["c4_temporal_coherence"]["score"],
        "gated_coverage": r["gated_coverage"]["score"],
        "vrc_r": r["vrc_r"],
        "vrc_r_product": r["vrc_r_product"],
    }


# ---------------------------------------------------------------------------
# E3 — synthetic validation with ground truth
# ---------------------------------------------------------------------------

def make_synthetic_ring(n_per_frame=40000, n_frames=10, seed=SEED):
    """
    Ground-truth-complete synthetic scene: a cylindrical surface band around
    the origin with azimuth-dependent colour (a hue wheel — deliberately
    asymmetric so cloned content is provably wrong) and a bright marker blob
    that orbits over time (provides temporal dynamics).
    """
    rng = np.random.default_rng(seed)
    # static surface geometry, sampled once (a temporally coherent scene must
    # not resample its geometry per frame — that would be flicker by design)
    theta = rng.uniform(-np.pi, np.pi, n_per_frame)
    r = rng.normal(1.0, 0.01, n_per_frame)
    y = rng.uniform(-0.4, 0.4, n_per_frame)
    base_xyz = np.stack([r * np.cos(theta), y, r * np.sin(theta)],
                        axis=1).astype(np.float32)
    # stationary procedural texture: per-sector colour histograms are nearly
    # identical (so appearance consistency holds for the true scene), but the
    # spatial pattern is azimuth-unique, so cloned content is provably wrong
    # under reference-based comparison
    # frequencies chosen high enough (>= 8 periods per 45-degree sector) that
    # every sector samples the full colour curve: per-sector histograms are
    # then nearly identical, satisfying C2's stationarity assumption while the
    # spatial phase remains azimuth-unique
    deg = (np.degrees(theta) + 360) % 360
    red = (128 + 127 * np.sin(64 * theta + 5 * y)).astype(np.uint8)
    grn = (128 + 127 * np.sin(96 * theta + 3 * y + 1.7)).astype(np.uint8)
    blu = (128 + 127 * np.sin(128 * theta + 9 * y + 0.9)).astype(np.uint8)
    base_rgb = np.stack([red, grn, blu], axis=1)

    xyz_l, rgb_l, frame_l = [], [], []
    for t in range(n_frames):
        jitter = rng.normal(0, 0.001, base_xyz.shape).astype(np.float32)
        rgb = base_rgb.copy()
        # orbiting bright marker provides temporal dynamics
        marker_deg = (360.0 * t / n_frames) % 360
        dist = np.abs(((deg - marker_deg + 180) % 360) - 180)
        rgb[dist < 15] = 255
        xyz_l.append(base_xyz + jitter)
        rgb_l.append(rgb)
        frame_l.append(np.full(n_per_frame, t, dtype=np.int32))
    return (np.concatenate(xyz_l), np.concatenate(rgb_l),
            np.concatenate(frame_l))


def ablate_to_sectors(xyz, rgb, frame, keep_angles):
    """Keep only points inside the listed 45-degree sectors (about origin)."""
    theta = np.degrees(np.arctan2(xyz[:, 2], xyz[:, 0]))
    mask = np.zeros(len(xyz), dtype=bool)
    for a in keep_angles:
        diff = np.abs(((theta - a + 180) % 360) - 180)
        mask |= diff <= 22.5
    return xyz[mask], rgb[mask], frame[mask]


def gt_quality_at_angles(gt, completed, angles, center=CAPTURE_CENTER):
    """Reference-based quality: render completed vs ground truth at the
    ablated angles (middle frame) and average the PSNR-mapped score."""
    gxyz, grgb, gframe = gt
    cxyz, crgb, cframe = completed
    t_mid = int(np.median(np.unique(gframe)))
    scores = []
    for a in angles:
        gm = gframe == t_mid
        cm = cframe == t_mid
        gt_img = render_turntable_frame(gxyz[gm], grgb[gm], a, center)
        pr_img = render_turntable_frame(cxyz[cm], crgb[cm], a, center)
        scores.append(reference_quality(pr_img, gt_img)["score"])
    return round(float(np.mean(scores)), 4) if scores else None


def e3_synthetic() -> dict:
    gt = make_synthetic_ring()
    gxyz, grgb, gframe = gt
    full = compute_vrc_r(gxyz, grgb, gframe, center=CAPTURE_CENTER)
    print(f"[E3] full synthetic ring: naive={full['naive_coverage']['score']} "
          f"VRC-R={full['vrc_r']}")

    curves = []
    for k in range(1, 9):
        keep = TEST_ANGLES_DEG[:k]
        ablated_angles = TEST_ANGLES_DEG[k:]
        axyz, argb, aframe = ablate_to_sectors(gxyz, grgb, gframe, keep)
        honest = compute_vrc_r(axyz, argb, aframe, center=CAPTURE_CENTER)
        row = {
            "true_coverage": k / 8.0,
            "kept_angles": keep,
            "honest": summarize(honest, 0),
            "honest_gt_quality": gt_quality_at_angles(
                gt, (axyz, argb, aframe), ablated_angles) if ablated_angles else None,
        }
        if k < 8:
            for attack in ["chaff_static", "clone"]:
                a = apply_attack(attack, axyz, argb, aframe,
                                 center=CAPTURE_CENTER, seed=SEED)
                r = compute_vrc_r(a["xyz"], a["rgb"], a["frame"],
                                  center=CAPTURE_CENTER)
                row[attack] = summarize(r, int(a["n_added"]))
                row[f"{attack}_gt_quality"] = gt_quality_at_angles(
                    gt, (a["xyz"], a["rgb"], a["frame"]), ablated_angles)
        curves.append(row)
        print(f"[E3] true={k}/8: honest naive={row['honest']['naive_coverage']} "
              f"VRC-R={row['honest']['vrc_r']}"
              + (f", chaff naive={row['chaff_static']['naive_coverage']} "
                 f"VRC-R={row['chaff_static']['vrc_r']}, "
                 f"clone VRC-R={row['clone']['vrc_r']} "
                 f"cloneQ={row['clone_gt_quality']}" if k < 8 else ""))
    return {"full_scene": summarize(full, 0), "ablation_curves": curves,
            "n_points_full": int(len(gxyz)),
            "n_frames": int(len(np.unique(gframe)))}


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes_dir", default="data/scenes")
    ap.add_argument("--out", default="results/session")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    print("=== E0: provenance audit ===")
    e0 = e0_provenance(args.scenes_dir)
    Path(args.out, "provenance.json").write_text(json.dumps(e0, indent=2))

    print("=== E1: diagnostic study ===")
    e1 = e1_diagnostic(args.scenes_dir)
    Path(args.out, "diagnostic.json").write_text(json.dumps(e1, indent=2))

    print("=== E2: gameability audit ===")
    e2 = e2_attacks(args.scenes_dir)
    Path(args.out, "attack_table.json").write_text(json.dumps(e2, indent=2))

    print("=== E3: synthetic validation ===")
    e3 = e3_synthetic()
    Path(args.out, "synthetic_validation.json").write_text(json.dumps(e3, indent=2))

    print(f"=== done in {(time.time()-t0)/60:.1f} min — artifacts in {args.out} ===")


if __name__ == "__main__":
    main()
