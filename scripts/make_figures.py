"""
BARF 4D — Paper figure generation.

Reads the raw result artifacts written by scripts/run_all_experiments.py
(plus the .npz scenes for the visualisation panels) and writes the paper
figures to paper/figures/. Attacks shown in figures are regenerated with
the same seed used in the experiments (they are deterministic).

Usage:
    python3 scripts/make_figures.py [--results results/session]
                                    [--scenes_dir data/scenes]
                                    [--out paper/figures]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attacks.degenerate_completions import apply_attack, TEST_ANGLES_DEG
from src.metrics.robust_vrc import render_turntable_frame

SCENES = ["01_mdn_flower", "02_w3schools_big_buck_bunny", "03_samplelib_5s"]
SCENE_LABELS = {"01_mdn_flower": "S1 flower",
                "02_w3schools_big_buck_bunny": "S2 bunny",
                "03_samplelib_5s": "S3 street"}
ATTACKS = ["dust", "chaff_static", "chaff_flicker", "clone"]
ATTACK_LABELS = {"original": "original", "dust": "dust",
                 "chaff_static": "chaff", "chaff_flicker": "flicker chaff",
                 "clone": "clone"}
CENTER = np.zeros(3)
SEED = 0

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.titlesize": 9, "axes.labelsize": 9,
                     "figure.dpi": 200})


def fig1_angular_profiles(results_dir, out_dir):
    """Polar per-sector point counts, capture-centred, for the 3 real scenes."""
    diag = json.loads(Path(results_dir, "diagnostic.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6),
                             subplot_kw={"projection": "polar"})
    for ax, name in zip(axes, SCENES):
        counts = diag[name]["capture_centered"]["sector_counts"]
        angles = np.radians(TEST_ANGLES_DEG)
        vals = np.array([counts[str(a)] for a in TEST_ANGLES_DEG], dtype=float)
        log_vals = np.log10(vals + 1)
        bars = ax.bar(angles, log_vals, width=np.radians(40), alpha=0.85,
                      color=["#2c5f2e" if v > 0 else "#cccccc" for v in vals])
        cov = diag[name]["capture_centered"]["naive_coverage"]["score"]
        ax.set_title(f"{SCENE_LABELS[name]}\nnaive coverage {cov*100:.1f}%",
                     pad=12)
        ax.set_yticks([2, 4, 6])
        ax.set_yticklabels(["$10^2$", "$10^4$", "$10^6$"], fontsize=6)
        ax.set_xticks(angles)
        ax.set_xticklabels([f"{a}°" for a in TEST_ANGLES_DEG], fontsize=6)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(out_dir, "fig1_angular_profiles.png"), bbox_inches="tight")
    plt.close(fig)
    print("[figures] fig1_angular_profiles.png")


def fig2_attack_panels(scenes_dir, out_dir, scene="02_w3schools_big_buck_bunny"):
    """Top-down (XZ) views: original scene and the three attack families."""
    d = np.load(Path(scenes_dir) / f"{scene}.npz")
    xyz, rgb, frame = d["xyz"], d["rgb"], d["frame"]
    rng = np.random.default_rng(1)

    variants = [("original", xyz, rgb)]
    for attack in ["dust", "chaff_static", "clone"]:
        a = apply_attack(attack, xyz, rgb, frame, center=CENTER, seed=SEED)
        variants.append((ATTACK_LABELS[attack], a["xyz"], a["rgb"]))

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.1))
    for ax, (label, vxyz, vrgb) in zip(axes, variants):
        n = len(vxyz)
        idx = rng.choice(n, min(60000, n), replace=False)
        ax.scatter(vxyz[idx, 0], vxyz[idx, 2], s=0.2,
                   c=vrgb[idx].astype(float) / 255.0, linewidths=0)
        ax.plot(0, 0, marker="^", color="red", markersize=5)
        ax.set_title(label)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4)
    fig.tight_layout()
    fig.savefig(Path(out_dir, "fig2_attack_panels.png"), bbox_inches="tight")
    plt.close(fig)
    print("[figures] fig2_attack_panels.png")


def fig3_synthetic_curves(results_dir, out_dir):
    """Metric response vs true coverage under ablation and attack."""
    syn = json.loads(Path(results_dir, "synthetic_validation.json").read_text())
    curves = syn["ablation_curves"]
    x = [c["true_coverage"] for c in curves]

    def series(variant, key):
        return [c[variant][key] if variant in c else np.nan for c in curves]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))

    ax1.plot(x, x, "k:", lw=0.8, label="ideal (y = true coverage)")
    ax1.plot(x, series("honest", "naive_coverage"), "o-", color="#888",
             ms=3, lw=1, label="naive, honest")
    ax1.plot(x[:-1], series("chaff_static", "naive_coverage")[:-1], "s-",
             color="#c0392b", ms=3, lw=1, label="naive, chaff attack")
    ax1.set_xlabel("true angular coverage")
    ax1.set_ylabel("coverage metric")
    ax1.set_title("(a) naive coverage is gamed flat to 1.0")
    ax1.legend(fontsize=6, loc="lower right")
    ax1.grid(alpha=0.3)

    ax2.plot(x, x, "k:", lw=0.8, label="ideal")
    ax2.plot(x, series("honest", "vrc_r"), "o-", color="#2c5f2e",
             ms=4, lw=2.2, label="VRC-R, honest")
    ax2.plot(x[:-1], series("chaff_static", "vrc_r")[:-1], "s--",
             color="#c0392b", ms=3, lw=1,
             label="VRC-R, chaff attack (coincides)")
    ax2.plot(x[:-1], series("clone", "vrc_r")[:-1], "^-", color="#e67e22",
             ms=3, lw=1, label="VRC-R, clone (blind spot)")
    clone_q = [c.get("clone_gt_quality", np.nan) for c in curves]
    ax2.plot(x[:-1], clone_q[:-1], "v--", color="#8e44ad",
             ms=3, lw=1, label="GT quality of clone")
    ax2.set_xlabel("true angular coverage")
    ax2.set_ylabel("score")
    ax2.set_title("(b) VRC-R resists chaff; reference catches clone")
    ax2.legend(fontsize=6, loc="upper left")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(Path(out_dir, "fig3_synthetic_curves.png"), bbox_inches="tight")
    plt.close(fig)
    print("[figures] fig3_synthetic_curves.png")


def fig4_metric_bars(results_dir, out_dir):
    """Per-variant sub-metric and composite scores, averaged over scenes."""
    table = json.loads(Path(results_dir, "attack_table.json").read_text())
    variants = ["original"] + ATTACKS
    metrics = [("naive_coverage", "naive coverage"),
               ("c1_relative_density", "C1 density"),
               ("c2_appearance", "C2 appearance"),
               ("c3_structure", "C3 structure"),
               ("c4_temporal_coherence", "C4 coherence"),
               ("gated_coverage", "gated coverage"),
               ("vrc_r", "VRC-R")]
    means = {v: [float(np.mean([table[s][v][m] for s in SCENES]))
                 for m, _ in metrics] for v in variants}

    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    width = 0.15
    xpos = np.arange(len(metrics))
    colors = {"original": "#2c5f2e", "dust": "#888888",
              "chaff_static": "#c0392b", "chaff_flicker": "#e74c3c",
              "clone": "#e67e22"}
    for i, v in enumerate(variants):
        ax.bar(xpos + (i - 2) * width, means[v], width * 0.92,
               label=ATTACK_LABELS[v], color=colors[v])
    ax.set_xticks(xpos)
    ax.set_xticklabels([lbl for _, lbl in metrics], fontsize=7)
    ax.set_ylabel("score (mean over 3 scenes)")
    ax.legend(fontsize=6, ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, 1.18))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(out_dir, "fig4_metric_bars.png"), bbox_inches="tight")
    plt.close(fig)
    print("[figures] fig4_metric_bars.png")


def fig5_turntable_montage(scenes_dir, out_dir, scene="02_w3schools_big_buck_bunny"):
    """Turntable renders (middle frame) at 4 azimuths: original vs clone."""
    d = np.load(Path(scenes_dir) / f"{scene}.npz")
    xyz, rgb, frame = d["xyz"], d["rgb"], d["frame"]
    clone = apply_attack("clone", xyz, rgb, frame, center=CENTER, seed=SEED)
    t_mid = int(np.median(np.unique(frame)))
    angles = [90, 180, 270, 0]

    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.6))
    for j, a in enumerate(angles):
        m = frame == t_mid
        img = render_turntable_frame(xyz[m], rgb[m], a, CENTER, img_size=192)
        axes[0, j].imshow(img)
        axes[0, j].set_title(f"{a}°")
        cm = clone["frame"] == t_mid
        img_c = render_turntable_frame(clone["xyz"][cm], clone["rgb"][cm], a,
                                       CENTER, img_size=192)
        axes[1, j].imshow(img_c)
        for i in (0, 1):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    axes[0, 0].set_ylabel("original", fontsize=8)
    axes[1, 0].set_ylabel("clone attack", fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(out_dir, "fig5_turntable_montage.png"), bbox_inches="tight")
    plt.close(fig)
    print("[figures] fig5_turntable_montage.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/session")
    ap.add_argument("--scenes_dir", default="data/scenes")
    ap.add_argument("--out", default="paper/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    fig1_angular_profiles(args.results, args.out)
    fig2_attack_panels(args.scenes_dir, args.out)
    fig3_synthetic_curves(args.results, args.out)
    fig4_metric_bars(args.results, args.out)
    fig5_turntable_montage(args.scenes_dir, args.out)
    print("[figures] all figures written to", args.out)


if __name__ == "__main__":
    main()
