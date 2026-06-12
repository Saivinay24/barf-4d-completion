"""
BARF 4D — Paper builder.

Substitutes every {{placeholder}} in paper/barf_paper_template.html with a
value computed from the committed result artifacts in results/session/, and
writes:
    paper/barf_paper.html        — the final paper
    paper/REPRODUCIBILITY.md     — auto-generated number -> artifact map

No number in the final paper is hand-typed: if a placeholder cannot be
resolved from an artifact, the build fails.

Usage:
    python3 scripts/build_paper.py --runtime_min 6 --n_tests 135
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "session"
SCENES = ["01_mdn_flower", "02_w3schools_big_buck_bunny", "03_samplelib_5s"]
SCENE_LABELS = {"01_mdn_flower": "S1",
                "02_w3schools_big_buck_bunny": "S2",
                "03_samplelib_5s": "S3"}
VARIANTS = ["original", "dust", "chaff_static", "chaff_flicker", "clone"]
VARIANT_LABELS = {"original": "original", "dust": "+dust",
                  "chaff_static": "+chaff", "chaff_flicker": "+flicker",
                  "clone": "+clone"}


def f2(x):
    return f"{x:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime_min", required=True,
                    help="end-to-end runtime of run_all_experiments.py in "
                         "minutes, from its own printed timer")
    ap.add_argument("--n_tests", required=True, type=int,
                    help="pytest test count (pytest --collect-only -q)")
    args = ap.parse_args()

    prov = json.loads((RES / "provenance.json").read_text())
    diag = json.loads((RES / "diagnostic.json").read_text())
    atk = json.loads((RES / "attack_table.json").read_text())
    syn = json.loads((RES / "synthetic_validation.json").read_text())

    fills = {}   # placeholder -> (value_str, artifact, how)

    def put(key, value, artifact, how):
        fills[key] = (str(value), artifact, how)

    # ---- provenance / diagnostic ----
    n_total = sum(prov[s]["n_points"] for s in SCENES)
    put("n_points_total", f"{n_total:,}", "provenance.json", "sum of n_points")
    for i, s in enumerate(SCENES, 1):
        put(f"s{i}_n", f"{prov[s]['n_points']:,}", "provenance.json",
            f"{s}.n_points")
        put(f"s{i}_delta", prov[s]["missing_vs_w_h_T"], "provenance.json",
            f"{s}.missing_vs_w_h_T")
        naive = diag[s]["capture_centered"]["naive_coverage"]["score"]
        put(f"s{i}_naive_pct", f"{naive*100:.1f}", "diagnostic.json",
            f"{s}.capture_centered.naive_coverage.score x100")
    naives = [diag[s]["capture_centered"]["naive_coverage"]["score"]
              for s in SCENES]
    put("naive_min_pct", f"{min(naives)*100:.1f}", "diagnostic.json",
        "min capture-centred naive coverage x100")
    put("naive_max_pct", f"{max(naives)*100:.1f}", "diagnostic.json",
        "max capture-centred naive coverage x100")
    cents = [diag[s]["centroid_centered"]["c1_relative_density"]["score"]
             for s in SCENES]
    assert all(c == 1.0 for c in cents), "centroid C1 expected saturated"
    put("centroid_c1_all", "1.00", "diagnostic.json",
        "centroid_centered.c1_relative_density.score, all scenes")

    # ---- attack table ----
    dust_added = [atk[s]["dust"]["n_added"] for s in SCENES]
    put("dust_min_added", min(dust_added), "attack_table.json", "dust.n_added min")
    put("dust_max_added", max(dust_added), "attack_table.json", "dust.n_added max")

    def vmean(variant, key):
        return float(np.mean([atk[s][variant][key] for s in SCENES]))

    put("orig_vrcr_mean", f2(vmean("original", "vrc_r")), "attack_table.json",
        "mean vrc_r, original")
    put("chaff_vrcr_mean", f2(vmean("chaff_static", "vrc_r")),
        "attack_table.json", "mean vrc_r, chaff_static")
    put("clone_vrcr_mean", f2(vmean("clone", "vrc_r")), "attack_table.json",
        "mean vrc_r, clone")
    put("clone_gated_mean", f2(vmean("clone", "gated_coverage")),
        "attack_table.json", "mean gated_coverage, clone")
    put("chaff_c2_mean", f2(vmean("chaff_static", "c2_appearance")),
        "attack_table.json", "mean c2, chaff_static")
    put("chaff_c3_mean", f2(vmean("chaff_static", "c3_structure")),
        "attack_table.json", "mean c3, chaff_static")
    put("chaff_c4_mean", f2(vmean("chaff_static", "c4_temporal_coherence")),
        "attack_table.json", "mean c4, chaff_static")
    put("flicker_c4_mean", f2(vmean("chaff_flicker", "c4_temporal_coherence")),
        "attack_table.json", "mean c4, chaff_flicker")

    rows = []
    for s in SCENES:
        for v in VARIANTS:
            r = atk[s][v]
            label = (f"{SCENE_LABELS[s]} {VARIANT_LABELS[v]}"
                     if v != "original" else f"<strong>{SCENE_LABELS[s]}</strong>")
            added = f"{r['n_added']:,}" if r["n_added"] else "&mdash;"
            rows.append(
                f"<tr><td>{label}</td><td>{added}</td>"
                f"<td>{f2(r['naive_coverage'])}</td>"
                f"<td>{f2(r['c1_relative_density'])}</td>"
                f"<td>{f2(r['c2_appearance'])}</td>"
                f"<td>{f2(r['c3_structure'])}</td>"
                f"<td>{f2(r['c4_temporal_coherence'])}</td>"
                f"<td>{f2(r['gated_coverage'])}</td>"
                f"<td><strong>{f2(r['vrc_r'])}</strong></td></tr>")
    put("attack_table_rows", "\n".join(rows), "attack_table.json",
        "all scenes x variants, fields as columns")

    # ---- synthetic validation ----
    curves = syn["ablation_curves"]
    put("syn_n_points", f"{syn['n_points_full']:,}",
        "synthetic_validation.json", "n_points_full")
    put("syn_n_frames", syn["n_frames"], "synthetic_validation.json", "n_frames")
    honest = [c["honest"]["vrc_r"] for c in curves]
    truth = [c["true_coverage"] for c in curves]
    put("syn_vrcr_k1", f2(honest[0]), "synthetic_validation.json",
        "ablation_curves[k=1].honest.vrc_r")
    put("syn_vrcr_k8", f2(honest[-1]), "synthetic_validation.json",
        "ablation_curves[k=8].honest.vrc_r")
    from scipy.stats import spearmanr
    rho = spearmanr(truth, honest).statistic
    put("syn_spearman", f2(rho), "synthetic_validation.json",
        "spearmanr(true_coverage, honest.vrc_r)")
    chaff_gaps = [abs(c["chaff_static"]["vrc_r"] - c["honest"]["vrc_r"])
                  for c in curves if "chaff_static" in c]
    put("syn_chaff_gap_max", f"{max(chaff_gaps):.3f}",
        "synthetic_validation.json",
        "max |chaff vrc_r - honest vrc_r| over ablation levels")
    chaff_naive = [c["chaff_static"]["naive_coverage"]
                   for c in curves if "chaff_static" in c]
    assert all(v == 1.0 for v in chaff_naive), "chaff naive expected 1.0"
    clone_q = [c["clone_gt_quality"] for c in curves if "clone_gt_quality" in c]
    put("clone_gtq_mean", f2(float(np.mean(clone_q))),
        "synthetic_validation.json", "mean clone_gt_quality over k=1..7")
    put("clone_gtq_syn_range", f"{min(clone_q):.2f}&ndash;{max(clone_q):.2f}",
        "synthetic_validation.json", "min..max clone_gt_quality")
    put("honest_q_at_full", "1.00 by construction",
        "synthetic_validation.json",
        "reference_quality(GT render, GT render) = 1 by definition")

    # ---- session facts ----
    put("total_runtime_min", args.runtime_min, "run_all_experiments.py output",
        "printed end-to-end timer")
    put("n_tests", args.n_tests, "pytest", "pytest --collect-only -q count")

    # ---- substitute ----
    template = (ROOT / "paper" / "barf_paper_template.html").read_text()
    unresolved = []

    def sub(m):
        key = m.group(1)
        if key not in fills:
            unresolved.append(key)
            return m.group(0)
        return fills[key][0]

    html = re.sub(r"\{\{(\w+)\}\}", sub, template)
    if unresolved:
        sys.exit(f"ERROR: unresolved placeholders: {sorted(set(unresolved))}")
    leftovers = re.findall(r"\{\{\w+\}\}", html)
    assert not leftovers, leftovers
    (ROOT / "paper" / "barf_paper.html").write_text(html)
    print(f"[build_paper] wrote paper/barf_paper.html "
          f"({len(fills)} placeholders resolved)")

    # ---- reproducibility map ----
    lines = [
        "# Reproducibility Map",
        "",
        "Every number in `paper/barf_paper.html` is substituted by",
        "`scripts/build_paper.py` from committed artifacts — no number is",
        "hand-typed. This file is auto-generated by the same script.",
        "",
        "## Pipeline",
        "",
        "```bash",
        "bash scripts/reproduce.sh   # end-to-end: data prep, tests, experiments, figures",
        "python3 scripts/build_paper.py --runtime_min <printed by experiments> \\",
        "    --n_tests $(python3 -m pytest --collect-only -q tests/ | tail -1 | grep -o '[0-9]*' | head -1)",
        "```",
        "",
        "Steps performed by `scripts/reproduce.sh`:",
        "1. `python3 scripts/prepare_data.py` — converts the Phase-0 Colab PLYs",
        "   (per-pixel depth unprojections; provenance audited in the paper's",
        "   Table 1) to `data/scenes/*.npz`. The npz files are committed, so",
        "   this step is a no-op unless rebuilding from the original PLYs.",
        "2. `python3 -m pytest tests/ -q` — unit tests for attacks and metrics.",
        "3. `python3 scripts/run_all_experiments.py` — E0 provenance audit,",
        "   E1 diagnostic, E2 gameability audit, E3 synthetic validation.",
        "   Writes `results/session/{provenance,diagnostic,attack_table,",
        "   synthetic_validation}.json`. Deterministic, seed 0, CPU-only.",
        "4. `python3 scripts/make_figures.py` — renders `paper/figures/*.png`",
        "   from the result artifacts (attacks regenerated with the same seed).",
        "",
        "PDF: rendered from the HTML with headless Chrome:",
        "```bash",
        '"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \\',
        "    --headless --print-to-pdf=paper/barf_paper.pdf \\",
        "    --no-pdf-header-footer paper/barf_paper.html",
        "```",
        "",
        "## Figures",
        "",
        "| Figure | Script | Input artifact |",
        "|---|---|---|",
        "| Fig 1 angular profiles | `scripts/make_figures.py:fig1_angular_profiles` | `results/session/diagnostic.json` |",
        "| Fig 2 attack panels | `scripts/make_figures.py:fig2_attack_panels` | `data/scenes/02_*.npz` + seed-0 attacks |",
        "| Fig 3 metric bars | `scripts/make_figures.py:fig4_metric_bars` | `results/session/attack_table.json` |",
        "| Fig 4 synthetic curves | `scripts/make_figures.py:fig3_synthetic_curves` | `results/session/synthetic_validation.json` |",
        "| Fig 5 turntable montage | `scripts/make_figures.py:fig5_turntable_montage` | `data/scenes/02_*.npz` + seed-0 clone |",
        "",
        "## Table 2 (gameability audit)",
        "",
        "All cells come from `results/session/attack_table.json`",
        "(`scenes x {original,dust,chaff_static,chaff_flicker,clone}`), fields",
        "`naive_coverage, c1_relative_density, c2_appearance, c3_structure,",
        "c4_temporal_coherence, gated_coverage, vrc_r, n_added`.",
        "",
        "## Every inline number",
        "",
        "| Placeholder | Value | Artifact | Derivation |",
        "|---|---|---|---|",
    ]
    for key in sorted(fills):
        if key == "attack_table_rows":
            continue
        v, artifact, how = fills[key]
        lines.append(f"| `{key}` | {v} | `results/session/{artifact}`"
                     if artifact.endswith(".json")
                     else f"| `{key}` | {v} | {artifact}")
        lines[-1] += f" | {how} |"
    lines += [
        "",
        "## Data provenance",
        "",
        "The three real scenes are per-pixel monocular depth unprojections",
        "(392x252x25 points each) produced by the project's Colab pipeline in",
        "June 2026 and audited in this session (point count = W*H*T minus",
        "invalid pixels). They are NOT optimised 4DGS reconstructions, and the",
        "paper describes them accordingly. Original PLYs:",
        "`/Users/saivinay/Downloads/colab_eval_uncapped/<scene>/neoverse/scene.ply`;",
        "committed npz copies: `data/scenes/*.npz` (with embedded provenance).",
        "",
        "## What was NOT run",
        "",
        "No GPU reconstruction (NeoVerse/D4RT), no diffusion completion",
        "(Vivid4D/See4D/Fillerbuster), no VR hardware, no user study. The paper",
        "claims none of these.",
    ]
    (ROOT / "paper" / "REPRODUCIBILITY.md").write_text("\n".join(lines) + "\n")
    print("[build_paper] wrote paper/REPRODUCIBILITY.md")


if __name__ == "__main__":
    main()
