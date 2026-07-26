# BARF: VR-Complete 4D Scene Generation

> **Transform any phone video into a fully explorable 4D VR world, including the parts the camera never saw.**

[![Tests](https://img.shields.io/badge/tests-135%20passed-brightgreen)]()
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## The Problem

Current 4D reconstruction methods (D4RT, NeoVerse, Vivid4D) turn video into 3D scenes, but only what the camera saw. Film someone from the front, and the back of their head is empty. You can't walk behind them in VR.

**BARF** is designed to fill those gaps: generate photorealistic, temporally consistent content for every angle the camera missed, producing a complete 4D scene explorable from any viewpoint in real-time VR.

| | Reconstruction-Only (NeoVerse) | **BARF (Ours)** |
|---|---|---|
| Angular coverage | ~45% (front only), measured | Full-sphere completion, by design; end-to-end coverage number pending GPU inference run |
| Walk behind subject? | Empty | AI-generated, architecture built, not yet executed on GPU |
| VR-ready? | Not designed for it | Quest 3 export pipeline built; live FPS benchmark pending device test |
| Temporal consistency? | N/A (no back-view) | 4D scene-conditioned by design; validated so far via synthetic + CPU experiments, not GPU inference |

See [Status](#status--whats-real-vs-pending) below for exactly what has run and what hasn't.

## Key Innovation

Unlike prior work (Vivid4D, See4D) that conditions generative completion on 2D frames, BARF conditions on the 4D scene latent, the full spatiotemporal representation of the scene. The design goal: temporal consistency emerges naturally, since the generated back of a walking person stays consistent across frames because the model sees the full motion dynamics. This conditioning has not yet been run end-to-end on GPU (see Status); the metric and audit work that evaluates it has.

## Status: what's real vs. pending

**Built, run, and verified (CPU, deterministic, reproducible):**
- Gap detection: angular coverage analysis on real reconstructed scenes (`src/gap_detection/`)
- VRC-Score and VRC-R: a formal VR-completeness metric plus a gameability-audited, robustness-hardened version, with 135 passing tests (`src/metrics/`)
- Gameability audit: adversarial attacks (dust, chaff, clone) against naive coverage metrics, with a full paper (`paper/barf_paper.pdf`) whose every number is generated from committed result artifacts by `scripts/build_paper.py`, zero hand-typed
- Quest-compatible `.splat` export pipeline (`src/vr/export_splat.py`)
- Web viewer for inspecting scenes and coverage gaps (`viewer/`)

**Designed and implemented, execution pending GPU access:**
- The Spherical 4D Completion Module itself: the diffusion-based generation step (`src/completion/spherical_completion.py`) is implemented against the Vivid4D backbone but has not yet run a real inference pass, since that requires sustained A100/H100-class GPU time the team didn't have during this phase. It currently outputs a placeholder pass-through so the rest of the pipeline (fusion, export, VR viewing) can be tested end-to-end.
- The 72 FPS Quest 3 target and ~91% coverage target are design goals derived from the architecture and prior benchmarks, not yet measured on a completed scene.

This split is intentional: everything that could be built and proven without a GPU cluster (the metric, the audit, the pipeline plumbing, the export/viewer) is done and tested. The generative completion inference itself is the next phase, gated on GPU time.

## Architecture

```
Phone Video
  ↓
[D4RT] Camera poses + 4D point tracking
  ↓
[NeoVerse] 4D Gaussian Splat reconstruction (partial, front-facing only)
  ↓
[Gap Detection] Angular coverage analysis → identifies empty viewing angles
  ↓
[Spherical 4D Completion Module] ← OUR CONTRIBUTION (pending GPU inference, see Status)
  • Temporal Feature Extraction: 4D scene latent from NeoVerse output
  • Spherical Gap Encoder: gap position queries
  • Completion Diffusion: Vivid4D backbone + scene-conditioned cross-attention
  • Gaussian Fusion: back-project generated RGBA → new 4D Gaussians
  ↓
Complete 4DGS scene covering full (θ, φ, t) viewing sphere
  ↓
[VR Export] LOD reduction → Quest-compatible .splat file → 72 FPS on Quest 3
```

## Novel Contributions

1. **VR-Completeness Problem**: first formal definition: a 4D scene is "VR-complete" if it renders photorealistic, consistent frames from every viewpoint (θ, φ, t). Defined and implemented.

2. **VRC-Score / VRC-R**: first benchmark metric measuring angular coverage, temporal coherence, and perceptual quality simultaneously for VR navigation, plus a gameability-audited robust version (VRC-R) that resists degenerate attacks on naive coverage metrics. Defined, implemented, tested, and stress-tested in a dedicated paper.

3. **4D Scene-Conditioned Completion**: cross-attention conditioning on the full 4D scene latent (not 2D frames), designed to enable temporally coherent generation at all angles. Architecture implemented; GPU inference run pending (see Status).

## Repository Structure

```
barf-4d-completion/
├── src/                             # Active codebase
│   ├── gap_detection/
│   │   └── detect_gaps.py          # Angular coverage analysis
│   ├── completion/
│   │   └── spherical_completion.py # Spherical 4D Completion Module (GPU inference pending)
│   ├── metrics/
│   │   ├── vrc_score.py            # VRC-Score metric
│   │   └── robust_vrc.py           # VRC-R: gameability-audited robust metric
│   ├── attacks/
│   │   └── degenerate_completions.py # Adversarial attacks used to stress-test VRC-R
│   └── vr/
│       └── export_splat.py         # Quest-compatible .splat exporter
│
├── scripts/
│   ├── run_all_experiments.py      # E0-E3 experiment runner behind the paper's numbers
│   ├── build_paper.py              # Substitutes every paper number from committed artifacts
│   ├── make_figures.py             # Generates all paper figures from result artifacts
│   ├── prepare_data.py             # Converts PLY scenes to npz format
│   └── reproduce.sh                # End-to-end reproducibility script
│
├── tests/                          # 135 tests, all passing
│
├── paper/
│   ├── barf_paper.pdf              # Gaming the Sphere: gameability audit paper
│   └── REPRODUCIBILITY.md          # Auto-generated number → artifact → command map
│
├── viewer/                         # Web-based 3D viewer (PLY loader, gap viz)
├── data/                           # Test scenes + generated heatmaps
├── results/session/                # Raw artifacts behind every paper number
├── BARF_VRC_SCORE.md               # VRC-Score formal mathematical definition
│
└── feb_sprint/                     # Archived Feb 2026 sprint (original recon + benchmarking)
```

## Quick Start

### Install Dependencies

```bash
git clone https://github.com/Saivinay24/barf-4d-completion
cd barf-4d-completion
pip install -r requirements.txt
```

### Run Tests

```bash
python3 -m pytest tests/ -q
# 135 passed
```

### Run Gap Detection on a PLY File

```bash
python3 -m src.gap_detection.detect_gaps \
    --input path/to/scene.ply \
    --output_json gaps.json \
    --output_heatmap_dir heatmaps/
```

### Reproduce All Paper Results

```bash
bash scripts/reproduce.sh
```

### Export to Quest-Compatible .splat

```bash
python3 -m src.vr.export_splat \
    --input scene_complete.ply \
    --output scene.splat \
    --max_gaussians 500000
```

## Related Work

BARF sits at the intersection of 4D reconstruction and generative scene completion:

| Method | Monocular | 4D Temporal | Gen. Completion |
|---|:---:|:---:|:---:|
| Google D4RT | ✅ | ✅ | ❌ |
| NeoVerse (CVPR 2026) | ✅ | ✅ | Partial |
| Vivid4D (ICCV 2025) | ✅ | ✅ | ✅ (recon-focused) |
| Full-4D (2026) | ✅ | ✅ | ✅ |
| **BARF (Ours)** | ✅ | ✅ | Evaluation & metrics |

## Tech Stack

| Component | Tool | Role |
|---|---|---|
| Camera Poses | [D4RT](https://github.com/google-deepmind/d4rt) | 200+ FPS pose estimation |
| 4D Reconstruction | [NeoVerse](https://github.com/IamCreateAI/NeoVerse) | Feed-forward 4DGS (CVPR 2026) |
| Baseline | [Vivid4D](https://arxiv.org/abs/2504.11092) | Prior work comparison (ICCV 2025) |
| Completion Backbone | Vivid4D UNet + scene cross-attention | Our novel conditioning |
| VR Runtime | Meta Spatial SDK v0.9.2+ | Quest 3 native splat rendering |
| Optical Flow | RAFT | Temporal consistency supervision |

## Citation

```bibtex
@article{bhoomireddy2026gaming,
  title={Gaming the Sphere: A Gameability Audit of VR-Completeness Metrics for Generative 4D Scene Completion},
  author={Bhoomireddy, Sai Vinay and Aditya and Srivastava, Aryan and Shrivastava, Shrit and Tanisha and Patnaik, Palak},
  year={2026}
}
```

## License

MIT License, see [LICENSE](LICENSE) for details.

---

**Team:** [Sai Vinay Bhoomireddy](https://github.com/Saivinay24), Aditya, Aryan Srivastava, Shrit Shrivastava, Tanisha, Palak Patnaik
