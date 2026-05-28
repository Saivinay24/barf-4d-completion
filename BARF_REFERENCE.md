# BARF 4D — Reference Document

## Scientific Thesis
BARF transforms a single monocular video into a temporally consistent, omnidirectionally complete 4D Gaussian Splat scene explorable in real-time VR from any viewpoint. We address the **VR-completeness problem** — no prior work does all four: monocular input + 4D temporal + generative completion + VR-ready output simultaneously.

## Key Differentiators vs Competitors
- **NeoVerse** (CVPR 2026 Highlight): our reconstruction backbone, not a competitor — bounded novel views, not VR-complete
- **Vivid4D** (ICCV 2025): most dangerous academic competitor — optimizes PSNR at trajectory-adjacent views, not omnidirectional; cite and beat specifically in far-angle (90°–270°) regime
- **World Labs Marble / NVIDIA Lyra 2.0**: static scenes only — no 4D dynamics
- **Our claim**: still unclaimed as of May 2026

## Three Novel Contributions
1. **VR-Completeness Problem** — new task definition + VRC-Score metric (angular coverage × temporal coherence × perceptual quality)
2. **Spherical 4D Completion Module** — diffusion conditioned on 4D scene latent (not 2D frames) → temporal consistency emerges naturally
3. **Real-time VR pipeline** — phone video → Quest 3 at 72 FPS

## Tech Stack (May 2026)
| Component | Tool |
|---|---|
| 4D Reconstruction | NeoVerse (https://github.com/IamCreateAI/NeoVerse) |
| Camera Poses | D4RT (DeepMind) |
| Completion Backbone | Vivid4D (conditioned on 4D scene latent via cross-attention) |
| Object Generation | TRELLIS 2 |
| VR Viewer | Meta Spatial SDK v0.9.2+ |
| Gap Detection | Voxel-based DBSCAN (existing code, wire to NeoVerse output) |
| Web Viewer | Palak's Three.js viewer (viewer/) — working |

## Pipeline Architecture
```
Video → D4RT (poses + tracks) → NeoVerse (4DGS) → Gap Detection
      → Spherical 4D Completion Module → 4DGS Fusion
      → LOD reduction → Quest .splat → VR Viewer (72 FPS)
```

## Completion Module (Core Innovation)
```
Input: 4DGS scene S + angular gap mask G
  → Temporal Feature Extraction (scene dynamics)
  → Spherical Gap Encoder (cluster by θ,φ region)
  → Completion Diffusion (Vivid4D backbone + 4D scene-conditioned cross-attention)
  → 4DGS Fusion (D4RT depth for geometry-consistent placement)
Output: Complete 4DGS scene S' covering full (θ,φ,t)
```
Use **Option A**: condition Vivid4D inpainting on NeoVerse scene features via cross-attention. Not full training from scratch.

## VRC-Score
```python
def vrc_score(scene_4dgs, test_angles=[0,45,90,135,180,225,270,315]):
    # Coverage(θ): % viewpoints with >threshold point density
    # Coherence(t): mean LPIPS between consecutive frames per viewpoint
    # Quality(θ,t): FID/LPIPS vs ground truth
    # VRC = Coverage × (1 - Coherence_loss) × Quality
```
Target: NeoVerse baseline ~0.41 → BARF ~0.82

## Current Repo State
- `main` — research artifacts, viewer, benchmark outputs
- `R4_Tanisha` — SV3D/Zero123++ novel view code (unconfirmed run)
- `pxlkele-patch-1` — Palak's viewer patch
- `vinay/sprint-completion` — integration work
- `viewer/` — working Three.js PLY viewer (Palak) ✅

## Immediate Action Priority
**Week 1:**
1. Merge all branches to main
2. Get NeoVerse running on cloud GPU (Vast.ai H100 ~$2/hr or Colab Pro A100), run on 3 test videos, produce PLY output
3. Run Vivid4D on same 3 videos (prior-work baseline)
4. Wire gap detection to NeoVerse output → occupancy heatmaps at 8 viewpoints → Figure 1 of paper

**Week 2:**
5. Write VRC-Score formal definition (one page)
6. Architecture whiteboard session — sketch completion module before any code
7. Read Vivid4D and See4D papers in full

## Paper Targets
1. **SIGGRAPH Asia 2026** — if Phase 0+1 done by August (submission ~June/July 2026)
2. **CVPR 2027** — primary (submission ~Nov 2026)
3. **ICCV 2027** — backup

## Paper Narrative Hook
> "We walk around people every day. But when you film someone and put them in VR, you can't walk behind them — the back of their head simply doesn't exist. BARF addresses the VR-completeness problem — generating the full omnidirectional 4D scene from a single monocular video — enabling true free-viewpoint navigation of dynamic real-world captures in VR for the first time."

## Key Sources
- NeoVerse: https://arxiv.org/abs/2601.00393 | https://github.com/IamCreateAI/NeoVerse
- D4RT: https://arxiv.org/pdf/2512.08924
- Vivid4D: https://arxiv.org/abs/2504.11092
- See4D: https://arxiv.org/pdf/2510.26796
- Fillerbuster: https://arxiv.org/abs/2502.05175
- NVIDIA Lyra 2.0: https://arxiv.org/html/2604.13036v1
