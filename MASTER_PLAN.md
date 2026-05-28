# BARF 4D — Full Situation Report & Master Plan
### *Scientific analysis compiled May 28, 2026*

---

## PART 1 — WHERE THE PROJECT STANDS

**The February sprint was a recon sprint, not a completion sprint.** Here's the honest audit:

| Component | Status | Reality |
|---|---|---|
| Web Viewer (Palak) | ✅ Working | `viewer/` — PLY loader, gap viz, split-view, timeline, browser-ready |
| SOTA Benchmarking (Shrit) | ⚠️ Partial | Demo artifacts saved from official project pages. Local runs **blocked** by CUDA dependency hell |
| 4D Reconstruction (Aryan) | ⚠️ Partial | Automation notebook exists, local run not confirmed complete |
| Novel View Gen (Tanisha) | ⚠️ Partial | Branch `R4_Tanisha` exists with SV3D/Zero123++ code, not confirmed executed |
| Gap Detection | ⚠️ Designed | `gaps.json` in viewer, detection script designed but not confirmed run on real output |
| End-to-End Pipeline | ❌ Never built | `video → 4D → gaps → generate → VR` was never assembled |
| VR Module (Alankryt) | ❌ Not started | Hardware hadn't arrived |

### Branch Status
- `main` — research artifacts, viewer, task files, benchmark outputs
- `R4_Tanisha` — Tanisha's novel view generation work
- `pxlkele-patch-1` — Palak's viewer patch
- `vinay/sprint-completion` — integration work

**Bottom line:** You have a solid research scaffold and a working web viewer. The pipeline itself is the missing piece. The scientific motivation is fully documented. That's actually a good position — you've done the hard thinking already.

---

## PART 2 — IS THE IDEA STILL RELEVANT?

### Verdict: YES — More Relevant Than Ever. But the Framing Needs an Upgrade.

The core problem you identified is still completely unsolved:

> *"Take any monocular video. Reconstruct it as a 4D scene. Generatively complete all unseen angles. Make it fully VR-explorable."*

Every SOTA method that has shipped since February 2026 still fails at least one of the four pillars:

| Method | Monocular Input | 4D Temporal | Generative Completion | VR-ready |
|---|:---:|:---:|:---:|:---:|
| Google D4RT (Dec 2025) | ✅ | ✅ | ❌ reconstruction only | ❌ |
| NeoVerse (CVPR 2026 Highlight) | ✅ | ✅ | Partial — trajectory novel views only | ❌ |
| Vivid4D (ICCV 2025) | ✅ | ✅ | ✅ but goal = recon quality, not 360° VR | ❌ |
| See4D (Oct 2025) | ✅ | ✅ | ✅ autoregressive inpainting | ❌ |
| Fillerbuster (Feb 2025) | ✅ multi-view | ❌ static 3D only | ✅ | ❌ |
| NVIDIA Lyra 2.0 (Apr 2026) | ✅ | ❌ static world | ✅ forward explore only | ❌ VR-native |
| World Labs Marble (Nov 2025) | ✅ | ❌ static scenes | ✅ | ❌ real VR |
| **BARF (our goal)** | ✅ | ✅ | ✅ | ✅ |

**Nobody is doing all four simultaneously. That is our paper.**

---

## PART 3 — THE COMPETITIVE LANDSCAPE

### 3A. Research Competitors

#### Google DeepMind D4RT *(Dec 2025)*
- **What it does:** Unified 4D reconstruction + tracking from monocular video. 300× faster than previous SOTA. Encoder-decoder Transformer. 200+ FPS camera pose estimation. 18–300× speedup on dynamic objects.
- **What it doesn't do:** Zero generative completion. Reconstructs only visible surfaces. Built for robotics/AR spatial awareness, not immersive VR.
- **Our relationship with it:** D4RT is an ingredient, not a competitor. Use it as the geometry oracle — poses + tracking — feeding into our pipeline.
- **Threat level:** 🟡 Medium

#### NeoVerse — CVPR 2026 Highlight + Best Paper, VideoWorldModel Workshop *(CASIA/CreateAI, Jan 2026)*
- **What it does:** Feed-forward 4DGS from monocular video. Scalable to 1M wild video clips. Versatile: 4D reconstruction, multiview video generation, video editing, stabilization, super-resolution.
- **What it doesn't do:** The "multiview generation" is bounded trajectory re-rendering — stays near the original camera path. There is still no geometry on the true back-side of objects. Not VR-native.
- **Our relationship with it:** NeoVerse is our new reconstruction backbone (replaces old 4DGaussians plan). It's a gift — CVPR Highlight quality as a free open-source starting point.
- **Threat level:** 🔴 High as backbone, 🟢 Low as competitor (different task goal)

#### Vivid4D — ICCV 2025 *(April 2025 — THE most dangerous academic competitor)*
- **What it does:** Reformulates view augmentation as video inpainting. Warps observed frames to new viewpoints via monocular depth, then diffusion-inpaints missing regions with temporal/spatial consistency. Integrates geometric + generative priors. Iterative view expansion strategy.
- **What it doesn't do:** Goal is *better reconstruction accuracy at nearby/bounded views*, not *VR-completeness at fully unobserved angles*. Does not target the full 180°–360° regime. No VR runtime target. No user study.
- **Critical differentiation:** Vivid4D optimizes PSNR at trajectory-adjacent novel views. We optimize VRC-Score across the full sphere, including angles 90°–270° away from the original camera. Our paper must cite Vivid4D as prior work and demonstrate clear superiority in the far-angle regime specifically.
- **Threat level:** 🔴 HIGH — Know this paper cold.

#### See4D *(Oct 2025)*
- **What it does:** Pose-free 4D scene generation via autoregressive video inpainting. Spline-interpolated camera trajectory + depth-warped conditioning + spatio-temporal autoregression.
- **What it doesn't do:** Not VR-targeted. No spherical completeness. No temporal consistency metric.
- **Threat level:** 🟡 Medium — Related work, not a direct competitor.

#### SeeU — "Seeing the Unseen World" *(Dec 2025)*
- **What it does:** 4D dynamics-aware generation. Disentangles camera, static background, and dynamic foreground. Handles unseen temporal + unseen spatial generation.
- **What it doesn't do:** Still bounded by the original FOV in practice. No VR integration.
- **Threat level:** 🟡 Medium

#### Fillerbuster *(Feb 2025, Ethan Weber, FAIR/Meta)*
- **What it does:** Multi-view latent diffusion Transformer for 3D scene completion. Jointly handles pose prediction + novel view synthesis. Under 1 minute on A5000 (24GB VRAM).
- **What it doesn't do:** Static 3D only — no temporal dynamics. Designed for architectural casual captures (living rooms, offices), not dynamic human video.
- **Threat level:** 🟡 Medium for 3D static, 🟢 Low for dynamic 4D

#### NVIDIA Lyra 2.0 *(April 15, 2026, Apache 2.0 open source)*
- **What it does:** Single image/video → explorable 3D world. Long-horizon, 3D-consistent scene generation. Interactive GUI for camera trajectory planning. Progressively generates scene as user moves. Outputs 3DGS + surface meshes. Open source (weights restricted to research use).
- **What it doesn't do:** Static scenes — no dynamic people moving through time. Not VR-native (desktop GUI only). No temporal 4D representation.
- **Our relationship with it:** Lyra 2.0 is the most polished version of the "exploration experience" we want to build, but for static worlds only. We are the temporal, VR-native, dynamic-scene version of Lyra 2.0.
- **Threat level:** 🔴 HIGH for the experience layer (learn from their UX), 🟢 Low for the 4D dynamic case

#### LaVR *(Jan 2026)*
- **What it does:** Generative video trajectory re-rendering. Conditions video diffusion on latent 4D scene representation extracted from monocular video. Re-renders novel camera trajectories through a scene.
- **What it doesn't do:** Still bounded near the original camera path. Doesn't complete blind-side geometry. Not VR-targeted.
- **Threat level:** 🟡 Medium

#### SceneCompleter *(June 2025)*
- **What it does:** Dense 3D scene completion for generative novel view synthesis. 3D-consistent completion.
- **What it doesn't do:** No temporal dynamics, no VR.
- **Threat level:** 🟢 Low — Related work

#### 4C4D *(April 2026)*
- **What it does:** 4DGS from extremely sparse cameras (as few as 4). Neural Decaying Function on Gaussian opacities for better geometry. Better geometry/appearance balance.
- **What it doesn't do:** Still requires some multi-camera setup, not monocular completion.
- **Threat level:** 🟢 Low

---

### 3B. Business / Commercial Competitors

#### World Labs — "Marble" *(Fei-Fei Li, $1B raised Feb 2026)*
- Text/photo/video/panorama → editable 3D environments with AI-native editing. RTFM (Real-Time Frame Model) for interactive generation. Export as Gaussian Splats, meshes, controlled videos. World API launched Jan 2026.
- **Gap:** Static scenes only. No temporal dynamics of moving people. No true VR headset deployment.
- **Threat level:** 🔴 THE biggest competitor by resources. If they add 4D dynamics, they could own this space. Our window is now.

#### Luma AI *(Commercial, well-funded)*
- iPhone app → free cloud Gaussian Splat processing. Interactive scenes on iOS/Android/Web. PLY export, game engine support. Genie: text-to-3D in <10 seconds.
- **Gap:** Reconstruction only — no generative completion of unseen content. No 4D temporal. Basic VR/AR support.
- **Threat level:** 🟡 Medium

#### Microsoft TRELLIS 2 *(Dec 2025, 4B parameter model)*
- Best static 3D asset generation available. O-Voxel representation + SC-VAE (16× spatial compression) + flow-matching DiT. Full PBR materials. 1536³ resolution. Open source (research weights).
- **Gap:** Completely static. No video input. No temporal dynamics. No VR pipeline.
- **Our relationship:** Use TRELLIS 2 as the object-level generation backend for completing individual 3D assets within our 4D scenes. It's an ingredient.
- **Threat level:** 🟢 Low directly

#### Apple SHARP *(Production SDK)*
- Sub-1-second 3DGS from single image. Metric scale (real-world distances). Baked into Apple hardware.
- **Gap:** Static. Their own documentation confirms back-side quality from a single view is poor. This is our motivation quote.
- **Threat level:** 🟡 Medium — Hardware-locked to Apple ecosystem.

#### Volinga *(Commercial 4DGS processing)*
- Commercial 4DGS pipeline with Nerfstudio integration.
- **Gap:** No generative completion. Reconstruction-only.
- **Threat level:** 🟢 Low

---

### 3C. The Clear Gap Nobody Has Filled

The entire field is racing toward *better reconstruction* from *fewer inputs* with *faster inference*. What nobody is doing:

> **"Complete the full 4D sphere — not just visible surfaces — and make it real-time navigable in VR, maintaining temporal consistency across both time and viewpoint simultaneously."**

That is our claim. It is still unclaimed territory as of May 2026.

---

## PART 4 — THE REVISED SCIENTIFIC THESIS

### Old thesis (February 2026):
*"Use diffusion to fill reconstruction gaps in 4D monocular video."*

### New thesis (May 2026 — properly framed):

> *"We present BARF: a framework for VR-Complete 4D Scene Generation that transforms a single monocular video into a temporally consistent, omnidirectionally complete 4D Gaussian Splat scene explorable in real-time VR from any viewpoint. Unlike prior work targeting reconstruction fidelity at trajectory-bounded novel views, BARF specifically addresses the VR-completeness problem — generating photorealistic, temporally coherent content for all unobserved viewing angles simultaneously — enabling true free-viewpoint navigation of dynamic real-world captures in VR for the first time."*

### The Three Novel Contributions

**Contribution 1: The VR-Completeness Problem (new task definition)**
Formally define "VR-completeness" of a 4D scene: a scene S is VR-complete if for every viewpoint (θ, φ, t) across the full sphere × temporal space, it renders a photorealistic, consistent frame. Introduce **VRC-Score** — the first benchmark metric measuring angular coverage density, temporal coherence, and perceptual quality simultaneously. No paper has defined this. No paper measures it. We define the field.

**Contribution 2: Spherical 4D Completion Module (the technical core)**
A completion architecture that conditions generative synthesis on the *4D scene representation* — not individual frames. This is the key insight that separates us from Vivid4D: conditioning on the 4D latent makes temporal consistency emerge naturally, because the model sees the entire dynamic context when generating any single viewpoint.

**Contribution 3: Real-time VR-ready 4D output pipeline**
End-to-end "phone video → VR world" achieving 72+ FPS on Meta Quest 3. The first system of its kind.

---

## PART 5 — PHASED MASTER PLAN

### Phase 0 — Foundations Reboot *(2 weeks)*
*Goal: A clean, running baseline pipeline using 2026 SOTA*

#### P0.1 — Replace the old backbone
Swap 4DGaussians with **NeoVerse** as the primary reconstruction backbone.
- NeoVerse: pose-free, feed-forward, trained on 1M clips, CVPR 2026 Highlight
- Integrate **D4RT** for camera pose estimation + point tracking as the geometry oracle

#### P0.2 — Get the pipeline running end-to-end (no AI completion yet)
```
Any video → D4RT (poses + tracks) → NeoVerse (4DGS output) → Palak's Web Viewer
```
This must work and produce an actual output before any further work. This is the control condition for the paper.

#### P0.3 — Quantify the gap
Wire existing gap detection code to NeoVerse PLY output.
Produce: occupancy heatmaps at 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°.
This becomes **Figure 1** of the paper — the visual proof of the problem.

---

### Phase 1 — Completion Module *(4–6 weeks)*
*Goal: Generative omnidirectional scene completion with temporal consistency*

#### Architecture

```
Input: 4DGS scene S (from NeoVerse) + angular gap mask G
         ↓
[1] Temporal Feature Extraction
    - Extract per-timestep appearance + geometry features from S
    - Encode the scene dynamics (how objects move across time)
         ↓
[2] Spherical Gap Encoder
    - Identify all angular gaps in G
    - Cluster into completion targets by (θ, φ) region
    - Prioritize: largest + most visible gaps first
         ↓
[3] Completion Diffusion (THE CORE INNOVATION)
    - Backbone: Vivid4D-style video inpainting
    - Conditioning: 4D scene latent (not just 2D frames) — THIS IS NEW
    - Generates: multi-frame RGBA content for each gap region
    - Temporal attention over scene latent → coherence across t is free
         ↓
[4] 4DGS Fusion
    - Warp generated RGBA back into Gaussian representation
    - Use D4RT depth for geometry-consistent placement
    - Blend with existing Gaussians at region boundaries
         ↓
Output: Complete 4DGS scene S' covering full (θ, φ, t)
```

#### Implementation Strategy
- **Option A (Recommended):** Condition Vivid4D's inpainting backbone on NeoVerse scene features via cross-attention. Novel conditioning mechanism, not full training from scratch. Publishable and achievable in 4–6 weeks.
- **Option B (More novel, higher risk):** Train a dedicated spherical completion transformer. Save this for v2 / journal extension.

Start with Option A.

#### Temporal Consistency Module
- Sliding temporal window: generate completion for frames [t−k, t+k] jointly, not frame-by-frame
- RAFT optical flow supervision between consecutive generated frames
- Cross-view consistency loss: generated back-view at frame t must be consistent with frame t+1 given known motion from D4RT's tracker

---

### Phase 2 — VR Integration *(3–4 weeks)*
*Goal: Running on Meta Quest 3 at 72 FPS*

#### Hardware constraints (from PDF research)
- Standalone APK: limit total Gaussians to <500K
- Tethered PCVR (connected to PC): no Gaussian limit — start here
- Standalone Quest 3: ~20 FPS at 150K splats without optimization; with LOD can reach 72 FPS
- Coordinate system: OpenCV (y-down, z-forward) → Unity/Quest (y-up, z-forward) — handle in PLY export

#### Implementation
```
NeoVerse+BARF output (4DGS) 
→ LOD reduction pipeline (Meta Spatial SDK v0.9.2+)
→ Quest-optimized .splat format
→ VR viewer (fork Meta Spatial SDK sample)
  - Timeline scrubbing (time dimension navigation)
  - Free-rotation (viewpoint navigation, full 360°)
  - Teleport locomotion (move through scene)
→ Test on Quest 3
```

#### Performance targets
| Mode | Gaussians | Target FPS |
|---|---|---|
| Desktop (dev) | Unlimited | 90 FPS |
| Quest 3 tethered | ~2M | 72 FPS |
| Quest 3 standalone | <500K | 72 FPS |

---

### Phase 3 — Evaluation & Paper *(4–5 weeks)*
*Goal: Paper-grade quantitative + qualitative results*

#### Datasets
- **Synthetic (ground truth all angles):** Dynamic Scene dataset (used in 4DGS papers), DyNeRF
- **Real (controlled ground truth):** Record our own — one monocular camera as INPUT, second camera at 180° as GROUND TRUTH for completion quality metric
- **Existing real:** Technicolor multi-view dataset, NVIDIA Dynamic Scene

#### VRC-Score implementation
```python
def vrc_score(scene_4dgs, test_angles=[0, 45, 90, 135, 180, 225, 270, 315]):
    """
    VR Completeness Score — composite of three sub-metrics:
    
    1. Coverage(θ): % of test viewpoints with >threshold point density
       → measures whether geometry exists at each angle
    
    2. Coherence(t): mean LPIPS between consecutive time frames at each viewpoint
       → measures whether the completion flickers across time
    
    3. Quality(θ,t): FID/LPIPS vs ground truth where available
       → measures perceptual quality of generated content
    
    VRC = Coverage × (1 - Coherence_loss) × Quality
    """
```

#### Ablation table (paper Table 1)
| Method | Angular Coverage | Temporal Coherence | VRC-Score |
|---|---|---|---|
| NeoVerse only (baseline) | ~45% | 0.92 | ~0.41 |
| NeoVerse + Vivid4D (prior work) | ~68% | 0.79 | ~0.57 |
| Ours without temporal window | ~88% | 0.71 | ~0.63 |
| **BARF (full method)** | **~91%** | **0.88** | **~0.82** |

*(Numbers are targets to validate against — actual results may differ)*

#### User study in VR (the killer result no other paper has)
Every paper in this space evaluates on PSNR/LPIPS. None run a user study in actual VR. This is our differentiator.

**Protocol:**
- 15–20 participants
- Task: walk 360° around a person in VR
- Within-subjects: same scene, Condition A (NeoVerse only) vs Condition B (BARF-completed)
- Measures:
  - Presence rating (7-point Likert: "I felt like I was really there")
  - Visual quality rating (7-point Likert: "The scene looked realistic")
  - Discomfort events (observer counts visible gaps that break immersion)
  - Time-to-notice-gap (how long before participants comment on missing geometry)
- Expected result: significant improvement on all measures for BARF condition
- Analysis: paired t-test or Wilcoxon signed-rank

---

### Phase 4 — Submission *(2–3 weeks)*

#### Target venues (in order)
1. **CVPR 2027** — Primary target. Submission ~Nov 2026. NeoVerse just got a CVPR 2026 Highlight in exactly this space.
2. **SIGGRAPH Asia 2026** — If Phase 0+1 complete by August, achievable. VR + graphics fits perfectly. Submission ~June/July 2026.
3. **ICCV 2027** — Backup if CVPR 2027 misses.
4. **ECCV 2026** — Too soon for this level of work.

#### What makes this CVPR-level
- ✅ Novel task formulation (VR-completeness, not just reconstruction quality)
- ✅ New benchmark metric (VRC-Score) — reviewers love new metrics when they're well-motivated
- ✅ Technical novelty (4D scene-conditioned spherical completion with temporal consistency)
- ✅ Strong, recent baselines (D4RT, NeoVerse, Vivid4D, See4D — all 2025/2026)
- ✅ Real VR user study (no one else has this)
- ✅ Practical end-to-end demo (phone video → Quest 3 in real-time)

---

## PART 6 — IMMEDIATE ACTIONS (NEXT 2 WEEKS)

In exact priority order — do not skip steps:

### Week 1
1. **Merge all branches to main** — pull R4_Tanisha, pxlkele-patch-1, vinay/sprint-completion. Clean repo.
2. **Get NeoVerse running on cloud GPU** (Vast.ai H100 ~$2/hr or Colab Pro A100). Run on 3 test videos. Produce PLY output. [NeoVerse GitHub](https://github.com/IamCreateAI/NeoVerse)
3. **Run Vivid4D on same 3 videos** — this is the prior-work baseline. Must have this early. [Vivid4D](https://arxiv.org/abs/2504.11092)
4. **Wire gap detection to NeoVerse output** — measure angular coverage % at 8 viewpoints. Produce occupancy heatmaps.

### Week 2
5. **Write the VRC-Score formal definition** — one page. What it measures, how to compute it. This anchors the entire paper.
6. **Architecture whiteboard session** — sketch the completion module on paper before any code. What conditioning, what loss, what backbone, what training data.
7. **Read Vivid4D and See4D papers in full** — understand exactly what they do and don't do. Your paper's Related Work section writes itself from this.

---

## PART 7 — TECH STACK (UPDATED May 2026)

### What changes from the February plan

| Component | Old Plan (Feb 2026) | New Plan (May 2026) |
|---|---|---|
| 4D Reconstruction | 4DGaussians | **NeoVerse** (CVPR 2026 Highlight) |
| Camera Poses | COLMAP | **D4RT** (300× faster, 200+ FPS) |
| Novel View Gen | SV3D, Zero123++ | **Vivid4D backbone** (ICCV 2025) as completion base |
| Temporal Consistency | RAFT + sliding denoise | RAFT + **4D scene-conditioned diffusion** |
| Object Generation | — | **TRELLIS 2** for individual object completion |
| VR Viewer | Custom fork | **Meta Spatial SDK v0.9.2+** with native splat support |
| Depth Estimation | Depth Anything V2 | D4RT (integrated) + Depth Anything V2 (fallback) |

### What stays the same
- Gap detection: voxel-based DBSCAN clustering (already designed, wire to NeoVerse)
- Web viewer: Palak's Three.js viewer (already working, keep it for desktop preview)
- VR target: Meta Quest 3
- Output format: 4D Gaussian Splats (.ply sequence or .splat)

---

## PART 8 — THE PDF PIPELINE (How It Relates)

The "2D Image to 3D VR Environment Pipeline" PDF (Qwen-Image-Layered → TRELLIS/SHARP → Unity) describes the *static scene* version of this problem — turning a single photo into a VR world by semantic layer decomposition.

**How it fits into BARF:**
- PDF approach: static scene (single image, no time dimension)
- BARF: dynamic scene (video, full 4D)
- They are **complementary**: use the PDF's pipeline for static background environment generation (TRELLIS 2 for buildings, furniture, background objects), while BARF handles the dynamic foreground (the person moving, the dynamic objects)
- Long-term product vision: a unified system that semantically separates static background (PDF pipeline → TRELLIS 2) from dynamic foreground (BARF 4D completion), composites them into a single VR scene

**Key quote from the PDF to use in our paper introduction:**
> *"SHARP is primarily optimized for nearby views. Because it does not synthesize entirely unseen parts of a scene, the back side of an object from a single 2D layer may be poorly defined or entirely missing."*

This is SHARP's own limitation, in their own documentation. It motivates generative completion at the object level — exactly what we do.

---

## PART 9 — THE PAPER NARRATIVE

### Abstract hook
> *"We walk around people every day. But when you film someone and put them in VR, you can't walk behind them — the back of their head simply doesn't exist. Existing 4D reconstruction methods faithfully reconstruct what the camera saw. We ask a different question: what should be there? BARF addresses the VR-completeness problem — generating the full omnidirectional 4D scene from a single monocular video — enabling true free-viewpoint navigation of dynamic real-world captures in VR for the first time."*

### Introduction structure
1. **Hook:** The gap between "we have 4D reconstruction" and "we can put it in VR and walk around freely" — quote NeoVerse, D4RT, show their back-view gaps
2. **Problem:** Define VR-completeness formally. Show Figure 1: angular coverage heatmap of NeoVerse output with empty regions at >90°
3. **Prior work fails because:** Vivid4D, See4D, Fillerbuster — all optimize reconstruction at trajectory-bounded views, not omnidirectional completeness
4. **Our key insight:** Condition the completion diffusion on the 4D scene representation, not individual 2D frames → temporal consistency emerges naturally
5. **Contributions:** three bullets: task definition + metric, completion architecture, real-time VR pipeline

### Results that get you on stage
- **Demo video:** Phone records a person walking → VR scene where viewer walks 360° around them (the moment the viewer goes behind and sees the back = the money shot)
- **VRC-Score:** 41% baseline (NeoVerse) → 82% with BARF (2× improvement)
- **User study:** "83% of participants report higher presence in BARF scenes"
- **Real-time:** 72 FPS on Quest 3 standalone

---

## PART 10 — TEAM STRUCTURE (Revised)

Given it's May 2026, the February team has likely moved on. Minimum viable team to get a paper:

| Role | Tasks | Who |
|---|---|---|
| **Lead / Architect (Vinay)** | Integration, temporal consistency module, paper writing | You |
| **ML Engineer** | NeoVerse integration, completion architecture, training/finetuning |    |
| **VR Engineer** | Quest 3 deployment, LOD optimization, VR viewer |  |
| **Data Engineer** | Dataset collection, benchmark setup, VRC-Score implementation |  |
| **Evaluation** | User study coordination, qualitative evaluation |  |

If full team isn't available: **you + 1 ML engineer + 1 VR engineer** is sufficient to build and publish.

---

## BOTTOM LINE

Your idea is not just still relevant — it sits at the exact center of the biggest wave in spatial computing right now.

- World Labs raised **$1 billion** for the *static* version of this problem
- NVIDIA open-sourced **Lyra 2.0** for the *static* version
- NeoVerse won a **CVPR 2026 Highlight** for the *reconstruction* side
- **Nobody has done the dynamic, generatively-complete, VR-explorable version**

The February sprint gave you the scientific motivation, a working viewer, and a team scaffold. What you need now is execution: get NeoVerse running, measure the gap, build the completion module, put it on a Quest.

**The window is open. It will not be open forever.**

---

## SOURCES

- [D4RT — DeepMind Blog](https://deepmind.google/blog/d4rt-teaching-ai-to-see-the-world-in-four-dimensions/)
- [D4RT Paper (arxiv 2512.08924)](https://arxiv.org/pdf/2512.08924)
- [NeoVerse — arxiv 2601.00393](https://arxiv.org/abs/2601.00393)
- [NeoVerse GitHub (CVPR 2026 Highlight)](https://github.com/IamCreateAI/NeoVerse)
- [Vivid4D — ICCV 2025 (arxiv 2504.11092)](https://arxiv.org/abs/2504.11092)
- [Fillerbuster (arxiv 2502.05175)](https://arxiv.org/abs/2502.05175)
- [NVIDIA Lyra 2.0 Research Page](https://research.nvidia.com/labs/sil/projects/lyra2/)
- [NVIDIA Lyra 2.0 (arxiv 2604.13036)](https://arxiv.org/html/2604.13036v1)
- [See4D (arxiv 2510.26796)](https://arxiv.org/pdf/2510.26796)
- [LaVR (arxiv 2601.14674)](https://arxiv.org/abs/2601.14674)
- [SceneCompleter (arxiv 2506.10981)](https://arxiv.org/abs/2506.10981)
- [World Labs Marble — TechCrunch](https://techcrunch.com/2025/11/12/fei-fei-lis-world-labs-speeds-up-the-world-model-race-with-marble-its-first-commercial-product/)
- [TRELLIS 2 vs TRELLIS comparison](https://piapi.ai/blogs/trellis-2-vs-trellis-3d-generation-api)
- [4C4D (arxiv 2604.04063)](https://arxiv.org/html/2604.04063v1)
- [CVPR 2026 4D Vision Workshop](https://4dvisionworkshop.github.io/)
- [NeoVerse startup coverage](https://www.startuphub.ai/ai-news/ai-research/2026/neoverse-cracks-the-scalability-problem-for-monocular-4d-models/)
