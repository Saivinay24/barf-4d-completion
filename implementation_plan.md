# BARF 4D — Implementation Plan

> **Last updated:** May 28, 2026
>
> This document is the single source of truth for what is built, what remains, and how to execute each task. If you are picking up a task, read the relevant phase section — it has everything you need.

---

## Project Overview

**BARF** transforms a monocular phone video into a fully explorable 4D VR scene — including the parts the camera never saw. Current SOTA (NeoVerse, D4RT, Vivid4D) only reconstructs visible surfaces. BARF adds a **generative completion module** that fills in every missing angle using a diffusion model conditioned on the 4D scene representation.

**The key innovation:** Instead of conditioning on 2D frames (like Vivid4D does), we condition on the full 4D scene latent. This means the generated back-of-head at frame t+1 is automatically consistent with frame t, because the model sees the full motion dynamics.

### What exists right now

```
✅ Gap detection module          — src/gap_detection/detect_gaps.py (528 lines, tested)
✅ Completion module architecture — src/completion/spherical_completion.py (661 lines, tested)
✅ VRC-Score metric               — src/metrics/vrc_score.py (480 lines, tested)
✅ VR export pipeline             — src/vr/export_splat.py (542 lines, tested)
✅ End-to-end pipeline script     — scripts/run_pipeline.sh
✅ Vivid4D baseline script        — scripts/run_vivid4d_baseline.sh
✅ Paper draft                    — paper/draft.md (all sections, 17 result placeholders)
✅ 104 unit tests                 — all passing
```

### What needs to happen

```
❌ Run NeoVerse + D4RT on real videos (GPU)
❌ Replace stub UNet with real Vivid4D backbone (GPU)
❌ Train cross-attention conditioning layers (GPU, 3-5 days)
❌ Build Quest 3 VR viewer app (Unity + Meta Spatial SDK)
❌ Run ablation study + fill paper results
❌ User study (15-20 participants)
```

---

## Progress at a Glance

| Phase | Status | Summary |
|---|---|---|
| **Phase 0** — Foundations | 🟡 60% | Scripts ready, need GPU runs |
| **Phase 1** — Completion Module | 🟡 40% | Architecture done, need real backbone + training |
| **Phase 2** — VR Integration | 🟡 30% | Export pipeline done, need Quest 3 app |
| **Phase 3** — Evaluation | 🟡 25% | Metric + paper template done, need real numbers |
| **Phase 4** — Submission | ⬜ 5% | Waiting on results |

---

## Phase 0 — Foundations (Target: 2 weeks)

**Goal:** Get working baseline outputs from SOTA methods on 3 test videos.

### What's already built

**Pipeline script** (`scripts/run_pipeline.sh`):
- Takes a video path and output directory as arguments
- 5 sequential steps: frame extraction → D4RT → NeoVerse → gap detection → output
- Frame extraction (Step 1) works locally on CPU using OpenCV
- Steps 2-3 (D4RT, NeoVerse) have `# TODO: GPU EXECUTION REQUIRED` markers with the exact commands to run
- Steps 4-5 (gap detection, output) work locally on CPU
- When no GPU is available, the script auto-generates synthetic placeholder outputs so downstream modules can be tested

**Gap detection** (`src/gap_detection/detect_gaps.py`):
- Pure numpy implementation — no open3d dependency
- Reads any PLY file (ASCII or binary)
- Computes point density in 8 angular sectors (0°, 45°, 90°, ..., 315°)
- Finds gap clusters via voxel-based BFS
- Generates occupancy heatmap PNGs at each angle
- Outputs `gaps.json` with gap positions, sizes, and a coverage summary
- CLI: `python3 -m src.gap_detection.detect_gaps --input scene.ply --output_json gaps.json --output_heatmap_dir heatmaps/`

**Vivid4D baseline script** (`scripts/run_vivid4d_baseline.sh`):
- Full setup: conda env, git clone, dependency install
- Two run modes: full video inference or frame-by-frame
- Documents expected output format and where to find results

**Synthetic test data** (`data/test_plys/synthetic_front_facing.ply`):
- ~3200 points forming a front-facing hemisphere
- Gap detection correctly reports 50% coverage (back half empty)
- 8 heatmap PNGs already generated in `data/gap_heatmaps/heatmaps/`

### Tasks remaining

#### P0.1 — Select 3 test videos
**Context:** We need diverse monocular videos to test the full pipeline. These become the benchmark scenes used throughout the paper.

**Requirements:**
- Video 1: Single person walking (tests dynamic motion completion)
- Video 2: Person sitting/gesturing (tests subtle motion + face/hands)
- Video 3: Multi-object scene (tests scene-level completion)
- All should be monocular (single phone camera), 5-15 seconds, decent lighting
- Options: record your own, or download from DyNeRF dataset or Dynamic Scene dataset

**Output:** 3 `.mp4` files in `data/test_videos/`

---

#### P0.2 — Run D4RT on 3 test videos
**Context:** D4RT (Google DeepMind) provides camera poses + 4D point tracking. It's our "geometry oracle" — we trust its poses and use its depth maps for back-projection later.

**Requirements:** GPU with 24GB+ VRAM (A100 or H100)

**Steps:**
```bash
# 1. Clone D4RT
git clone https://github.com/google-deepmind/d4rt
cd d4rt && pip install -e .

# 2. Run on each video
python d4rt/run_d4rt.py \
    --input_video ../data/test_videos/video_01.mp4 \
    --output_dir ../outputs/video_01/d4rt/

# 3. Check outputs exist:
#    outputs/video_01/d4rt/poses.json       — camera trajectory
#    outputs/video_01/d4rt/tracks.npz       — 4D point tracks
#    outputs/video_01/d4rt/depth/           — per-frame depth maps
```

**Expected output:**
- `poses.json` — camera intrinsics + extrinsic matrices per frame
- `tracks.npz` — 3D point positions tracked across all frames
- `depth/*.png` — per-frame estimated depth maps

**Time estimate:** ~30 min per video

---

#### P0.3 — Run NeoVerse on 3 test videos
**Context:** NeoVerse (CVPR 2026 Highlight) is our reconstruction backbone. It produces the initial 4DGS (4D Gaussian Splat) scene from monocular video. This is the "partial" scene that BARF then completes.

**Requirements:** GPU with 40GB+ VRAM (needs large model checkpoint)

**Steps:**
```bash
# 1. Clone and setup NeoVerse
git clone https://github.com/IamCreateAI/NeoVerse
cd NeoVerse && pip install -e .
# Download checkpoints (see NeoVerse README for links)

# 2. Run inference on each video
python inference.py \
    --video ../data/test_videos/video_01.mp4 \
    --output_dir ../outputs/video_01/neoverse/

# 3. Check outputs exist:
#    outputs/video_01/neoverse/scene.ply      — full scene point cloud
#    outputs/video_01/neoverse/frame_*.ply    — per-timestep point clouds
```

**Expected output:**
- `scene.ply` — combined 4DGS scene (this is what gap detection analyzes)
- `frame_XXXX.ply` — per-timestep point clouds (for 4D temporal analysis)

**Time estimate:** ~20 min per video

**Validation:** After running, immediately run gap detection to verify the PLY is valid:
```bash
python3 -m src.gap_detection.detect_gaps \
    --input outputs/video_01/neoverse/scene.ply \
    --output_json outputs/video_01/gaps/gaps.json \
    --output_heatmap_dir outputs/video_01/gaps/heatmaps/
```
Expected result: ~45-55% angular coverage (front-facing only). The heatmaps should show dense points at 0° and sparse/empty at 180°.

---

#### P0.4 — Run Vivid4D baseline on 3 test videos
**Context:** Vivid4D (ICCV 2025) is the closest prior work. Running it on the same videos gives us the "prior work" baseline row in our ablation table. We need to show BARF beats Vivid4D specifically on back-view angles.

**Requirements:** GPU with 24GB+ VRAM

**Steps:** Follow `scripts/run_vivid4d_baseline.sh` — it has complete setup instructions.

**Expected output:**
- `vivid4d_scene.ply` — Vivid4D's augmented reconstruction
- Rendered views at 8 angles for comparison

**Validation:** Run gap detection on Vivid4D output and compare coverage % against NeoVerse-only output. Vivid4D should improve trajectory-adjacent views (~60-70% coverage) but still miss back-angles.

---

#### P0.5 — Run gap detection + VRC-Score on all outputs
**Context:** This produces the baseline numbers that the entire paper is built on.

**Steps:**
```bash
# For each method (NeoVerse, Vivid4D):
python3 -m src.gap_detection.detect_gaps \
    --input outputs/video_01/neoverse/scene.ply \
    --output_json outputs/video_01/gaps/neoverse_gaps.json \
    --output_heatmap_dir outputs/video_01/gaps/neoverse_heatmaps/

python3 -m src.metrics.vrc_score \
    --scene_ply outputs/video_01/neoverse/scene.ply \
    --output outputs/video_01/vrc/neoverse_vrc.json
```

**Expected output:** JSON files with per-angle coverage percentages. Record these in `results/ablation_table.md` (rows 1-2).

---

### GPU Setup (One-Time)

```bash
# Recommended: Vast.ai H100 (~$2/hr) or Colab Pro A100
# Budget for Phase 0: ~$10-15 total

# On the GPU machine:
git clone https://github.com/Saivinay24/barf-4d-completion
cd barf-4d-completion
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Phase 1 — Completion Module (Target: 4-6 weeks)

**Goal:** Replace the stub UNet with a real Vivid4D backbone, train the cross-attention conditioning layers, and produce completed scenes.

### What's already built

**Architecture** (`src/completion/spherical_completion.py`) — 5 components, all implemented and unit-tested:

1. **TemporalFeatureExtractor** (lines 77-122):
   - PointNet-style encoder that takes per-timestep Gaussian parameters `(T, N, 6)` where 6 = `[x, y, z, r, g, b]`
   - Mean-pools over N Gaussians per timestep to produce a scene latent `(T, 512)`
   - This latent captures *how the scene looks and moves* at each timestep

2. **SphericalGapEncoder** (lines 129-184):
   - Takes gap dicts from `detect_gaps.py` and encodes them as query vectors
   - Input features: `[sin(θ), cos(θ), sin(φ), cos(φ), log(size)]` → MLP → `(G, 512)`
   - These queries tell the model *where* it needs to generate content

3. **SceneConditionedCrossAttention** (lines 191-233):
   - **THE KEY INNOVATION** — this is what differentiates BARF from Vivid4D
   - Standard multi-head cross-attention: `query = diffusion activations`, `key/value = [scene_latent; gap_queries]`
   - By attending to the full temporal scene latent, the model "sees" all timesteps when generating any single gap region
   - This makes temporal consistency emerge naturally without explicit optical flow loss

4. **CompletionDiffusion** (lines 236-349):
   - Currently uses a **stub UNet** (MLP encoder/decoder) for unit testing
   - The stub has the correct interface: accepts `(B, T, C, H, W)` noisy frames + scene_latent + gap_queries + timestep
   - The `SceneConditionedCrossAttention` layer is already wired in at the bottleneck
   - **Lines 297-300** have the exact TODO comment for swapping in the real Vivid4D UNet

5. **GaussianFusion** (lines 356-442):
   - Back-projects generated RGBA frames to 3D Gaussian positions using camera intrinsics/extrinsics
   - Merges new Gaussians with the existing scene, filtering by confidence (alpha > 0.5)

6. **SphericalCompletionPipeline** (lines 449-629):
   - End-to-end wrapper class
   - `complete_synthetic()` — runs a full forward pass with random tensors on CPU (for testing)
   - `complete()` — reads real PLY + gaps JSON, runs inference, writes output PLY (currently writes placeholder)

### Tasks remaining

#### P1.1 — Swap stub UNet for real Vivid4D backbone
**Context:** The stub encoder/decoder in `CompletionDiffusion` (lines 274-288) is a placeholder MLP. For real inference, we need the actual Vivid4D UNet that knows how to denoise video frames.

**What to do:**
1. Clone Vivid4D: `git clone https://github.com/vivid4d/vivid4d` (check paper for exact repo)
2. Download the pre-trained UNet checkpoint
3. In `src/completion/spherical_completion.py`, class `CompletionDiffusion.__init__()`:
   - Replace `self.stub_encoder` and `self.stub_decoder` with the loaded Vivid4D UNet
   - The UNet should accept `(B, C, H, W)` noisy frames and return denoised frames
4. Register `SceneConditionedCrossAttention` hooks at each UNet downsampling block:
   - Vivid4D's UNet has ~4 downsampling blocks
   - At each block, inject our cross-attention layer that attends to `scene_context`
   - The existing `self.scene_cross_attn` is only at the bottleneck — for full quality, add one per block
5. Update the `forward()` method to route through the real UNet instead of the stub

**Key constraint:** The `SceneConditionedCrossAttention` expects `dim=512` (feature_dim). Match this to the UNet's internal channel dimension at each block. You may need different `dim` values per block.

**Validation:** Run `complete_synthetic()` with the real UNet on GPU. The output shape should still be `(B, T, 4, H, W)`. Compare with stub outputs to verify the cross-attention is receiving gradients.

---

#### P1.2 — Implement DDIM sampling loop
**Context:** The current `forward()` does a single denoising step. Real diffusion inference needs 20 iterative DDIM steps going from pure noise → clean image.

**What to do:**
1. In `SphericalCompletionPipeline.complete()` (around line 581), add DDIM sampling:
```python
# Pseudocode for DDIM loop:
x_t = torch.randn(B, T, 4, H, W)  # start from noise
for step in reversed(range(n_diffusion_steps)):
    timestep = torch.full((B,), step)
    predicted_noise = self.diffusion(x_t, scene_latent, gap_queries, timestep)
    x_t = ddim_step(x_t, predicted_noise, step)  # standard DDIM update
```
2. Use Vivid4D's existing noise scheduler if available, or implement DDIM from `diffusers` library
3. Set `n_diffusion_steps = 20` (configurable via CLI `--n_steps`)

**Validation:** Generate frames for one gap region. Even without training, the output should be spatially coherent (not random noise) because the pre-trained UNet provides a strong prior.

---

#### P1.3 — Fine-tune cross-attention layers
**Context:** The Vivid4D UNet is pre-trained for 2D frame conditioning. We need to teach the cross-attention layers to condition on our 4D scene latent instead. This is where the novel research happens.

**Training strategy (Option A — recommended, fastest):**
1. **Freeze** all Vivid4D UNet weights
2. **Train only:** `SceneConditionedCrossAttention` layers + `TemporalFeatureExtractor` + `SphericalGapEncoder`
3. **Training data:** Use NeoVerse reconstructions as scene input. For each scene, render "ground truth" back-views from the NeoVerse output at angles where data exists, and use those as supervision targets.
4. **Loss:** Standard diffusion MSE loss on the predicted noise, plus optional LPIPS perceptual loss on denoised frames
5. **Hardware:** H100 GPU, batch_size=2 (limited by T × H × W memory)
6. **Estimated time:** 3-5 days of training

**Validation:** After training, run completion on a NeoVerse scene PLY. Compare the gap detection coverage before and after — it should jump from ~45% to ~85%+.

---

#### P1.4 — Add temporal sliding window
**Context:** Even with 4D scene conditioning, there may be residual frame-to-frame flicker. The sliding window generates frames [t-k, t+k] jointly and applies optical flow consistency.

**What to do:**
1. During DDIM sampling, process temporal windows of 2k+1 frames together (k=5)
2. After denoising each window, compute RAFT optical flow between consecutive generated frames
3. Add a flow consistency loss: penalize generated frames where the flow-warped frame t doesn't match frame t+1
4. Overlap windows by k frames and blend in the overlap region

**Implementation:** Add as a method in `SphericalCompletionPipeline`. This is described in paper Section 4.4.

**Validation:** Measure VRC-Coherence before and after adding the temporal window. This populates the "BARF w/o temporal" vs "BARF (full)" rows in the ablation.

---

#### P1.5 — Run completion on real NeoVerse output
**Context:** This is the first real end-to-end test.

**Steps:**
```bash
python3 -m src.completion.spherical_completion \
    --scene_ply outputs/video_01/neoverse/scene.ply \
    --gaps_json outputs/video_01/gaps/gaps.json \
    --output_ply outputs/video_01/completion/scene_complete.ply \
    --device cuda \
    --n_steps 20
```

**Validation:**
```bash
# Before completion:
python3 -m src.metrics.vrc_score --scene_ply outputs/video_01/neoverse/scene.ply --output results/vrc_before.json

# After completion:
python3 -m src.metrics.vrc_score --scene_ply outputs/video_01/completion/scene_complete.ply --output results/vrc_after.json

# Compare: coverage should jump from ~45% to ~91%
```

---

## Phase 2 — VR Integration (Target: 3-4 weeks)

**Goal:** Running completed 4D scenes on Meta Quest 3 at 72 FPS standalone.

### What's already built

**VR export pipeline** (`src/vr/export_splat.py`):
- Converts PLY → Quest-compatible `.splat` binary format (antimatter15/splat spec)
- Two LOD reduction strategies:
  - `lod_reduce_uniform()` — random subsample with optional voxel pre-filter
  - `lod_reduce_importance()` — inverse-density sampling that preserves sparse boundary regions
- Coordinate transform: OpenCV (y-down) → Unity/Quest (y-up)
- `.splat` format: 32 bytes per Gaussian (position, scale, color, opacity, rotation quaternion)
- `SplatExporter` class handles the full flow: load PLY → reduce → transform → encode → write
- `export_4d_sequence()` batch-exports per-timestep PLYs for temporal playback
- Quest 3 standalone limit: 500K Gaussians for stable 72 FPS

**CLI:**
```bash
python3 -m src.vr.export_splat \
    --input scene_complete.ply \
    --output scene.splat \
    --max_gaussians 500000 \
    --lod importance
```

### Tasks remaining

#### P2.1 — Build Quest 3 viewer app
**Context:** We need a Unity app that loads `.splat` files and lets the user walk around the scene with joystick controls.

**Tech stack:**
- Unity 2022.3 LTS
- Meta Spatial SDK v0.9.2+ (has native Gaussian Splat rendering)
- Meta Quest Developer Hub (for sideloading APK)

**Requirements:**
- Load a single `.splat` file and render it
- Free-rotation camera: left joystick = move, right joystick = look
- Display current FPS in a debug HUD
- For 4D: load a sequence of `.splat` files and play them back with a timeline scrubber (D-pad or UI slider)

**Reference:** The web viewer in `viewer/` has a working PLY loader + gap visualization in Three.js. The Quest app is a native Unity port of the same concept.

---

#### P2.2 — Test with real .splat files
After Phase 1 produces completed PLYs, export them:
```bash
python3 -m src.vr.export_splat --input scene_complete.ply --output scene.splat --max_gaussians 500000
```
Sideload onto Quest 3 via `adb install` and verify rendering.

---

#### P2.3 — Benchmark FPS at different Gaussian counts
Export the same scene at 100K, 200K, 300K, 500K Gaussians. Measure FPS for each on Quest 3 standalone. Find the sweet spot. Fill in the VR Performance table in `results/ablation_table.md`.

---

## Phase 3 — Evaluation & Paper (Target: 4-5 weeks)

**Goal:** Paper-grade quantitative results, figures, and user study.

### What's already built

**VRC-Score** (`src/metrics/vrc_score.py`):
- `VRC-Score = Coverage(θ) × Coherence(t) × Quality(θ,t)` — multiplicative (any zero collapses the score)
- Coverage: fraction of 8 test angles with ≥5 points in the angular sector (pure numpy, works on CPU)
- Coherence: MAE or LPIPS between consecutive frames at a fixed viewpoint (needs rendered frames)
- Quality: PSNR/LPIPS vs ground truth (needs GT renders from dual-camera or synthetic dataset)
- CLI: `python3 -m src.metrics.vrc_score --scene_ply scene.ply --output vrc.json`

**Paper draft** (`paper/draft.md`):
- 440 lines, all sections: Abstract, Introduction, Related Work, Method (4.1-4.5), Experiments (5.1-5.5), Conclusion
- 12 references (D4RT, NeoVerse, Vivid4D, See4D, SeeU, Fillerbuster, Lyra 2.0, LaVR, etc.)
- 17 `[RESULT: ...]` placeholders waiting for real numbers

**Ablation table template** (`results/ablation_table.md`):
- 4-row comparison: NeoVerse only → +Vivid4D → BARF w/o temporal → BARF full
- Target values already filled for reference
- Exact GPU commands to populate each row

### Tasks remaining

#### P3.1 — Collect benchmark dataset
**Options:**
- **(a) DyNeRF** — real multi-view dataset, downloadable, 6 scenes
- **(b) Dynamic Scene dataset** — synthetic, has ground truth at all angles
- **(c) Custom dual-camera** — record with two phones (front + back), gives real GT back-views
- Minimum: 3 scenes. Ideal: 6+ for statistical rigor.

#### P3.2 — Run full ablation study
For each of 3+ scenes, run all 4 configurations and compute VRC-Score:
1. NeoVerse only (baseline)
2. NeoVerse + Vivid4D (prior work)
3. BARF without temporal window
4. BARF full (with temporal window)

Copy numbers into `results/ablation_table.md`.

#### P3.3 — Generate paper figures
- **Figure 1:** Angular coverage heatmaps side-by-side (NeoVerse vs BARF)
- **Figure 2:** Per-angle rendered views at 0°, 45°, 90°, 135°, 180°
- **Figure 3:** Temporal consistency at 180° across 10 timesteps (Vivid4D flickering vs BARF smooth)

#### P3.4 — Fill paper `[RESULT]` placeholders
Search `paper/draft.md` for `[RESULT:` — there are 17 placeholders. Replace each with the real number from ablation.

#### P3.5 — User study
- 15-20 participants
- Each experiences 4 scenes (2 BARF, 2 NeoVerse-only) in counterbalanced order
- Measures: presence (Likert 7pt), visual quality (Likert 7pt), discomfort events
- Analysis: paired t-test or Wilcoxon signed-rank test

#### P3.6 — Record demo video
60-90 second video showing: phone filming → pipeline → VR 360° walkable result.

---

## Phase 4 — Submission (Target: 2-3 weeks)

**Goal:** Submit to CVPR 2027 or SIGGRAPH Asia 2026.

### Tasks
- P4.1 — Replace all `[RESULT]` placeholders with real numbers
- P4.2 — Add figures to paper
- P4.3 — Format in CVPR LaTeX or SIGGRAPH template
- P4.4 — Internal review (2-3 readers)
- P4.5 — Create supplementary material (demo video, code link)
- P4.6 — Submit

### Timeline options

**SIGGRAPH Asia 2026 (aggressive):**
```
May 28 → Jun 15:  Phase 0 + Phase 1
Jun 15 → Jun 30:  Phase 2 + Phase 3
Jul 1  → Jul 10:  Phase 4 (submit)
```

**CVPR 2027 (recommended):**
```
May 28 → Jul 15:  Phase 0 + Phase 1
Jul 15 → Aug 31:  Phase 2 + Phase 3
Sep 1  → Oct 15:  Phase 4 (polish)
Oct 15 → Nov 15:  Buffer + submit
```

---

## File Reference

| File | What it does | Lines |
|---|---|---|
| `src/gap_detection/detect_gaps.py` | Angular coverage analysis, gap clustering, heatmap generation | 528 |
| `src/completion/spherical_completion.py` | 5-component completion architecture (stub UNet, needs real backbone) | 661 |
| `src/metrics/vrc_score.py` | VRC-Score: Coverage × Coherence × Quality | 480 |
| `src/vr/export_splat.py` | PLY → Quest-compatible .splat with LOD reduction | 542 |
| `scripts/run_pipeline.sh` | End-to-end: video → D4RT → NeoVerse → gaps → output | 271 |
| `scripts/run_vivid4d_baseline.sh` | Vivid4D baseline setup + run instructions | 135 |
| `paper/draft.md` | Full paper draft, 17 result placeholders | 440 |
| `results/ablation_table.md` | Ablation table template with target values | 135 |
| `MASTER_PLAN.md` | Competitive analysis + full research roadmap | — |
| `BARF_VRC_SCORE.md` | Formal mathematical definition of VRC-Score | — |
| `BARF_REFERENCE.md` | Tech stack + architecture quick-reference | 90 |

## Dependencies

```
# Core (CPU)
numpy
opencv-python
pillow

# GPU (for Phase 1+)
torch
torchvision

# Optional (for LPIPS coherence metric)
lpips

# Testing
pytest
```

Install: `pip install -r requirements.txt`

Run tests: `python3 -m pytest tests/ -q` (should report 104 passed)
