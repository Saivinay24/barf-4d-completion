# VRC-Score: VR Completeness Score
### Formal Definition — BARF 4D (Contribution 1)

---

## 1. Motivation

Existing 4D reconstruction metrics (PSNR, SSIM, LPIPS, Chamfer Distance) measure
quality at observed or trajectory-adjacent views. They are blind to the VR
navigation problem: **can a user freely walk 360° around a reconstructed scene
without encountering empty geometry?**

We introduce the **VRC-Score (VR Completeness Score)** — the first metric
specifically designed to evaluate omnidirectional completeness of a 4D scene
for VR free-viewpoint navigation.

---

## 2. Formal Definition

### Problem Setup

Let S be a 4D scene defined over:
- Spatial domain: R³
- Temporal domain: T = {t₁, t₂, ..., tₙ}
- Viewpoint domain: Ω = S² × T (unit sphere × time)

**Definition (VR-Complete Scene):**
A scene S is *VR-complete* if for every viewpoint (θ, φ, t) ∈ Ω, the rendered
frame Rθ,φ,t is photorealistic (high perceptual quality) and temporally
consistent with adjacent frames.

### The VRC-Score

```
VRC-Score(S) = VRC-Coverage(S) × VRC-Coherence(S) × VRC-Quality(S)
```

where each sub-metric is in [0, 1].

---

## 3. Sub-Metrics

### 3.1 VRC-Coverage C(S) ∈ [0,1]

Measures what fraction of the full viewing sphere has sufficient geometry.

**Test angles:** Θ = {0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°} (8 azimuth angles)

**Sector definition:** Each test angle θᵢ covers the sector [θᵢ − 22.5°, θᵢ + 22.5°].

**Coverage indicator:**
```
covered(θᵢ) = 1  if  |{p ∈ S : azimuth(p) ∈ sector(θᵢ)}| ≥ τ
              0  otherwise
```
where τ = 5 points (default density threshold).

**VRC-Coverage:**
```
C(S) = (1/|Θ|) × Σᵢ covered(θᵢ)
```

**Interpretation:**
- C = 1.0: all 8 angles have geometry → perfect spherical coverage
- C = 0.5: only front 4 angles covered → typical monocular reconstruction gap
- C = 0.0: no geometry → completely empty scene

---

### 3.2 VRC-Coherence H(S) ∈ [0,1]

Measures temporal consistency of the scene across time at all viewpoints.
High coherence = no flickering when the user holds a viewpoint while the scene
evolves (e.g., a walking person).

**Per-angle coherence** (at viewpoint θᵢ over T timesteps):
```
flicker(θᵢ) = (1/(T-1)) × Σₜ LPIPS(Rθᵢ,t, Rθᵢ,t+1)
```
where LPIPS uses AlexNet backbone (perceptual distance, lower = more similar).

**VRC-Coherence:**
```
H(S) = 1 - mean_{θᵢ ∈ Θ} flicker(θᵢ) / flicker_max
```
where flicker_max = 0.5 (normalisation constant — corresponds to completely
unrelated consecutive frames).

**CPU approximation** (when renders unavailable):
Replace LPIPS with MAE (mean absolute pixel error), normalised by 0.2.

---

### 3.3 VRC-Quality Q(S) ∈ [0,1]

Measures perceptual quality of rendered frames vs ground truth.

**Requires ground truth:** either synthetic dataset with multi-view renders,
or two-camera capture (monocular front + back camera for ground truth).

**Per-viewpoint quality:**
```
Q(θᵢ) = (1/T) × Σₜ (1 - LPIPS(Rθᵢ,t, GT_θᵢ,t))
```

**VRC-Quality:**
```
Q(S) = mean_{θᵢ ∈ Θ} Q(θᵢ)
```

**When GT not available:**
VRC-Score reduces to VRC-Score-NoGT:
```
VRC-Score-NoGT(S) = 2 × C(S) × H(S) / (C(S) + H(S))   [harmonic mean]
```

---

## 4. Composite Score

```
VRC-Score(S) = C(S) × H(S) × Q(S)
```

**Properties:**
1. Range: [0, 1]
2. Multiplicative: any sub-metric near 0 collapses the score → all three must be good
3. Decomposable: each sub-metric is interpretable independently
4. Comparable: allows direct comparison across methods on the same test set

---

## 5. Expected Values — Ablation Table Targets

| Method | C(S) Coverage | H(S) Coherence | Q(S) Quality | VRC-Score |
|---|---|---|---|---|
| NeoVerse only (baseline) | ~0.45 | ~0.92 | ~0.50 | ~0.21 |
| NeoVerse + Vivid4D | ~0.68 | ~0.79 | ~0.60 | ~0.32 |
| BARF (ours, no temp. window) | ~0.88 | ~0.71 | ~0.70 | ~0.44 |
| **BARF (ours, full)** | **~0.91** | **~0.88** | **~0.78** | **~0.62** |

*Note: Q(S) shown here as back-view quality. Full composite will differ.
These are target values — real GPU numbers go in results/ablation_table.md.*

---

## 6. Implementation

```python
from src.metrics.vrc_score import VRCScore, compute_coverage, composite_vrc_score

# From PLY (coverage only, CPU):
scorer = VRCScore()
result = scorer.compute_from_ply("scene.ply", output_path="vrc_score.json")
# → result["vrc_coverage"] = 0.91

# From rendered frames (full score, GPU recommended):
renders = {0: frames_0deg, 90: frames_90deg, 180: frames_180deg, ...}
gt = {0: gt_0deg, 90: gt_90deg, ...}
result = scorer.compute_from_frames(renders, ground_truth=gt)
# → result["vrc_score"] = 0.82
```

---

## 7. Benchmark Protocol

To produce paper-grade VRC-Score numbers:

### 7.1 Synthetic evaluation (DyNeRF / Dynamic Scene dataset)
1. Run each method on test video
2. Render from all 8 test angles at all T timesteps
3. Compare with dataset ground truth renders
4. Report per-method Coverage, Coherence, Quality, VRC-Score

### 7.2 Real-world evaluation (two-camera captures)
1. Record test scenes with Camera A (monocular front — this is the INPUT)
   and Camera B (positioned at 180° — this is the GROUND TRUTH for back-view)
2. Run each method with Camera A input only
3. Evaluate back-view render vs Camera B ground truth for Q(S)
4. Evaluate all angles for C(S) and H(S)

### 7.3 VR user study (qualitative validation)
See paper/draft.md Section 5.3 for full protocol.

---

## 8. Relationship to Existing Metrics

| Metric | What it measures | VRC-Score relationship |
|---|---|---|
| PSNR / SSIM | Pixel-level quality at seen views | Captured by Q(S) but at unseen angles |
| LPIPS | Perceptual quality at seen views | Backbone of Q(S) and H(S) |
| Chamfer Distance | 3D geometry accuracy | Correlated with C(S) but not equivalent |
| FID | Distribution of generated images | Partial overlap with Q(S) |
| **VRC-Score** | Full-sphere VR navigability | **New — not captured by any above** |

---

*Defined by Sai Vinay Bhoomireddy, BARF 4D project, May 2026.*
*Implementation: src/metrics/vrc_score.py*
