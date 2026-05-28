# Gap Analysis — Partial 4D Reconstructions

## Overview

This document presents the 360° gap analysis of partial 4D reconstructions produced by 4DGaussians. Each reconstruction was loaded, rotated in 45° increments, and assessed for missing geometry.

---

## Video: v01_person_walk (Custom, 1080p)

### Rotation Analysis

| Angle | Density | Status | Notes |
|-------|---------|--------|-------|
| 0° (Front) | Dense | ✅ Good | Person face/torso clearly reconstructed |
| 45° | Dense | ✅ Good | Right side visible, slight edge thinning |
| 90° (Side) | Medium | ⚠️ Thinning | Left arm incomplete, torso edge sparse |
| 135° | Sparse | ❌ Degraded | Major geometry loss, fragments only |
| 180° (Back) | Nearly empty | ❌ Critical | ~70% missing — back of head, torso absent |
| 225° | Sparse | ❌ Degraded | Mirror of 135°, significant gaps |
| 270° (Side) | Medium | ⚠️ Thinning | Better than opposite side (more visible from camera) |
| 315° | Dense | ✅ Good | Approaching camera-facing side |

**Gap Coverage Estimate:** ~55% of 360° sphere has usable geometry.
**Primary missing region:** 120° arc centered at 180° (back).
**Gap Detection Results:** 12 gap clusters identified, largest at back of head.

---

## Video: DAVIS/bear (480p, 82 frames)

### Rotation Analysis

| Angle | Density | Status | Notes |
|-------|---------|--------|-------|
| 0° (Front) | Dense | ✅ Good | Bear face and front legs clear |
| 45° | Dense | ✅ Good | Side profile well-reconstructed |
| 90° (Side) | Medium | ⚠️ Thinning | Belly starting to thin |
| 135° | Sparse | ❌ Degraded | Rear haunches fragmentary |
| 180° (Back) | Nearly empty | ❌ Critical | Tail area and back ~65% missing |
| 225° | Sparse | ❌ Degraded | Mirror of 135° |
| 270° (Side) | Medium | ⚠️ Thinning | Similar to 90° |
| 315° | Dense | ✅ Good | Approaching front again |

**Gap Coverage Estimate:** ~58% of 360° sphere.
**Primary missing region:** 110° arc centered at 180°.
**Gap Detection Results:** 8 gap clusters, largest along spine/back.

---

## Video: DAVIS/parkour (480p, 100 frames)

### Rotation Analysis

| Angle | Density | Status | Notes |
|-------|---------|--------|-------|
| 0° (Front) | Dense | ✅ Good | Person mid-movement, well-captured |
| 90° (Side) | Medium-Dense | ⚠️ Okay | Motion blur causes some artifacts |
| 180° (Back) | Nearly empty | ❌ Critical | ~75% missing — fast motion worsens gaps |
| 270° (Side) | Medium | ⚠️ Thinning | Better than 90° (more camera visibility) |

**Gap Coverage Estimate:** ~50% of 360°.
**Note:** Fast motion makes gaps MORE severe — reconstruction confidence drops at edges.

---

## Summary Statistics

| Video | 360° Coverage | Gap Clusters | Largest Gap (m³) | Primary Gap Region |
|-------|---------------|--------------|-------------------|--------------------|
| v01_person_walk | 55% | 12 | 0.0042 | Back of head/torso |
| DAVIS/bear | 58% | 8 | 0.0031 | Back/spine |
| DAVIS/parkour | 50% | 15 | 0.0056 | Entire rear hemisphere |

## Key Observations

1. **Gap location is predictable:** Always opposite the camera's viewing direction.
2. **Gap severity correlates with:**
   - Object complexity (people > simple objects)
   - Camera motion (static camera → worse gaps)
   - Video length (more frames → slightly better coverage)
3. **Transition zone (90°-135°):** Geometry degrades gradually, not abruptly.
4. **Gap detection script** reliably identifies these regions with voxel-based analysis.

## Implications for BARF

- **Target: Fill 35-50% of the sphere** with AI-generated geometry
- **Priority order:** Back (180°) → sides (135°, 225°) → transition zones
- **Temporal aspect:** Gaps are consistent across frames (same regions missing in every frame) — this actually helps temporal consistency
