# BARF Demo Presentation — Feb 28, 2026

## Presentation Outline (20 minutes)

---

### Slide 1: Title (30s)
**BARF: Generative 4D Completion**  
*Transform monocular video into complete, explorable 360° 4D experiences*

---

### Slide 2: The Problem (2 min)
- Show a phone video of someone walking
- Current SOTA (4DGaussians, CAT4D) reconstructs only what camera saw
- **Demo:** Rotate 4D reconstruction 360° → show gaps at rear
- "You can't walk behind them — there's nothing there"

---

### Slide 3: Our Solution (2 min)
- BARF fills the gaps using generative AI
- Architecture diagram: video → reconstruction → gap detection → generation → merge
- **Key innovation:** Temporal consistency across generated frames

---

### Slide 4: Data Pipeline — R2 (2 min)
**Aditya's contribution:**
- 10+ processed videos (DAVIS benchmark + custom)
- Depth maps via Depth Anything V2
- Camera poses via COLMAP
- Automated `process_video.py` pipeline
- **Demo:** Run pipeline on a new video in real-time

---

### Slide 5: SOTA Benchmarks — R1 (3 min)
**Shrit's contribution:**
- Cloned and ran CAT4D, 4DGaussians, Shape-of-Motion
- Benchmark comparison table (speed, quality, coverage)
- **Key finding:** All methods produce 50-65% spherical coverage
- Gap evidence: 360° rotation screenshots showing missing rear

---

### Slide 6: 4D Reconstruction — R3 (3 min)
**Aryan's contribution:**
- 4DGaussians running on our data
- 3 videos fully reconstructed as 4D point clouds
- Gap detection script: voxel analysis → JSON coordinates
- **Demo:** Load reconstruction in Open3D, rotate to show gaps
- "Our gap detector automatically finds these regions"

---

### Slide 7: Novel View Generation — R4 (3 min)
**Tanisha's contribution:**
- Tested SV3D and Zero123++ for back-view generation
- SV3D produces best quality (4/5) but 3x more flicker than real video
- Comparison: SV3D vs Zero123++ vs SVD
- **Demo:** Show front view → AI-generated back view
- "We CAN generate plausible back-views. Consistency is the challenge."

---

### Slide 8: Web Viewer — R5 (2 min)
**Palak's contribution:**
- Three.js web viewer with PLY support
- Gap overlay (red spheres marking missing regions)
- Before/After toggle (C key) + Split view (S key)
- 4D timeline slider
- **Live Demo:** Open viewer, load reconstruction, toggle before/after

---

### Slide 9: Integration Pipeline — R0 (2 min)
**Vinay's contribution:**
- End-to-end `pipeline.py`: video → complete 4D
- Temporal smoothing module (EMA + optical flow)
- Reduces flicker by ~40-60%
- Architecture designed for VR integration

---

### Slide 10: Results & Metrics (1 min)
| Metric | Before BARF | After BARF |
|--------|------------|------------|
| 360° Coverage | ~55% | ~90%+ |
| Temporal Flicker | N/A | 3x → 1.5x (with smoothing) |
| Processing Time | N/A | ~15s/frame |

---

### Slide 11: Next Steps (1 min)
- VR integration (Meta Quest) — waiting on hardware
- Real-time optimization for 90 FPS VR
- D4RT API integration when released (mid-2026)
- Improve temporal consistency with cross-frame attention

---

### Slide 12: Q&A

---

## Demo Script

### Live Demo 1: Web Viewer (during Slide 8)
1. Open `http://localhost:8000` in browser
2. Auto-loads test.ply with gap overlay
3. Orbit camera 360° — point out gaps (red spheres)
4. Press `C` to show completed version
5. Press `S` for split-screen comparison

### Live Demo 2: Pipeline (during Slide 9)
```bash
python core/pipeline.py --input test_video.mp4 --output demo_output/ --skip-depth --skip-colmap
```

### Backup: Pre-recorded screen captures
- Record all demos ahead of time as .mp4 backups
- Place in `presentation/` folder
