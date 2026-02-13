# BARF: Generative 4D Completion

**Transform monocular video into complete, explorable 360° 4D experiences.**

## What is BARF?

Current 4D reconstruction methods (like Google D4RT, NeoVerse, 4DGaussians) can turn a phone video into a 3D scene... but **only what the camera saw**. Film someone from the front? The back of their head is **empty**. You can't walk behind them in VR.

**BARF (Binarily Augmented Reality Footage)** fills those gaps using generative AI, creating a complete 360° 4D reconstruction you can freely explore.

## The Problem We're Solving

| Current SOTA (D4RT, 4DGS) | BARF (Our Solution) |
|---------------------------|---------------------|
| Reconstructs visible surfaces only | Generates missing back-sides with AI |
| ~180° field of view from camera | Complete 360° explorable scene |
| Empty gaps when you rotate around | Temporally consistent fills |
| Can't "walk behind" objects in VR | Full free-viewpoint navigation |

## Architecture

```
Input: Monocular video (1 camera angle)
   ↓
[1] 4D Reconstruction (4DGaussians / D4RT)
   ↓
[2] Gap Detection (voxel-based)
   ↓
[3] Novel View Generation (Zero123++ / SV3D)
   ↓
[4] Temporal Consistency (sliding denoising) ← OUR INNOVATION
   ↓
Output: Complete 360° 4D scene
```

## Team Structure (Feb 14-28, 2026 Sprint)

| Role | Person | Focus |
|------|--------|-------|
| 🎯 **Vinay (R0)** | Lead | Architecture + Integration + Temporal Consistency |
| 📚 **Shrit (R1)** | Repo Hunter | Run CAT4D/Vivid4D/NeoVerse, benchmark |
| 📦 **Aditya (R2)** | Data Engineer | DAVIS/Sintel + Depth Maps + COLMAP |
| 🔍 **Aryan (R3)** | 4D Reconstruction | Run 4DGaussians, produce partial 4D |
| 🎨 **Tanisha (R4)** | Novel View Gen | Zero123++/SV3D for back-view generation |
| 🖥️ **Palak (R5)** | Viewer Engineer | Web viewer with gap visualization |

## Repository Structure

```
barf-4d-completion/
├── tasks/                  # Individual task assignments (READ YOUR FILE!)
│   ├── 00_README.md       # Project overview
│   ├── R1_repo_hunter_shrit.md  # Shrit's tasks
│   ├── R2_data_engineer_aditya.md # Aditya's tasks
│   ├── R3_4d_reconstruction_aryan.md # Aryan's tasks
│   ├── R4_novel_view_generator_tanisha.md # Tanisha's tasks
│   └── R5_viewer_engineer_palak.md # Palak's tasks
│
├── research/              # Shrit: benchmark outputs, comparisons
├── datasets/              # Aditya: DAVIS, Sintel, depth maps, COLMAP
├── reconstructions/       # Aryan: 4D point clouds, gap detection
├── diffusion_experiments/ # Tanisha: generated views, consistency tests
├── viewer/                # Palak: web-based 3D viewer
├── vr_viewer/             # (Future) VR viewer for Quest
└── core/                  # Vinay: integration pipeline, temporal consistency
```

## Quick Start

### For Team Members

1. **Read your task file:** `tasks/[YOUR_NAME]_[ROLE].md`
2. **Clone this repo:**
   ```bash
   git clone https://github.com/Saivinay24/barf-4d-completion
   cd barf-4d-completion
   ```
3. **Work in YOUR folder only** (to avoid conflicts)
4. **Push daily:**
   ```bash
   git add [YOUR_FOLDER]/
   git commit -m "[YOUR_NAME]: what you did today"
   git pull
   git push
   ```

### Git Workflow Rules

- ✅ **DO:** Only edit files in your assigned folder
- ✅ **DO:** Commit at end of each day with descriptive messages
- ✅ **DO:** Pull before pushing to get others' updates
- ❌ **DON'T:** Edit other people's folders (ask first)
- ❌ **DON'T:** Commit large binary files (use Git LFS or Drive)

## Timeline

**Week 1 (Feb 14-21):** Clone repos, run SOTA methods, produce outputs  
**Week 2 (Feb 22-28):** Integration, benchmarking, demo prep  
**Feb 28:** Final demo presentation

## Tech Stack

- **4D Reconstruction:** 4DGaussians, Shape-of-Motion, (future: D4RT API)
- **Depth Estimation:** Depth Anything V2
- **Camera Poses:** COLMAP
- **Novel View Synthesis:** Zero123++, SV3D, Stable Video Diffusion
- **3D Viewing:** Three.js, antimatter15/splat viewer
- **Temporal Consistency:** Optical flow (RAFT), sliding denoising

## Key Deliverables (Feb 28)

1. Working end-to-end pipeline (video → complete 4D)
2. Benchmark comparison vs CAT4D/Vivid4D/NeoVerse
3. Web viewer showing before/after gap filling
4. Quantitative metrics (gap coverage %, temporal consistency)
5. Demo video + presentation

## Research Questions We're Tackling

1. **Can diffusion models generate plausible back-views from monocular video?**
2. **How do we maintain temporal consistency across generated frames?**
3. **What's the gap between reconstruction-only vs generative completion?**
4. **Is this fast enough for practical VR applications?**

## Future Work (Post-Sprint)

- VR integration (Meta Quest) once hardware arrives
- Real-time optimization for 90 FPS VR
- D4RT API integration when released (mid-2026)
- Explore business models: SaaS, API, plugin marketplace

## Contact

**Lead:** Vinay (Saivinay24)  
**Project Duration:** Feb 14 - Feb 28, 2026  
**License:** TBD (likely MIT for research components)

---

**Philosophy:** Don't build. Fork. Run. Produce.
