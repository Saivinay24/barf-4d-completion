# BARF Project: Sprint Overview (Feb 14-28, 2026)

## What We're Building

**BARF (Binarily Augmented Reality Footage)** — the generative 4D completion layer that fills the gaps in monocular video reconstructions.

**The Problem:**
- Google D4RT can turn a phone video into a 4D reconstruction (3D + time)
- BUT it only reconstructs what the camera SAW
- If you film someone from the front, the back of their head is EMPTY
- You can't "walk behind" them in VR — there's nothing there

**Our Solution:**
- Use AI generation (diffusion models) to create the missing back-sides
- Ensure temporal consistency (no flickering across frames)
- Deliver a complete 360° 4D scene you can explore in VR

## Team Structure

| Role | Person | What They Do |
|------|--------|-------------|
| 🎯 **Vinay** | Lead | Architecture, integration, temporal consistency research |
| 📚 **Shrit** | Repo Hunter | Clone & run CAT4D/Vivid4D/NeoVerse, produce benchmarks |
| 📦 **Aditya** | Data Engineer | DAVIS/Sintel datasets + depth maps + COLMAP |
| 🔍 **Aryan** | 4D Reconstruction | Run 4DGaussians, produce actual partial reconstructions |
| 🎨 **Tanisha** | Novel View Generator | Run Zero123++/SV3D, generate missing views |
| 🖥️ **Palak** | Viewer Engineer | Fork splat viewer, add gap visualization |

## Remember: Don't Build. Fork. Run. Produce.

**Every person produces a working artifact by end of Week 1** — not tutorials, but SOTA output on real data.

## Timeline

```
Week 1 (Feb 14-21): CLONE + RUN + PRODUCE OUTPUTS
Week 2 (Feb 22-28): INTEGRATE + BENCHMARK + DEMO PREP

Feb 28: Demo Day — 20-minute presentation
```

## GPU Access

We need GPUs for R1, R3, R4:
contact me for access. 

## Daily Standup (15 min, non-negotiable)

**Format — each person, 30 seconds:**
1. "I got [X] working" or "I'm stuck on [Y]"
2. Lead unblocks immediately or reassigns

**Rule:** If stuck for more than 2 hours → switch to helping someone else while I fix your blocker.

## Demo Day (Feb 28)

**presentation:**
1. Show phone video of someone walking
2. Show R3's 4D reconstruction — rotate around, point out gaps
3. Show R4's AI-generated back-views
4. Show R5's web viewer: before (gaps) vs after (filled), 360° rotation
5. Show my pipeline: video → reconstruction → gap detection → generation → viewer
6. Show R1's benchmark: comparison vs CAT4D/Vivid4D
7. Show next steps: VR integration (once Quest hardware arrives)


## Your Individual Task File

Each person has a detailed task file:
- `R1_repo_hunter_shrit.md` — Shrit: Clone SOTA repos, run benchmarks
- `R2_data_engineer_aditya.md` — Aditya: Datasets + depth + COLMAP
- `R3_4d_reconstruction_aryan.md` — Aryan: Run 4DGaussians
- `R4_novel_view_generator_tanisha.md` — Tanisha: Run Zero123++/SV3D
- `R5_viewer_engineer_palak.md` — Palak: Fork splat viewer
- `R6_vr_engineer_alankryt.md` — Alankryt: VR integration
**Read your file. Execute. Ship.**
