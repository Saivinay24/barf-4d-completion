# Gap Evidence - Why BARF Matters

This document summarizes visible reconstruction gaps in currently available benchmark artifacts.

## CAT4D / CAP4D Output Analysis

Source assets:
- `research/benchmark_outputs/cat4d/front_view.png`
- `research/benchmark_outputs/cat4d/side_view.png`
- `research/benchmark_outputs/cat4d/back_view.png`

### Front view (~0 deg)
![CAT4D front](research/benchmark_outputs/cat4d/front_view.png)
Observed: frontal face/subject structure is stable and detailed.

### Side view (~90 deg)
![CAT4D side](research/benchmark_outputs/cat4d/side_view.png)
Observed: silhouette begins to thin around boundaries.

### Rear view (~180 deg)
![CAT4D rear](research/benchmark_outputs/cat4d/back_view.png)
Observed: geometry confidence drops in unseen regions; rear contour appears weaker.

Conclusion: CAT4D/CAP4D appears strongest near observed frontal views and weaker in less observed rear geometry.

BARF opportunity: fill missing/uncertain rear structure using generative priors.

## Vidu4D Output Analysis

Source assets:
- `research/benchmark_outputs/Vidu4D/front_view.png`
- `research/benchmark_outputs/Vidu4D/back_view.png`
- `research/benchmark_outputs/Vidu4D/rear_probe.png`

### Front view (~0 deg)
![Vidu4D front](research/benchmark_outputs/Vidu4D/front_view.png)
Observed: good photometric quality in visible regions.

### Rear probe (~180 deg approximate)
![Vidu4D rear](research/benchmark_outputs/Vidu4D/rear_probe.png)
Observed: rear/occluded structures are less certain and may appear smoothed or incomplete.

Conclusion: Vidu4D produces compelling visible-side renderings, but unseen-region completeness remains uncertain.

BARF opportunity: improve consistency and completion for occluded/rear geometry.

## NeoVerse Output Analysis

Source assets:
- `research/benchmark_outputs/neoverse/front_view.png`
- `research/benchmark_outputs/neoverse/side_view.png`
- `research/benchmark_outputs/neoverse/back_view.png`

### Front view (~0 deg)
![NeoVerse front](research/benchmark_outputs/neoverse/front_view.png)
Observed: strong fidelity and consistent geometry in showcased view.

### Side view (~90 deg)
![NeoVerse side](research/benchmark_outputs/neoverse/side_view.png)
Observed: good continuity, though strict geometric completeness is hard to verify from single demo trajectory.

### Back view (~180 deg)
![NeoVerse rear](research/benchmark_outputs/neoverse/back_view.png)
Observed: rear quality appears plausible but still needs same-input controlled benchmark for definitive gap quantification.

Conclusion: NeoVerse has high visual quality; controlled local comparisons are still needed to prove full unseen-region robustness.

BARF opportunity: provide explicit rear completion guarantees and measurable reduction in gap regions.

## Next Step To Strengthen Evidence

For each method, run the exact same real videos and capture fixed camera angles (0, 90, 180, 270 deg) from reconstructed outputs. Then annotate per-angle missing regions and compute simple completeness metrics (e.g., occupied area consistency across viewpoints).
