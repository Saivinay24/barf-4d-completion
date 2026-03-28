# BARF 4D — Web Viewer

Web-based point cloud viewer for the BARF 4D completion pipeline. Loads `.ply` files, visualises gap regions, and supports 4D timeline playback with before/after comparison.

## Running Locally

```bash
cd viewer/
python3 -m http.server 8000
# Open http://localhost:8000
```

## Loading Data

| Action | How |
|--------|-----|
| Load partial reconstruction | Drag & drop one or more `.ply` files onto the viewer |
| Load 4D frame sequence | Drag & drop **multiple** `.ply` files — sorted by filename, played as a timeline |
| Load completed reconstruction | Click **Load Completed** button, then select `.ply` file(s) |
| Load gaps JSON | Click **Load Gaps JSON** button, or drag & drop a `.json` file |

The bundled `gaps.json` and `test.ply` are auto-loaded when served locally via HTTP.

## Controls

### Mouse / Touch
| Input | Action |
|-------|--------|
| Left drag | Orbit |
| Scroll wheel | Zoom |
| Right drag | Pan |
| Pinch (touch) | Zoom |

### Keyboard
| Key | Action |
|-----|--------|
| `G` | Toggle gap overlay on/off |
| `S` | Toggle split view (left = partial + gaps, right = completed) |
| `C` | Toggle between partial and completed view (full screen) |
| `Space` | Play / pause timeline |
| `← →` | Step one frame back / forward |
| `R` | Reset camera |

### Buttons (bottom bar)
| Button | Action |
|--------|--------|
| Gaps ON/OFF | Show/hide red gap spheres |
| Split View | Side-by-side comparison |
| Load Completed | Load completed `.ply` frame(s) for comparison |
| Load Gaps JSON | Load gap region data from a `.json` file |
| Reset View | Re-centre camera |

## PLY Format Support

The parser reads the PLY header to auto-detect vertex layout and supports:

- **Standard RGB** — `r`/`g`/`b` (uint8, 0–255) or `red`/`green`/`blue`
- **4DGaussians / 3DGS** — `f_dc_0`, `f_dc_1`, `f_dc_2` (SH DC coefficients, converted to RGB)
- **Any property order** — stride and offsets computed from the header
- **Large files** — uses typed arrays throughout; no intermediate allocations

## gaps.json Format

```json
[
  { "center": [x, y, z], "size": 1.0 },
  ...
]
```

`size` controls the sphere radius: `radius = cbrt(size) * 0.15`.

## File Structure

```
viewer/
├── index.html       ← Main viewer (all logic inline, no build step)
├── gaps.json        ← Sample gap data (auto-loaded)
├── test.ply         ← Sample point cloud for testing
├── ply_loader.js    ← Standalone PLY loader (reference; logic is inlined in index.html)
├── gap_viz.js       ← Gap sphere renderer (reference; logic is inlined in index.html)
└── split_view.js    ← Split-view reference code (integrated into render loop in index.html)
```

## Demo Day Script (2 min)

1. Open viewer in browser (`python3 -m http.server 8000`)
2. Drag R3's `.ply` reconstruction onto the viewer → point cloud appears
3. Drag `gaps.json` onto the viewer → red spheres mark hollow regions
4. Orbit 360° to show gaps visually
5. Press `C` to switch to completed view → gaps filled by BARF
6. Press `S` for split-view — left shows gaps, right shows completion side-by-side
7. Drag multiple frames → scrub the timeline slider or press Space to play 4D animation
