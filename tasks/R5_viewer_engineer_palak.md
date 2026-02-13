# R5 — Viewer Engineer

**Read `00_README.md` first for project overview.**

## Your Role

Don't build a viewer from scratch. Fork the best Gaussian Splatting viewer and extend it.

## Week 1: Fork + Extend Viewer

### Day 1 (Feb 14): Fork Existing Viewer
```bash
# Option 1: antimatter15/splat (most popular)
git clone https://github.com/antimatter15/splat
cd splat

# OR Option 2: gsplat.js (more actively maintained)
git clone https://github.com/dylanebert/gsplat.js
cd gsplat.js

# OR Option 3: PlayCanvas SuperSplat
git clone https://github.com/playcanvas/super-splat
```

**Try running it locally:**
```bash
# Most use simple HTTP server
python3 -m http.server 8000
# Open http://localhost:8000 in browser
```

**Load a sample .splat file** (many repos include examples).

**Output:** Forked viewer running locally with sample data

### Day 2-3 (Feb 15-16): Add PLY Support + Gap Overlay
Most viewers support `.splat` format. Add `.ply` support if missing:

```javascript
// viewer/ply_loader.js
async function loadPLY(url) {
    const response = await fetch(url);
    const buffer = await response.arrayBuffer();
    
    // Parse PLY format (header + binary data)
    const text = new TextDecoder().decode(buffer.slice(0, 1000));
    const headerEnd = text.indexOf('end_header') + 'end_header'.length + 1;
    
    // Read vertex count from header
    const vertexCountMatch = text.match(/element vertex (\d+)/);
    const vertexCount = parseInt(vertexCountMatch[1]);
    
    // Parse binary vertex data (x, y, z, r, g, b)
    const dataView = new DataView(buffer, headerEnd);
    const points = [];
    const colors = [];
    
    for (let i = 0; i < vertexCount; i++) {
        const offset = i * 24;  // Assuming float32 for xyz, uint8 for rgb
        points.push(
            dataView.getFloat32(offset, true),
            dataView.getFloat32(offset + 4, true),
            dataView.getFloat32(offset + 8, true)
        );
        colors.push(
            dataView.getUint8(offset + 12) / 255,
            dataView.getUint8(offset + 13) / 255,
            dataView.getUint8(offset + 14) / 255
        );
    }
    
    return { points, colors };
}
```

**Add gap visualization:**
Load R3's `gaps.json` and render gaps as red spheres/highlights.

```javascript
// viewer/gap_viz.js
async function loadGaps(gapsJsonUrl) {
    const response = await fetch(gapsJsonUrl);
    const gaps = await response.json();
    
    // For each gap, create a red sphere at gap.center
    gaps.forEach(gap => {
        const sphere = createSphere(gap.center, Math.cbrt(gap.size) * 0.1);
        sphere.material.color = new THREE.Color(1, 0, 0);  // Red
        sphere.material.opacity = 0.5;
        sphere.material.transparent = true;
        scene.add(sphere);
    });
}

function createSphere(center, radius) {
    const geometry = new THREE.SphereGeometry(radius, 16, 16);
    const material = new THREE.MeshBasicMaterial();
    const sphere = new THREE.Mesh(geometry, material);
    sphere.position.set(center[0], center[1], center[2]);
    return sphere;
}
```

**Output:** Viewer loads PLY + visualizes gaps

### Day 4-5 (Feb 17-18): Split-View Mode
Add before/after comparison:

```javascript
// viewer/split_view.js
let splitMode = false;
let originalScene = null;
let completedScene = null;

function toggleSplitView() {
    splitMode = !splitMode;
    
    if (splitMode) {
        // Render two viewports side by side
        renderer.setScissorTest(true);
        
        // Left half: original (with gaps)
        renderer.setViewport(0, 0, canvas.width / 2, canvas.height);
        renderer.setScissor(0, 0, canvas.width / 2, canvas.height);
        renderer.render(originalScene, camera);
        
        // Right half: completed (gaps filled)
        renderer.setViewport(canvas.width / 2, 0, canvas.width / 2, canvas.height);
        renderer.setScissor(canvas.width / 2, 0, canvas.width / 2, canvas.height);
        renderer.render(completedScene, camera);
        
        renderer.setScissorTest(false);
    } else {
        // Normal full-screen view
        renderer.setViewport(0, 0, canvas.width, canvas.height);
        renderer.render(currentScene, camera);
    }
}

// Bind to keyboard
document.addEventListener('keydown', (e) => {
    if (e.key === 's') {
        toggleSplitView();
    }
});
```

**Output:** Split-view mode working (press 'S' to toggle)

### Day 6-7 (Feb 19-20): Timeline Slider for 4D
Load multiple `.ply` files (one per frame) and scrub through them:

```html
<!-- viewer/index.html -->
<div id="timeline-controls">
    <input type="range" id="frame-slider" min="0" max="100" value="0">
    <span id="frame-display">Frame: 0 / 100</span>
    <button id="play-btn">Play</button>
</div>
```

```javascript
// viewer/timeline.js
let frames = [];  // Array of loaded point clouds
let currentFrame = 0;
let playing = false;

// Load all frames
async function loadFrameSequence(urlPattern) {
    // urlPattern: "reconstructions/v01_person_walk/frames_{i:04d}.ply"
    for (let i = 0; i < 100; i++) {
        const url = urlPattern.replace('{i:04d}', i.toString().padStart(4, '0'));
        const frame = await loadPLY(url);
        frames.push(frame);
    }
    
    document.getElementById('frame-slider').max = frames.length - 1;
}

// Update displayed frame
function setFrame(frameIndex) {
    currentFrame = Math.max(0, Math.min(frameIndex, frames.length - 1));
    
    // Update 3D scene with this frame's point cloud
    updateScene(frames[currentFrame]);
    
    document.getElementById('frame-display').textContent = `Frame: ${currentFrame} / ${frames.length}`;
}

// Slider interaction
document.getElementById('frame-slider').addEventListener('input', (e) => {
    setFrame(parseInt(e.target.value));
});

// Play/pause
document.getElementById('play-btn').addEventListener('click', () => {
    playing = !playing;
    if (playing) {
        playAnimation();
    }
});

function playAnimation() {
    if (!playing) return;
    
    currentFrame = (currentFrame + 1) % frames.length;
    setFrame(currentFrame);
    
    requestAnimationFrame(playAnimation);
}
```

**Output:** 4D playback with timeline slider

---

## Week 2: Real Data + Integration + Polish

### Day 8-10 (Feb 22-24): Load R3's Real Data
- Get R3's actual reconstructions (`.ply` files)
- Get R3's gap JSON files
- Load them in your viewer
- Verify everything works

```javascript
// Example usage
loadPLY('../../reconstructions/v01_person_walk/frames_0000.ply');
loadGaps('../../gap_detection/gaps_v01_0000.json');
```

**Output:** Viewer showing real project data

### Day 11-12 (Feb 25-26): Before/After Toggle
Add keyboard shortcut to toggle between partial (R3's output) and completed (R3 + R4's generated fills):

```javascript
let showCompleted = false;

document.addEventListener('keydown', (e) => {
    if (e.key === 'c') {
        showCompleted = !showCompleted;
        
        if (showCompleted) {
            loadPLY('../../reconstructions/v01_person_walk_completed/frames_0000.ply');
        } else {
            loadPLY('../../reconstructions/v01_person_walk/frames_0000.ply');
        }
    }
});

// Add UI indicator
const indicator = document.createElement('div');
indicator.id = 'mode-indicator';
indicator.style.position = 'absolute';
indicator.style.top = '10px';
indicator.style.right = '10px';
indicator.style.padding = '10px';
indicator.style.background = 'rgba(0,0,0,0.7)';
indicator.style.color = 'white';
indicator.textContent = showCompleted ? 'COMPLETED' : 'PARTIAL';
document.body.appendChild(indicator);
```

**Output:** Toggle between before/after with visual proof of BARF's fill

### Day 13-14 (Feb 27-28): Polish + Demo Prep
- Performance optimize (reduce point count if laggy)
- Add loading indicator
- Add instructions overlay
- Record screen demo showing:
  - Load partial reconstruction
  - Rotate 360° (show gaps)
  - Load gaps visualization (red highlights)
  - Toggle to completed version
  - Scrub through timeline
- Write `viewer/README.md`

**Output:** Polished demo-ready viewer

---

## Your Deliverables by Feb 28

```
✅ Forked Gaussian Splatting viewer running locally
✅ PLY loading support (if not native)
✅ Gap overlay visualization (red highlights)
✅ Split-view OR before/after toggle
✅ Timeline slider for 4D playback
✅ Loaded with real data from R3
✅ Screen recording demo
```

---

## Demo Day — Your Part (2 min)

Show:
1. "Here's our web viewer" (load page)
2. "Load R3's reconstruction" (show point cloud)
3. "Rotate to see gaps" (360° spin, gaps visible)
4. "Toggle to see BARF's completion" (press C, gaps filled)
5. "Scrub through time" (timeline slider)

**Message:** "Visual proof that BARF fills the gaps."
