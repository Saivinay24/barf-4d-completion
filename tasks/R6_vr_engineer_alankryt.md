# R6 — VR Developer

**Read `00_README.md` first for project overview.**

## Your Role

Get 4D Gaussian Splatting running in Quest VR. Fork, don't build from scratch.

## Tools You Need
- Meta Quest 2/3/Pro
- Quest browser OR SideQuest for local dev
- Three.js + WebXR knowledge (or willingness to learn fast)

## Week 1: Find + Fork VR Viewer

### Day 1 (Feb 14): Research Existing Viewers
Search for:
- "WebXR Gaussian Splatting"
- "Three.js WebXR splat viewer"
- Check: `luma.ai` (they have WebXR demos), `playcanvas/super-splat`, `splatapult`

**Find ONE that:**
1. Works in Quest browser
2. Has WebXR support
3. Can load `.splat` or `.ply` files

```bash
# Example: if you find a repo
git clone [WEBXR_SPLAT_VIEWER_URL]
cd vr_viewer
python3 -m http.server 8000
```

Test on your computer first, then on Quest browser.

**Output:** Forked WebXR viewer running

### Day 2-3 (Feb 15-16): Get Running on Quest
**Test on Quest:**
1. Find your computer's local IP: `ifconfig` (macOS/Linux) or `ipconfig` (Windows)
2. On Quest browser, navigate to `http://[YOUR_IP]:8000`
3. Click "Enter VR" button
4. Verify you can see the scene in VR

**Common issues:**
- HTTPS required: Use `ngrok` to tunnel
  ```bash
  ngrok http 8000
  # Use the https:// URL on Quest
  ```
- CORS errors: Add headers to your server

**Output:** Basic VR viewer working on Quest

### Day 4-5 (Feb 17-18): Add Locomotion
Critical for "walking behind" the reconstruction.

```javascript
// vr_viewer/locomotion.js
import * as THREE from 'three';

let player = new THREE.Object3D();
player.position.set(0, 1.6, 5);  // Start 5m away
scene.add(player);
camera.parent = player;  // Camera moves with player

// Controller input
const controllers = [];

function setupControllers() {
    for (let i = 0; i < 2; i++) {
        const controller = renderer.xr.getController(i);
        controller.addEventListener('selectstart', onSelectStart);
        controller.addEventListener('selectend', onSelectEnd);
        player.add(controller);
        controllers.push(controller);
    }
}

// Thumbstick movement
function updateLocomotion() {
    const session = renderer.xr.getSession();
    if (!session) return;
    
    for (const source of session.inputSources) {
        if (source.gamepad) {
            const axes = source.gamepad.axes;
            
            // Left thumbstick: move forward/backward, strafe
            if (source.handedness === 'left') {
                const speedForward = -axes[3] * 0.05;  // Y axis
                const speedStrafe = axes[2] * 0.05;    // X axis
                
                // Move in camera's forward direction
                const forward = new THREE.Vector3(0, 0, -1);
                forward.applyQuaternion(camera.quaternion);
                forward.y = 0;  // Keep horizontal
                forward.normalize();
                
                const right = new THREE.Vector3(1, 0, 0);
                right.applyQuaternion(camera.quaternion);
                right.y = 0;
                right.normalize();
                
                player.position.addScaledVector(forward, speedForward);
                player.position.addScaledVector(right, speedStrafe);
            }
            
            // Right thumbstick: snap turn
            if (source.handedness === 'right' && Math.abs(axes[2]) > 0.8) {
                // Snap turn 30 degrees
                if (!source.turning) {
                    player.rotateY(axes[2] > 0 ? -Math.PI / 6 : Math.PI / 6);
                    source.turning = true;
                }
            } else {
                source.turning = false;
            }
        }
    }
}

// Call in animation loop
renderer.setAnimationLoop(() => {
    updateLocomotion();
    renderer.render(scene, camera);
});
```

**Test:** Can you walk around the scene? Can you rotate?

**Output:** Smooth VR locomotion working

### Day 6-7 (Feb 19-20): File Loading via URL Param
Allow specifying which `.splat` file to load:

```javascript
// vr_viewer/file_loader.js
const urlParams = new URLSearchParams(window.location.search);
const fileUrl = urlParams.get('file') || 'default.splat';

loadSplatFile(fileUrl).then(() => {
    console.log('Loaded:', fileUrl);
});

// Usage: http://localhost:8000/?file=../../reconstructions/v01_person_walk/frames_0000.ply
```

Test with R3's outputs (once they have them).

**Output:** Can load different files via URL

---

## Week 2: 4D Playback + Before/After Demo

### Day 8-10 (Feb 22-24): 4D Playback in VR
Load a sequence of `.ply` frames and play through them:

```javascript
// vr_viewer/timeline_vr.js
let frames = [];
let currentFrame = 0;
let playing = false;

// Load frame sequence
async function loadFrameSequence(pattern, count) {
    for (let i = 0; i < count; i++) {
        const url = pattern.replace('{i}', i.toString().padStart(4, '0'));
        const frame = await loadPLY(url);
        frames.push(frame);
    }
}

// Controller button to play/pause
function onSelectStart(event) {
    playing = !playing;
    if (playing) {
        playFrames();
    }
}

function playFrames() {
    if (!playing) return;
    
    currentFrame = (currentFrame + 1) % frames.length;
    updateSceneWithFrame(frames[currentFrame]);
    
    setTimeout(playFrames, 1000 / 30);  // 30 FPS playback
}
```

**Add visual indicator in VR:**
- Floating panel showing current frame number
- Play/pause icon

**Output:** 4D playback working in VR

### Day 11-12 (Feb 25-26): The Killer Demo
**Before/After in VR:**
1. Load R3's partial reconstruction
2. Walk around it in VR
3. Walk to the back — see the gap (empty space)
4. Press controller button → load completed version (with R4's fills)
5. Same back view now has geometry

```javascript
// vr_viewer/before_after.js
let showCompleted = false;

function toggleCompletion(event) {
    showCompleted = !showCompleted;
    
    const fileUrl = showCompleted 
        ? '../../reconstructions/v01_person_walk_completed/frames_0000.ply'
        : '../../reconstructions/v01_person_walk/frames_0000.ply';
    
    loadPLY(fileUrl);
    
    // Update floating text panel
    updateTextPanel(showCompleted ? 'COMPLETED' : 'PARTIAL');
}

// Bind to controller button
controller.addEventListener('selectstart', toggleCompletion);
```

**Output:** VR demo showing BARF's value

### Day 13-14 (Feb 27-28): Record Demo + Polish
**Record from Quest:**
- Use Quest's built-in screen recording (hold Oculus + trigger)
- OR use Quest Link + OBS to record from PC

**Demo script:**
1. Put on Quest
2. Start recording
3. Load partial reconstruction
4. Walk around front (looks good)
5. Walk to back (gaps visible — just empty space)
6. Press button to load completed version
7. Walk to back again (now filled!)
8. Scrub through timeline
9. Stop recording

**Output:** Quest screen recording showing before/after

---

## Your Deliverables by Feb 28

```
✅ WebXR Gaussian Splatting viewer running on Quest
✅ Locomotion (thumbstick movement, snap turning)
✅ 4D playback with controller controls
✅ Before/after toggle in VR
✅ Real data from R3 loaded and viewable
✅ Quest screen recording demo
```

---

## What to Do If Stuck

**If no WebXR splat viewer exists (Day 2):**
- Use Three.js WebXR template + add point cloud rendering
- Or use A-Frame WebXR + custom point cloud component

**If Quest browser doesn't support WebXR:**
- Use Oculus Browser (native app)
- Or build for SideQuest (more advanced)

**If performance is terrible:**
- Reduce point count (subsample)
- Use lower-res reconstruction
- Optimize geometry (instancing, etc.)

---

## Demo Day — Your Part (3 min)

**Put on the Quest in front of everyone. Screen-cast to laptop.**

Show:
1. "I'm in VR looking at R3's reconstruction"
2. "Front view looks good" (walk around front)
3. "Now I'll walk behind..." (walk to back, gap visible)
4. "This is the gap BARF needs to fill" (point at empty space)
5. "Press button → load completed version"
6. "Now the back is filled!" (walk around, now has geometry)
7. "Scrub through time" (timeline playback)

**Message:** "You can literally SEE the gap in VR, then see BARF fill it."
