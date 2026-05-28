# R4 — Novel View Generator

**Read `00_README.md` first for project overview.**

## Your Role

Generate what the camera DIDN'T see — the AI core of BARF.  
**Needs GPU** — use Colab Pro (T4/A100 works).

## Week 1: Setup + Novel View Generation

### Day 1 (Feb 14): Set Up Zero123++
Open Google Colab, create new notebook `diffusion_experiments/01_zero123++.ipynb`:

```python
# Cell 1: Install
!pip install diffusers transformers accelerate torch pillow

# Cell 2: Load model
from diffusers import DiffusionPipeline
import torch
from PIL import Image

pipe = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2",
    torch_dtype=torch.float16
).to("cuda")

# Cell 3: Test
# Upload a test image (front view of an object/person)
image = Image.open("test_front.jpg").resize((256, 256))
result = pipe(image, num_inference_steps=75).images[0]
result.save("multiview_output.png")
result  # Display in Colab
```

**Run this.** Verify you get 6 novel views in a grid.

**Output:** Zero123++ working in Colab

### Day 2-3 (Feb 15-16): Set Up SV3D
Create `diffusion_experiments/02_sv3d.ipynb`:

```python
# SV3D (Stable Video 3D) — generates orbital video from single image
!pip install diffusers transformers torch

from diffusers import StableVideo3DPipeline
import torch
from PIL import Image

pipe = StableVideo3DPipeline.from_pretrained(
    "stabilityai/sv3d",
    torch_dtype=torch.float16
).to("cuda")

image = Image.open("test_front.jpg").resize((512, 512))

# Generate 21-frame orbital video
frames = pipe(image, num_frames=21, num_inference_steps=25).frames[0]

# Save as GIF
frames[0].save(
    "orbital_video.gif",
    save_all=True,
    append_images=frames[1:],
    duration=100,
    loop=0
)
```

**Test both Zero123++ and SV3D.** Which gives better back-views?

**Output:** SV3D working, comparison of both methods

### Day 4-5 (Feb 17-18): Real Data Test — Back-View Generation
Get frames from R2. Extract person/object using masks.

```python
# 03_backview_test.ipynb
from PIL import Image
import torch
from diffusers import StableVideo3DPipeline

pipe = StableVideo3DPipeline.from_pretrained("stabilityai/sv3d", torch_dtype=torch.float16).to("cuda")

# Load a front-view frame from R2's data
front_view = Image.open("../datasets/custom/v01_person_walk/frames/frame_0050.jpg")

# Crop to person (use R2's mask if available, or manual crop)
# For now, assume centered person
front_view = front_view.crop((400, 100, 1500, 900)).resize((512, 512))

# Generate orbital views
orbital = pipe(front_view, num_frames=21).frames[0]

# Frame 10-11 should be ~180° (back view)
back_view = orbital[10]
back_view.save("generated_backview.png")

# Save full sequence
for i, frame in enumerate(orbital):
    frame.save(f"diffusion_experiments/backview_sequence/frame_{i:02d}.png")
```

**Test on 10 different frames** from R2's videos.

**Output:** `diffusion_experiments/backview_results/` with input-output pairs

### Day 6-7 (Feb 19-20): Video Consistency Test
**Critical:** Do generated frames flicker across time?

```python
# 04_consistency_test.ipynb
import os
from PIL import Image
import torch
import numpy as np

pipe =  StableVideo3DPipeline.from_pretrained("stabilityai/sv3d", torch_dtype=torch.float16).to("cuda")

# Get 30 consecutive frames from R2
frames_dir = "../datasets/custom/v01_person_walk/frames/"
frame_files = sorted(os.listdir(frames_dir))[:30]

generated_backs = []
for frame_file in frame_files:
    front = Image.open(os.path.join(frames_dir, frame_file)).crop((400,100,1500,900)).resize((512,512))
    orbital = pipe(front, num_frames=21).frames[0]
    back_view = orbital[10]  # 180° view
    generated_backs.append(np.array(back_view))
    back_view.save(f"diffusion_experiments/consistency_test/generated_{frame_file}")

# Measure flicker: average pixel difference between consecutive frames
diffs = []
for i in range(1, len(generated_backs)):
    diff = np.abs(generated_backs[i].astype(float) - generated_backs[i-1].astype(float)).mean()
    diffs.append(diff)

flicker_score = np.mean(diffs)
print(f"Average flicker: {flicker_score:.2f}")

# Compare with original video flicker
original_diffs = []
original_frames = [np.array(Image.open(os.path.join(frames_dir, f))) for f in frame_files]
for i in range(1, len(original_frames)):
    diff = np.abs(original_frames[i].astype(float) - original_frames[i-1].astype(float)).mean()
    original_diffs.append(diff)

original_flicker = np.mean(original_diffs)
print(f"Original flicker: {original_flicker:.2f}")
print(f"Generated is {flicker_score / original_flicker:.1f}x more flickery")
```

**Output:** Flicker measurements, generated video sequence

---

## Week 2: Benchmark + Temporal Consistency + Integration

### Day 8-9 (Feb 22-23): Try Stable Video Diffusion for Consistency
Test if **Stable Video Diffusion** (image-to-video model) produces more consistent back-views:

```python
# 05_svd_temporal.ipynb
from diffusers import StableVideoDiffusionPipeline
import torch
from PIL import Image

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16
).to("cuda")

# Input: front view
# Prompt: "rotate 180 degrees to back view"  
# (SVD doesn't directly support prompts, but try conditioning it)

front = Image.open("front_view.jpg").resize((512, 512))
video_frames = pipe(front, num_frames=25).frames[0]

# Check if any frames show back view
for i, frame in enumerate(video_frames):
    frame.save(f"svd_test/frame_{i:02d}.png")
```

Compare SV3D vs SVD for temporal consistency.

**Output:** Comparison report in `diffusion_experiments/methods_comparison.md`

### Day 10-11 (Feb 24-25): Benchmark vs Ground Truth (if available)
If R2 has videos with 360° coverage (e.g., walked around a person), use those for comparison:

```python
# 06_benchmark.ipynb
from PIL import Image
import lpips
import torch

# Load LPIPS metric (perceptual similarity)
lpips_fn = lpips.LPIPS(net='alex').cuda()

# Ground truth: actual back view from R2's data
gt_back = Image.open("../datasets/custom/v01_360/frames/back_view.jpg").resize((512,512))

# Generated: your AI-generated back view
generated_back = Image.open("diffusion_experiments/backview_results/frame_0050_back.png")

# Convert to tensors
gt_tensor = torch.from_numpy(np.array(gt_back)).permute(2,0,1).float() / 255.0
gen_tensor = torch.from_numpy(np.array(generated_back)).permute(2,0,1).float() / 255.0

# Calculate perceptual similarity
similarity = lpips_fn(gt_tensor.unsqueeze(0).cuda(), gen_tensor.unsqueeze(0).cuda())
print(f"LPIPS distance: {similarity.item():.4f}  (lower is better)")
```

**If no ground truth available:** Just document qualitative assessment (does it LOOK plausible?).

**Output:** `diffusion_experiments/benchmark.md` with measurements or qualitative ratings

### Day 12-14 (Feb 26-28): Package for Pipeline + Integration
Create callable function for R0 to use:

```python
# diffusion_experiments/generator.py
"""
Novel view generator for BARF pipeline
"""
import torch
from PIL import Image
from diffusers import StableVideo3DPipeline

class NovelViewGenerator:
    def __init__(self, model="stabilityai/sv3d"):
        self.pipe = StableVideo3DPipeline.from_pretrained(
            model,
            torch_dtype=torch.float16
        ).to("cuda")
    
    def generate_backview(self, front_view_image, angle=180):
        """
        Generate view from specified angle
        angle: 0-360 degrees (0 = front, 180 = back)
        """
        # Ensure PIL Image
        if isinstance(front_view_image, str):
            front_view_image = Image.open(front_view_image)
        
        # Resize to 512x512
        img = front_view_image.resize((512, 512))
        
        # Generate 21-frame orbital sequence
        orbital = self.pipe(img, num_frames=21).frames[0]
        
        # Map angle to frame index
        frame_idx = int((angle / 360) * 21)
        
        return orbital[frame_idx]
    
    def generate_backview_sequence(self, front_view_frames):
        """
        Generate back-views for a sequence of frames
        Attempts to maintain temporal consistency
        """
        results = []
        for frame in front_view_frames:
            back = self.generate_backview(frame, angle=180)
            results.append(back)
        return results

# Test
if __name__ == "__main__":
    gen = NovelViewGenerator()
    front = Image.open("test_front.jpg")
    back = gen.generate_backview(front)
    back.save("test_back.png")
    print("Generated back view!")
```

**Integrate with R0's pipeline.**

**Output:** `diffusion_experiments/generator.py` ready for integration

---

## Your Deliverables by Feb 28

```
✅ Zero123++ AND SV3D running, compared
✅ Back-view generation tested on 10+ images
✅ Video consistency measured (flicker score)
✅ Benchmark vs ground truth (or qualitative assessment)
✅ Callable generator.py for pipeline
✅ All experiments in Colab notebooks (shared links)
```

---

## Demo Day — Your Part (3 min)

Show:
1. "Here's a front-view frame" (show image)
2. "Here's what Zero123++ generates for the back" (show result)
3. "Here's the consistency problem" (play flickering video)
4. "Best method: SV3D with X flicker score" (show metric)
5. "This integrates into the pipeline" (show generator.py call)

**Message:** "We CAN generate back-views. Consistency is the challenge."
