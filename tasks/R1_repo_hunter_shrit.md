# R1 — Repo Hunter + Benchmark Runner

**Read `00_README.md` first for project overview.**

## Your Role

Get SOTA 4D reconstruction repos running. Produce comparison benchmarks.  
**Not reading papers. Running code.**

## Week 1: Clone + Install + Run

### Day 1 (Feb 14 ): Clone CAT4D
```bash
# Search GitHub for CAT4D repo (might be cat4d-project/CAT4D or similar)
git clone [CAT4D_REPO_URL]
cd CAT4D
pip install -r requirements.txt
# Read their README — what GPU do they need? What data format?
```

**Output:** CAT4D repo cloned, dependencies installing

### Day 2 (Feb 15): Clone Vivid4D
```bash
# Search: "Vivid4D improving 4D reconstruction" on GitHub/HuggingFace
git clone [Vivid4D_REPO_URL]
cd Vivid4D
pip install -r requirements.txt
```

**Output:** Vivid4D repo cloned

### Day 3 (Feb 16): Clone NeoVerse
```bash
# Search: "NeoVerse CASIA CreateAI 4D" — might be on GitHub or coming soon
git clone [NeoVerse_REPO_URL]
cd NeoVerse
pip install -r requirements.txt
```

**Output:** NeoVerse repo cloned (or documented if not released yet)

### Day 4-5 (Feb 17-18): Get ONE Running
**This is the hard part.** Pick whichever repo has the best documentation.  
Fight through:
- CUDA version mismatches
- Missing dependencies
- Broken install scripts
- Unclear README

**Document every fix you make** — save commands that worked.

```bash
# Example fixes you might need:
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install ninja  # often missing
export CUDA_HOME=/usr/local/cuda-11.8
```

**Goal:** Get at least ONE repo to produce output on a sample video.

**Output:** One successful run with output files

### Day 6-7 (Feb 19-20): Run on Real Data
- Get sample videos from R2 (they shouldhave DAVIS downloaded by now)
- Run the working repo on 3 different videos
- Save all outputs to `research/benchmark_outputs/[method_name]/`

**Output:** 3 videos processed, outputs saved

---

## Week 2: Benchmarks + Comparison + Gap Evidence

### Day 8-9 (Feb 22-23): Try Second Repo
If time allows, get a SECOND repo running.  
Compare outputs side-by-side.

**Output:** Comparison screenshots/video

### Day 10-11 (Feb 24-25): Benchmark Table
Create `research/benchmark_comparison.md`:

| Method | Speed (sec/frame) | Quality (1-5) | Input Requirements | Output Format | Gaps Present? |
|--------|------------------|---------------|-------------------|---------------|---------------|
| **CAT4D** | [measure] | [rate visually] | Monocular video | .ply / .splat | YES / NO |
| **Vivid4D** | [measure] | [rate] | Monocular video | .ply | YES / NO |
| **4DGaussians** | [from R3] | [rate] | Frames + COLMAP | .ply | YES |

For each method you got running:
- Measure processing time
- Rate visual quality (1-5 subjectively)
- Document input requirements
- Note output format
- **Critical:** Does it have gaps in unseen regions?

**Output:** `research/benchmark_comparison.md`

### Day 12-14 (Feb 26-28): Gap Evidence Document
This is CRITICAL for proving BARF's value.

Create `research/gap_evidence.md`:

For each method's output:
1. Load the 4D reconstruction
2. Rotate the view 360° (use Open3D or R5's viewer)
3. **Screenshot angles where gaps are visible**
4. Annotate screenshots: "Missing back of head", "No rear geometry", etc.

Example structure:
```markdown
# Gap Evidence — Why We Need BARF

## CAT4D Output Analysis

### Video: DAVIS/bear

**Front view (0°):**
![front view](research/gap_evidence/cat4d_bear_front.png)
✅ Reconstruction looks good

**Side view (90°):**
![side view](research/gap_evidence/cat4d_bear_side.png)
⚠️ Starting to thin out

**Rear view (180°):**
![rear view](research/gap_evidence/cat4d_bear_rear.png)
❌ MAJOR GAPS — barely any geometry

**Conclusion:** CAT4D only reconstructs ~180° field of view from camera.  
**BARF's opportunity:** Fill the missing 180° with generative AI.
```

Repeat for all methods you tested.

**Output:** `research/gap_evidence.md` with annotated screenshots

---

## Your Deliverables by Feb 28

```
✅ At least 1 SOTA 4D repo running end-to-end (preferably 2)
✅ Benchmark outputs on 3+ real videos
✅ Comparison table with measurements
✅ Gap evidence document with 360° rotation screenshots
✅ Documentation of installation fixes (for future reference)
```

---

## What to Do If Stuck

**If CAT4D won't install (Day 4):**
- Try Vivid4D instead
- Or ask R3 to share their 4DGaussians outputs (reuse theirs)

**If NO repos install:**
- Document what you tried
- Create theoretical comparison from papers
- Help R2 with data processing instead
- Use R3's 4DGaussians outputs for gap analysis

**Success = at least ONE working method + gap evidence.**

---

## Demo Day — Your Part (3 min)

Show:
1. "Here are the SOTA methods we benchmarked" (table)
2. "Here's output from [X method] on our test video" (play/screenshot)
3. "Here's the gap problem" (360° rotation showing missing rear)
4. "This is why BARF is needed" (gap evidence screenshots)

**Message:** "Current methods can do front-view 4D, but they can't fill the back. BARF does."
