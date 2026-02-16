# Research Outputs — Shrit (R1)

This folder contains benchmark outputs from running SOTA 4D reconstruction repos.

## What Goes Here

- `benchmark_outputs/` — Output files from CAT4D, Vivid4D, NeoVerse
- `benchmark_comparison.md` — Speed, quality, format comparison table
- `gap_evidence.md` — Screenshots showing gaps in existing methods
- Installation notes and fixes for getting repos running

## Current Status

### 16th Feburary

While setting up Cap4d, several issues arose due to compatibility and package support. Here’s a summary of the troubleshooting and solutions applied:

1. **`decord`**
   - **Issue:** No macOS ARM64 wheels available.
   - **Solution:** Replaced `decord` with `eva-decord` in `requirements.txt`. `eva-decord` has the same API and supports ARM64.

2. **`chumpy`**
   - **Issue:** Fails under pip’s build isolation (requires `pip` and `wheel` during build).
   - **Solution:** Removed from main requirements. Added `wheel` to `requirements.txt`. Added a secondary step:
     ```bash
     pip install -r requirements-chumpy.txt --no-build-isolation
     ```
     (This step is scripted to run automatically in `install.sh` after the primary install.)

3. **`xformers`**
   - **Issue:** Required PyTorch ≥ 2.10, but only 2.8 was available. The code runs without it.
   - **Solution:** `xformers` was removed from `requirements.txt`.

4. **`triton`**
   - **Issue:** Not supported on macOS (only available for Linux/NVIDIA).
   - **Solution:** Removed `triton` from `requirements.txt`.

**Installation Summary:**  
You only need to run:

```bash
bash install.sh
```

This script installs all main dependencies from `requirements.txt`, then installs `chumpy` using `--no-build-isolation`.
