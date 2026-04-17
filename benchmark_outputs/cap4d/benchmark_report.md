# CAP4D Quick Benchmark (Fallback Path)

## Status
- Repo used: `cap4d` (`https://github.com/felixtaubner/cap4d`)
- Same input video used for local run attempt: `/Users/shritaake/Desktop/Code/BARF/video.mp4`
- Local run attempt: **blocked** on this machine
- Main blockers:
  - missing `decord` wheel for this local Python setup
  - FLAME asset requirement (`data/assets/flame/flame2023_no_jaw.pkl`) not available locally
  - Pixel3DMM dependency repo/scripts not installed (`$PIXEL3DMM_PATH/scripts/...`)

## Commands Executed (local)
```bash
python3 -m pip install omegaconf pytorch-lightning einops scipy==1.13.1 chumpy gsplat plyfile roma matplotlib
PYTHONPATH=$(realpath "./"):$PYTHONPATH python3 cap4d/inference/generate_images.py --config_path configs/generation/debug.yaml --reference_data_path examples/input/tesla/ --output_path examples/debug_output/tesla/
CAP4D_PATH=$(realpath "./") PIXEL3DMM_PATH=$(realpath "../pixel3dmm") bash scripts/generate_avatar.sh /Users/shritaake/Desktop/Code/BARF/video.mp4 /Users/shritaake/Desktop/Code/BARF/research/benchmark_outputs/cap4d/local_run debug
```

## Errors That Stop Local Execution
```text
ModuleNotFoundError: No module named 'decord'
FileNotFoundError: [Errno 2] No such file or directory: 'data/assets/flame/flame2023_no_jaw.pkl'
python: can't open file '.../pixel3dmm/scripts/run_preprocessing.py': [Errno 2] No such file or directory
```

## Backup Evidence (Official Demo)
- Source page: `https://felixtaubner.github.io/cap4d/`
- Downloaded sample: `official_demo_felix.mp4`
- Extracted screenshots:
  - `front_view.png`
  - `side_view.png`
  - `back_view.png`

## Quick Table

| Method | Speed | Quality | Gaps |
|---|---|---|---|
| cap4d | heavy setup (FLAME + Pixel3DMM + model assets), runtime not measured locally due blockers | promising in official demo video | yes (rear/head contour still shows thin or uncertain regions in sampled back view) |

## Gap Proof (Eyeball)
- front = stable face/details
- side = minor thin geometry around silhouette/hair boundary
- back = less reliable structure than front, some broken/thin edges around rear contour

## Output Folder
All artifacts saved in:
`research/benchmark_outputs/cap4d/`
