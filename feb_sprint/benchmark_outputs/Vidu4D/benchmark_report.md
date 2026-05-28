# Vidu4D Quick Benchmark (Fallback Path)

## Status
- Repo used: `Vidu4D` (`https://github.com/yikaiw/Vidu4D`)
- Local run attempt: **blocked** on this machine
- Blocker: CUDA-only extension compile in `lab4d/third_party/quaternion/backend.py` (`.cu` sources, requires CUDA toolkit / `CUDA_HOME`)
- What worked: dependency bootstrap (`torch`, `torchvision`, `torchaudio`, `ninja`) and official demo artifact download

## Commands Executed (local)
```bash
python3 -m pip install torch torchvision torchaudio ninja
python3 scripts/run_preprocess.py video video other "0"
```

## Error That Stops Local Execution
```text
OSError: CUDA_HOME environment variable is not set. Please set it to your CUDA install root.
```

## Backup Evidence (Official Demo)
- Source page: `https://vidu4d-dgs.github.io/`
- Downloaded sample: `official_demo_cat2.mp4`
- Extracted frames:
  - `front_view.png`
  - `back_view.png`
  - `rear_probe.png`

## Quick Table

| Method | Speed | Quality | Gaps |
|---|---|---|---|
| Vidu4D | slow setup (GPU/CUDA required), fast demo playback | good appearance and stable silhouette in sampled views | yes (rear/unseen geometry confidence is limited from sampled views) |

## Gap Proof (Eyeball)
- front = good
- back = partially broken / uncertain (rear structure not fully validated in this clip; shape around occluded fur/neck areas appears smoothed and less reliable)

## Output Folder
All artifacts saved in:
`research/benchmark_outputs/Vidu4D/`
