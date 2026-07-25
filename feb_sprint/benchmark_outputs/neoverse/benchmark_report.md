# NeoVerse Quick Benchmark (Fallback Path)

## Reality Check
- GitHub: `https://github.com/IamCreateAI/NeoVerse` (public, repo reachable)
- Hugging Face: `https://huggingface.co/Yuppie1204/NeoVerse` (model page referenced on project site)
- Local repo: cloned from `https://github.com/IamCreateAI/NeoVerse`

## Status
- Time-optimized path used: **official demo evidence**
- Full local inference run (`inference.py`) was **not executed** in this pass to avoid long install/setup time.
- Reason: this benchmark sweep prioritizes comparable visual evidence under limited time.

## Evidence Source
- Project page: `https://neoverse-4d.github.io/`
- Demo video URL: `https://neoverse-4d.github.io/resources/NeoVerse_public.mp4`
- Saved video: `official_demo.mp4`

## Extracted Screenshots
- `front_view.png`
- `side_view.png`
- `back_view.png`

## Quick Table

| Method | Speed | Quality | Gaps |
|---|---|---|---|
| NeoVerse | fast to evaluate via official demo; local inference not timed here | strong multi-view consistency in showcased clips | rear/unseen areas still need controlled same-input local A/B run for strict quantitative judgment |

## Output Folder
All artifacts saved in:
`research/benchmark_outputs/neoverse/`
