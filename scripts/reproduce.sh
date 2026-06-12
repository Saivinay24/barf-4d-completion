#!/bin/bash
# BARF 4D — end-to-end reproduction of every number and figure in
# paper/barf_paper.html, on CPU (Apple Silicon, no GPU required).
#
# Prerequisites: python3 with numpy, scipy, pandas, matplotlib, Pillow, pytest
#   pip install -r requirements.txt
#
# Inputs: the three Phase-0 Colab scene PLYs. Default location:
#   /Users/saivinay/Downloads/colab_eval_uncapped/<scene>/neoverse/scene.ply
# If data/scenes/*.npz already exist (committed in this repo), step 1 is a
# no-op and the original PLYs are not needed.
#
# Usage:  bash scripts/reproduce.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 1/4 prepare data (PLY -> npz; skipped if npz present) ==="
python3 scripts/prepare_data.py

echo "=== 2/4 run unit tests ==="
python3 -m pytest tests/ -q

echo "=== 3/4 run all experiments (E0-E3) ==="
python3 scripts/run_all_experiments.py --scenes_dir data/scenes --out results/session

echo "=== 4/4 generate paper figures ==="
python3 scripts/make_figures.py --results results/session --scenes_dir data/scenes --out paper/figures

echo "=== done. Raw results: results/session/  Figures: paper/figures/ ==="
