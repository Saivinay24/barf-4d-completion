"""
Convert the Phase-0 Colab ASCII PLY outputs into compressed .npz scene files.

Provenance note (audited 2026-06-13): each source PLY contains ~392*252*25
points — exactly one point per pixel per frame (minus invalid depth pixels).
These are per-pixel monocular depth unprojections of 25 video frames produced
by the Colab pipeline (labelled "NeoVerse reconstructor" in its export
comment), NOT optimised 4D Gaussian Splat reconstructions. The .npz files
carry this provenance in their metadata.

Usage:
    python3 scripts/prepare_data.py \
        --src /Users/saivinay/Downloads/colab_eval_uncapped \
        --dst data/scenes
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

SCENES = ["01_mdn_flower", "02_w3schools_big_buck_bunny", "03_samplelib_5s"]


def load_ply_fast(ply_path: str):
    """Parse an ASCII PLY with properties x y z red green blue frame_index."""
    with open(ply_path) as f:
        names = []
        n_header = 0
        for line in f:
            n_header += 1
            line = line.strip()
            if line.startswith("property"):
                names.append(line.split()[-1])
            if line == "end_header":
                break
    df = pd.read_csv(ply_path, sep=" ", skiprows=n_header, header=None,
                     names=names, engine="c", na_filter=False)
    xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    rgb = df[["red", "green", "blue"]].to_numpy(dtype=np.uint8)
    frame = df["frame_index"].to_numpy(dtype=np.int32)
    return xyz, rgb, frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/Users/saivinay/Downloads/colab_eval_uncapped")
    ap.add_argument("--dst", default="data/scenes")
    args = ap.parse_args()
    os.makedirs(args.dst, exist_ok=True)

    for scene in SCENES:
        ply = Path(args.src) / scene / "neoverse" / "scene.ply"
        meta_path = Path(args.src) / scene / "neoverse" / "metadata.json"
        out = Path(args.dst) / f"{scene}.npz"
        if out.exists():
            print(f"[prepare_data] {out} already exists, skipping")
            continue
        print(f"[prepare_data] parsing {ply} ...")
        xyz, rgb, frame = load_ply_fast(str(ply))
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        w = meta.get("resolution", {}).get("width")
        h = meta.get("resolution", {}).get("height")
        T = meta.get("num_frames")
        print(f"[prepare_data]   n={len(xyz)}  frames={frame.max()+1}  "
              f"w*h*T={w}*{h}*{T}={w*h*T if w else 'n/a'}")
        np.savez_compressed(
            out, xyz=xyz, rgb=rgb, frame=frame,
            provenance=np.array(json.dumps({
                "source_ply": str(ply),
                "source_metadata": meta,
                "artifact_type": "per-pixel monocular depth unprojection "
                                 "(one point per pixel per frame); "
                                 "NOT an optimised 4DGS reconstruction",
                "audit": "n_points ~= width*height*num_frames",
            })),
        )
        print(f"[prepare_data]   wrote {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
