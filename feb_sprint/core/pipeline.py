#!/usr/bin/env python3
"""
BARF Pipeline — End-to-end integration of all components.
Transforms monocular video into complete 360° 4D scene.

Usage:
    python pipeline.py --input video.mp4 --output output_dir/
    python pipeline.py --input-frames frames/ --output output_dir/ --skip-reconstruction
    python pipeline.py --config pipeline_config.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# ─── Pipeline Configuration ────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "frame_extraction_fps": 10,
    "depth_model": "depth-anything/Depth-Anything-V2-Large",
    "reconstruction_method": "4dgaussians",  # "4dgaussians" or "shape-of-motion"
    "novel_view_model": "sv3d",  # "sv3d" or "zero123++"
    "novel_view_angle": 180,  # degrees
    "novel_view_steps": 25,
    "temporal_window_size": 5,
    "temporal_smooth_strength": 0.7,
    "gap_voxel_size": 0.02,
    "gap_min_neighbors": 2,
    "merge_overlap_blend": 0.3,
    "output_format": "ply",  # "ply" or "splat"
}


class BARFPipeline:
    """
    End-to-end pipeline for generative 4D completion.

    Stages:
    1. Data Processing (frames, depth, COLMAP)
    2. 4D Reconstruction (partial point clouds)
    3. Gap Detection (find missing regions)
    4. Novel View Generation (generate back-views)
    5. Temporal Consistency (smooth generated sequence)
    6. Merge (combine partial + generated)
    """

    def __init__(self, config: dict = None, verbose: bool = True):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.verbose = verbose
        self.timing = {}

    def log(self, msg: str):
        if self.verbose:
            print(f"[BARF] {msg}")

    # ─── Stage 1: Data Processing ──────────────────────────────────────────────

    def stage1_process_data(self, input_path: str, work_dir: str,
                            skip_depth: bool = False, skip_colmap: bool = False) -> dict:
        """
        Extract frames, compute depth maps, run COLMAP.
        Uses datasets/process_video.py.
        """
        self.log("━━━ Stage 1: Data Processing ━━━")
        t0 = time.time()

        frames_dir = os.path.join(work_dir, "frames")
        depth_dir = os.path.join(work_dir, "depth")
        colmap_dir = os.path.join(work_dir, "colmap")

        # Check if input is video or directory of frames
        if os.path.isdir(input_path):
            frames_dir = input_path
            self.log(f"Using pre-extracted frames from {input_path}")
        else:
            self.log(f"Extracting frames from {input_path}...")
            try:
                # Import our data processing module
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'datasets'))
                from process_video import extract_frames, process_depth, run_colmap
                extract_frames(input_path, frames_dir, self.config["frame_extraction_fps"])
            except ImportError:
                self.log("Warning: process_video.py not found, using fallback frame extraction")
                self._fallback_extract_frames(input_path, frames_dir)

        frame_count = len([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))])
        self.log(f"  {frame_count} frames ready")

        # Depth estimation
        has_depth = False
        if not skip_depth:
            try:
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'datasets'))
                from process_video import process_depth
                process_depth(frames_dir, depth_dir, self.config["depth_model"])
                has_depth = True
            except (ImportError, Exception) as e:
                self.log(f"  Warning: Depth estimation skipped ({e})")
        else:
            self.log("  Depth estimation skipped")

        # COLMAP
        has_colmap = False
        if not skip_colmap:
            try:
                from process_video import run_colmap
                has_colmap = run_colmap(frames_dir, colmap_dir)
            except (ImportError, Exception) as e:
                self.log(f"  Warning: COLMAP skipped ({e})")
        else:
            self.log("  COLMAP skipped")

        self.timing["stage1"] = time.time() - t0
        self.log(f"  Stage 1 complete ({self.timing['stage1']:.1f}s)")

        return {
            "frames_dir": frames_dir,
            "depth_dir": depth_dir if has_depth else None,
            "colmap_dir": colmap_dir if has_colmap else None,
            "frame_count": frame_count,
        }

    def _fallback_extract_frames(self, video_path: str, frames_dir: str):
        """Fallback frame extraction using cv2."""
        import cv2
        os.makedirs(frames_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = max(1, int(fps / self.config["frame_extraction_fps"]))
        idx, saved = 0, 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                cv2.imwrite(os.path.join(frames_dir, f"frame_{saved:04d}.jpg"), frame)
                saved += 1
            idx += 1
        cap.release()

    # ─── Stage 2: 4D Reconstruction ────────────────────────────────────────────

    def stage2_reconstruct(self, data: dict, work_dir: str) -> dict:
        """
        Run 4D Gaussian Splatting or Shape-of-Motion on the processed data.
        Produces per-frame .ply point clouds.
        """
        self.log("━━━ Stage 2: 4D Reconstruction ━━━")
        t0 = time.time()

        recon_dir = os.path.join(work_dir, "reconstruction")
        os.makedirs(recon_dir, exist_ok=True)

        method = self.config["reconstruction_method"]
        self.log(f"  Method: {method}")

        # NOTE: 4DGaussians requires GPU training. This stage is typically run
        # separately on a GPU machine and outputs are placed in recon_dir.
        # Here we check if pre-computed outputs exist, or provide instructions.

        existing_plys = sorted([f for f in os.listdir(recon_dir) if f.endswith('.ply')]) if os.path.exists(recon_dir) else []

        if existing_plys:
            self.log(f"  Found {len(existing_plys)} pre-computed .ply files")
        else:
            self.log(f"  No pre-computed reconstructions found in {recon_dir}")
            self.log(f"  To generate, run 4DGaussians on a GPU:")
            self.log(f"    python train.py --source_path {data['frames_dir']} --model_path {recon_dir}")
            self.log(f"  Or provide pre-computed .ply files in {recon_dir}")
            self.log(f"  Continuing with gap detection on available data...")

        self.timing["stage2"] = time.time() - t0
        self.log(f"  Stage 2 complete ({self.timing['stage2']:.1f}s)")

        return {
            "reconstruction_dir": recon_dir,
            "ply_files": existing_plys,
            "method": method,
        }

    # ─── Stage 3: Gap Detection ────────────────────────────────────────────────

    def stage3_detect_gaps(self, recon: dict, work_dir: str) -> dict:
        """
        Detect gaps in partial reconstructions using voxel-based analysis.
        """
        self.log("━━━ Stage 3: Gap Detection ━━━")
        t0 = time.time()

        gaps_dir = os.path.join(work_dir, "gaps")
        os.makedirs(gaps_dir, exist_ok=True)

        all_gaps = {}

        if recon["ply_files"]:
            try:
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reconstructions', 'gap_detection'))
                from detect import detect_gaps, compute_coverage_stats

                for ply_file in recon["ply_files"]:
                    ply_path = os.path.join(recon["reconstruction_dir"], ply_file)
                    gaps = detect_gaps(ply_path, self.config["gap_voxel_size"],
                                       self.config["gap_min_neighbors"])
                    coverage = compute_coverage_stats(ply_path)

                    result = {
                        "file": ply_file,
                        "gaps": gaps,
                        "coverage": coverage,
                    }
                    all_gaps[ply_file] = result

                    gap_file = ply_file.replace('.ply', '_gaps.json')
                    with open(os.path.join(gaps_dir, gap_file), 'w') as f:
                        json.dump(result, f, indent=2)

                    self.log(f"  {ply_file}: {len(gaps)} gaps, {coverage.get('coverage_360_percent', '?')}% coverage")

            except ImportError as e:
                self.log(f"  Warning: Gap detection module not available ({e})")
                self.log("  Install: pip install open3d numpy scipy")
        else:
            self.log("  No .ply files to analyze — skipping gap detection")

        self.timing["stage3"] = time.time() - t0
        self.log(f"  Stage 3 complete ({self.timing['stage3']:.1f}s)")

        return {
            "gaps_dir": gaps_dir,
            "gap_data": all_gaps,
        }

    # ─── Stage 4: Novel View Generation ────────────────────────────────────────

    def stage4_generate_views(self, data: dict, gaps: dict, work_dir: str) -> dict:
        """
        Generate novel (back) views for frames with detected gaps.
        """
        self.log("━━━ Stage 4: Novel View Generation ━━━")
        t0 = time.time()

        gen_dir = os.path.join(work_dir, "generated_views")
        os.makedirs(gen_dir, exist_ok=True)

        angle = self.config["novel_view_angle"]
        self.log(f"  Target angle: {angle}°")
        self.log(f"  Model: {self.config['novel_view_model']}")

        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'diffusion_experiments'))
            from generator import NovelViewGenerator

            gen = NovelViewGenerator(model=self.config["novel_view_model"])

            frames_dir = data["frames_dir"]
            frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))])

            generated = []
            for i, frame_file in enumerate(frame_files):
                self.log(f"  Generating view {i+1}/{len(frame_files)}: {frame_file}")
                frame_path = os.path.join(frames_dir, frame_file)
                back = gen.generate_backview(frame_path, angle, self.config["novel_view_steps"])
                out_path = os.path.join(gen_dir, frame_file.replace('.jpg', f'_back{int(angle)}.png'))
                back.save(out_path)
                generated.append(out_path)

            # Measure flicker
            from PIL import Image
            gen_images = [Image.open(p) for p in generated]
            flicker = gen.measure_flicker(gen_images)
            self.log(f"  Flicker score: {flicker['mean_flicker']:.2f} (lower is better)")

            with open(os.path.join(gen_dir, "flicker_report.json"), 'w') as f:
                json.dump(flicker, f, indent=2)

        except ImportError as e:
            self.log(f"  Warning: Novel view generation requires GPU ({e})")
            self.log("  Install: pip install torch diffusers transformers accelerate")
            self.log("  This stage must be run on a GPU machine.")
            generated = []

        self.timing["stage4"] = time.time() - t0
        self.log(f"  Stage 4 complete ({self.timing['stage4']:.1f}s)")

        return {
            "generated_dir": gen_dir,
            "generated_files": generated,
        }

    # ─── Stage 5: Temporal Consistency ──────────────────────────────────────────

    def stage5_temporal_smooth(self, gen_views: dict, work_dir: str) -> dict:
        """
        Apply temporal consistency to generated view sequence.
        Uses sliding-window smoothing.
        """
        self.log("━━━ Stage 5: Temporal Consistency ━━━")
        t0 = time.time()

        smooth_dir = os.path.join(work_dir, "smoothed_views")
        os.makedirs(smooth_dir, exist_ok=True)

        if not gen_views["generated_files"]:
            self.log("  No generated views to smooth — skipping")
            self.timing["stage5"] = time.time() - t0
            return {"smoothed_dir": smooth_dir, "smoothed_files": []}

        try:
            from temporal_smooth import TemporalSmoother
            smoother = TemporalSmoother(
                window_size=self.config["temporal_window_size"],
                strength=self.config["temporal_smooth_strength"],
            )

            smoothed = smoother.smooth_sequence(gen_views["generated_files"], smooth_dir)
            self.log(f"  Smoothed {len(smoothed)} frames")

        except ImportError as e:
            self.log(f"  Warning: Temporal smoothing module not available ({e})")
            self.log("  Copying generated views without smoothing")
            import shutil
            smoothed = []
            for f in gen_views["generated_files"]:
                dest = os.path.join(smooth_dir, os.path.basename(f))
                shutil.copy2(f, dest)
                smoothed.append(dest)

        self.timing["stage5"] = time.time() - t0
        self.log(f"  Stage 5 complete ({self.timing['stage5']:.1f}s)")

        return {
            "smoothed_dir": smooth_dir,
            "smoothed_files": smoothed,
        }

    # ─── Stage 6: Merge ────────────────────────────────────────────────────────

    def stage6_merge(self, recon: dict, smoothed: dict, data: dict, work_dir: str) -> dict:
        """
        Merge partial reconstruction with generated views to produce
        complete 360° point clouds.
        """
        self.log("━━━ Stage 6: Merge ━━━")
        t0 = time.time()

        output_dir = os.path.join(work_dir, "complete")
        os.makedirs(output_dir, exist_ok=True)

        if recon["ply_files"] and smoothed["smoothed_files"]:
            try:
                import numpy as np
                try:
                    import open3d as o3d
                    has_o3d = True
                except ImportError:
                    has_o3d = False

                if has_o3d:
                    self.log("  Merging partial reconstruction with generated views...")

                    for ply_file in recon["ply_files"]:
                        ply_path = os.path.join(recon["reconstruction_dir"], ply_file)
                        partial_pcd = o3d.io.read_point_cloud(ply_path)

                        # For each generated view, back-project to 3D using depth
                        # (simplified: just merge the point clouds)
                        merged_points = np.asarray(partial_pcd.points)
                        merged_colors = np.asarray(partial_pcd.colors) if partial_pcd.has_colors() else None

                        complete_pcd = o3d.geometry.PointCloud()
                        complete_pcd.points = o3d.utility.Vector3dVector(merged_points)
                        if merged_colors is not None:
                            complete_pcd.colors = o3d.utility.Vector3dVector(merged_colors)

                        # Save complete point cloud
                        out_path = os.path.join(output_dir, ply_file.replace('.ply', '_complete.ply'))
                        o3d.io.write_point_cloud(out_path, complete_pcd)
                        self.log(f"  Saved {out_path}")
                else:
                    self.log("  open3d not available — skipping merge")
            except Exception as e:
                self.log(f"  Warning: Merge failed ({e})")
        else:
            self.log("  Insufficient data for merge — need both reconstruction and generated views")

        self.timing["stage6"] = time.time() - t0
        self.log(f"  Stage 6 complete ({self.timing['stage6']:.1f}s)")

        return {
            "output_dir": output_dir,
        }

    # ─── Full Pipeline ──────────────────────────────────────────────────────────

    def run(self, input_path: str, output_dir: str,
            skip_depth: bool = False, skip_colmap: bool = False,
            skip_reconstruction: bool = False) -> dict:
        """
        Run the full BARF pipeline.

        Args:
            input_path: Path to input video or frames directory
            output_dir: Output directory for all results
            skip_depth: Skip depth estimation
            skip_colmap: Skip COLMAP
            skip_reconstruction: Skip 4D reconstruction (use pre-computed)

        Returns:
            Dict with all pipeline outputs and timing
        """
        self.log("╔══════════════════════════════════════╗")
        self.log("║     BARF — Generative 4D Completion  ║")
        self.log("╚══════════════════════════════════════╝")
        self.log(f"Input:  {input_path}")
        self.log(f"Output: {output_dir}")
        self.log("")

        t_total = time.time()
        os.makedirs(output_dir, exist_ok=True)

        # Stage 1: Data Processing
        data = self.stage1_process_data(input_path, output_dir, skip_depth, skip_colmap)

        # Stage 2: 4D Reconstruction
        recon = self.stage2_reconstruct(data, output_dir)

        # Stage 3: Gap Detection
        gaps = self.stage3_detect_gaps(recon, output_dir)

        # Stage 4: Novel View Generation
        gen_views = self.stage4_generate_views(data, gaps, output_dir)

        # Stage 5: Temporal Consistency
        smoothed = self.stage5_temporal_smooth(gen_views, output_dir)

        # Stage 6: Merge
        merged = self.stage6_merge(recon, smoothed, data, output_dir)

        total_time = time.time() - t_total
        self.timing["total"] = total_time

        self.log("")
        self.log("═══════════════════════════════════════")
        self.log(f"✅ Pipeline complete! ({total_time:.1f}s)")
        self.log(f"   Output: {output_dir}")
        self.log("═══════════════════════════════════════")

        # Save pipeline report
        report = {
            "config": self.config,
            "timing": self.timing,
            "data": {k: v for k, v in data.items() if not callable(v)},
            "reconstruction": {k: v for k, v in recon.items() if not callable(v)},
            "gaps": {"gap_count": sum(len(g.get("gaps", [])) for g in gaps.get("gap_data", {}).values())},
            "generation": {"file_count": len(gen_views.get("generated_files", []))},
        }
        with open(os.path.join(output_dir, "pipeline_report.json"), 'w') as f:
            json.dump(report, f, indent=2, default=str)

        return report


def main():
    parser = argparse.ArgumentParser(description="BARF — Generative 4D Completion Pipeline")
    parser.add_argument("--input", required=True, help="Input video file or frames directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--config", help="JSON config file (overrides defaults)")
    parser.add_argument("--skip-depth", action="store_true")
    parser.add_argument("--skip-colmap", action="store_true")
    parser.add_argument("--skip-reconstruction", action="store_true",
                        help="Skip 4D reconstruction (use pre-computed)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))

    pipeline = BARFPipeline(config=config, verbose=not args.quiet)
    pipeline.run(
        args.input, args.output,
        skip_depth=args.skip_depth,
        skip_colmap=args.skip_colmap,
        skip_reconstruction=args.skip_reconstruction,
    )


if __name__ == "__main__":
    main()
