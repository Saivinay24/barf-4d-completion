#!/usr/bin/env python3
"""
Novel view generator for the BARF pipeline.
Generates back-views and arbitrary-angle views from front-facing images
using SV3D (Stable Video 3D) or Zero123++.

Usage:
    # Single image
    python generator.py --input front_view.jpg --output back_view.png --angle 180

    # Sequence of frames (batch processing)
    python generator.py --input-dir frames/ --output-dir backviews/ --angle 180

    # Generate full orbital sequence
    python generator.py --input front_view.jpg --output-dir orbital/ --full-orbit
"""
import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from PIL import Image


class NovelViewGenerator:
    """
    Generates novel views from a single input image using diffusion models.
    Supports SV3D (Stable Video 3D) and Zero123++ backends.
    """

    def __init__(self, model: str = "sv3d", device: str = "cuda", half_precision: bool = True):
        """
        Initialize the generator.

        Args:
            model: Model backend - "sv3d" or "zero123++"
            device: Torch device ("cuda" or "cpu")
            half_precision: Use float16 for faster inference
        """
        self.model_name = model
        self.device = device
        self.half_precision = half_precision
        self.pipe = None
        self._load_model()

    def _load_model(self):
        """Load the diffusion pipeline."""
        try:
            import torch
            from diffusers import DiffusionPipeline
        except ImportError:
            print("Error: Install dependencies: pip install torch diffusers transformers accelerate")
            sys.exit(1)

        dtype = torch.float16 if self.half_precision and self.device == "cuda" else torch.float32

        if self.model_name == "sv3d":
            try:
                from diffusers import StableVideo3DPipeline
                print("Loading SV3D pipeline...")
                self.pipe = StableVideo3DPipeline.from_pretrained(
                    "stabilityai/sv3d",
                    torch_dtype=dtype,
                ).to(self.device)
                self.num_orbital_frames = 21
                print("SV3D loaded successfully.")
            except Exception as e:
                print(f"Warning: Could not load SV3D: {e}")
                print("Falling back to Zero123++...")
                self.model_name = "zero123++"
                self._load_zero123()

        elif self.model_name == "zero123++":
            self._load_zero123()

        else:
            raise ValueError(f"Unknown model: {self.model_name}. Use 'sv3d' or 'zero123++'")

    def _load_zero123(self):
        """Load Zero123++ pipeline."""
        import torch
        from diffusers import DiffusionPipeline

        dtype = torch.float16 if self.half_precision and self.device == "cuda" else torch.float32

        print("Loading Zero123++ pipeline...")
        self.pipe = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.2",
            torch_dtype=dtype,
        ).to(self.device)
        self.num_orbital_frames = 6  # Zero123++ generates 6 views
        print("Zero123++ loaded successfully.")

    def generate_orbital(self, image: Union[str, Image.Image],
                         num_inference_steps: int = 25) -> List[Image.Image]:
        """
        Generate a full orbital sequence from a single image.

        Args:
            image: Input image (path or PIL Image)
            num_inference_steps: Denoising steps (more = better quality, slower)

        Returns:
            List of PIL Images representing views at different angles
        """
        if isinstance(image, str):
            image = Image.open(image)

        image = image.convert("RGB").resize((512, 512))

        if self.model_name == "sv3d":
            result = self.pipe(image, num_frames=self.num_orbital_frames,
                               num_inference_steps=num_inference_steps)
            return result.frames[0]
        else:
            # Zero123++ returns a grid image
            result = self.pipe(image, num_inference_steps=num_inference_steps)
            grid = result.images[0]
            # Split grid into individual views (2x3 grid of 256x256 images)
            views = []
            w, h = 256, 256
            for row in range(2):
                for col in range(3):
                    crop = grid.crop((col * w, row * h, (col + 1) * w, (row + 1) * h))
                    views.append(crop)
            return views

    def generate_backview(self, image: Union[str, Image.Image],
                          angle: float = 180,
                          num_inference_steps: int = 25) -> Image.Image:
        """
        Generate a view at a specific angle.

        Args:
            image: Input front-view image
            angle: Target angle in degrees (0=front, 180=back)
            num_inference_steps: Denoising steps

        Returns:
            PIL Image of the generated view
        """
        orbital = self.generate_orbital(image, num_inference_steps)

        # Map angle to frame index
        frame_idx = int((angle / 360) * len(orbital)) % len(orbital)
        return orbital[frame_idx]

    def generate_backview_sequence(self, frames: List[Union[str, Image.Image]],
                                    angle: float = 180,
                                    num_inference_steps: int = 25) -> List[Image.Image]:
        """
        Generate back-views for a sequence of frames.

        Args:
            frames: List of front-view images (paths or PIL Images)
            angle: Target angle
            num_inference_steps: Denoising steps

        Returns:
            List of generated back-view images
        """
        results = []
        for i, frame in enumerate(frames):
            print(f"  Generating view {i+1}/{len(frames)} at {angle}°...")
            back = self.generate_backview(frame, angle, num_inference_steps)
            results.append(back)
        return results

    def measure_flicker(self, generated_frames: List[Image.Image]) -> dict:
        """
        Measure temporal flicker in a sequence of generated frames.

        Args:
            generated_frames: List of generated PIL Images

        Returns:
            Dict with flicker metrics
        """
        arrays = [np.array(f).astype(float) for f in generated_frames]

        diffs = []
        for i in range(1, len(arrays)):
            diff = np.abs(arrays[i] - arrays[i - 1]).mean()
            diffs.append(diff)

        return {
            "mean_flicker": float(np.mean(diffs)) if diffs else 0,
            "max_flicker": float(np.max(diffs)) if diffs else 0,
            "min_flicker": float(np.min(diffs)) if diffs else 0,
            "std_flicker": float(np.std(diffs)) if diffs else 0,
            "num_frames": len(generated_frames),
        }


def main():
    parser = argparse.ArgumentParser(description="BARF novel view generator")
    parser.add_argument("--input", help="Input image file")
    parser.add_argument("--input-dir", help="Input directory of frames")
    parser.add_argument("--output", help="Output image file (single image mode)")
    parser.add_argument("--output-dir", help="Output directory (batch/orbital mode)")
    parser.add_argument("--angle", type=float, default=180, help="Target view angle (default: 180 = back)")
    parser.add_argument("--full-orbit", action="store_true", help="Generate full orbital sequence")
    parser.add_argument("--model", default="sv3d", choices=["sv3d", "zero123++"],
                        help="Model backend (default: sv3d)")
    parser.add_argument("--steps", type=int, default=25, help="Inference steps (default: 25)")
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument("--measure-flicker", action="store_true", help="Measure temporal flicker")
    args = parser.parse_args()

    if not args.input and not args.input_dir:
        parser.error("Either --input or --input-dir is required")

    gen = NovelViewGenerator(model=args.model, device=args.device)

    if args.full_orbit and args.input:
        # Generate full orbital sequence
        os.makedirs(args.output_dir or "orbital_output", exist_ok=True)
        out_dir = args.output_dir or "orbital_output"

        print(f"Generating full orbit from {args.input}...")
        t0 = time.time()
        orbital = gen.generate_orbital(args.input, args.steps)
        elapsed = time.time() - t0

        for i, frame in enumerate(orbital):
            angle = int(i / len(orbital) * 360)
            frame.save(os.path.join(out_dir, f"view_{angle:03d}deg.png"))

        print(f"Saved {len(orbital)} views to {out_dir} ({elapsed:.1f}s)")

    elif args.input_dir:
        # Batch process directory
        out_dir = args.output_dir or "backview_output"
        os.makedirs(out_dir, exist_ok=True)

        frame_files = sorted([f for f in os.listdir(args.input_dir)
                              if f.endswith(('.jpg', '.png'))])
        frame_paths = [os.path.join(args.input_dir, f) for f in frame_files]

        print(f"Processing {len(frame_paths)} frames at {args.angle}°...")
        t0 = time.time()
        results = gen.generate_backview_sequence(frame_paths, args.angle, args.steps)
        elapsed = time.time() - t0

        for frame_file, result in zip(frame_files, results):
            out_name = frame_file.replace('.jpg', f'_back{int(args.angle)}.png')
            result.save(os.path.join(out_dir, out_name))

        if args.measure_flicker:
            flicker = gen.measure_flicker(results)
            print(f"\nFlicker Report:")
            print(f"  Mean: {flicker['mean_flicker']:.2f}")
            print(f"  Max:  {flicker['max_flicker']:.2f}")
            print(f"  Std:  {flicker['std_flicker']:.2f}")

            import json
            with open(os.path.join(out_dir, "flicker_report.json"), 'w') as f:
                json.dump(flicker, f, indent=2)

        print(f"Saved {len(results)} back-views to {out_dir} ({elapsed:.1f}s)")

    elif args.input:
        # Single image
        print(f"Generating {args.angle}° view from {args.input}...")
        t0 = time.time()
        result = gen.generate_backview(args.input, args.angle, args.steps)
        elapsed = time.time() - t0

        output_path = args.output or "backview_output.png"
        result.save(output_path)
        print(f"Saved to {output_path} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
