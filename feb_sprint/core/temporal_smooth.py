#!/usr/bin/env python3
"""
Temporal Consistency Module for BARF.
Reduces flicker in independently generated novel view sequences.

Methods:
1. Sliding-window exponential moving average (EMA)
2. Optical flow-guided warping (RAFT-based)
3. Combined: warp + blend

Usage:
    python temporal_smooth.py --input generated_views/ --output smoothed_views/
    python temporal_smooth.py --input generated_views/ --output smoothed_views/ --method flow --strength 0.7
"""
import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image


class TemporalSmoother:
    """
    Temporal consistency module for sequences of generated images.

    Supports multiple smoothing strategies:
    - 'ema': Exponential moving average (fast, simple)
    - 'flow': Optical flow warping (better quality, requires RAFT)
    - 'combined': Flow-guided EMA (best quality)
    """

    def __init__(self, window_size: int = 5, strength: float = 0.7,
                 method: str = "ema"):
        """
        Args:
            window_size: Number of frames in sliding window
            strength: Smoothing strength (0.0 = no smoothing, 1.0 = max)
            method: "ema", "flow", or "combined"
        """
        self.window_size = window_size
        self.strength = np.clip(strength, 0.0, 1.0)
        self.method = method
        self.flow_model = None

    def _load_flow_model(self):
        """Load RAFT optical flow model if needed."""
        if self.flow_model is not None:
            return

        try:
            import torch
            import torchvision.models.optical_flow as of

            self.flow_model = of.raft_large(pretrained=True)
            self.flow_model.eval()
            if torch.cuda.is_available():
                self.flow_model = self.flow_model.cuda()
            print("  RAFT optical flow model loaded")
        except (ImportError, Exception) as e:
            print(f"  Warning: Could not load RAFT ({e}). Falling back to EMA.")
            self.method = "ema"

    def _compute_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """Compute optical flow between two frames using RAFT."""
        import torch
        import torchvision.transforms.functional as F

        # Convert to tensors
        t1 = torch.from_numpy(frame1).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        t2 = torch.from_numpy(frame2).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        if torch.cuda.is_available():
            t1, t2 = t1.cuda(), t2.cuda()

        with torch.no_grad():
            flows = self.flow_model(t1, t2)
            flow = flows[-1][0].cpu().numpy()  # (2, H, W)

        return flow.transpose(1, 2, 0)  # (H, W, 2)

    def _warp_frame(self, frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Warp a frame using optical flow."""
        h, w = frame.shape[:2]
        grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)

        map_x = (grid_x + flow[:, :, 0]).astype(np.float32)
        map_y = (grid_y + flow[:, :, 1]).astype(np.float32)

        # Clip to valid range
        map_x = np.clip(map_x, 0, w - 1)
        map_y = np.clip(map_y, 0, h - 1)

        try:
            import cv2
            warped = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
        except ImportError:
            # Fallback: nearest neighbor remap
            warped = frame[map_y.astype(int), map_x.astype(int)]

        return warped

    def smooth_ema(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply exponential moving average smoothing.
        Simple but effective for reducing high-frequency flicker.
        """
        if not frames:
            return []

        alpha = 1.0 - self.strength  # Higher strength = more smoothing = lower alpha
        smoothed = [frames[0].astype(np.float64)]

        for i in range(1, len(frames)):
            # EMA: new = alpha * current + (1-alpha) * previous_smoothed
            blended = alpha * frames[i].astype(np.float64) + (1 - alpha) * smoothed[-1]
            smoothed.append(blended)

        # Bidirectional pass for better quality
        backward = [frames[-1].astype(np.float64)]
        for i in range(len(frames) - 2, -1, -1):
            blended = alpha * frames[i].astype(np.float64) + (1 - alpha) * backward[-1]
            backward.append(blended)
        backward.reverse()

        # Average forward and backward passes
        result = []
        for fwd, bwd in zip(smoothed, backward):
            avg = (fwd + bwd) / 2.0
            result.append(np.clip(avg, 0, 255).astype(np.uint8))

        return result

    def smooth_flow(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply optical flow-guided warping + blending.
        Warps previous smoothed frame to current, then blends with current.
        """
        self._load_flow_model()

        if self.method == "ema":
            # Fell back to EMA
            return self.smooth_ema(frames)

        if not frames:
            return []

        smoothed = [frames[0]]

        for i in range(1, len(frames)):
            # Compute flow from previous to current
            flow = self._compute_flow(frames[i - 1], frames[i])

            # Warp previous smoothed frame
            warped_prev = self._warp_frame(smoothed[-1], flow)

            # Blend warped previous with current
            alpha = 1.0 - self.strength
            blended = (alpha * frames[i].astype(np.float64) +
                        (1 - alpha) * warped_prev.astype(np.float64))
            smoothed.append(np.clip(blended, 0, 255).astype(np.uint8))

        return smoothed

    def smooth_combined(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Combined approach: flow-guided warping + EMA for residual smoothing.
        """
        # First pass: flow-guided
        flow_smoothed = self.smooth_flow(frames)
        # Second pass: light EMA on the result
        old_strength = self.strength
        self.strength = self.strength * 0.3  # Light EMA
        result = self.smooth_ema(flow_smoothed)
        self.strength = old_strength
        return result

    def smooth_sequence(self, input_paths: List[str], output_dir: str) -> List[str]:
        """
        Smooth a sequence of image files.

        Args:
            input_paths: List of input image file paths
            output_dir: Directory to save smoothed images

        Returns:
            List of output file paths
        """
        os.makedirs(output_dir, exist_ok=True)

        # Load all frames
        print(f"  Loading {len(input_paths)} frames...")
        frames = [np.array(Image.open(p).convert("RGB")) for p in input_paths]

        # Apply smoothing
        print(f"  Smoothing with method={self.method}, strength={self.strength}, window={self.window_size}")

        if self.method == "flow":
            smoothed = self.smooth_flow(frames)
        elif self.method == "combined":
            smoothed = self.smooth_combined(frames)
        else:
            smoothed = self.smooth_ema(frames)

        # Save output
        output_paths = []
        for input_path, smooth_frame in zip(input_paths, smoothed):
            filename = os.path.basename(input_path)
            out_path = os.path.join(output_dir, filename)
            Image.fromarray(smooth_frame).save(out_path)
            output_paths.append(out_path)

        # Compute improvement metrics
        orig_flicker = self._compute_flicker(frames)
        smooth_flicker = self._compute_flicker(smoothed)
        improvement = (1 - smooth_flicker / max(orig_flicker, 1e-8)) * 100

        print(f"  Original flicker:  {orig_flicker:.2f}")
        print(f"  Smoothed flicker:  {smooth_flicker:.2f}")
        print(f"  Improvement:       {improvement:.1f}%")

        return output_paths

    def _compute_flicker(self, frames: List[np.ndarray]) -> float:
        """Compute mean frame-to-frame difference."""
        if len(frames) < 2:
            return 0.0
        diffs = []
        for i in range(1, len(frames)):
            diff = np.abs(frames[i].astype(float) - frames[i - 1].astype(float)).mean()
            diffs.append(diff)
        return float(np.mean(diffs))


def main():
    parser = argparse.ArgumentParser(description="BARF temporal consistency smoothing")
    parser.add_argument("--input", required=True, help="Input directory of generated views")
    parser.add_argument("--output", required=True, help="Output directory for smoothed views")
    parser.add_argument("--method", default="ema", choices=["ema", "flow", "combined"],
                        help="Smoothing method (default: ema)")
    parser.add_argument("--strength", type=float, default=0.7, help="Smoothing strength 0-1 (default: 0.7)")
    parser.add_argument("--window", type=int, default=5, help="Window size (default: 5)")
    args = parser.parse_args()

    input_dir = args.input
    frame_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))])
    input_paths = [os.path.join(input_dir, f) for f in frame_files]

    if not input_paths:
        print(f"No image files found in {input_dir}")
        sys.exit(1)

    smoother = TemporalSmoother(
        window_size=args.window,
        strength=args.strength,
        method=args.method,
    )

    output_paths = smoother.smooth_sequence(input_paths, args.output)
    print(f"\nSaved {len(output_paths)} smoothed frames to {args.output}")


if __name__ == "__main__":
    main()
