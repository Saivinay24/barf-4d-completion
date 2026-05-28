"""
BARF 4D — Spherical 4D Completion Module
==========================================
Core innovation: generative completion of all unobserved angular regions
in a 4D Gaussian Splat scene, conditioned on the 4D scene latent (not 2D frames).

Architecture (Option A — condition Vivid4D inpainting on NeoVerse scene features):
    Input: 4DGS scene S (PLY) + angular gap mask G (from detect_gaps)
      ↓
    TemporalFeatureExtractor: extracts per-timestep appearance + dynamics features
      ↓
    SphericalGapEncoder: clusters gaps by (θ, φ) region, builds completion targets
      ↓
    CompletionDiffusion: Vivid4D-style inpainting backbone
        conditioned on 4D scene latent via cross-attention (THE KEY INNOVATION)
        generates multi-frame RGBA for each gap region
      ↓
    GaussianFusion: warps generated RGBA back into 4DGS representation
      ↓
    Output: complete 4DGS scene S' covering full (θ, φ, t)

GPU/Training notes:
    - Full inference requires GPU (CUDA, ≥24GB VRAM recommended)
    - This module implements the architecture and runs unit-testable
      forward passes with synthetic tensors on CPU
    - Real inference: run on Vast.ai H100 after loading real checkpoints
    - TODO markers indicate where GPU-only code paths live

Usage (unit test, CPU):
    python -m pytest tests/test_completion.py -v

Usage (real inference, GPU required):
    # TODO: GPU EXECUTION REQUIRED
    python -m src.completion.spherical_completion \
        --scene_ply outputs/neoverse/scene.ply \
        --gaps_json outputs/gaps/gaps.json \
        --output_ply outputs/completion/scene_complete.ply \
        --device cuda
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Dependency handling — PyTorch is GPU-optional
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Provide stub so unit tests can import without torch
    class _TorchStub:
        class Tensor: pass
        class nn:
            class Module: pass
            class Linear: pass
            class LayerNorm: pass
            class MultiheadAttention: pass
    torch = _TorchStub()


# ---------------------------------------------------------------------------
# 1. Temporal Feature Extractor
# ---------------------------------------------------------------------------

class TemporalFeatureExtractor(nn.Module if TORCH_AVAILABLE else object):
    """
    Extracts per-timestep appearance and dynamics features from the 4DGS scene.

    In full training: this is a small 3D CNN / PointNet operating on the
    Gaussian parameters (position, covariance, color, opacity) across time.

    For Option A (condition existing backbone, no full training):
    - Encodes the PLY point cloud into a scene latent vector via mean pooling
      of per-point features, then projects to a cross-attention-compatible dim.

    Input:  scene_points  (T, N, C) — T timesteps, N Gaussians, C features
    Output: scene_latent  (T, D)    — D-dimensional scene representation per timestep
    """

    def __init__(self, input_dim: int = 6, hidden_dim: int = 256, output_dim: int = 512):
        if TORCH_AVAILABLE:
            super().__init__()
            self.point_encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
            )
        self.output_dim = output_dim

    def forward(self, scene_points):
        """
        Args:
            scene_points: (T, N, C) tensor — per-timestep Gaussian parameters
                          C = [x, y, z, r, g, b] (or more with opacity, cov, etc.)
        Returns:
            scene_latent: (T, D) tensor
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for forward pass")
        T, N, C = scene_points.shape
        # Encode each point, then mean-pool over points
        flat = scene_points.reshape(T * N, C)
        encoded = self.point_encoder(flat)         # (T*N, D)
        encoded = encoded.reshape(T, N, -1)
        latent = encoded.mean(dim=1)               # (T, D) — mean pool over Gaussians
        return latent


# ---------------------------------------------------------------------------
# 2. Spherical Gap Encoder
# ---------------------------------------------------------------------------

class SphericalGapEncoder(nn.Module if TORCH_AVAILABLE else object):
    """
    Encodes angular gap regions into completion query tokens.

    For each gap cluster (from GapDetector), produces a query vector that
    encodes: (a) the angular position (θ, φ), (b) the gap size, and
    (c) a learnable embedding for the gap type.

    These query tokens attend to the scene latent in the CompletionDiffusion
    cross-attention layers.

    Input:  gaps_list  list of gap dicts from detect_gaps
    Output: gap_queries  (G, D) tensor — one query per gap cluster
    """

    def __init__(self, output_dim: int = 512):
        if TORCH_AVAILABLE:
            super().__init__()
            # Input: [azimuth_sin, azimuth_cos, elevation_sin, elevation_cos, log_size]
            self.gap_encoder = nn.Sequential(
                nn.Linear(5, 128),
                nn.GELU(),
                nn.Linear(128, output_dim),
            )
        self.output_dim = output_dim

    def encode_gaps(self, gaps: List[Dict]) -> "torch.Tensor":
        """
        Encode gap dicts into query tensors.

        Args:
            gaps: list of gap dicts (from GapDetector.detect)
        Returns:
            (G, D) tensor of gap query vectors
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for forward pass")

        if len(gaps) == 0:
            return torch.zeros(0, self.output_dim)

        features = []
        for gap in gaps:
            az_rad = math.radians(gap["azimuth_deg"])
            el_rad = math.radians(gap.get("elevation_deg", 0.0))
            log_size = math.log(gap["size_m3"] + 1e-6)
            features.append([
                math.sin(az_rad),
                math.cos(az_rad),
                math.sin(el_rad),
                math.cos(el_rad),
                log_size,
            ])

        feat_tensor = torch.tensor(features, dtype=torch.float32)
        return self.gap_encoder(feat_tensor)  # (G, D)


# ---------------------------------------------------------------------------
# 3. Completion Diffusion (core innovation)
# ---------------------------------------------------------------------------

class SceneConditionedCrossAttention(nn.Module if TORCH_AVAILABLE else object):
    """
    Cross-attention layer that conditions diffusion denoising on the 4D scene latent.

    This is the KEY INNOVATION vs Vivid4D:
    - Vivid4D conditions on 2D frame features
    - We condition on the 4D scene latent (scene geometry + appearance + dynamics)
    - This means temporal consistency emerges naturally: the model "sees" all
      timesteps when generating content for any single gap region

    query:  diffusion intermediate activations  (B, L, D)
    key/v:  scene latent + gap queries           (B, T+G, D)
    output: scene-conditioned activations        (B, L, D)
    """

    def __init__(self, dim: int = 512, num_heads: int = 8, dropout: float = 0.0):
        if TORCH_AVAILABLE:
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.norm_q = nn.LayerNorm(dim)
            self.norm_kv = nn.LayerNorm(dim)
            self.proj_out = nn.Linear(dim, dim)

    def forward(
        self,
        x: "torch.Tensor",          # (B, L, D) — diffusion activations
        scene_context: "torch.Tensor",  # (B, T+G, D) — scene latent + gap queries
    ) -> "torch.Tensor":
        """
        Returns: (B, L, D) — scene-conditioned activations
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for forward pass")

        xq = self.norm_q(x)
        kv = self.norm_kv(scene_context)
        attn_out, _ = self.attn(xq, kv, kv)
        return x + self.proj_out(attn_out)


class CompletionDiffusion(nn.Module if TORCH_AVAILABLE else object):
    """
    Simplified diffusion backbone for spherical completion.

    In the full system: this wraps the Vivid4D inpainting UNet and injects
    SceneConditionedCrossAttention layers at every downsampling block.

    For unit testing: implements the same interface with synthetic tensors,
    using a 3-layer MLP as a stub UNet that accepts the same conditioning.

    The full Vivid4D integration requires:
    1. Load Vivid4D checkpoint (TODO: GPU)
    2. Register SceneConditionedCrossAttention hooks at each UNet block
    3. Pass scene_latent as conditioning to those hooks at inference time

    Input:
        noisy_frames:   (B, T, C, H, W) — noisy target frames for gap regions
        scene_latent:   (B, T, D)       — 4D scene features from TemporalFeatureExtractor
        gap_queries:    (B, G, D)       — gap position queries from SphericalGapEncoder
        timestep:       (B,)            — diffusion timestep
    Output:
        denoised_frames: (B, T, C, H, W) — predicted clean frames for gap regions
    """

    def __init__(
        self,
        feature_dim: int = 512,
        img_channels: int = 4,   # RGBA
        img_size: int = 64,      # spatial resolution (full: 256 or 512)
        n_timesteps: int = 1000,
    ):
        if TORCH_AVAILABLE:
            super().__init__()
            self.feature_dim = feature_dim
            self.img_channels = img_channels
            self.img_size = img_size
            self.n_timesteps = n_timesteps

            # Stub "UNet" for unit testing (replace with Vivid4D UNet for real inference)
            flat_dim = img_channels * img_size * img_size
            # Encoder: flat frame → feature_dim (the "bottleneck")
            self.stub_encoder = nn.Sequential(
                nn.Linear(flat_dim + feature_dim, feature_dim),
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim),
            )
            # Decoder: feature_dim → flat frame
            self.stub_decoder = nn.Sequential(
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim),
                nn.GELU(),
                nn.Linear(feature_dim, flat_dim),
            )

            # Scene-conditioned cross-attention (injected at UNet middle block)
            # operates entirely in feature_dim space — no dimension mismatch
            self.scene_cross_attn = SceneConditionedCrossAttention(dim=feature_dim)

            # Timestep embedding
            self.time_embed = nn.Embedding(n_timesteps, feature_dim)

            # TODO: GPU — for real inference, replace stub_unet with:
            # from vivid4d.models import Vivid4DUNet
            # self.unet = Vivid4DUNet.from_pretrained("vivid4d/checkpoints/unet")
            # Then inject SceneConditionedCrossAttention at each downsampling block

    def forward(
        self,
        noisy_frames: "torch.Tensor",    # (B, T, C, H, W)
        scene_latent: "torch.Tensor",    # (B, T, D)
        gap_queries: "torch.Tensor",     # (B, G, D)
        timestep: "torch.Tensor",        # (B,)
    ) -> "torch.Tensor":
        """
        Forward pass: denoise gap frames conditioned on 4D scene.
        Returns: (B, T, C, H, W)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for forward pass")

        B, T, C, H, W = noisy_frames.shape

        # 1. Combine scene latent and gap queries into context
        #    scene_latent: (B, T, D), gap_queries: (B, G, D)
        scene_context = torch.cat([scene_latent, gap_queries], dim=1)  # (B, T+G, D)

        # 2. Timestep embedding
        t_emb = self.time_embed(timestep)  # (B, D)

        # 3. Process each timestep frame
        outputs = []
        for t_idx in range(T):
            frame = noisy_frames[:, t_idx]          # (B, C, H, W)
            flat = frame.reshape(B, -1)             # (B, C*H*W)

            # Concatenate with timestep embedding
            t_feat = t_emb + scene_latent[:, t_idx]  # (B, D)
            x = torch.cat([flat, t_feat], dim=1)      # (B, C*H*W + D)

            # Stub encoder: flat frame + time → feature_dim bottleneck
            h = self.stub_encoder(x)                 # (B, D)

            # Apply scene-conditioned cross-attention at the bottleneck
            # h is already (B, D) — reshape to (B, 1, D) for attention
            h_seq = h.unsqueeze(1)                            # (B, 1, D)
            h_attended = self.scene_cross_attn(h_seq, scene_context)  # (B, 1, D)
            h_out = h_attended.squeeze(1)                     # (B, D)

            # Stub decoder: feature_dim → flat frame
            out_flat = self.stub_decoder(h_out)              # (B, C*H*W)
            out_frame = out_flat.reshape(B, C, H, W)
            outputs.append(out_frame)

        return torch.stack(outputs, dim=1)  # (B, T, C, H, W)


# ---------------------------------------------------------------------------
# 4. Gaussian Fusion
# ---------------------------------------------------------------------------

class GaussianFusion:
    """
    Fuses generated RGBA frames back into the 4DGS point cloud representation.

    For each gap region:
    1. Back-project generated pixels to 3D using estimated depth (from D4RT)
    2. Create new Gaussian primitives at the back-projected positions
    3. Blend with existing Gaussians at region boundaries (opacity-weighted)

    Full implementation requires depth maps from D4RT.
    This class handles the geometry for unit testing; full GPU run uses D4RT depth.
    """

    def __init__(self, depth_scale: float = 1.0):
        self.depth_scale = depth_scale

    def backproject_rgba_to_gaussians(
        self,
        rgba_frames: np.ndarray,   # (T, H, W, 4) — RGBA frames for gap region
        depth_map: np.ndarray,     # (T, H, W)    — depth from D4RT (or synthetic)
        camera_pose: np.ndarray,   # (4, 4)       — camera extrinsic matrix
        intrinsics: np.ndarray,    # (3, 3)        — camera intrinsic matrix
    ) -> np.ndarray:
        """
        Back-project RGBA pixels to 3D Gaussian positions.
        Returns: (T*H*W, 7) array of [x, y, z, r, g, b, alpha]
        """
        T, H, W, _ = rgba_frames.shape
        gaussians = []

        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        # Camera-to-world matrix
        R = camera_pose[:3, :3]
        t = camera_pose[:3, 3]

        for t_idx in range(T):
            for v in range(H):
                for u in range(W):
                    depth = depth_map[t_idx, v, u] * self.depth_scale
                    if depth <= 0:
                        continue

                    # Back-project to camera space
                    x_cam = (u - cx) * depth / fx
                    y_cam = (v - cy) * depth / fy
                    z_cam = depth
                    p_cam = np.array([x_cam, y_cam, z_cam])

                    # Transform to world space
                    p_world = R @ p_cam + t

                    rgba = rgba_frames[t_idx, v, u]
                    gaussians.append([
                        p_world[0], p_world[1], p_world[2],
                        float(rgba[0]), float(rgba[1]), float(rgba[2]),
                        float(rgba[3]),
                    ])

        return np.array(gaussians, dtype=np.float32) if gaussians else np.zeros((0, 7), dtype=np.float32)

    def merge_with_scene(
        self,
        existing_points: np.ndarray,   # (N, 6) — existing [x,y,z,r,g,b]
        new_gaussians: np.ndarray,      # (M, 7) — new [x,y,z,r,g,b,alpha]
        blend_radius: float = 0.1,
    ) -> np.ndarray:
        """
        Merge new Gaussians into existing scene with boundary blending.
        Returns: (N+M', 6) merged point cloud
        """
        if len(new_gaussians) == 0:
            return existing_points

        # Only add new Gaussians with alpha > 0.5 (confident predictions)
        high_conf = new_gaussians[new_gaussians[:, 6] > 0.5]
        if len(high_conf) == 0:
            return existing_points

        new_xyz_rgb = high_conf[:, :6]

        # Simple merge (no deduplication for now — TODO: add KD-tree dedup)
        merged = np.concatenate([existing_points, new_xyz_rgb], axis=0)
        return merged


# ---------------------------------------------------------------------------
# 5. Full SphericalCompletionPipeline
# ---------------------------------------------------------------------------

class SphericalCompletionPipeline:
    """
    End-to-end spherical completion pipeline.
    Chains: TemporalFeatureExtractor → SphericalGapEncoder →
            CompletionDiffusion → GaussianFusion

    Example (unit test, CPU):
        pipeline = SphericalCompletionPipeline(device="cpu")
        result = pipeline.complete_synthetic()

    Example (real inference, GPU required):
        # TODO: GPU EXECUTION REQUIRED
        pipeline = SphericalCompletionPipeline(device="cuda")
        result = pipeline.complete(
            scene_ply="outputs/neoverse/scene.ply",
            gaps_json="outputs/gaps/gaps.json",
            output_ply="outputs/completion/scene_complete.ply",
        )
    """

    def __init__(
        self,
        feature_dim: int = 512,
        img_size: int = 64,
        n_diffusion_steps: int = 20,
        device: str = "cpu",
    ):
        self.feature_dim = feature_dim
        self.img_size = img_size
        self.n_diffusion_steps = n_diffusion_steps
        self.device = device

        if TORCH_AVAILABLE:
            self.feature_extractor = TemporalFeatureExtractor(
                input_dim=6, output_dim=feature_dim
            )
            self.gap_encoder = SphericalGapEncoder(output_dim=feature_dim)
            self.diffusion = CompletionDiffusion(
                feature_dim=feature_dim,
                img_channels=4,
                img_size=img_size,
                n_timesteps=1000,
            )
            self.fusion = GaussianFusion()

            # Move to device
            if TORCH_AVAILABLE:
                self.feature_extractor = self.feature_extractor.to(device)
                self.gap_encoder = self.gap_encoder.to(device)
                self.diffusion = self.diffusion.to(device)

    def complete_synthetic(self) -> Dict:
        """
        Run a synthetic forward pass for unit testing (CPU, no real data needed).
        Returns dict with shapes of all intermediate tensors.
        """
        if not TORCH_AVAILABLE:
            # Return expected shapes without torch
            return {
                "status": "torch_not_available",
                "scene_latent_shape": (2, 10, 512),
                "gap_queries_shape": (2, 3, 512),
                "denoised_shape": (2, 10, 4, 64, 64),
            }

        B, T, N = 2, 10, 100   # batch=2, timesteps=10, points=100
        G = 3                   # gap clusters

        # Synthetic scene points: (B, T, N, 6) — [x,y,z,r,g,b]
        scene_pts = torch.randn(B * T, N, 6, device=self.device)

        # Extract scene latent
        latent = self.feature_extractor(scene_pts)   # (B*T, D)
        scene_latent = latent.reshape(B, T, -1)      # (B, T, D)

        # Synthetic gaps
        synthetic_gaps = [
            {"azimuth_deg": 180.0, "elevation_deg": 0.0, "size_m3": 0.5},
            {"azimuth_deg": 90.0,  "elevation_deg": 10.0, "size_m3": 0.3},
            {"azimuth_deg": 270.0, "elevation_deg": -5.0, "size_m3": 0.2},
        ]
        gap_q = self.gap_encoder.encode_gaps(synthetic_gaps).to(self.device)  # (G, D)
        gap_queries = gap_q.unsqueeze(0).expand(B, -1, -1)                    # (B, G, D)

        # Synthetic noisy frames for gap regions: (B, T, 4, H, W)
        noisy = torch.randn(B, T, 4, self.img_size, self.img_size, device=self.device)
        timestep = torch.zeros(B, dtype=torch.long, device=self.device)

        # Completion diffusion forward
        denoised = self.diffusion(noisy, scene_latent, gap_queries, timestep)

        return {
            "status": "ok",
            "scene_latent_shape": tuple(scene_latent.shape),
            "gap_queries_shape": tuple(gap_queries.shape),
            "denoised_shape": tuple(denoised.shape),
            "denoised_mean": float(denoised.mean().item()),
            "denoised_std": float(denoised.std().item()),
        }

    def complete(
        self,
        scene_ply: str,
        gaps_json: str,
        output_ply: str,
        # TODO: GPU EXECUTION REQUIRED for real inference
        # Pass D4RT depth maps and camera poses for proper back-projection
        depth_dir: Optional[str] = None,
        poses_json: Optional[str] = None,
    ) -> Dict:
        """
        Full completion pipeline. Requires GPU for diffusion inference.

        TODO: GPU EXECUTION REQUIRED
        This method loads real NeoVerse PLY + gap JSON and runs the full
        diffusion-based completion. Steps:
          1. Load PLY scene
          2. Load gap clusters
          3. Extract scene latent (TemporalFeatureExtractor)
          4. Encode gap queries (SphericalGapEncoder)
          5. Run DDIM sampling (n_diffusion_steps iterations)
          6. Back-project generated frames to Gaussians (GaussianFusion)
          7. Merge into complete PLY and save
        """
        print(f"[SphericalCompletion] Loading scene: {scene_ply}")
        print(f"[SphericalCompletion] Loading gaps:  {gaps_json}")

        if not os.path.exists(scene_ply):
            raise FileNotFoundError(f"Scene PLY not found: {scene_ply}")
        if not os.path.exists(gaps_json):
            raise FileNotFoundError(f"Gaps JSON not found: {gaps_json}")

        # TODO: GPU EXECUTION REQUIRED
        # Below is a placeholder that copies the input PLY as output
        # and marks it as needing GPU completion.

        with open(gaps_json) as f:
            gaps_data = json.load(f)

        n_gaps = len(gaps_data.get("gaps", []))
        empty_angles = gaps_data.get("summary", {}).get("empty_angles", [])

        print(f"[SphericalCompletion] Found {n_gaps} gap clusters")
        print(f"[SphericalCompletion] Empty angles: {empty_angles}")
        print(f"[SphericalCompletion] [TODO: GPU] Running diffusion completion...")
        print(f"[SphericalCompletion] [TODO: GPU] Load Vivid4D checkpoint + NeoVerse scene features")
        print(f"[SphericalCompletion] [TODO: GPU] Run {self.n_diffusion_steps} DDIM steps per gap cluster")
        print(f"[SphericalCompletion] [TODO: GPU] Back-project RGBA → Gaussians via D4RT depth")

        os.makedirs(os.path.dirname(output_ply) or ".", exist_ok=True)

        # Placeholder: copy input PLY with a completion-pending header
        from src.gap_detection.detect_gaps import load_ply_xyz
        pts = load_ply_xyz(scene_ply)
        n = len(pts)

        header = (
            f"ply\nformat ascii 1.0\n"
            f"comment BARF completion output — {n_gaps} gap clusters targeted\n"
            f"comment TODO: replace with GPU-completed scene\n"
            f"comment Empty angles: {empty_angles}\n"
            f"element vertex {n}\n"
            f"property float x\nproperty float y\nproperty float z\n"
            f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
            f"end_header\n"
        )
        with open(output_ply, "w") as f:
            f.write(header)
            for i in range(n):
                f.write(f"{pts[i,0]:.6f} {pts[i,1]:.6f} {pts[i,2]:.6f} 128 200 255\n")

        print(f"[SphericalCompletion] Placeholder output written: {output_ply}")
        print(f"[SphericalCompletion] Replace with GPU run to get real completion.")

        return {
            "output_ply": output_ply,
            "n_input_points": n,
            "n_gap_clusters": n_gaps,
            "empty_angles": empty_angles,
            "status": "placeholder_gpu_required",
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BARF Spherical 4D Completion Module"
    )
    parser.add_argument("--scene_ply", required=True, help="Input NeoVerse scene PLY")
    parser.add_argument("--gaps_json", required=True, help="Input gaps JSON from detect_gaps")
    parser.add_argument("--output_ply", required=True, help="Output completed scene PLY")
    parser.add_argument("--device", default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--n_steps", type=int, default=20, help="Diffusion DDIM steps")
    args = parser.parse_args()

    pipeline = SphericalCompletionPipeline(
        n_diffusion_steps=args.n_steps,
        device=args.device,
    )
    result = pipeline.complete(
        scene_ply=args.scene_ply,
        gaps_json=args.gaps_json,
        output_ply=args.output_ply,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
