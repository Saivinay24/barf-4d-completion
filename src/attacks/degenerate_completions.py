"""
BARF 4D — Degenerate Completion Attacks
========================================
A family of content-free "completions" that maximise angular coverage
metrics without adding any genuine scene content. These are adversarial
probes for stress-testing VR-completeness metrics: a useful completeness
metric must NOT reward these.

Scene representation (consistent across the eval suite):
    xyz   : (N, 3) float32 — point positions
    rgb   : (N, 3) uint8   — point colours
    frame : (N,)   int32   — temporal frame index of each point

Attack taxonomy:
    dust    — the minimum number of points (exactly the naive density
              threshold tau) placed in each empty sector. Flips a
              threshold-based coverage metric to 100% with ~tau points
              per sector.
    chaff   — volumetric uniform-random points filling each empty sector
              at full observed density, with uniform-random colours.
              "static" variant: the same chaff points appear in every
              frame (temporally stable). "flicker" variant: chaff is
              re-sampled independently per frame (temporally unstable).
    clone   — the observed point cloud is rotated about the vertical axis
              so that a copy of the observed content fills each empty
              sector. Keeps real colours, real local structure, and real
              temporal dynamics. This is simultaneously an "attack" and a
              naive symmetry-prior completion baseline.

All attacks are deterministic given `seed`.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

TEST_ANGLES_DEG = [0, 45, 90, 135, 180, 225, 270, 315]
SECTOR_WIDTH_DEG = 45.0


def sector_azimuth_deg(xyz: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Azimuth (degrees, [-180, 180]) of each point about `center` in the XZ plane."""
    rel = xyz - center.reshape(1, 3)
    return np.degrees(np.arctan2(rel[:, 2], rel[:, 0]))


def sector_mask(theta_deg: np.ndarray, angle_deg: float,
                width_deg: float = SECTOR_WIDTH_DEG) -> np.ndarray:
    diff = np.abs(((theta_deg - angle_deg + 180.0) % 360.0) - 180.0)
    return diff <= width_deg / 2.0


def sector_counts(xyz: np.ndarray, center: np.ndarray,
                  angles: List[int] = TEST_ANGLES_DEG) -> Dict[int, int]:
    theta = sector_azimuth_deg(xyz, center)
    return {a: int(sector_mask(theta, a).sum()) for a in angles}


def empty_sectors(xyz: np.ndarray, center: np.ndarray,
                  density_threshold: int = 5,
                  angles: List[int] = TEST_ANGLES_DEG) -> List[int]:
    counts = sector_counts(xyz, center, angles)
    return [a for a in angles if counts[a] < density_threshold]


def _radial_stats(xyz: np.ndarray, center: np.ndarray) -> Tuple[float, float, float, float]:
    """25th/75th percentile radius and y-range of the observed cloud."""
    rel = xyz - center.reshape(1, 3)
    r = np.sqrt(rel[:, 0] ** 2 + rel[:, 2] ** 2)
    return (float(np.percentile(r, 25)), float(np.percentile(r, 75)),
            float(np.percentile(xyz[:, 1], 10)), float(np.percentile(xyz[:, 1], 90)))


def _sample_sector_points(rng: np.random.Generator, n: int, angle_deg: float,
                          center: np.ndarray, r_lo: float, r_hi: float,
                          y_lo: float, y_hi: float,
                          width_deg: float = SECTOR_WIDTH_DEG) -> np.ndarray:
    """Uniform-random points inside the angular sector wedge (cylindrical shell)."""
    theta = np.radians(angle_deg + rng.uniform(-width_deg / 2, width_deg / 2, n))
    r = rng.uniform(r_lo, r_hi, n)
    y = rng.uniform(y_lo, y_hi, n)
    x = center[0] + r * np.cos(theta)
    z = center[2] + r * np.sin(theta)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def dust_attack(xyz: np.ndarray, rgb: np.ndarray, frame: np.ndarray,
                center: Optional[np.ndarray] = None,
                density_threshold: int = 5,
                seed: int = 0) -> Dict[str, np.ndarray]:
    """Place exactly `density_threshold` grey points in each empty sector."""
    rng = np.random.default_rng(seed)
    if center is None:
        center = xyz.mean(axis=0)
    targets = empty_sectors(xyz, center, density_threshold)
    r_lo, r_hi, y_lo, y_hi = _radial_stats(xyz, center)

    new_xyz, new_rgb, new_frame = [xyz], [rgb], [frame]
    for angle in targets:
        pts = _sample_sector_points(rng, density_threshold, angle, center,
                                    r_lo, r_hi, y_lo, y_hi)
        new_xyz.append(pts)
        new_rgb.append(np.full((len(pts), 3), 128, dtype=np.uint8))
        new_frame.append(np.zeros(len(pts), dtype=np.int32))

    return {"xyz": np.concatenate(new_xyz), "rgb": np.concatenate(new_rgb),
            "frame": np.concatenate(new_frame), "n_added": sum(len(a) for a in new_xyz[1:])}


def chaff_attack(xyz: np.ndarray, rgb: np.ndarray, frame: np.ndarray,
                 center: Optional[np.ndarray] = None,
                 density_threshold: int = 5,
                 temporal_mode: str = "static",
                 seed: int = 0) -> Dict[str, np.ndarray]:
    """
    Fill each empty sector with uniform-random points at full observed density.

    temporal_mode:
        "static"  — one chaff point set, replicated into every frame
        "flicker" — chaff re-sampled independently for every frame
    """
    if temporal_mode not in ("static", "flicker"):
        raise ValueError(f"unknown temporal_mode: {temporal_mode}")
    rng = np.random.default_rng(seed)
    if center is None:
        center = xyz.mean(axis=0)
    counts = sector_counts(xyz, center)
    targets = [a for a in TEST_ANGLES_DEG if counts[a] < density_threshold]
    covered_counts = [c for c in counts.values() if c >= density_threshold]
    per_sector_total = int(np.median(covered_counts)) if covered_counts else 1000

    frames_unique = np.unique(frame)
    T = max(len(frames_unique), 1)
    per_frame = max(per_sector_total // T, 1)
    r_lo, r_hi, y_lo, y_hi = _radial_stats(xyz, center)

    new_xyz, new_rgb, new_frame = [xyz], [rgb], [frame]
    for angle in targets:
        if temporal_mode == "static":
            pts = _sample_sector_points(rng, per_frame, angle, center,
                                        r_lo, r_hi, y_lo, y_hi)
            cols = rng.integers(0, 256, (per_frame, 3), dtype=np.uint8)
            for t in frames_unique:
                new_xyz.append(pts)
                new_rgb.append(cols)
                new_frame.append(np.full(per_frame, t, dtype=np.int32))
        elif temporal_mode == "flicker":
            for t in frames_unique:
                pts = _sample_sector_points(rng, per_frame, angle, center,
                                            r_lo, r_hi, y_lo, y_hi)
                new_xyz.append(pts)
                new_rgb.append(rng.integers(0, 256, (per_frame, 3), dtype=np.uint8))
                new_frame.append(np.full(per_frame, t, dtype=np.int32))

    return {"xyz": np.concatenate(new_xyz), "rgb": np.concatenate(new_rgb),
            "frame": np.concatenate(new_frame), "n_added": sum(len(a) for a in new_xyz[1:])}


def clone_attack(xyz: np.ndarray, rgb: np.ndarray, frame: np.ndarray,
                 center: Optional[np.ndarray] = None,
                 density_threshold: int = 5,
                 seed: int = 0) -> Dict[str, np.ndarray]:
    """
    Rotate a copy of the densest observed sector's content about the vertical
    axis through `center` so that it fills each empty sector. Preserves real
    colours, local structure, and per-frame dynamics.
    """
    if center is None:
        center = xyz.mean(axis=0)
    counts = sector_counts(xyz, center)
    targets = [a for a in TEST_ANGLES_DEG if counts[a] < density_threshold]
    src_angle = max(counts, key=counts.get)

    theta = sector_azimuth_deg(xyz, center)
    src_mask = sector_mask(theta, src_angle)
    src_xyz = xyz[src_mask] - center.reshape(1, 3)
    src_rgb = rgb[src_mask]
    src_frame = frame[src_mask]

    new_xyz, new_rgb, new_frame = [xyz], [rgb], [frame]
    for angle in targets:
        rot = np.radians(angle - src_angle)
        c, s = np.cos(rot), np.sin(rot)
        # rotation about the y axis (XZ plane)
        rx = src_xyz[:, 0] * c - src_xyz[:, 2] * s
        rz = src_xyz[:, 0] * s + src_xyz[:, 2] * c
        pts = np.stack([rx, src_xyz[:, 1], rz], axis=1).astype(np.float32)
        new_xyz.append(pts + center.reshape(1, 3))
        new_rgb.append(src_rgb.copy())
        new_frame.append(src_frame.copy())

    return {"xyz": np.concatenate(new_xyz), "rgb": np.concatenate(new_rgb),
            "frame": np.concatenate(new_frame), "n_added": sum(len(a) for a in new_xyz[1:])}


ATTACK_REGISTRY = {
    "dust": lambda x, r, f, **kw: dust_attack(x, r, f, **kw),
    "chaff_static": lambda x, r, f, **kw: chaff_attack(x, r, f, temporal_mode="static", **kw),
    "chaff_flicker": lambda x, r, f, **kw: chaff_attack(x, r, f, temporal_mode="flicker", **kw),
    "clone": lambda x, r, f, **kw: clone_attack(x, r, f, **kw),
}


def apply_attack(name: str, xyz: np.ndarray, rgb: np.ndarray, frame: np.ndarray,
                 **kwargs) -> Dict[str, np.ndarray]:
    if name not in ATTACK_REGISTRY:
        raise ValueError(f"unknown attack: {name}. Available: {list(ATTACK_REGISTRY)}")
    return ATTACK_REGISTRY[name](xyz, rgb, frame, **kwargs)
