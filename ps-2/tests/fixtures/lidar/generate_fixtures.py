#!/usr/bin/env python3
"""
Fixture Generator: Deterministic Demo LiDAR Dataset (Phase 17 Part C)

Generates 5 small but spatially rich .npy point cloud frames that are used as
the built-in demo fixture for `run_dataset_replay_demo.py` and related tests.

All frames are 100% deterministic (fixed seed per frame). Re-running this script
will always produce identical output files.

Data layout (per .npy file):
    NumPy array of shape (N, 4), dtype float32
    Columns: [x (m), y (m), z (m), intensity (0.0–1.0)]

Scene description:
    The five frames simulate a vehicle driving slowly through a structured scene:
    - Flat ground plane covering ±15 m in X and Y
    - Several rectangular elevated obstacles (walls, curbs, objects) that shift
      slightly between frames to represent sensor motion and scene dynamics
    - Point density ≈ 2 000 pts/frame (fast processing, visible spatial variation)

Usage:
    python tests/fixtures/lidar/generate_fixtures.py

Output:
    tests/fixtures/lidar/frame_0000.npy  (frame 0 — initial scene)
    tests/fixtures/lidar/frame_0001.npy  (frame 1 — minor scene shift)
    tests/fixtures/lidar/frame_0002.npy  (frame 2 — obstacle appears)
    tests/fixtures/lidar/frame_0003.npy  (frame 3 — large scene change)
    tests/fixtures/lidar/frame_0004.npy  (frame 4 — stable scene again)
    tests/fixtures/lidar/README.md       (format documentation)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Resolve project root so this script can be run from anywhere
# --------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
FIXTURE_DIR = SCRIPT_DIR  # save .npy files next to this script


# --------------------------------------------------------------------------
# Scene generation helpers
# --------------------------------------------------------------------------

def _flat_ground(rng: np.random.Generator, n: int, x_range: float, y_range: float,
                 z_noise: float = 0.03) -> np.ndarray:
    """Flat ground plane with small elevation noise."""
    x = rng.uniform(-x_range, x_range, n).astype(np.float32)
    y = rng.uniform(-y_range, y_range, n).astype(np.float32)
    z = rng.normal(0.0, z_noise, n).astype(np.float32)
    intensity = rng.uniform(0.2, 0.5, n).astype(np.float32)
    return np.column_stack([x, y, z, intensity])


def _rectangular_object(rng: np.random.Generator, n: int,
                         cx: float, cy: float, width: float, depth: float,
                         height: float, z_base: float = 0.0) -> np.ndarray:
    """Solid rectangular obstacle at (cx, cy) with given dimensions."""
    x = rng.uniform(cx - width / 2, cx + width / 2, n).astype(np.float32)
    y = rng.uniform(cy - depth / 2, cy + depth / 2, n).astype(np.float32)
    z = (rng.uniform(z_base, z_base + height, n)).astype(np.float32)
    intensity = rng.uniform(0.5, 0.9, n).astype(np.float32)
    return np.column_stack([x, y, z, intensity])


def _wall(rng: np.random.Generator, n: int,
          x0: float, y0: float, x1: float, y1: float,
          height: float, z_base: float = 0.0) -> np.ndarray:
    """Thin vertical wall from (x0,y0) to (x1,y1)."""
    t = rng.uniform(0, 1, n).astype(np.float32)
    x = (x0 + t * (x1 - x0)).astype(np.float32)
    y = (y0 + t * (y1 - y0)).astype(np.float32)
    z = rng.uniform(z_base, z_base + height, n).astype(np.float32)
    intensity = rng.uniform(0.6, 0.95, n).astype(np.float32)
    return np.column_stack([x, y, z, intensity])


def _make_frame(frame_idx: int, n_total: int = 2000) -> np.ndarray:
    """
    Build one deterministic frame.

    Each frame shares the same basic scene but modifies obstacle positions/heights
    slightly to exercise the pipeline's incremental update logic.
    """
    base_seed = 42 + frame_idx * 7          # unique, deterministic seed per frame
    rng = np.random.default_rng(base_seed)

    parts = []

    # Ground plane ≈ 60 % of points
    n_ground = int(n_total * 0.60)
    parts.append(_flat_ground(rng, n_ground, x_range=15.0, y_range=15.0))

    # Static low curb / road edge (present in all frames)
    n_curb = int(n_total * 0.08)
    parts.append(_wall(rng, n_curb, -14.0, -14.0, 14.0, -14.0, height=0.15))
    parts.append(_wall(rng, n_curb, -14.0,  14.0, 14.0,  14.0, height=0.15))

    # Building / tall obstacle: shifts slightly between frames
    n_bldg = int(n_total * 0.12)
    offset_x = float(frame_idx) * 0.5          # small sensor-motion shift
    parts.append(_rectangular_object(
        rng, n_bldg,
        cx=5.0 + offset_x, cy=5.0,
        width=4.0, depth=3.0, height=2.5,
    ))

    # Smaller object: appears in frames 2+ to trigger INCREMENTAL_UPDATE / FULL_REBUILD
    n_small = int(n_total * 0.10)
    if frame_idx >= 2:
        parts.append(_rectangular_object(
            rng, n_small,
            cx=-4.0, cy=-3.0,
            width=1.5, depth=1.5, height=0.8,
        ))
    else:
        # Pad with more ground so total point count is stable
        parts.append(_flat_ground(rng, n_small, x_range=6.0, y_range=6.0))

    # Frame 3: additional large scene change (tall structure in new region)
    if frame_idx == 3:
        n_extra = int(n_total * 0.10)
        parts.append(_rectangular_object(
            rng, n_extra,
            cx=-8.0, cy=8.0,
            width=5.0, depth=5.0, height=4.0,
        ))

    pts = np.vstack(parts).astype(np.float32)

    # Shuffle rows so there is no accidental sequential spatial ordering
    rng.shuffle(pts)

    return pts


def main() -> None:
    n_frames = 5
    n_points = 2000

    print("=" * 60)
    print("LIDAR FIXTURE GENERATOR — Phase 17 Part C")
    print("=" * 60)
    print(f"Output directory : {FIXTURE_DIR}")
    print(f"Frames           : {n_frames}")
    print(f"Points per frame : ~{n_points}")
    print()

    for i in range(n_frames):
        pts = _make_frame(i, n_total=n_points)
        out_path = FIXTURE_DIR / f"frame_{i:04d}.npy"
        np.save(out_path, pts)
        min_xyz = pts[:, :3].min(axis=0)
        max_xyz = pts[:, :3].max(axis=0)
        print(
            f"  frame_{i:04d}.npy  shape={pts.shape}  "
            f"x=[{min_xyz[0]:.1f}, {max_xyz[0]:.1f}]  "
            f"y=[{min_xyz[1]:.1f}, {max_xyz[1]:.1f}]  "
            f"z=[{min_xyz[2]:.2f}, {max_xyz[2]:.2f}]"
        )

    print()
    print("Fixture files written successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
