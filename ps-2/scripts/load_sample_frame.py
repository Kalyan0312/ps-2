#!/usr/bin/env python3
"""
Sample Frame Loader Script
Loads a sample synthetic LiDAR frame and displays its structural metrics.
"""

import sys
from pathlib import Path

# Ensure project root is on path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.data import SyntheticLidarLoader


def main():
    print("=" * 60)
    print("        Adaptive LiDAR - Phase 1 Sample Frame Loader        ")
    print("=" * 60)

    loader = SyntheticLidarLoader(num_frames=3, seed=42)
    print(f"[*] Total frames initialized: {len(loader)}")
    
    # Load frame 0
    frame = loader[0]
    min_xyz, max_xyz = frame.bounds

    print(f"[*] Frame ID: {frame.frame_id} (Timestamp: {frame.timestamp:.2f}s)")
    print(f"[*] Number of Points: {frame.num_points}")
    print(f"[*] Data Shape: {frame.points.shape} (Schema: [x, y, z, intensity])")
    print("[*] Coordinate Extents (min -> max):")
    print(f"    X: {min_xyz[0]:.3f} m  ->  {max_xyz[0]:.3f} m")
    print(f"    Y: {min_xyz[1]:.3f} m  ->  {max_xyz[1]:.3f} m")
    print(f"    Z: {min_xyz[2]:.3f} m  ->  {max_xyz[2]:.3f} m")
    print(f"[*] Intensity Range: {frame.intensity.min():.3f} -> {frame.intensity.max():.3f}")
    if frame.metadata:
        print("[*] Frame Metadata:")
        for k, v in frame.metadata.items():
            print(f"    - {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
