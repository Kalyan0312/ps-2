#!/usr/bin/env python3
"""
Phase 2 Preprocessing Sample Frame Runner.
Loads a Phase 1 sample frame, executes the preprocessing pipeline using config.yaml,
and reports filtering diagnostics.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.data import SyntheticLidarLoader
from backend.preprocessing import PreprocessingPipeline, PreprocessingConfig


def main():
    print("=" * 65)
    print("       Adaptive LiDAR - Phase 2 Preprocessing Runner       ")
    print("=" * 65)

    config_path = ROOT_DIR / "configs" / "config.yaml"
    pipeline = PreprocessingPipeline.from_config_file(config_path)

    loader = SyntheticLidarLoader(num_frames=1, seed=42)
    raw_frame = loader[0]

    raw_min, raw_max = raw_frame.bounds
    raw_points = raw_frame.num_points

    print(f"[*] Loaded Raw Frame ID: {raw_frame.frame_id}")
    print(f"[*] Original number of points: {raw_points}")
    print("[*] Original coordinate bounds (min -> max):")
    print(f"    X: {raw_min[0]:.3f} m  ->  {raw_max[0]:.3f} m")
    print(f"    Y: {raw_min[1]:.3f} m  ->  {raw_max[1]:.3f} m")
    print(f"    Z: {raw_min[2]:.3f} m  ->  {raw_max[2]:.3f} m")
    print("-" * 65)

    # Run preprocessing
    cleaned_frame = pipeline.process(raw_frame)
    clean_min, clean_max = cleaned_frame.bounds
    clean_points = cleaned_frame.num_points
    removed_points = raw_points - clean_points

    print("[*] Preprocessing completed.")
    print(f"[*] Remaining number of points: {clean_points}")
    print(f"[*] Points removed: {removed_points} ({removed_points / raw_points * 100:.2f}%)")
    print("[*] Preprocessed coordinate bounds (min -> max):")
    print(f"    X: {clean_min[0]:.3f} m  ->  {clean_max[0]:.3f} m")
    print(f"    Y: {clean_min[1]:.3f} m  ->  {clean_max[1]:.3f} m")
    print(f"    Z: {clean_min[2]:.3f} m  ->  {clean_max[2]:.3f} m")
    print("-" * 65)

    stages = cleaned_frame.metadata.get("preprocessing", {}).get("stages", {})
    print("[*] Stage-by-stage removal breakdown:")
    for stage_name, count in stages.items():
        print(f"    - {stage_name}: {count} points")
    print("=" * 65)


if __name__ == "__main__":
    main()
