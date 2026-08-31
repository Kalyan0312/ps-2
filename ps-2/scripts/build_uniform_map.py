#!/usr/bin/env python3
"""
Phase 3A Uniform 2.5D Elevation Grid Map Runner.
Loads a sample frame, applies preprocessing, constructs a uniform 2.5D map,
and prints grid statistics.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.data import SyntheticLidarLoader
from backend.preprocessing import PreprocessingPipeline
from backend.mapping import Uniform25DMapBuilder


def main():
    print("=" * 65)
    print("      Adaptive LiDAR - Phase 3A Uniform 2.5D Grid Mapping      ")
    print("=" * 65)

    config_path = ROOT_DIR / "configs" / "config.yaml"

    # Step 1: Load Phase 1 Frame
    loader = SyntheticLidarLoader(num_frames=1, seed=42)
    raw_frame = loader[0]
    raw_point_count = raw_frame.num_points

    # Step 2: Run Phase 2 Preprocessing Pipeline
    prep_pipeline = PreprocessingPipeline.from_config_file(config_path)
    clean_frame = prep_pipeline.process(raw_frame)
    clean_point_count = clean_frame.num_points

    # Step 3: Build Uniform 2.5D Map
    map_builder = Uniform25DMapBuilder.from_config_file(config_path)
    grid_map = map_builder.build_map(clean_frame)

    summary = grid_map.get_summary()

    # Step 4: Display required statistics
    print("[*] Uniform 2.5D Map Results:")
    print(f"    - Original/input point count : {raw_point_count}")
    print(f"    - Preprocessed point count   : {clean_point_count}")
    print(f"    - Map resolution             : {grid_map.resolution:.2f} m/cell")
    print(f"    - Map bounds X               : [{grid_map.min_x:.1f} m, {grid_map.max_x:.1f} m]")
    print(f"    - Map bounds Y               : [{grid_map.min_y:.1f} m, {grid_map.max_y:.1f} m]")
    print(f"    - Map dimensions             : {grid_map.rows} rows x {grid_map.cols} cols (shape: {grid_map.shape})")
    print(f"    - Total number of grid cells : {grid_map.total_cells}")
    print(f"    - Occupied cells             : {grid_map.num_occupied_cells}")
    print(f"    - Empty cells                : {grid_map.num_empty_cells}")
    print(f"    - Occupancy percentage       : {grid_map.occupancy_percentage:.2f}%")
    if summary["min_elevation"] is not None:
        print(f"    - Elevation range (Z)        : {summary['min_elevation']:.3f} m -> {summary['max_elevation']:.3f} m")
    print("=" * 65)


if __name__ == "__main__":
    main()
