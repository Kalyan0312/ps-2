#!/usr/bin/env python3
"""
Phase 3B Multi-Resolution 2.5D Elevation Grid Map Runner.
Loads sample data, runs preprocessing, builds multi-scale 2.5D maps,
and displays a formatted comparison table across all resolution levels.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.data import SyntheticLidarLoader
from backend.preprocessing import PreprocessingPipeline
from backend.mapping import MultiResolution25DMapBuilder


def main():
    print("=" * 80)
    print("   Adaptive LiDAR - Phase 3B Multi-Resolution 2.5D Elevation Mapping   ")
    print("=" * 80)

    config_path = ROOT_DIR / "configs" / "config.yaml"

    # Step 1: Load Phase 1 Sample Frame
    loader = SyntheticLidarLoader(num_frames=1, seed=42)
    raw_frame = loader[0]
    raw_count = raw_frame.num_points

    # Step 2: Run Phase 2 Preprocessing Pipeline
    prep_pipeline = PreprocessingPipeline.from_config_file(config_path)
    clean_frame = prep_pipeline.process(raw_frame)
    clean_count = clean_frame.num_points

    print(f"[*] Input Frame Points: {raw_count} (Raw) -> {clean_count} (Preprocessed)")
    print(f"[*] Map Bounds X: [{prep_pipeline.config.min_range:.1f}m - {prep_pipeline.config.max_range:.1f}m range config]")
    print("-" * 80)

    # Step 3: Build Multi-Resolution Maps
    builder = MultiResolution25DMapBuilder.from_config_file(config_path)
    multires_map = builder.build_multires_map(clean_frame)

    # Step 4: Print Comparison Table
    print(f"{'Level':<12} {'Resolution':<12} {'Grid Dims':<14} {'Total Cells':<14} {'Occupied Cells':<16} {'Occupancy %':<12}")
    print("-" * 80)

    for item in multires_map.get_summary():
        lvl_name = item['level_name'] or 'custom'
        res_str = f"{item['resolution']:.2f} m"
        dims_str = f"{item['rows']}x{item['cols']}"
        total_str = f"{item['total_cells']:,}"
        occ_str = f"{item['occupied_cells']:,}"
        pct_str = f"{item['occupancy_percentage']:.2f}%"

        print(f"{lvl_name:<12} {res_str:<12} {dims_str:<14} {total_str:<14} {occ_str:<16} {pct_str:<12}")

    print("=" * 80)


if __name__ == "__main__":
    main()
