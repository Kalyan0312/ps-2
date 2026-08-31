#!/usr/bin/env python3
"""
Phase 5 Adaptive Resolution Decision Runner.
Loads sample data, runs preprocessing, builds a uniform 2.5D map,
computes terrain complexity features, determines resolution allocation per cell,
and outputs the resolution distribution.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.data import SyntheticLidarLoader
from backend.preprocessing import PreprocessingPipeline
from backend.mapping import Uniform25DMapBuilder
from backend.terrain import TerrainAnalyzer
from backend.priority import AdaptiveResolutionSelector


def main():
    print("=" * 75)
    print("       Adaptive LiDAR - Phase 5 Adaptive Resolution Decision Engine       ")
    print("=" * 75)

    config_path = ROOT_DIR / "configs" / "config.yaml"

    # Step 1: Load Phase 1 Sample Frame
    loader = SyntheticLidarLoader(num_frames=1, seed=42)
    raw_frame = loader[0]

    # Step 2: Run Phase 2 Preprocessing Pipeline
    prep_pipeline = PreprocessingPipeline.from_config_file(config_path)
    clean_frame = prep_pipeline.process(raw_frame)

    # Step 3: Build Uniform 2.5D Map (Phase 3A)
    map_builder = Uniform25DMapBuilder.from_config_file(config_path)
    grid_map = map_builder.build_map(clean_frame)

    # Step 4: Run Terrain Analysis (Phase 4)
    analyzer = TerrainAnalyzer.from_config_file(config_path)
    terrain_result = analyzer.analyze(grid_map)

    # Step 5: Run Adaptive Resolution Decision (Phase 5)
    selector = AdaptiveResolutionSelector.from_config_file(config_path)
    decision = selector.select_resolution(grid_map, terrain_result)

    counts = decision.counts
    pcts = decision.percentages
    res_levels = decision.resolution_levels

    # Step 6: Print Required Output Fields
    print("[*] Resolution Allocation Decision Results:")
    print(f"    - Total occupied cells  : {grid_map.num_occupied_cells}")
    print(f"    - Analyzed cells        : {decision.num_decided_cells}")
    print("-" * 75)
    print(f"{'Tier':<14} {'Resolution':<12} {'Cell Count':<14} {'Percentage':<12}")
    print("-" * 75)

    for tier in ["coarse", "medium", "fine", "ultra_fine"]:
        res_val = f"{res_levels.get(tier, 0.0):.2f} m"
        c_val = f"{counts.get(tier, 0):,}"
        p_val = f"{pcts.get(tier, 0.0):.2f}%"
        print(f"{tier:<14} {res_val:<12} {c_val:<14} {p_val:<12}")

    print("=" * 75)


if __name__ == "__main__":
    main()
