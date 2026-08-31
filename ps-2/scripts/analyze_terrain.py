#!/usr/bin/env python3
"""
Phase 4 Terrain Complexity and Roughness Analysis Runner.
Loads sample data, runs preprocessing, builds a uniform 2.5D map,
computes terrain complexity metrics, and outputs statistics.
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


def main():
    print("=" * 70)
    print("      Adaptive LiDAR - Phase 4 Terrain Complexity & Roughness      ")
    print("=" * 70)

    config_path = ROOT_DIR / "configs" / "config.yaml"

    # Step 1: Load Phase 1 Sample Frame
    loader = SyntheticLidarLoader(num_frames=1, seed=42)
    raw_frame = loader[0]

    # Step 2: Run Phase 2 Preprocessing Pipeline
    prep_pipeline = PreprocessingPipeline.from_config_file(config_path)
    clean_frame = prep_pipeline.process(raw_frame)

    # Step 3: Build Uniform 2.5D Elevation Map
    map_builder = Uniform25DMapBuilder.from_config_file(config_path)
    grid_map = map_builder.build_map(clean_frame)

    # Step 4: Run Terrain Analysis
    analyzer = TerrainAnalyzer.from_config_file(config_path)
    result = analyzer.analyze(grid_map)
    stats = result.get_stats()

    # Step 5: Print Required Output Fields
    print("[*] Terrain Complexity Analysis Results:")
    print(f"    - Map resolution             : {grid_map.resolution:.2f} m/cell")
    print(f"    - Occupied cells             : {grid_map.num_occupied_cells}")
    print(f"    - Analyzed terrain cells     : {result.num_analyzed_cells}")
    print(f"    - Minimum roughness (std dev): {stats['roughness']['min']:.4f} m")
    print(f"    - Maximum roughness (std dev): {stats['roughness']['max']:.4f} m")
    print(f"    - Mean roughness (std dev)   : {stats['roughness']['mean']:.4f} m")
    print(f"    - Minimum slope (gradient)   : {stats['slope']['min']:.4f} m/m")
    print(f"    - Maximum slope (gradient)   : {stats['slope']['max']:.4f} m/m")
    print(f"    - Mean slope (gradient)      : {stats['slope']['mean']:.4f} m/m")
    print(f"    - Mean elevation range       : {stats['elevation_range']['mean']:.4f} m")
    print("=" * 70)


if __name__ == "__main__":
    main()
