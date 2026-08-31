#!/usr/bin/env python3
"""
Phase 8: Real LiDAR Point Cloud End-to-End Pipeline Execution Script.

Loads a real LiDAR scan (.npy or KITTI .bin), passes it through the exact existing pipeline:
  Raw Real LiDAR Frame
  → PreprocessingPipeline
  → Uniform25DMapBuilder (Base Grid)
  → TerrainAnalyzer (Roughness, Slope, Elevation Range)
  → AdaptiveResolutionSelector (Tier Decisions)
  → AdaptiveMap25DBuilder (Multi-Tier Composite Map)
  → Metrics & Evaluation Summary

Usage:
  python scripts/run_real_lidar_pipeline.py --file /path/to/kitti/000000.bin
  python scripts/run_real_lidar_pipeline.py --file data/sample_kitti.bin
  python scripts/run_real_lidar_pipeline.py --file data/sample_pointcloud.npy
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.data.factory import load_lidar_file
from backend.preprocessing.pipeline import PreprocessingPipeline
from backend.preprocessing.config import PreprocessingConfig
from backend.mapping.config import MappingConfig
from backend.mapping.uniform_builder import Uniform25DMapBuilder
from backend.mapping.adaptive_builder import AdaptiveMap25DBuilder
from backend.terrain.analyzer import TerrainAnalyzer
from backend.terrain.config import TerrainConfig
from backend.priority.selector import AdaptiveResolutionSelector
from backend.priority.config import PriorityConfig
from backend.metrics.map_metrics import calculate_adaptive_map_metrics, calculate_grid_map_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run complete Adaptive 2.5D Mapping pipeline on real LiDAR scans."
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Path to real LiDAR point cloud file (.npy or KITTI .bin)",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--default-intensity",
        type=float,
        default=0.5,
        help="Default intensity if Nx3 points are loaded",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    target_file = args.file
    if target_file is None:
        sample_bin = PROJECT_ROOT / "data" / "sample_kitti.bin"
        if sample_bin.is_file():
            print("[*] No --file specified. Using built-in sample fixture: data/sample_kitti.bin")
            print("    To run on your own dataset: python scripts/run_real_lidar_pipeline.py --file /path/to/velodyne/000000.bin\n")
            target_file = str(sample_bin)
        else:
            print("[!] Error: No LiDAR file provided. Use --file /path/to/scan.bin", file=sys.stderr)
            sys.exit(1)

    path = Path(target_file)
    if not path.is_file():
        print(f"[!] Error: File does not exist: {path.resolve()}", file=sys.stderr)
        sys.exit(1)

    config_path = Path(args.config)

    # 1. Ingest real LiDAR data
    print("============================================================")
    print("REAL LiDAR ADAPTIVE 2.5D MAPPING PIPELINE")
    print("============================================================")
    print(f"[*] Ingesting file: {path.name} (Format: {path.suffix})")

    raw_frame = load_lidar_file(
        file_path=path,
        default_intensity=args.default_intensity,
    )
    print(f"    Raw points loaded : {raw_frame.num_points:,}")
    print(f"    Sensor coordinates: {raw_frame.metadata.get('sensor_coordinates', 'cartesian')}")

    # 2. Preprocessing
    prep_cfg = PreprocessingConfig.from_yaml(config_path) if config_path.is_file() else PreprocessingConfig()
    preprocessor = PreprocessingPipeline(config=prep_cfg)
    prep_frame = preprocessor.process(raw_frame)
    retained_pct = (prep_frame.num_points / raw_frame.num_points * 100.0) if raw_frame.num_points > 0 else 0.0
    print(f"[*] Preprocessing filtered to: {prep_frame.num_points:,} points ({retained_pct:.1f}% retained)")

    # 3. Base grid map construction
    map_cfg = MappingConfig.from_yaml(config_path) if config_path.is_file() else MappingConfig()
    base_builder = Uniform25DMapBuilder(config=map_cfg)
    base_grid = base_builder.build_map(prep_frame)
    print(f"[*] Base grid constructed at {base_grid.resolution:.2f} m/cell (Shape: {base_grid.shape[0]}x{base_grid.shape[1]})")

    # 4. Terrain complexity analysis
    terrain_cfg = TerrainConfig.from_yaml(config_path) if config_path.is_file() else TerrainConfig()
    terrain_analyzer = TerrainAnalyzer(config=terrain_cfg)
    terrain_result = terrain_analyzer.analyze(base_grid)
    print(f"[*] Terrain analysis completed across {terrain_result.num_analyzed_cells:,} occupied cells")

    # 5. Adaptive resolution selection
    priority_cfg = PriorityConfig.from_yaml(config_path) if config_path.is_file() else PriorityConfig()
    adaptive_selector = AdaptiveResolutionSelector(config=priority_cfg)
    decision_result = adaptive_selector.select_resolution(base_grid, terrain_result)

    counts = decision_result.counts
    pcts = decision_result.percentages
    print("\n--- Resolution Tier Decisions ---")
    for tier in ["coarse", "medium", "fine", "ultra_fine"]:
        c = counts.get(tier, 0)
        p = pcts.get(tier, 0.0)
        res = map_cfg.multi_resolution_levels.get(tier, 0.0)
        print(f"  {tier:<12} @ {res:.2f} m : {c:>6,} cells ({p:>5.1f}%)")

    # 6. Adaptive map reconstruction
    adaptive_builder = AdaptiveMap25DBuilder(config=map_cfg)
    adaptive_map = adaptive_builder.build(prep_frame, decision_result)
    print(f"\n[*] Adaptive composite map built successfully across {len(adaptive_map.tier_maps)} active tiers")

    # 7. Metrics calculation
    adaptive_metrics = calculate_adaptive_map_metrics(adaptive_map, name="Adaptive Map")
    uniform_coarse = base_builder.build_map(prep_frame)
    coarse_metrics = calculate_grid_map_metrics(uniform_coarse, name="Uniform Coarse")

    print("\n--- Map Complexity & Memory Metrics ---")
    print(f"  Base decision cells          : {adaptive_metrics.adaptive_breakdown.base_decision_cells:,}")
    print(f"  Represented base cells       : {adaptive_metrics.adaptive_breakdown.base_represented_cells:,}")
    print(f"  Tier-allocated grid cells    : {adaptive_metrics.adaptive_breakdown.tier_allocated_cells:,}")
    print(f"  Tier-occupied cells          : {adaptive_metrics.adaptive_breakdown.tier_occupied_cells:,}")
    print(f"  Actual array memory (ndarray): {adaptive_metrics.memory.array_kb:.2f} KB ({adaptive_metrics.memory.array_mb:.4f} MB)")

    # 8. Query verification
    if prep_frame.num_points > 0:
        sample_pt = prep_frame.points[0]
        query_res = adaptive_map.world_query(float(sample_pt[0]), float(sample_pt[1]))
        print(f"\n--- Point Query Test at ({sample_pt[0]:.2f}, {sample_pt[1]:.2f}) ---")
        print(f"  Represented  : {query_res.is_represented}")
        print(f"  Tier         : {query_res.tier} ({query_res.resolution} m)")
        print(f"  Mean Z       : {query_res.mean_z:.3f} m (Point count: {query_res.point_count})")

    print("============================================================")
    print("Real LiDAR end-to-end adaptive mapping pipeline complete.")
    print("============================================================")


if __name__ == "__main__":
    main()
