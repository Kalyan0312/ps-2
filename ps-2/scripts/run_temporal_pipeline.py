#!/usr/bin/env python3
"""
Phase 9: Temporal Multi-Frame Processing and Change Detection Script.

Executes sequential processing over consecutive LiDAR frames, evaluates spatial
stability, identifies new/removed obstacles, and tracks elevation variations across time.

Usage:
  python scripts/run_temporal_pipeline.py
  python scripts/run_temporal_pipeline.py --scenario dynamic
  python scripts/run_temporal_pipeline.py --scenario static
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.data.synthetic import SyntheticLidarGenerator
from backend.preprocessing.pipeline import PreprocessingPipeline
from backend.preprocessing.config import PreprocessingConfig
from backend.mapping.config import MappingConfig
from backend.mapping.uniform_builder import Uniform25DMapBuilder
from backend.temporal.config import TemporalConfig
from backend.temporal.processor import TemporalProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Temporal Multi-Frame Processing and Change Detection."
    )
    parser.add_argument(
        "--scenario", "-s",
        type=str,
        default="dynamic",
        choices=["dynamic", "static", "new_obstacle", "removed_obstacle", "elevation_change"],
        help="Temporal scenario to simulate ('dynamic', 'static', 'new_obstacle', 'removed_obstacle', 'elevation_change')",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.15,
        help="Elevation difference threshold in meters",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    prep_cfg = PreprocessingConfig.from_yaml(config_path) if config_path.is_file() else PreprocessingConfig()
    map_cfg = MappingConfig.from_yaml(config_path) if config_path.is_file() else MappingConfig()
    temporal_cfg = TemporalConfig.from_yaml(config_path) if config_path.is_file() else TemporalConfig()
    temporal_cfg.elevation_change_threshold = args.threshold

    preprocessor = PreprocessingPipeline(config=prep_cfg)
    builder = Uniform25DMapBuilder(config=map_cfg)
    processor = TemporalProcessor(config=temporal_cfg)

    generator = SyntheticLidarGenerator(seed=42)
    frames = generator.generate_temporal_sequence(
        scenario=args.scenario,
        num_ground_points=4500,
    )

    print("==================================================================================")
    print(f"ADAPTIVE LiDAR MAPPING — PHASE 9 TEMPORAL PROCESSING & CHANGE DETECTION")
    print("==================================================================================")
    print(f"Scenario                     : {args.scenario}")
    print(f"Total Frames Ingested        : {len(frames)}")
    print(f"Elevation Change Threshold   : {temporal_cfg.elevation_change_threshold:.2f} m")
    print(f"Base Grid Resolution         : {map_cfg.resolution:.2f} m/cell")
    print("==================================================================================\n")

    # Header for tabular summary
    header = (
        f"{'Frame':<7} {'Points':>8} {'Occupied':>10} {'Changed':>9} "
        f"{'Stable':>8} {'New Occ':>9} {'Rem Occ':>9} {'Elev Chg':>10} "
        f"{'Change %':>10} {'Reusable %':>12}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for idx, frame in enumerate(frames):
        # 1. Preprocessing
        prep_frame = preprocessor.process(frame)

        # 2. Build 2.5D Grid Map
        grid_map = builder.build_map(prep_frame)

        # 3. Temporal Processing & Change Detection
        result = processor.process_grid_map(grid_map, frame_id=frame.frame_id)
        results.append(result)

        stage_desc = frame.metadata.get("stage", "")

        print(
            f"{result.frame_index:<7} {prep_frame.num_points:>8,} {grid_map.num_occupied_cells:>10,} "
            f"{result.num_changed_cells:>9,} {result.num_stable_cells:>8,} "
            f"{result.num_new_occupied_cells:>9,} {result.num_removed_occupied_cells:>9,} "
            f"{result.num_elevation_changed_cells:>10,} "
            f"{result.change_percentage:>9.1f}% {result.stable_percentage:>11.1f}%"
        )

    print("-" * len(header))
    print("\n==================================================================================")
    print("TEMPORAL STAGE BREAKDOWN:")
    print("==================================================================================")

    for idx, res in enumerate(results):
        stage_name = frames[idx].metadata.get("stage", f"Frame {idx}")
        print(f"\n--- Frame {res.frame_index} ({stage_name}) ---")
        if res.is_initial_frame:
            print("  State               : Initialized temporal baseline")
            print(f"  Initial Active Cells: {res.total_active_cells:,} cells (100% baseline stable)")
        else:
            print(f"  Previous Frame ID   : {res.previous_frame_id}")
            print(f"  Total Active Cells  : {res.total_active_cells:,}")
            print(f"  Stable / Reusable   : {res.num_stable_cells:,} cells ({res.stable_percentage:.1f}% spatial reuse)")
            print(f"  Total Changed Cells : {res.num_changed_cells:,} cells ({res.change_percentage:.1f}%)")
            print(f"    - New Occupancy   : {res.num_new_occupied_cells:,} cells")
            print(f"    - Removed Occ.    : {res.num_removed_occupied_cells:,} cells")
            print(f"    - Elevation Delta : {res.num_elevation_changed_cells:,} cells (>= {temporal_cfg.elevation_change_threshold:.2f} m)")

    print("\n==================================================================================")
    print("Temporal multi-frame processing pipeline complete.")
    print("==================================================================================")


if __name__ == "__main__":
    main()
