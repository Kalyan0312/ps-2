#!/usr/bin/env python3
"""
Demonstration Script for Phase 13 Part A: Temporal Change Detection Between Consecutive LiDAR Maps.

Demonstrates:
  1. Generating consecutive temporal LiDAR frames with controlled spatial & elevation changes.
  2. Building AdaptiveMap25D instances through the full mapping pipeline.
  3. Processing the map stream through TemporalMapManager.
  4. Detecting unchanged, changed, newly observed, and no longer observed regions.
  5. Reporting quantitative change statistics and execution performance.
"""

import sys
import time
from pathlib import Path
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.data.frame import LidarFrame
from backend.data.synthetic import SyntheticLidarGenerator
from backend.preprocessing.pipeline import PreprocessingPipeline
from backend.mapping.config import MappingConfig
from backend.mapping.uniform_builder import Uniform25DMapBuilder
from backend.terrain.analyzer import TerrainAnalyzer
from backend.priority.selector import AdaptiveResolutionSelector
from backend.mapping.adaptive_builder import AdaptiveMap25DBuilder
from backend.temporal.config import TemporalChangeConfig
from backend.temporal.temporal_manager import TemporalMapManager


def build_adaptive_pipeline(frame: LidarFrame, map_cfg: MappingConfig):
    """Executes the full pipeline to build an AdaptiveMap25D from a raw LiDAR frame."""
    clean_frame = PreprocessingPipeline().process(frame)
    base_grid = Uniform25DMapBuilder(config=map_cfg).build_map(clean_frame)
    terrain_result = TerrainAnalyzer().analyze(base_grid)
    decision = AdaptiveResolutionSelector().select_resolution(base_grid, terrain_result)
    adaptive_map = AdaptiveMap25DBuilder(config=map_cfg).build(clean_frame, decision)
    return adaptive_map


def main():
    map_cfg = MappingConfig.from_yaml("configs/config.yaml")
    temporal_cfg = TemporalChangeConfig(
        elevation_change_threshold=0.20,
        min_points_for_comparison=1,
    )
    manager = TemporalMapManager(config=temporal_cfg)
    gen = SyntheticLidarGenerator(seed=42)

    # ------------------------------------------------------------------
    # Step 1: Ingest Frame 1 (Baseline Environment)
    # ------------------------------------------------------------------
    raw_frame_1 = gen.generate_frame(frame_id=1, num_ground_points=6000)
    amap_1 = build_adaptive_pipeline(raw_frame_1, map_cfg)

    t0 = time.perf_counter()
    res_1 = manager.update(amap_1)
    t_det_1 = (time.perf_counter() - t0) * 1000

    # ------------------------------------------------------------------
    # Step 2: Generate Frame 2 (Controlled Dynamic Changes)
    # ------------------------------------------------------------------
    # Duplicate base points and introduce:
    # 1. Elevation change (+0.85m berm in center region)
    # 2. Newly observed obstacle cluster at (8, 8)
    # 3. Removed points at (-10, -10)
    pts2 = raw_frame_1.points.copy()

    # Elevation modification in central radius
    dist_center = np.hypot(pts2[:, 0], pts2[:, 1])
    berm_mask = dist_center < 3.0
    pts2[berm_mask, 2] += 0.85

    # Filter out a quadrant (simulate sensor occlusion / removal)
    keep_mask = ~((pts2[:, 0] < -8.0) & (pts2[:, 1] < -8.0))
    pts2 = pts2[keep_mask]

    # Add a new distinct object cluster
    new_cluster = np.array([
        [8.0 + dx, 8.0 + dy, 1.75, 120.0]
        for dx in np.linspace(-0.6, 0.6, 6)
        for dy in np.linspace(-0.6, 0.6, 6)
    ], dtype=np.float64)
    pts2 = np.vstack([pts2, new_cluster])

    raw_frame_2 = LidarFrame(points=pts2, frame_id=2, timestamp=0.1)
    amap_2 = build_adaptive_pipeline(raw_frame_2, map_cfg)

    # ------------------------------------------------------------------
    # Step 3: Run Temporal Change Detection
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    res_2 = manager.update(amap_2)
    t_det_2 = (time.perf_counter() - t0) * 1000

    summary_2 = res_2.get_summary()

    # ------------------------------------------------------------------
    # Output Summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print("TEMPORAL CHANGE DETECTION — PHASE 13 PART A")
    print("=" * 60)
    print()
    print("Frame 1 (Initial Baseline):")
    print(f"  Represented Cells     : {amap_1.num_represented_cells:,}")
    print(f"  Initial Frame         : {res_1.is_initial_frame}")
    print(f"  Processing Time       : {t_det_1:.3f} ms")
    print()
    print("Frame 2 (Dynamic Update):")
    print(f"  Compared Cells        : {res_2.total_compared_cells:,}")
    print(f"  Unchanged Cells       : {res_2.unchanged_cells:,} ({res_2.stable_percentage:.1f}%)")
    print(f"  Changed Cells         : {res_2.changed_cells:,} ({res_2.change_percentage:.1f}%)")
    print(f"    - Elevation Changed : {res_2.num_elevation_changed_cells:,}")
    print(f"    - Newly Observed    : {res_2.newly_observed_cells:,}")
    print(f"    - No Longer Observed: {res_2.no_longer_observed_cells:,}")
    print()
    print("Change Statistics:")
    print(f"  Threshold             : {temporal_cfg.elevation_change_threshold:.2f} m")
    print(f"  Max Elevation Delta   : {summary_2['max_elevation_difference']:.3f} m")
    print(f"  Mean Elevation Delta  : {summary_2['mean_elevation_difference']:.3f} m")
    print()
    print("Performance:")
    print(f"  Base Grid Cells       : {res_2.total_grid_cells:,} ({res_2.rows}x{res_2.cols} @ {res_2.resolution} m)")
    print(f"  Comparison Time       : {t_det_2:.3f} ms")
    print("=" * 60)
    print("Temporal change detection demonstration completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
