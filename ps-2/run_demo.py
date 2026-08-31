#!/usr/bin/env python3
"""
Adaptive Variable-Resolution 2.5D LiDAR Mapping — End-to-End System Demonstration.

Executes the complete pipeline:
  1. Synthetic LiDAR Ingestion / Generation
  2. Preprocessing & Range / Height / Outlier Filtering
  3. Uniform 2.5D Elevation Grid Mapping (Base Map)
  4. Vectorized Terrain Complexity Analysis (Roughness, Slope, Elevation Range)
  5. Vectorized Adaptive Resolution Selection (Coarse / Medium / Fine / Ultra-Fine)
  6. Compute Budgeting & Resource-Aware Optimization
  7. Composite Adaptive Map Reconstruction (AdaptiveMap25D)
  8. O(1) Spatial Querying & Metrics Summary
"""

import sys
import time
from pathlib import Path
import numpy as np
import yaml

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: Path | str) -> dict:
    """Loads YAML configuration file into a dictionary."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

from backend.data.synthetic import SyntheticLidarGenerator
from backend.preprocessing.pipeline import PreprocessingPipeline
from backend.preprocessing.config import PreprocessingConfig
from backend.mapping.config import MappingConfig
from backend.mapping.uniform_builder import Uniform25DMapBuilder
from backend.mapping.adaptive_builder import AdaptiveMap25DBuilder
from backend.terrain.analyzer import TerrainAnalyzer
from backend.terrain.config import TerrainConfig
from backend.priority.selector import AdaptiveResolutionSelector
from backend.priority.config import PriorityConfig
from backend.budget.config import BudgetConfig
from backend.budget.optimizer import BudgetAwareResolutionOptimizer
from backend.metrics.map_metrics import calculate_adaptive_map_metrics


def run():
    config_file = PROJECT_ROOT / "configs" / "config.yaml"

    print("=" * 72)
    print("      ADAPTIVE VARIABLE-RESOLUTION 2.5D LiDAR MAPPING DEMO      ")
    print("=" * 72)

    # 0. Configuration loading
    prep_cfg = PreprocessingConfig.from_yaml(config_file) if config_file.is_file() else PreprocessingConfig()
    map_cfg = MappingConfig.from_yaml(config_file) if config_file.is_file() else MappingConfig()
    terrain_cfg = TerrainConfig.from_yaml(config_file) if config_file.is_file() else TerrainConfig()
    priority_cfg = PriorityConfig.from_yaml(config_file) if config_file.is_file() else PriorityConfig()
    budget_cfg = BudgetConfig.from_yaml(config_file) if config_file.is_file() else BudgetConfig()

    print(f"[*] Workspace Root       : {PROJECT_ROOT}")
    print(f"[*] Python Version       : {sys.version.split()[0]}")
    print(f"[*] Base Resolution     : {map_cfg.resolution:.2f} m/cell")
    print(f"[*] Multi-Res Tiers      : {map_cfg.multi_resolution_levels}")
    print("-" * 72)

    total_start = time.perf_counter()

    # 1. LiDAR Point Cloud Generation
    t0 = time.perf_counter()
    generator = SyntheticLidarGenerator(seed=42)
    raw_frame = generator.generate_frame(frame_id=0, num_ground_points=6000)
    t_gen = (time.perf_counter() - t0) * 1000

    print(f"[1] Ingested Raw Frame   : {raw_frame.num_points:,} points ({t_gen:.2f} ms)")

    # 2. Preprocessing
    t0 = time.perf_counter()
    preprocessor = PreprocessingPipeline(config=prep_cfg)
    clean_frame = preprocessor.process(raw_frame)
    t_prep = (time.perf_counter() - t0) * 1000
    prep_meta = clean_frame.metadata.get("preprocessing", {})
    retained_pct = (clean_frame.num_points / raw_frame.num_points * 100.0) if raw_frame.num_points > 0 else 0.0

    print(f"[2] Preprocessing        : {clean_frame.num_points:,} points retained ({retained_pct:.1f}%, {t_prep:.2f} ms)")

    # 3. Base Uniform Grid Mapping
    t0 = time.perf_counter()
    base_builder = Uniform25DMapBuilder(config=map_cfg)
    base_grid = base_builder.build_map(clean_frame)
    t_base = (time.perf_counter() - t0) * 1000

    print(f"[3] Uniform Base Map     : {base_grid.rows}x{base_grid.cols} grid ({base_grid.total_cells:,} cells, {base_grid.num_occupied_cells:,} occupied, {t_base:.2f} ms)")

    # 4. Vectorized Terrain Complexity Analysis
    t0 = time.perf_counter()
    terrain_analyzer = TerrainAnalyzer(config=terrain_cfg)
    terrain_result = terrain_analyzer.analyze(base_grid)
    t_terrain = (time.perf_counter() - t0) * 1000

    rough_max = float(np.nanmax(terrain_result.roughness)) if terrain_result.num_analyzed_cells > 0 else 0.0
    slope_max = float(np.nanmax(terrain_result.slope)) if terrain_result.num_analyzed_cells > 0 else 0.0

    print(f"[4] Terrain Analysis     : {terrain_result.num_analyzed_cells:,} cells analyzed (Max Roughness: {rough_max:.3f}m, Max Slope: {slope_max:.2f}, {t_terrain:.2f} ms)")

    # 5. Vectorized Adaptive Resolution Selection
    t0 = time.perf_counter()
    selector = AdaptiveResolutionSelector(config=priority_cfg)
    decision = selector.select_resolution(base_grid, terrain_result)
    t_select = (time.perf_counter() - t0) * 1000

    print(f"[5] Resolution Decision  : {decision.num_decided_cells:,} decisions made ({t_select:.2f} ms)")
    for tier, count in decision.counts.items():
        pct = decision.percentages.get(tier, 0.0)
        res_val = decision.resolution_levels.get(tier, 0.0)
        print(f"      - {tier:<12} ({res_val:.2f} m): {count:>5,} cells ({pct:>5.1f}%)")

    # 6. Budget Optimization
    t0 = time.perf_counter()
    optimizer = BudgetAwareResolutionOptimizer(config=budget_cfg)
    budget_result = optimizer.optimize(decision=decision, terrain_result=terrain_result)
    t_budget = (time.perf_counter() - t0) * 1000

    print(f"[6] Budget Optimization  : Status '{budget_result.status}', {budget_result.num_downgraded_cells} downgrades ({t_budget:.2f} ms)")

    # 7. Adaptive Map Reconstruction
    t0 = time.perf_counter()
    adaptive_builder = AdaptiveMap25DBuilder(config=map_cfg)
    adaptive_map = adaptive_builder.build(clean_frame, budget_result.optimized_decision)
    t_build = (time.perf_counter() - t0) * 1000

    total_time = (time.perf_counter() - total_start) * 1000

    metrics = calculate_adaptive_map_metrics(adaptive_map)
    abd = metrics.adaptive_breakdown

    print(f"[7] Adaptive Map Built   : {len(adaptive_map.tier_maps)} active tiers, {adaptive_map.num_represented_cells:,} represented cells ({t_build:.2f} ms)")
    print("-" * 72)
    print("PIPELINE PERFORMANCE:")
    print(f"  - Total Execution Time : {total_time:.2f} ms")
    print(f"  - Preprocessing        : {t_prep:.2f} ms")
    print(f"  - Terrain Analysis     : {t_terrain:.2f} ms")
    print(f"  - Resolution Selection : {t_select:.2f} ms")
    print(f"  - Adaptive Map Build   : {t_build:.2f} ms")
    print("-" * 72)

    # 8. Spatial Query Demonstrations
    print("SPATIAL QUERY TESTS:")
    for tier in ["ultra_fine", "fine", "medium", "coarse"]:
        mask = adaptive_map.index_tier == tier
        indices = np.argwhere(mask)
        for r, c in indices:
            min_bx = base_grid.min_x + c * base_grid.resolution
            max_bx = min_bx + base_grid.resolution
            min_by = base_grid.min_y + r * base_grid.resolution
            max_by = min_by + base_grid.resolution
            in_cell = (
                (clean_frame.points[:, 0] >= min_bx) & (clean_frame.points[:, 0] <= max_bx) &
                (clean_frame.points[:, 1] >= min_by) & (clean_frame.points[:, 1] <= max_by)
            )
            if np.any(in_cell):
                pt = clean_frame.points[in_cell][0]
                info = adaptive_map.world_query(float(pt[0]), float(pt[1]))
                if info.is_represented and info.tier == tier:
                    print(f"  * Tier '{tier:<10}' query @ ({pt[0]:>6.2f}, {pt[1]:>6.2f}) -> Res: {info.resolution:.2f}m, Elevation: {info.mean_z:.3f}m, Points: {info.point_count}")
                    break

    # Out of bounds query
    oob_info = adaptive_map.world_query(100.0, 100.0)
    print(f"  * Out-of-bounds query  @ (100.00, 100.00) -> is_represented={oob_info.is_represented}")

    print("-" * 72)
    print("RESOURCE & STORAGE FOOTPRINT:")
    if abd:
        print(f"  - Tier Dense Grid Cells: {abd.tier_allocated_cells:,} cells (across {len(adaptive_map.tier_maps)} bounding boxes)")
        print(f"  - Base Index Matrix    : {abd.base_decision_cells:,} cells")
        print(f"  - Total Storage Cells  : {metrics.size.total_cells:,} cells")
    print(f"  - Actual Array Memory  : {metrics.memory.array_kb:.2f} KB ({metrics.memory.array_mb:.3f} MB)")
    print("=" * 72)
    print("End-to-end demonstration completed successfully.")
    print("=" * 72)


if __name__ == "__main__":
    run()
