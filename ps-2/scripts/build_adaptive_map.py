#!/usr/bin/env python3
"""
Phase 6: Build Adaptive 2.5D Map.

Executes the full Adaptive LiDAR pipeline:

  Phase 1: Generate sample LiDAR frame
  Phase 2: Preprocess the frame
  Phase 3A: Build uniform 2.5D base map
  Phase 4: Terrain complexity analysis
  Phase 5: Adaptive resolution decision
  Phase 6: Construct composite adaptive 2.5D map

Then prints the required ADAPTIVE 2.5D MAP RESULTS report.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from backend.data import SyntheticLidarLoader
from backend.preprocessing import PreprocessingPipeline
from backend.mapping import Uniform25DMapBuilder, AdaptiveMap25DBuilder
from backend.terrain import TerrainAnalyzer
from backend.priority import AdaptiveResolutionSelector


def main():
    config_path = ROOT_DIR / "configs" / "config.yaml"

    # -----------------------------------------------------------------------
    # Step 1: Generate sample LiDAR frame (Phase 1)
    # -----------------------------------------------------------------------
    loader = SyntheticLidarLoader(num_frames=1, seed=42)
    raw_frame = loader[0]
    raw_points = raw_frame.num_points

    # -----------------------------------------------------------------------
    # Step 2: Preprocess (Phase 2)
    # -----------------------------------------------------------------------
    pipeline = PreprocessingPipeline.from_config_file(config_path)
    clean_frame = pipeline.process(raw_frame)
    prep_points = clean_frame.num_points

    # -----------------------------------------------------------------------
    # Step 3: Build uniform 2.5D base map (Phase 3A)
    # -----------------------------------------------------------------------
    map_builder = Uniform25DMapBuilder.from_config_file(config_path)
    base_grid = map_builder.build_map(clean_frame)

    # -----------------------------------------------------------------------
    # Step 4: Terrain complexity analysis (Phase 4)
    # -----------------------------------------------------------------------
    analyzer = TerrainAnalyzer.from_config_file(config_path)
    terrain_result = analyzer.analyze(base_grid)

    # -----------------------------------------------------------------------
    # Step 5: Adaptive resolution decision (Phase 5)
    # -----------------------------------------------------------------------
    selector = AdaptiveResolutionSelector.from_config_file(config_path)
    decision = selector.select_resolution(base_grid, terrain_result)

    counts = decision.counts
    pcts = decision.percentages
    res_levels = decision.resolution_levels

    # -----------------------------------------------------------------------
    # Step 6: Build adaptive composite map (Phase 6)
    # -----------------------------------------------------------------------
    adaptive_builder = AdaptiveMap25DBuilder.from_config_file(config_path)
    adaptive_map = adaptive_builder.build(clean_frame, decision)
    summary = adaptive_map.get_summary()

    # -----------------------------------------------------------------------
    # Example coordinate query
    # -----------------------------------------------------------------------
    # Find an occupied base-grid cell to demonstrate a meaningful query
    occ_indices = np.argwhere(base_grid.occupied_mask)
    sample_results = []
    for r, c in occ_indices[:50]:
        wx, wy = base_grid.grid_to_world(int(r), int(c))
        qr = adaptive_map.world_query(float(wx), float(wy))
        if qr.is_represented:
            sample_results.append(qr)
            if len(sample_results) == 3:
                break

    # -----------------------------------------------------------------------
    # Print required output
    # -----------------------------------------------------------------------
    sep = "=" * 52

    print(sep)
    print("ADAPTIVE 2.5D MAP RESULTS")
    print(sep)
    print()

    print("--- Input Pipeline ---")
    print(f"  Raw point count        : {raw_points:,}")
    print(f"  Preprocessed points    : {prep_points:,}  ({prep_points/raw_points*100:.1f}% retained)")
    print(f"  Base map resolution    : {base_grid.resolution:.2f} m/cell")
    print(f"  Total analyzed cells   : {decision.num_decided_cells:,}")
    print()

    print("--- Resolution Allocation (Phase 5 decisions) ---")
    print(f"  {'Tier':<12} {'Resolution':>10}   {'Cells':>7}   {'%':>6}")
    print(f"  {'-'*12} {'-'*10}   {'-'*7}   {'-'*6}")
    for tier in ["coarse", "medium", "fine", "ultra_fine"]:
        rv = f"{res_levels.get(tier, 0.0):.2f} m"
        cv = counts.get(tier, 0)
        pv = pcts.get(tier, 0.0)
        print(f"  {tier:<12} {rv:>10}   {cv:>7,}   {pv:>5.1f}%")
    print()

    print("--- Final Adaptive Map (Phase 6) ---")
    print(f"  Represented adaptive cells : {adaptive_map.num_represented_cells:,}")
    print(f"  Tiers built                : {len(adaptive_map.tier_maps)}")
    print(f"  Available tier names       : {', '.join(adaptive_map.available_tiers)}")
    print()
    print("  Spatial allocation by tier:")
    tier_counts_map = adaptive_map.tier_cell_counts
    tier_pcts_map = adaptive_map.tier_percentages
    for tier in ["coarse", "medium", "fine", "ultra_fine"]:
        tc = tier_counts_map.get(tier, 0)
        tp = tier_pcts_map.get(tier, 0.0)
        built = "(built)" if tier in adaptive_map.tier_maps else "(no cells)"
        tier_res = adaptive_map.tier_resolutions.get(tier, 0.0)
        print(f"    {tier:<12}  @ {tier_res:.2f} m  →  {tc:>7,} base cells  ({tp:.1f}%)  {built}")
    print()

    print("--- Example Query Results ---")
    if sample_results:
        for i, qr in enumerate(sample_results, 1):
            print(f"  Query {i}: world ({qr.query_x:.2f}, {qr.query_y:.2f})")
            print(f"    represented  : {qr.is_represented}")
            print(f"    tier         : {qr.tier}")
            print(f"    resolution   : {qr.resolution:.2f} m")
            print(f"    min_z / mean_z / max_z : "
                  f"{qr.min_z:.3f} / {qr.mean_z:.3f} / {qr.max_z:.3f} m")
            print(f"    point_count  : {qr.point_count}")
    else:
        print("  (No represented cells found at sampled coordinates)")
    print()

    print("--- Comparison vs Uniform Fine-Resolution Map ---")
    uf_total = summary["uniform_fine_total_cells"]
    uf_res = summary["uniform_fine_resolution_m"]
    actual_tier_cells = summary["actual_tier_occupied_cells"]
    repr_cells = adaptive_map.num_represented_cells

    # Tier-allocated dense cells in adaptive bounding boxes
    tier_allocated = sum(grid.total_cells for grid in adaptive_map.tier_maps.values())
    total_allocated_adaptive = tier_allocated + adaptive_map.base_shape[0] * adaptive_map.base_shape[1]

    print(f"  Uniform ultra-fine ({uf_res:.2f} m) total cells  : {uf_total:>10,}")
    print(f"  Adaptive map dense allocated cells         : {total_allocated_adaptive:>10,} ({tier_allocated:,} tier + {adaptive_map.base_shape[0]*adaptive_map.base_shape[1]:,} index)")
    print(f"  Adaptive map represented base cells        : {repr_cells:>10,}")
    print(f"  Adaptive map actual tier occupied cells    : {actual_tier_cells:>10,}")
    if uf_total > 0:
        alloc_reduction = (1.0 - total_allocated_adaptive / uf_total) * 100.0
        print(f"  Dense grid allocation reduction            :      {alloc_reduction:>5.1f}%")
        print(f"  (adaptive map allocates {total_allocated_adaptive:,} cells vs uniform's {uf_total:,} cells)")
    print()
    print(sep)


if __name__ == "__main__":
    main()
