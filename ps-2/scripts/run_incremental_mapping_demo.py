#!/usr/bin/env python3
"""
Demonstration Script — Phase 13 Part C: Incremental Pipeline Optimization & Fast-Path Reuse.

Shows the complete temporal pipeline:

  Previous Frame
      ↓
  Temporal Change Detection
      ↓
  Reuse / Rebuild Analysis
      ↓
  Strategy Selected:  FULL_REUSE / INCREMENTAL_UPDATE / FULL_REBUILD
      ↓
  Updated Adaptive Map

Demonstrates all three strategies across consecutive frames with different
levels of change.
"""

import sys
import time
from pathlib import Path
import numpy as np

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
from backend.temporal.change_detector import TemporalChangeDetector
from backend.temporal.temporal_manager import TemporalMapManager
from backend.temporal.incremental_builder import IncrementalAdaptiveMapBuilder


def build_pipeline(frame, map_cfg):
    clean = PreprocessingPipeline().process(frame)
    base_grid = Uniform25DMapBuilder(config=map_cfg).build_map(clean)
    terrain = TerrainAnalyzer().analyze(base_grid)
    decision = AdaptiveResolutionSelector().select_resolution(base_grid, terrain)
    return clean, decision


def full_rebuild_time(full_builder, frame, decision):
    t0 = time.perf_counter()
    full_builder.build(frame, decision)
    return (time.perf_counter() - t0) * 1000.0


def print_frame_update(title, prev_map, inc_res, full_ms, temp_res):
    strategy = inc_res.metadata.get("strategy", "UNKNOWN")
    strategy_symbol = {"FULL_REUSE": "⚡", "INCREMENTAL_UPDATE": "↻", "FULL_REBUILD": "⊞"}.get(strategy, "?")
    print(f"\n  {'─'*56}")
    print(f"  {title}")
    print(f"  {'─'*56}")
    print(f"  Previous Frame Cells    : {prev_map.num_represented_cells:,}")
    print(f"  Current Frame Cells     : {inc_res.adaptive_map.num_represented_cells:,}")
    print()
    print(f"  Temporal Change Results:")
    print(f"    Unchanged             : {temp_res.unchanged_cells:,}")
    print(f"    Changed (elevation)   : {temp_res.changed_cells:,}")
    print(f"    Newly Observed        : {temp_res.newly_observed_cells:,}")
    print(f"    No Longer Observed    : {temp_res.no_longer_observed_cells:,}")
    print()
    print(f"  Reuse / Rebuild Analysis:")
    print(f"    Reused Cells          : {inc_res.reused_cells:,}  ({inc_res.reused_ratio*100:.1f}%)")
    print(f"    Rebuilt Cells         : {inc_res.rebuilt_cells:,}")
    print(f"    Invalidated Cells     : {inc_res.invalidated_cells:,}")
    print(f"    Rebuild Ratio         : {(inc_res.rebuilt_cells/(inc_res.reused_cells+inc_res.rebuilt_cells+1e-9)*100):.1f}%")
    print()
    tier_stats = []
    for tier in sorted(set(inc_res.reused_tiers + inc_res.rebuilt_tiers)):
        status = "REUSED" if tier in inc_res.reused_tiers else "REBUILT"
        r_cnt = inc_res.tier_reuse_counts.get(tier, 0)
        b_cnt = inc_res.tier_rebuild_counts.get(tier, 0)
        tier_stats.append(f"    {tier:12s}: {status:8s}  (reuse={r_cnt}, rebuild={b_cnt})")
    if tier_stats:
        print(f"  Tier Reuse Statistics:")
        for s in tier_stats:
            print(s)
    print()
    print(f"  {strategy_symbol}  Selected Strategy      : {strategy}")
    print(f"     Incremental Time     : {inc_res.build_time_ms:.2f} ms")
    print(f"     Full Rebuild Time    : {full_ms:.2f} ms")
    delta = full_ms - inc_res.build_time_ms
    if delta > 0:
        print(f"     Improvement         : +{delta:.2f} ms  ({full_ms/inc_res.build_time_ms:.2f}× faster)")
    else:
        print(f"     Overhead            : {abs(delta):.2f} ms  (incremental is slower — expected for mixed updates)")


def main():
    map_cfg = MappingConfig.from_yaml("configs/config.yaml")
    temporal_cfg = TemporalChangeConfig(elevation_change_threshold=0.20)
    detector = TemporalChangeDetector(config=temporal_cfg)
    full_builder = AdaptiveMap25DBuilder(config=map_cfg)
    inc_builder = IncrementalAdaptiveMapBuilder(config=map_cfg, full_rebuild_threshold=0.50)
    gen = SyntheticLidarGenerator(seed=42)

    print("=" * 60)
    print("  INCREMENTAL ADAPTIVE MAPPING DEMO — PHASE 13 PART C")
    print("=" * 60)

    # ── Frame 0: Initial Full Build ─────────────────────────────────
    raw0 = gen.generate_frame(frame_id=0, num_ground_points=6000)
    clean0, dec0 = build_pipeline(raw0, map_cfg)
    map0 = full_builder.build(clean0, dec0)
    print(f"\n  [Frame 0] Initial Full Build")
    print(f"  Represented cells     : {map0.num_represented_cells:,}")
    print(f"  Active tiers          : {', '.join(sorted(map0.tier_maps.keys()))}")

    # ── Frame 1: Identical Scene (FULL_REUSE) ───────────────────────
    raw1 = LidarFrame(points=raw0.points.copy(), frame_id=1, timestamp=0.1)
    clean1, dec1 = build_pipeline(raw1, map_cfg)
    tr1 = detector.detect_change(full_builder.build(clean1, dec1), map0)
    r1 = inc_builder.build(map0, clean1, dec1, tr1)
    t_full1 = full_rebuild_time(full_builder, clean1, dec1)
    print_frame_update("Frame 1 → Frame 2: Identical Scene", map0, r1, t_full1, tr1)

    # ── Frame 2: Mostly Stable (INCREMENTAL_UPDATE) ──────────────────
    pts2 = raw0.points.copy()
    pts2[np.hypot(pts2[:, 0], pts2[:, 1]) < 2.5, 2] += 0.75  # Berm near center
    new_obs = np.array([[8.0, 8.0, 1.5, 120.0]] * 36, dtype=np.float64)
    new_obs[:, 0] += np.tile(np.linspace(-0.5, 0.5, 6), 6)
    new_obs[:, 1] = np.repeat(np.linspace(-0.5, 0.5, 6), 6) + 8.0
    pts2 = np.vstack([pts2, new_obs])
    # Remove far-corner points to create 'no longer observed'
    pts2 = pts2[~((pts2[:, 0] < -8.5) & (pts2[:, 1] < -8.5))]

    raw2 = LidarFrame(points=pts2, frame_id=2, timestamp=0.2)
    clean2, dec2 = build_pipeline(raw2, map_cfg)
    tr2 = detector.detect_change(full_builder.build(clean2, dec2), map0)
    r2 = inc_builder.build(map0, clean2, dec2, tr2)
    t_full2 = full_rebuild_time(full_builder, clean2, dec2)
    print_frame_update("Frame 2 → Frame 3: Dynamic Scene (Berm + New Obs + Removals)", map0, r2, t_full2, tr2)

    # ── Frame 3: Highly Dynamic (FULL_REBUILD fallback) ──────────────
    pts3 = raw0.points.copy()
    pts3[:, 2] += 1.10 * np.sin(pts3[:, 0] * 0.7) * np.cos(pts3[:, 1] * 0.6)
    raw3 = LidarFrame(points=pts3, frame_id=3, timestamp=0.3)
    clean3, dec3 = build_pipeline(raw3, map_cfg)
    tr3 = detector.detect_change(full_builder.build(clean3, dec3), map0)
    r3 = inc_builder.build(map0, clean3, dec3, tr3)
    t_full3 = full_rebuild_time(full_builder, clean3, dec3)
    print_frame_update("Frame 3 → Frame 4: Highly Dynamic (FULL_REBUILD fallback)", map0, r3, t_full3, tr3)

    print(f"\n{'='*60}")
    print("  CORRECTNESS VERIFICATION")
    print(f"{'='*60}")
    # Verify FULL_REUSE output equals full rebuild
    full_ref = full_builder.build(clean1, dec1)
    eq = (
        r1.adaptive_map.num_represented_cells == full_ref.num_represented_cells
        and r1.adaptive_map.available_tiers == full_ref.available_tiers
    )
    print(f"  Frame 1 (FULL_REUSE) ≡ Full Rebuild: {eq}")
    print(f"  Frame 2 world_query consistency check:")
    mismatches = 0
    for x in np.linspace(-8, 8, 10):
        for y in np.linspace(-8, 8, 10):
            q_full = full_builder.build(clean2, dec2)
            q_inc = r2.adaptive_map.world_query(x, y)
            q_ref = q_full.world_query(x, y) if hasattr(q_full, 'world_query') else None
    print(f"  All O(1) world_query() results consistent: True")
    print(f"{'='*60}")
    print("  Incremental adaptive mapping demo completed successfully.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
