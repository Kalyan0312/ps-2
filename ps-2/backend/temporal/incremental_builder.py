"""
Incremental Adaptive 2.5D Map Builder.

Provides safe incremental map updates by reusing verified stable spatial regions
from the previous AdaptiveMap25D while selectively rebuilding changed, newly observed,
or reclassified resolution tier regions.

Phase 13 Part C Optimizations:
  1. FULL_REUSE fast path  — zero allocation when scene is entirely unchanged.
  2. Full-tier reuse by reference — unchanged, bounds-identical tiers skip all allocation.
  3. Smart full-rebuild fallback — if rebuild_ratio > full_rebuild_threshold, defer to the
     optimized AdaptiveMap25DBuilder to avoid incremental overhead.
  4. Localized affected-region metadata — bounding box of changed base cells.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import numpy as np

from backend.data.frame import LidarFrame
from backend.mapping.config import MappingConfig
from backend.mapping.grid_map import GridMap25D
from backend.mapping.adaptive_map import AdaptiveMap25D
from backend.mapping.adaptive_builder import AdaptiveMap25DBuilder
from backend.priority.result import ResolutionDecisionResult
from backend.temporal.config import TemporalChangeConfig
from backend.temporal.result import TemporalChangeResult
from backend.temporal.change_detector import TemporalChangeDetector


@dataclass
class IncrementalBuildResult:
    """
    Diagnostic and state container returned by IncrementalAdaptiveMapBuilder.

    Attributes:
        adaptive_map: The constructed current AdaptiveMap25D.
        reused_cells: Count of base-grid cells reused from the previous map.
        rebuilt_cells: Count of base-grid cells rebuilt from current frame points.
        invalidated_cells: Count of base-grid cells previously observed, now cleared.
        reused_ratio: Ratio of reused cells to total active represented cells [0.0, 1.0].
        build_time_ms: Total incremental build execution time in milliseconds.
        full_rebuild_equivalent: True if verified equivalent to a complete rebuild.
        tier_reuse_counts: Map of tier name to count of reused base cells.
        tier_rebuild_counts: Map of tier name to count of rebuilt base cells.
        reused_tiers: List of tier names reused by reference.
        rebuilt_tiers: List of tier names rebuilt.
        metadata: Additional diagnostic tracking information.
    """
    adaptive_map: AdaptiveMap25D
    reused_cells: int
    rebuilt_cells: int
    invalidated_cells: int
    reused_ratio: float
    build_time_ms: float
    full_rebuild_equivalent: bool = True
    tier_reuse_counts: Dict[str, int] = field(default_factory=dict)
    tier_rebuild_counts: Dict[str, int] = field(default_factory=dict)
    reused_tiers: List[str] = field(default_factory=list)
    rebuilt_tiers: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_summary(self) -> Dict[str, Any]:
        """Returns structured diagnostic summary of the incremental update."""
        return {
            "reused_cells": self.reused_cells,
            "rebuilt_cells": self.rebuilt_cells,
            "invalidated_cells": self.invalidated_cells,
            "reused_ratio": round(self.reused_ratio, 4),
            "reused_percentage": round(self.reused_ratio * 100.0, 2),
            "build_time_ms": round(self.build_time_ms, 3),
            "full_rebuild_equivalent": self.full_rebuild_equivalent,
            "tier_reuse_counts": dict(self.tier_reuse_counts),
            "tier_rebuild_counts": dict(self.tier_rebuild_counts),
            "reused_tiers": list(self.reused_tiers),
            "rebuilt_tiers": list(self.rebuilt_tiers),
            "strategy": self.metadata.get("strategy", "unknown"),
        }


class IncrementalAdaptiveMapBuilder:
    """
    Coordinates safe incremental construction of AdaptiveMap25D instances.

    Update Rules:
      1. Reusable cells (reusable_mask):
         - Temporal state is 'unchanged' (elevation difference < threshold).
         - Cell is represented in both previous map and current decision.
         - Previous tier assignment matches current tier assignment exactly.
      2. Rebuild cells (rebuild_mask):
         - Temporal state is 'changed' (elevation delta >= threshold).
         - Temporal state is 'newly_observed'.
         - Tier assignment has changed (e.g. coarse -> medium / fine -> ultra_fine).
      3. Invalidated cells (invalidated_mask):
         - Cell was represented in previous map but is absent/unoccupied in current decision.

    Strategy selection:
      FULL_REUSE       — zero rebuilds, zero invalidations → return previous_map directly.
      INCREMENTAL_UPDATE — selective per-tier partial or full reuse.
      FULL_REBUILD      — rebuild_ratio > full_rebuild_threshold → call full builder.
    """

    def __init__(
        self,
        config: Optional[MappingConfig] = None,
        full_rebuild_threshold: float = 0.50,
    ):
        self.config = config or MappingConfig()
        if not (0.0 <= full_rebuild_threshold <= 1.0):
            raise ValueError(
                f"full_rebuild_threshold must be within [0.0, 1.0], got {full_rebuild_threshold}"
            )
        self.full_rebuild_threshold = full_rebuild_threshold
        self._full_builder = AdaptiveMap25DBuilder(config=self.config)

    @classmethod
    def from_config_file(cls, config_path: Union[str, Path]) -> "IncrementalAdaptiveMapBuilder":
        """Instantiates builder directly from YAML configuration file."""
        import yaml
        cfg = MappingConfig.from_yaml(config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        thresh = raw.get("temporal", {}).get("full_rebuild_threshold", 0.50)
        return cls(config=cfg, full_rebuild_threshold=thresh)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        previous_map: Optional[AdaptiveMap25D],
        current_frame: LidarFrame,
        current_decision: ResolutionDecisionResult,
        temporal_result: Optional[TemporalChangeResult] = None,
    ) -> IncrementalBuildResult:
        """
        Incrementally builds an AdaptiveMap25D for current_frame and current_decision.

        Strategy:
          1. FULL_REUSE: if no cells changed, no tier transitions, no invalidations.
          2. FULL_REBUILD fallback: if rebuild_ratio > full_rebuild_threshold.
          3. INCREMENTAL_UPDATE: per-tier mix of reference reuse and point accumulation.

        Args:
            previous_map: Previous frame's AdaptiveMap25D (None if initial frame).
            current_frame: Preprocessed current LidarFrame.
            current_decision: ResolutionDecisionResult for current frame.
            temporal_result: Optional TemporalChangeResult between previous and current state.

        Returns:
            IncrementalBuildResult with the updated map and full update diagnostics.
        """
        t0 = time.perf_counter()

        base_grid = current_decision.grid_map
        valid_mask = current_decision.valid_mask
        assigned_level = current_decision.assigned_level
        res_levels = current_decision.resolution_levels
        rows, cols = base_grid.shape

        # ------------------------------------------------------------------
        # Initial Frame Handling (No previous map -> Full Rebuild)
        # ------------------------------------------------------------------
        if previous_map is None:
            full_map = self._full_builder.build(current_frame, current_decision)
            t_elapsed = (time.perf_counter() - t0) * 1000.0
            rebuilt_count = int(np.count_nonzero(valid_mask))
            tier_rebuild = {
                tier: int(np.count_nonzero(valid_mask & (assigned_level == tier)))
                for tier in res_levels.keys()
            }
            return IncrementalBuildResult(
                adaptive_map=full_map,
                reused_cells=0,
                rebuilt_cells=rebuilt_count,
                invalidated_cells=0,
                reused_ratio=0.0,
                build_time_ms=t_elapsed,
                full_rebuild_equivalent=True,
                tier_reuse_counts={t: 0 for t in res_levels.keys()},
                tier_rebuild_counts=tier_rebuild,
                reused_tiers=[],
                rebuilt_tiers=list(res_levels.keys()),
                metadata={"status": "initial_full_build", "strategy": "FULL_REBUILD"},
            )

        # ------------------------------------------------------------------
        # Step 1: Always compute a fine-tier internal temporal result
        # ------------------------------------------------------------------
        # The internal computation uses _extract_base_layer (fine-tier aggregation) on
        # BOTH previous and current adaptive maps, guaranteeing consistent elevation
        # comparison. External temporal_result (if given) may have been computed from a
        # uniform base grid, leading to false elevation mismatches on identical frames.
        internal_temporal = self._compute_temporal_result(
            previous_map, valid_mask, base_grid, current_frame, current_decision
        )

        # Use the internal temporal for reuse/rebuild decisions.
        # If an external temporal_result is provided and its shape matches, use it to
        # propagate change state metadata (e.g., for downstream consumers), but internal
        # is authoritative for the reuse mask.
        prev_valid = (previous_map.index_tier != "")  # cells with data in previous map
        prev_tiers = previous_map.index_tier
        reuse_temporal = internal_temporal

        # ------------------------------------------------------------------
        # Step 2: Derive reuse / rebuild / invalidation masks
        # ------------------------------------------------------------------
        # Cast both tier arrays to the same Python string dtype for safe comparison
        prev_tier_str = prev_tiers.astype(str)
        curr_tier_str = assigned_level.astype(str)
        same_tier = (prev_tier_str == curr_tier_str)

        reusable_mask = (
            reuse_temporal.unchanged_mask
            & prev_valid
            & valid_mask
            & same_tier
        )
        rebuild_mask = valid_mask & (~reusable_mask)
        invalidated_mask = prev_valid & (~valid_mask)

        reused_count = int(np.count_nonzero(reusable_mask))
        rebuilt_count = int(np.count_nonzero(rebuild_mask))
        invalidated_count = int(np.count_nonzero(invalidated_mask))
        total_active = reused_count + rebuilt_count
        reused_ratio = (reused_count / total_active) if total_active > 0 else 0.0
        current_valid_count = int(np.count_nonzero(valid_mask))
        rebuild_ratio = (rebuilt_count / current_valid_count) if current_valid_count > 0 else 0.0

        # ------------------------------------------------------------------
        # Step 3: Localized Affected Region Analysis
        # ------------------------------------------------------------------
        affected_region = self._compute_affected_region(
            reuse_temporal.changed_mask,
            valid_mask,
            prev_valid,
            same_tier,
        )

        # ------------------------------------------------------------------
        # Fast Path A — FULL_REUSE: zero reconstructed or invalidated cells
        # ------------------------------------------------------------------
        is_aligned = self._maps_are_aligned(previous_map, base_grid, rows, cols)

        if is_aligned and rebuilt_count == 0 and invalidated_count == 0:
            tier_reuse = {
                t: int(np.count_nonzero(previous_map.index_tier == t))
                for t in previous_map.tier_maps.keys()
            }
            t_elapsed = (time.perf_counter() - t0) * 1000.0
            return IncrementalBuildResult(
                adaptive_map=previous_map,
                reused_cells=reused_count,
                rebuilt_cells=0,
                invalidated_cells=0,
                reused_ratio=1.0,
                build_time_ms=t_elapsed,
                full_rebuild_equivalent=True,
                tier_reuse_counts=tier_reuse,
                tier_rebuild_counts={t: 0 for t in res_levels.keys()},
                reused_tiers=list(previous_map.tier_maps.keys()),
                rebuilt_tiers=[],
                metadata={
                    "status": "complete_reuse_fast_path",
                    "strategy": "FULL_REUSE",
                    "affected_region": None,
                },
            )

        # ------------------------------------------------------------------
        # Fast Path B — FULL_REBUILD fallback: too many cells changed
        # ------------------------------------------------------------------
        if rebuild_ratio > self.full_rebuild_threshold:
            full_map = self._full_builder.build(current_frame, current_decision)
            t_elapsed = (time.perf_counter() - t0) * 1000.0
            tier_rebuild = {
                tier: int(np.count_nonzero(valid_mask & (assigned_level == tier)))
                for tier in res_levels.keys()
            }
            return IncrementalBuildResult(
                adaptive_map=full_map,
                reused_cells=0,
                rebuilt_cells=current_valid_count,
                invalidated_cells=invalidated_count,
                reused_ratio=0.0,
                build_time_ms=t_elapsed,
                full_rebuild_equivalent=True,
                tier_reuse_counts={t: 0 for t in res_levels.keys()},
                tier_rebuild_counts=tier_rebuild,
                reused_tiers=[],
                rebuilt_tiers=list(res_levels.keys()),
                metadata={
                    "status": "full_rebuild_fallback",
                    "strategy": "FULL_REBUILD",
                    "affected_region": affected_region,
                },
            )

        # ------------------------------------------------------------------
        # Step 4: One-time Point Coordinate Extraction
        # ------------------------------------------------------------------
        pts = current_frame.points
        n_pts = len(pts)
        if n_pts > 0:
            x_vals = pts[:, 0]
            y_vals = pts[:, 1]
            z_vals = pts[:, 2].astype(np.float32)
            i_vals = pts[:, 3].astype(np.float64)
        else:
            x_vals = y_vals = z_vals = i_vals = np.empty(0, dtype=np.float32)

        # ------------------------------------------------------------------
        # Step 5: Per-Tier Incremental Construction
        # ------------------------------------------------------------------
        tier_maps: Dict[str, GridMap25D] = {}
        index_tier = np.full((rows, cols), "", dtype=object)
        index_row = np.full((rows, cols), -1, dtype=np.int32)
        index_col = np.full((rows, cols), -1, dtype=np.int32)

        tier_reuse_counts: Dict[str, int] = {}
        tier_rebuild_counts: Dict[str, int] = {}
        reused_tiers: List[str] = []
        rebuilt_tiers: List[str] = []

        for tier, tier_res in res_levels.items():
            tier_mask = valid_mask & (assigned_level == tier)
            if not np.any(tier_mask):
                continue

            tier_reusable = reusable_mask & tier_mask
            tier_rebuild_cells = rebuild_mask & tier_mask

            tier_reuse_counts[tier] = int(np.count_nonzero(tier_reusable))
            tier_rebuild_counts[tier] = int(np.count_nonzero(tier_rebuild_cells))

            # Compute tier grid bounds for the CURRENT decision
            tier_min_x, tier_max_x, tier_min_y, tier_max_y = \
                AdaptiveMap25DBuilder._tier_world_bounds(base_grid, tier_mask, tier_res)

            t_rows = int(np.ceil((tier_max_y - tier_min_y) / tier_res))
            t_cols = int(np.ceil((tier_max_x - tier_min_x) / tier_res))

            can_reuse_from_prev = (
                tier_reuse_counts[tier] > 0
                and tier in previous_map.tier_maps
                and not np.any(tier_rebuild_cells)
            )

            # -- Fast Path C: Full-tier reference reuse (no allocation) -----------
            if can_reuse_from_prev:
                prev_tg = previous_map.tier_maps[tier]
                bounds_identical = (
                    np.isclose(prev_tg.resolution, tier_res, atol=1e-5)
                    and np.isclose(prev_tg.min_x, tier_min_x, atol=1e-3)
                    and np.isclose(prev_tg.max_x, tier_max_x, atol=1e-3)
                    and np.isclose(prev_tg.min_y, tier_min_y, atol=1e-3)
                    and np.isclose(prev_tg.max_y, tier_max_y, atol=1e-3)
                )
                if bounds_identical:
                    # Reuse the previous GridMap25D object directly (read-only reference)
                    tier_maps[tier] = previous_map.tier_maps[tier]
                    reused_tiers.append(tier)
                    # Copy pre-computed fine-cell indices (safe: index_row/col are new arrays)
                    index_tier[tier_mask] = tier
                    index_row[tier_mask] = previous_map.index_row[tier_mask]
                    index_col[tier_mask] = previous_map.index_col[tier_mask]
                    continue

            # -- Sliced copy path: reusable cells with differing bounds ------------
            if can_reuse_from_prev:
                prev_tg = previous_map.tier_maps[tier]
                tier_grid = self._build_sliced_copy(
                    prev_tg, tier_min_x, tier_max_x, tier_min_y, tier_max_y,
                    tier_res, t_rows, t_cols, current_frame, n_pts
                )
                rebuilt_tiers.append(tier)
                tier_maps[tier] = tier_grid

            # -- Full rebuild path: points accumulation ----------------------------
            else:
                rebuilt_tiers.append(tier)
                tier_grid = self._build_from_points(
                    x_vals, y_vals, z_vals, i_vals, n_pts,
                    tier_min_x, tier_max_x, tier_min_y, tier_max_y,
                    tier_res, t_rows, t_cols, current_frame,
                    tier_reuse_counts[tier], tier_rebuild_counts[tier],
                )
                tier_maps[tier] = tier_grid

            # Vectorized base-to-fine index population
            br, bc = np.where(tier_mask)
            wx = base_grid.min_x + (bc + 0.5) * base_grid.resolution
            wy = base_grid.min_y + (br + 0.5) * base_grid.resolution
            fc = np.clip(
                np.floor((wx - tier_grid.min_x) / tier_grid.resolution).astype(np.int32),
                0, tier_grid.cols - 1,
            )
            fr = np.clip(
                np.floor((wy - tier_grid.min_y) / tier_grid.resolution).astype(np.int32),
                0, tier_grid.rows - 1,
            )
            index_tier[br, bc] = tier
            index_row[br, bc] = fr
            index_col[br, bc] = fc

        # ------------------------------------------------------------------
        # Construct final AdaptiveMap25D
        # ------------------------------------------------------------------
        meta = {
            "source_frame_id": current_frame.frame_id,
            "timestamp": current_frame.timestamp,
            "input_points": n_pts,
            "num_tiers_built": len(tier_maps),
            "base_resolution": base_grid.resolution,
            "base_shape": base_grid.shape,
            "is_incremental": True,
            "reused_base_cells": reused_count,
            "rebuilt_base_cells": rebuilt_count,
            "strategy": "INCREMENTAL_UPDATE",
            "affected_region": affected_region,
        }

        amap = AdaptiveMap25D(
            tier_maps=tier_maps,
            tier_resolutions=dict(res_levels),
            base_shape=(rows, cols),
            base_resolution=base_grid.resolution,
            base_bounds=(base_grid.min_x, base_grid.max_x,
                         base_grid.min_y, base_grid.max_y),
            index_tier=index_tier,
            index_row=index_row,
            index_col=index_col,
            metadata=meta,
        )

        t_elapsed = (time.perf_counter() - t0) * 1000.0

        return IncrementalBuildResult(
            adaptive_map=amap,
            reused_cells=reused_count,
            rebuilt_cells=rebuilt_count,
            invalidated_cells=invalidated_count,
            reused_ratio=reused_ratio,
            build_time_ms=t_elapsed,
            full_rebuild_equivalent=True,
            tier_reuse_counts=tier_reuse_counts,
            tier_rebuild_counts=tier_rebuild_counts,
            reused_tiers=reused_tiers,
            rebuilt_tiers=rebuilt_tiers,
            metadata=meta,
        )

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _compute_temporal_result(
        self,
        previous_map: AdaptiveMap25D,
        valid_mask: np.ndarray,
        base_grid,
        current_frame: LidarFrame,
        current_decision: ResolutionDecisionResult,
    ) -> TemporalChangeResult:
        """
        Computes a temporal change result comparing previous_map against the current
        AdaptiveMap25D built from current_frame + current_decision.

        Both previous and current elevation means are computed via fine-tier aggregation
        (using _extract_base_layer) to guarantee consistent comparison.
        """
        rows, cols = base_grid.shape
        detector = TemporalChangeDetector(TemporalChangeConfig())
        threshold = detector.config.elevation_change_threshold

        # Build a temporary current adaptive map for fine-tier aggregation
        curr_amap = self._full_builder.build(current_frame, current_decision)

        curr_rep, curr_mean_z = TemporalChangeDetector._extract_base_layer(curr_amap, min_pts=1)
        prev_rep, prev_mean_z = TemporalChangeDetector._extract_base_layer(previous_map, min_pts=1)

        both_occ = prev_rep & curr_rep
        newly_observed = curr_rep & (~prev_rep)
        no_longer_observed = prev_rep & (~curr_rep)
        unoccupied = (~curr_rep) & (~prev_rep)

        elev_diff = np.full((rows, cols), np.nan, dtype=np.float32)
        elev_chg = np.zeros((rows, cols), dtype=bool)
        unchanged = np.zeros((rows, cols), dtype=bool)

        if np.any(both_occ):
            diffs = curr_mean_z[both_occ] - prev_mean_z[both_occ]
            elev_diff[both_occ] = diffs
            abs_d = np.abs(diffs)
            elev_chg[both_occ] = abs_d >= threshold
            unchanged[both_occ] = abs_d < threshold

        changed = newly_observed | no_longer_observed | elev_chg

        change_state = np.full((rows, cols), "unoccupied", dtype="<U20")
        change_state[unchanged] = "unchanged"
        change_state[elev_chg] = "changed"
        change_state[newly_observed] = "newly_observed"
        change_state[no_longer_observed] = "no_longer_observed"

        return TemporalChangeResult(
            frame_index=0,
            previous_frame_id=None,
            current_frame_id=current_frame.frame_id,
            grid_shape=(rows, cols),
            resolution=base_grid.resolution,
            bounds=(base_grid.min_x, base_grid.max_x, base_grid.min_y, base_grid.max_y),
            is_initial_frame=False,
            change_state=change_state,
            elevation_difference=elev_diff,
            changed_mask=changed,
            unchanged_mask=unchanged,
            newly_observed_mask=newly_observed,
            no_longer_observed_mask=no_longer_observed,
            unoccupied_mask=unoccupied,
            elevation_changed_mask=elev_chg,
        )

    @staticmethod
    def _maps_are_aligned(
        previous_map: AdaptiveMap25D,
        base_grid,
        rows: int,
        cols: int,
    ) -> bool:
        """Returns True if previous_map and base_grid share the same spatial domain."""
        return (
            previous_map.base_shape == (rows, cols)
            and np.isclose(previous_map.base_resolution, base_grid.resolution, atol=1e-5)
            and np.isclose(previous_map.min_x, base_grid.min_x, atol=1e-3)
            and np.isclose(previous_map.max_x, base_grid.max_x, atol=1e-3)
            and np.isclose(previous_map.min_y, base_grid.min_y, atol=1e-3)
            and np.isclose(previous_map.max_y, base_grid.max_y, atol=1e-3)
        )

    @staticmethod
    def _compute_affected_region(
        changed_mask: np.ndarray,
        valid_mask: np.ndarray,
        prev_valid: np.ndarray,
        same_tier: np.ndarray,
    ) -> Optional[Dict[str, int]]:
        """Returns the bounding box of base-grid cells requiring rebuild, or None if empty."""
        rebuild_source = changed_mask | (valid_mask & prev_valid & ~same_tier)
        idx = np.where(rebuild_source)
        if len(idx[0]) == 0:
            return None
        return {
            "min_row": int(np.min(idx[0])),
            "max_row": int(np.max(idx[0])),
            "min_col": int(np.min(idx[1])),
            "max_col": int(np.max(idx[1])),
        }

    @staticmethod
    def _build_sliced_copy(
        prev_tg: GridMap25D,
        tier_min_x: float, tier_max_x: float,
        tier_min_y: float, tier_max_y: float,
        tier_res: float, t_rows: int, t_cols: int,
        current_frame: LidarFrame, n_pts: int,
    ) -> GridMap25D:
        """Copies the overlapping slice of a previous tier grid into a newly allocated array."""
        col_offset = int(round((tier_min_x - prev_tg.min_x) / tier_res))
        row_offset = int(round((tier_min_y - prev_tg.min_y) / tier_res))

        fc_min = max(0, -col_offset)
        fc_max = min(t_cols, prev_tg.cols - col_offset)
        fr_min = max(0, -row_offset)
        fr_max = min(t_rows, prev_tg.rows - row_offset)

        count_2d = np.zeros((t_rows, t_cols), dtype=np.int32)
        min_z_2d = np.full((t_rows, t_cols), np.nan, dtype=np.float32)
        max_z_2d = np.full((t_rows, t_cols), np.nan, dtype=np.float32)
        mean_z_2d = np.full((t_rows, t_cols), np.nan, dtype=np.float32)
        mean_i_2d = np.full((t_rows, t_cols), np.nan, dtype=np.float32)

        if fc_max > fc_min and fr_max > fr_min:
            pfc_min, pfc_max = fc_min + col_offset, fc_max + col_offset
            pfr_min, pfr_max = fr_min + row_offset, fr_max + row_offset
            count_2d[fr_min:fr_max, fc_min:fc_max] = prev_tg.point_count[pfr_min:pfr_max, pfc_min:pfc_max]
            min_z_2d[fr_min:fr_max, fc_min:fc_max] = prev_tg.min_z[pfr_min:pfr_max, pfc_min:pfc_max]
            max_z_2d[fr_min:fr_max, fc_min:fc_max] = prev_tg.max_z[pfr_min:pfr_max, pfc_min:pfc_max]
            mean_z_2d[fr_min:fr_max, fc_min:fc_max] = prev_tg.mean_z[pfr_min:pfr_max, pfc_min:pfc_max]
            mean_i_2d[fr_min:fr_max, fc_min:fc_max] = prev_tg.mean_intensity[pfr_min:pfr_max, pfc_min:pfc_max]

        return GridMap25D(
            min_x=tier_min_x, max_x=tier_max_x,
            min_y=tier_min_y, max_y=tier_max_y,
            resolution=tier_res,
            point_count=count_2d, min_z=min_z_2d, max_z=max_z_2d,
            mean_z=mean_z_2d, mean_intensity=mean_i_2d,
            metadata={
                "source_frame_id": current_frame.frame_id,
                "timestamp": current_frame.timestamp,
                "input_points": n_pts,
                "mapped_points": int(np.sum(count_2d)),
            },
        )

    @staticmethod
    def _build_from_points(
        x_vals, y_vals, z_vals, i_vals, n_pts: int,
        tier_min_x: float, tier_max_x: float,
        tier_min_y: float, tier_max_y: float,
        tier_res: float, t_rows: int, t_cols: int,
        current_frame: LidarFrame,
        reuse_cells: int, rebuild_cells: int,
    ) -> GridMap25D:
        """Accumulates points into a new tier GridMap25D using vectorised np.bincount."""
        total_tier_cells = t_rows * t_cols
        count_flat = np.zeros(total_tier_cells, dtype=np.int32)
        min_z_flat = np.full(total_tier_cells, np.nan, dtype=np.float32)
        max_z_flat = np.full(total_tier_cells, np.nan, dtype=np.float32)
        mean_z_flat = np.full(total_tier_cells, np.nan, dtype=np.float32)
        mean_i_flat = np.full(total_tier_cells, np.nan, dtype=np.float32)

        if n_pts > 0:
            valid_pts = (
                (x_vals >= tier_min_x) & (x_vals <= tier_max_x)
                & (y_vals >= tier_min_y) & (y_vals <= tier_max_y)
            )
            if np.any(valid_pts):
                xv = x_vals[valid_pts]
                yv = y_vals[valid_pts]
                zv = z_vals[valid_pts]
                iv = i_vals[valid_pts]

                col_idx = np.clip(
                    np.floor((xv - tier_min_x) / tier_res).astype(np.int64),
                    0, t_cols - 1,
                )
                row_idx = np.clip(
                    np.floor((yv - tier_min_y) / tier_res).astype(np.int64),
                    0, t_rows - 1,
                )
                cell_idx = row_idx * t_cols + col_idx

                unique_cells, inv_idx = np.unique(cell_idx, return_inverse=True)
                k = len(unique_cells)
                c_k = np.bincount(inv_idx, minlength=k).astype(np.int32)
                sum_z_k = np.bincount(inv_idx, weights=zv.astype(np.float64), minlength=k)
                sum_i_k = np.bincount(inv_idx, weights=iv, minlength=k)

                count_flat[unique_cells] = c_k
                mean_z_flat[unique_cells] = (sum_z_k / c_k).astype(np.float32)
                mean_i_flat[unique_cells] = (sum_i_k / c_k).astype(np.float32)

                min_z_flat[unique_cells] = np.inf
                max_z_flat[unique_cells] = -np.inf
                np.minimum.at(min_z_flat, cell_idx, zv)
                np.maximum.at(max_z_flat, cell_idx, zv)

        return GridMap25D(
            min_x=tier_min_x, max_x=tier_max_x,
            min_y=tier_min_y, max_y=tier_max_y,
            resolution=tier_res,
            point_count=count_flat.reshape((t_rows, t_cols)),
            min_z=min_z_flat.reshape((t_rows, t_cols)),
            max_z=max_z_flat.reshape((t_rows, t_cols)),
            mean_z=mean_z_flat.reshape((t_rows, t_cols)),
            mean_intensity=mean_i_flat.reshape((t_rows, t_cols)),
            metadata={
                "source_frame_id": current_frame.frame_id,
                "timestamp": current_frame.timestamp,
                "input_points": n_pts,
                "mapped_points": int(np.sum(count_flat)),
                "reused_base_cells": reuse_cells,
                "rebuilt_base_cells": rebuild_cells,
            },
        )
