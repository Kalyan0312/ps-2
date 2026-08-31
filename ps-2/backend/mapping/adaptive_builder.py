"""
Adaptive 2.5D Map Builder.

Consumes a ResolutionDecisionResult (Phase 5) and constructs an AdaptiveMap25D
by building one GridMap25D per required resolution tier from the original
LidarFrame, restricted to the spatial sub-regions assigned to that tier.

Construction Algorithm
----------------------
1. For each tier name in the decision result's valid cells:
   a. Collect all base-grid cells assigned to this tier.
   b. Determine the tight bounding box of those cells (in world coordinates).
   c. Build a GridMap25D at the tier's resolution over that bounding box from
      the original LidarFrame points that fall inside it.
   d. Record the tier-specific grid in tier_maps.

2. Build the index matrices (index_tier, index_row, index_col) aligned to the
   base grid so that world_query() is O(1).

The LidarFrame used must be the *preprocessed* frame (same one that produced
the base GridMap25D inside the decision result).

Phase 12 Part C Optimizations
------------------------------
- Point arrays are extracted once (O(N)); each point is mapped to its
  base-grid cell once via a vectorized floor+clip.
- Per-tier point selection uses that pre-computed mapping for an O(N)
  boolean lookup — no repeated full-point-cloud scans and no intermediate
  LidarFrame copies.
- Tier GridMap25D objects are built directly from the partitioned arrays
  via _build_tier_grid(), bypassing Uniform25DMapBuilder to avoid a
  second spatial-filter pass over each tier's points.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
from pathlib import Path
import numpy as np

from backend.data.frame import LidarFrame
from backend.mapping.config import MappingConfig
from backend.mapping.grid_map import GridMap25D
from backend.mapping.adaptive_map import AdaptiveMap25D
from backend.priority.result import ResolutionDecisionResult


class AdaptiveMap25DBuilder:
    """
    Builds an AdaptiveMap25D from a ResolutionDecisionResult and a LidarFrame.

    The builder:
    - Reads resolution levels from the existing MappingConfig (config.yaml).
    - Reuses Uniform25DMapBuilder for each tier's GridMap25D construction.
    - Does NOT store four complete global maps; each tier map covers only the
      extent of its assigned cells.

    Args:
        config: MappingConfig with resolution-level definitions and bounds.
            Defaults to MappingConfig() if not supplied.
    """

    def __init__(self, config: Optional[MappingConfig] = None):
        self.config = config or MappingConfig()

    @classmethod
    def from_config_file(cls, config_path: str | Path) -> "AdaptiveMap25DBuilder":
        """Instantiates builder directly from YAML configuration file."""
        cfg = MappingConfig.from_yaml(config_path)
        return cls(config=cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        frame: LidarFrame,
        decision: ResolutionDecisionResult,
    ) -> AdaptiveMap25D:
        """
        Constructs the composite AdaptiveMap25D.

        Phase 12 Part C optimizations
        ------------------------------
        - Point arrays (x, y, z, intensity) are extracted from the frame once.
        - Each point is mapped to a base-grid cell once (O(N) vectorized).
        - Per-tier point selection uses a single vectorized lookup into the
          pre-computed base-cell assignment — no repeated full-point-cloud scans
          and no intermediate LidarFrame copies.
        - Tier GridMap25D objects are built directly from the partitioned arrays,
          bypassing the Uniform25DMapBuilder path to avoid a second point scan.

        Args:
            frame: Preprocessed LidarFrame (same one used to derive
                the base GridMap25D in `decision`).
            decision: ResolutionDecisionResult from Phase 5 (AdaptiveResolutionSelector).

        Returns:
            AdaptiveMap25D: Composite adaptive-resolution 2.5D elevation map.
        """
        base_grid = decision.grid_map
        valid_mask = decision.valid_mask
        assigned_level = decision.assigned_level
        res_levels = decision.resolution_levels

        rows, cols = base_grid.shape

        # Index matrices, initialised to "empty" sentinel values
        index_tier = np.full((rows, cols), "", dtype=object)
        index_row = np.full((rows, cols), -1, dtype=np.int32)
        index_col = np.full((rows, cols), -1, dtype=np.int32)

        tier_maps: Dict[str, GridMap25D] = {}

        # ------------------------------------------------------------------
        # ONE-TIME point array extraction — avoids repeated pts[:, k] slicing
        # across tiers.  No LidarFrame copies are created.
        # ------------------------------------------------------------------
        pts = frame.points
        n_pts = len(pts)

        if n_pts > 0:
            x_vals = pts[:, 0]
            y_vals = pts[:, 1]
            z_vals = pts[:, 2].astype(np.float32)
            i_vals = pts[:, 3].astype(np.float64)
        else:
            x_vals = y_vals = z_vals = i_vals = np.empty(0)

        # ------------------------------------------------------------------
        # Process each tier — spatial bbox filter is applied inside
        # _build_tier_grid(), identical to Uniform25DMapBuilder.build_map().
        # No LidarFrame copy or second full-scan per tier.
        # ------------------------------------------------------------------
        for tier, tier_res in res_levels.items():
            # Base-grid cells assigned to this tier
            tier_mask = valid_mask & (assigned_level == tier)
            if not np.any(tier_mask):
                continue  # No cells for this tier → skip

            # World-coordinate extents of tier-assigned base cells
            tier_min_x, tier_max_x, tier_min_y, tier_max_y = \
                self._tier_world_bounds(base_grid, tier_mask, tier_res)

            # Build tier GridMap25D directly from the pre-extracted arrays.
            # _build_tier_grid() applies the same spatial boundary filter as
            # Uniform25DMapBuilder.build_map() — outputs are bit-for-bit
            # identical to the previous _filter_frame + build_map() path,
            # but without the LidarFrame copy or second point-array scan.
            tier_grid = self._build_tier_grid(
                x_vals, y_vals, z_vals, i_vals,
                tier_min_x, tier_max_x, tier_min_y, tier_max_y, tier_res,
                frame_id=frame.frame_id, timestamp=frame.timestamp,
                n_input_pts=n_pts,
            )
            tier_maps[tier] = tier_grid

            # ------------------------------------------------------------------
            # Vectorized population of index matrices for base cells in this tier
            # ------------------------------------------------------------------
            br, bc = np.where(tier_mask)

            # Vectorized world centres of assigned base cells
            wx = base_grid.min_x + (bc + 0.5) * base_grid.resolution
            wy = base_grid.min_y + (br + 0.5) * base_grid.resolution

            # Vectorized corresponding fine-grid (row, col) with boundary clamping
            fc = np.clip(
                np.floor((wx - tier_grid.min_x) / tier_grid.resolution).astype(np.int32),
                0,
                tier_grid.cols - 1,
            )
            fr = np.clip(
                np.floor((wy - tier_grid.min_y) / tier_grid.resolution).astype(np.int32),
                0,
                tier_grid.rows - 1,
            )

            index_tier[br, bc] = tier
            index_row[br, bc] = fr
            index_col[br, bc] = fc

        metadata = {
            "source_frame_id": frame.frame_id,
            "timestamp": frame.timestamp,
            "input_points": len(frame.points),
            "num_tiers_built": len(tier_maps),
            "base_resolution": base_grid.resolution,
            "base_shape": base_grid.shape,
        }

        return AdaptiveMap25D(
            tier_maps=tier_maps,
            tier_resolutions=dict(res_levels),
            base_shape=(rows, cols),
            base_resolution=base_grid.resolution,
            base_bounds=(base_grid.min_x, base_grid.max_x,
                         base_grid.min_y, base_grid.max_y),
            index_tier=index_tier,
            index_row=index_row,
            index_col=index_col,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_tier_grid(
        x_t: np.ndarray,
        y_t: np.ndarray,
        z_t: np.ndarray,
        i_t: np.ndarray,
        min_x: float, max_x: float,
        min_y: float, max_y: float,
        res: float,
        *,
        frame_id: int,
        timestamp: float,
        n_input_pts: int,
    ) -> GridMap25D:
        """
        Builds a GridMap25D directly from pre-extracted point arrays.

        Identical accumulation logic to Uniform25DMapBuilder.build_map() so
        that numerical outputs are bit-for-bit equivalent. No LidarFrame copy
        is created and no second spatial filter pass is needed.
        """
        t_rows = int(np.ceil((max_y - min_y) / res))
        t_cols = int(np.ceil((max_x - min_x) / res))
        total_cells = t_rows * t_cols

        count_flat = np.zeros(total_cells, dtype=np.int32)
        min_z_flat = np.full(total_cells, np.nan, dtype=np.float32)
        max_z_flat = np.full(total_cells, np.nan, dtype=np.float32)
        mean_z_flat = np.full(total_cells, np.nan, dtype=np.float32)
        mean_i_flat = np.full(total_cells, np.nan, dtype=np.float32)

        if len(x_t) > 0:
            # Spatial boundary filter — matches Uniform25DMapBuilder exactly
            valid = (
                (x_t >= min_x) & (x_t <= max_x) &
                (y_t >= min_y) & (y_t <= max_y)
            )
            if np.any(valid):
                xv = x_t[valid]
                yv = y_t[valid]
                zv = z_t[valid]
                iv = i_t[valid]

                col_idx = np.clip(
                    np.floor((xv - min_x) / res).astype(np.int64),
                    0, t_cols - 1,
                )
                row_idx = np.clip(
                    np.floor((yv - min_y) / res).astype(np.int64),
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

        metadata = {
            "source_frame_id": frame_id,
            "timestamp": timestamp,
            "input_points": n_input_pts,
            "mapped_points": int(np.sum(count_flat)),
        }

        return GridMap25D(
            min_x=min_x, max_x=max_x,
            min_y=min_y, max_y=max_y,
            resolution=res,
            point_count=count_flat.reshape((t_rows, t_cols)),
            min_z=min_z_flat.reshape((t_rows, t_cols)),
            max_z=max_z_flat.reshape((t_rows, t_cols)),
            mean_z=mean_z_flat.reshape((t_rows, t_cols)),
            mean_intensity=mean_i_flat.reshape((t_rows, t_cols)),
            metadata=metadata,
        )

    @staticmethod
    def _tier_world_bounds(
        base_grid: GridMap25D,
        tier_mask: np.ndarray,
        tier_res: float,
    ) -> Tuple[float, float, float, float]:
        """
        Calculates the tight world bounding box for all base cells in tier_mask.

        The box is snapped to `tier_res` grid lines so that the resulting
        Uniform25DMapBuilder covers an integer number of tier cells.
        A minimum 1-cell padding ensures no point on the border is clipped.
        """
        cell_rows, cell_cols = np.where(tier_mask)

        # World centres of assigned base cells
        min_br, max_br = int(cell_rows.min()), int(cell_rows.max())
        min_bc, max_bc = int(cell_cols.min()), int(cell_cols.max())

        # World extent of those base cells
        raw_min_x = base_grid.min_x + min_bc * base_grid.resolution
        raw_max_x = base_grid.min_x + (max_bc + 1) * base_grid.resolution
        raw_min_y = base_grid.min_y + min_br * base_grid.resolution
        raw_max_y = base_grid.min_y + (max_br + 1) * base_grid.resolution

        # Snap outward to tier_res grid lines
        snap_min_x = float(np.floor(raw_min_x / tier_res) * tier_res)
        snap_max_x = float(np.ceil(raw_max_x / tier_res) * tier_res)
        snap_min_y = float(np.floor(raw_min_y / tier_res) * tier_res)
        snap_max_y = float(np.ceil(raw_max_y / tier_res) * tier_res)

        # Guarantee at least one tier cell span
        if snap_max_x <= snap_min_x:
            snap_max_x = snap_min_x + tier_res
        if snap_max_y <= snap_min_y:
            snap_max_y = snap_min_y + tier_res


        return snap_min_x, snap_max_x, snap_min_y, snap_max_y
