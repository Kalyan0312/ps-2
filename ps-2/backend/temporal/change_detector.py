"""
Temporal Change Detector.

Compares consecutive 2.5D elevation grid maps or AdaptiveMap25D instances
to identify occupancy transitions, elevation variations, and stability across
spatial regions.
"""

from __future__ import annotations

from typing import Optional, Any, Union, Tuple
import numpy as np

from backend.mapping.grid_map import GridMap25D
from backend.mapping.adaptive_map import AdaptiveMap25D
from backend.temporal.config import TemporalConfig, TemporalChangeConfig
from backend.temporal.result import (
    TemporalChangeResult,
    STATE_UNOCCUPIED,
    STATE_UNCHANGED,
    STATE_CHANGED,
    STATE_NEWLY_OBSERVED,
    STATE_NO_LONGER_OBSERVED,
)


class TemporalChangeDetector:
    """
    Evaluates temporal changes between two consecutive AdaptiveMap25D or GridMap25D instances.

    The comparison operates in a common spatial reference domain (the base index grid).
    It compares corresponding spatial regions across any resolution tiers and classifies
    each cell as:
      - 'unchanged': represented in both with |delta_z| < threshold
      - 'changed': represented in both with |delta_z| >= threshold
      - 'newly_observed': represented in current map, absent in previous
      - 'no_longer_observed': represented in previous map, absent in current
      - 'unoccupied': empty in both maps
    """

    def __init__(self, config: Optional[Union[TemporalConfig, TemporalChangeConfig]] = None):
        self.config = config or TemporalConfig()

    def detect_change(
        self,
        current_grid: Union[AdaptiveMap25D, GridMap25D],
        previous_grid: Optional[Union[AdaptiveMap25D, GridMap25D]] = None,
        frame_index: int = 0,
        current_frame_id: Any = None,
        previous_frame_id: Any = None,
    ) -> TemporalChangeResult:
        """
        Performs temporal change detection between previous_grid and current_grid.

        Args:
            current_grid: Current frame's AdaptiveMap25D or GridMap25D.
            previous_grid: Previous frame's AdaptiveMap25D or GridMap25D (None if initial).
            frame_index: Sequence index of current frame.
            current_frame_id: Optional identifier of current frame.
            previous_frame_id: Optional identifier of previous frame.

        Returns:
            TemporalChangeResult: Classified temporal states, masks, and delta stats.
        """
        if isinstance(current_grid, AdaptiveMap25D):
            return self._detect_adaptive_change(
                current_map=current_grid,
                previous_map=previous_grid,
                frame_index=frame_index,
                current_frame_id=current_frame_id,
                previous_frame_id=previous_frame_id,
            )
        else:
            return self._detect_uniform_grid_change(
                current_grid=current_grid,
                previous_grid=previous_grid,
                frame_index=frame_index,
                current_frame_id=current_frame_id,
                previous_frame_id=previous_frame_id,
            )

    # ------------------------------------------------------------------
    # Adaptive Map Temporal Comparison (Phase 13)
    # ------------------------------------------------------------------

    def _detect_adaptive_change(
        self,
        current_map: AdaptiveMap25D,
        previous_map: Optional[AdaptiveMap25D] = None,
        frame_index: int = 0,
        current_frame_id: Any = None,
        previous_frame_id: Any = None,
    ) -> TemporalChangeResult:
        """Compares two AdaptiveMap25D instances in a common spatial reference domain."""
        threshold = self.config.elevation_change_threshold
        min_pts = self.config.min_points_per_cell
        cid = current_frame_id if current_frame_id is not None else current_map.metadata.get("source_frame_id", frame_index)

        # --------------------------------------------------------------
        # 1. Initial Frame Handling (No previous state)
        # --------------------------------------------------------------
        if previous_map is None:
            shape = current_map.base_shape
            res = current_map.base_resolution
            bounds = current_map.base_bounds

            curr_rep, _ = self._extract_base_layer(current_map, min_pts)

            change_state = np.full(shape, STATE_UNOCCUPIED, dtype="<U20")
            change_state[curr_rep] = STATE_NEWLY_OBSERVED

            new_occ = curr_rep.copy()
            rem_occ = np.zeros(shape, dtype=bool)
            elev_chg = np.zeros(shape, dtype=bool)
            unchanged = np.zeros(shape, dtype=bool)
            elev_diff = np.full(shape, np.nan, dtype=np.float32)
            changed = np.zeros(shape, dtype=bool)
            unoccupied = ~curr_rep

            metadata = {
                "threshold_elevation_m": threshold,
                "status": "initial_frame",
                "min_points_for_comparison": min_pts,
                "comparison_type": "adaptive_base_grid",
            }

            return TemporalChangeResult(
                frame_index=frame_index,
                previous_frame_id=None,
                current_frame_id=cid,
                grid_shape=shape,
                resolution=res,
                bounds=bounds,
                is_initial_frame=True,
                change_state=change_state,
                elevation_difference=elev_diff,
                changed_mask=changed,
                unchanged_mask=unchanged,
                newly_observed_mask=new_occ,
                no_longer_observed_mask=rem_occ,
                unoccupied_mask=unoccupied,
                elevation_changed_mask=elev_chg,
                metadata=metadata,
            )

        # --------------------------------------------------------------
        # 2. Consecutive Frames Comparison
        # --------------------------------------------------------------
        pid = previous_frame_id if previous_frame_id is not None else previous_map.metadata.get("source_frame_id", frame_index - 1)

        # Check if base bounds and resolution are aligned
        same_bounds = (
            np.isclose(current_map.min_x, previous_map.min_x, atol=1e-3) and
            np.isclose(current_map.max_x, previous_map.max_x, atol=1e-3) and
            np.isclose(current_map.min_y, previous_map.min_y, atol=1e-3) and
            np.isclose(current_map.max_y, previous_map.max_y, atol=1e-3) and
            np.isclose(current_map.base_resolution, previous_map.base_resolution, atol=1e-5) and
            current_map.base_shape == previous_map.base_shape
        )

        if same_bounds:
            # Fast vectorized comparison over aligned base grid
            shape = current_map.base_shape
            res = current_map.base_resolution
            bounds = current_map.base_bounds

            curr_rep, curr_mean_z = self._extract_base_layer(current_map, min_pts)
            prev_rep, prev_mean_z = self._extract_base_layer(previous_map, min_pts)

            both_rep = curr_rep & prev_rep
            newly_observed = curr_rep & (~prev_rep)
            no_longer_observed = prev_rep & (~curr_rep)
            unoccupied = (~curr_rep) & (~prev_rep)

            elev_diff = np.full(shape, np.nan, dtype=np.float32)
            elev_chg = np.zeros(shape, dtype=bool)
            unchanged = np.zeros(shape, dtype=bool)

            if np.any(both_rep):
                diffs = curr_mean_z[both_rep] - prev_mean_z[both_rep]
                elev_diff[both_rep] = diffs
                abs_diffs = np.abs(diffs)
                elev_chg[both_rep] = abs_diffs >= threshold
                unchanged[both_rep] = abs_diffs < threshold

            changed = newly_observed | no_longer_observed | elev_chg

            change_state = np.full(shape, STATE_UNOCCUPIED, dtype="<U20")
            change_state[unchanged] = STATE_UNCHANGED
            change_state[elev_chg] = STATE_CHANGED
            change_state[newly_observed] = STATE_NEWLY_OBSERVED
            change_state[no_longer_observed] = STATE_NO_LONGER_OBSERVED

        else:
            # Non-identical or partially overlapping bounds: construct union domain
            res = current_map.base_resolution
            min_x = min(current_map.min_x, previous_map.min_x)
            max_x = max(current_map.max_x, previous_map.max_x)
            min_y = min(current_map.min_y, previous_map.min_y)
            max_y = max(current_map.max_y, previous_map.max_y)
            bounds = (min_x, max_x, min_y, max_y)

            rows = int(np.ceil((max_y - min_y) / res))
            cols = int(np.ceil((max_x - min_x) / res))
            shape = (rows, cols)

            change_state = np.full(shape, STATE_UNOCCUPIED, dtype="<U20")
            elev_diff = np.full(shape, np.nan, dtype=np.float32)
            newly_observed = np.zeros(shape, dtype=bool)
            no_longer_observed = np.zeros(shape, dtype=bool)
            elev_chg = np.zeros(shape, dtype=bool)
            unchanged = np.zeros(shape, dtype=bool)
            unoccupied = np.ones(shape, dtype=bool)

            # Evaluate each cell in union space via O(1) world_query
            for r in range(rows):
                wy = min_y + (r + 0.5) * res
                for c in range(cols):
                    wx = min_x + (c + 0.5) * res
                    cq = current_map.world_query(wx, wy)
                    pq = previous_map.world_query(wx, wy)

                    c_rep = cq.is_represented and (cq.point_count is not None and cq.point_count >= min_pts)
                    p_rep = pq.is_represented and (pq.point_count is not None and pq.point_count >= min_pts)

                    if c_rep and p_rep:
                        unoccupied[r, c] = False
                        diff = float(cq.mean_z - pq.mean_z)
                        elev_diff[r, c] = diff
                        if abs(diff) >= threshold:
                            elev_chg[r, c] = True
                            change_state[r, c] = STATE_CHANGED
                        else:
                            unchanged[r, c] = True
                            change_state[r, c] = STATE_UNCHANGED
                    elif c_rep and not p_rep:
                        unoccupied[r, c] = False
                        newly_observed[r, c] = True
                        change_state[r, c] = STATE_NEWLY_OBSERVED
                    elif p_rep and not c_rep:
                        unoccupied[r, c] = False
                        no_longer_observed[r, c] = True
                        change_state[r, c] = STATE_NO_LONGER_OBSERVED

            changed = newly_observed | no_longer_observed | elev_chg

        metadata = {
            "threshold_elevation_m": threshold,
            "status": "consecutive_comparison",
            "min_points_for_comparison": min_pts,
            "comparison_type": "adaptive_base_grid",
        }

        return TemporalChangeResult(
            frame_index=frame_index,
            previous_frame_id=pid,
            current_frame_id=cid,
            grid_shape=shape,
            resolution=res,
            bounds=bounds,
            is_initial_frame=False,
            change_state=change_state,
            elevation_difference=elev_diff,
            changed_mask=changed,
            unchanged_mask=unchanged,
            newly_observed_mask=newly_observed,
            no_longer_observed_mask=no_longer_observed,
            unoccupied_mask=unoccupied,
            elevation_changed_mask=elev_chg,
            metadata=metadata,
        )

    @staticmethod
    def _extract_base_layer(
        amap: AdaptiveMap25D,
        min_pts: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts base-grid representation mask and mean elevation from an AdaptiveMap25D
        by aggregating across all fine cells in each assigned base cell.
        """
        rows, cols = amap.base_shape
        base_counts = np.zeros(rows * cols, dtype=np.int32)
        base_sum_z = np.zeros(rows * cols, dtype=np.float64)

        for tier, tg in amap.tier_maps.items():
            cols_grid, rows_grid = np.meshgrid(np.arange(tg.cols), np.arange(tg.rows))
            wx_all = tg.min_x + (cols_grid + 0.5) * tg.resolution
            wy_all = tg.min_y + (rows_grid + 0.5) * tg.resolution

            base_c = np.clip(
                np.floor((wx_all - amap.min_x) / amap.base_resolution).astype(np.int32),
                0, cols - 1,
            )
            base_r = np.clip(
                np.floor((wy_all - amap.min_y) / amap.base_resolution).astype(np.int32),
                0, rows - 1,
            )
            base_lin = base_r * cols + base_c

            occ = (tg.point_count > 0) & (~np.isnan(tg.mean_z))
            if np.any(occ):
                occ_lin = base_lin[occ]
                occ_pts = tg.point_count[occ]
                occ_zs = tg.mean_z[occ].astype(np.float64) * occ_pts
                np.add.at(base_counts, occ_lin, occ_pts)
                np.add.at(base_sum_z, occ_lin, occ_zs)

        rep_mask = (base_counts >= min_pts).reshape((rows, cols))
        mean_z = np.full((rows, cols), np.nan, dtype=np.float32)

        flat_rep = base_counts >= min_pts
        if np.any(flat_rep):
            flat_mean = (base_sum_z[flat_rep] / np.maximum(base_counts[flat_rep], 1)).astype(np.float32)
            mean_z_flat = mean_z.reshape(-1)
            mean_z_flat[flat_rep] = flat_mean
            mean_z = mean_z_flat.reshape((rows, cols))

        return rep_mask, mean_z

    # ------------------------------------------------------------------
    # Uniform GridMap25D Temporal Comparison (Phase 9 Compatibility)
    # ------------------------------------------------------------------

    def _detect_uniform_grid_change(
        self,
        current_grid: GridMap25D,
        previous_grid: Optional[GridMap25D] = None,
        frame_index: int = 0,
        current_frame_id: Any = None,
        previous_frame_id: Any = None,
    ) -> TemporalChangeResult:
        """Uniform 2.5D grid change detector preserving exact Phase 9 behavior."""
        rows, cols = current_grid.shape
        shape = (rows, cols)
        res = current_grid.resolution
        bounds = (current_grid.min_x, current_grid.max_x, current_grid.min_y, current_grid.max_y)
        min_pts = self.config.min_points_per_cell
        threshold = self.config.elevation_change_threshold

        cid = current_frame_id if current_frame_id is not None else current_grid.metadata.get("frame_id", frame_index)

        if previous_grid is None:
            curr_occ = (current_grid.point_count >= min_pts) & (~np.isnan(current_grid.mean_z))
            new_occ = np.zeros(shape, dtype=bool)
            rem_occ = np.zeros(shape, dtype=bool)
            elev_chg = np.zeros(shape, dtype=bool)
            elev_diff = np.full(shape, np.nan, dtype=np.float32)
            changed = np.zeros(shape, dtype=bool)
            stable = curr_occ.copy()
            unoccupied = ~curr_occ

            change_state = np.full(shape, STATE_UNOCCUPIED, dtype="<U20")
            change_state[curr_occ] = STATE_UNCHANGED

            metadata = {
                "threshold_elevation_m": threshold,
                "status": "initial_frame",
                "min_points_per_cell": min_pts,
                "comparison_type": "uniform_grid",
            }

            return TemporalChangeResult(
                frame_index=frame_index,
                previous_frame_id=None,
                current_frame_id=cid,
                grid_shape=shape,
                resolution=res,
                bounds=bounds,
                is_initial_frame=True,
                change_state=change_state,
                elevation_difference=elev_diff,
                changed_mask=changed,
                unchanged_mask=stable,
                newly_observed_mask=new_occ,
                no_longer_observed_mask=rem_occ,
                unoccupied_mask=unoccupied,
                elevation_changed_mask=elev_chg,
                metadata=metadata,
            )

        self._validate_compatibility(current_grid, previous_grid)
        pid = previous_frame_id if previous_frame_id is not None else previous_grid.metadata.get("frame_id", frame_index - 1)

        curr_occ = (current_grid.point_count >= min_pts) & (~np.isnan(current_grid.mean_z))
        prev_occ = (previous_grid.point_count >= min_pts) & (~np.isnan(previous_grid.mean_z))

        new_occ = (~prev_occ) & curr_occ
        rem_occ = prev_occ & (~curr_occ)
        both_occ = prev_occ & curr_occ
        unoccupied = (~prev_occ) & (~curr_occ)

        elev_diff = np.full(shape, np.nan, dtype=np.float32)
        elev_chg = np.zeros(shape, dtype=bool)
        stable = np.zeros(shape, dtype=bool)

        if np.any(both_occ):
            diffs = current_grid.mean_z[both_occ] - previous_grid.mean_z[both_occ]
            elev_diff[both_occ] = diffs
            abs_diffs = np.abs(diffs)
            elev_chg[both_occ] = abs_diffs >= threshold
            stable[both_occ] = abs_diffs < threshold

        changed = new_occ | rem_occ | elev_chg

        change_state = np.full(shape, STATE_UNOCCUPIED, dtype="<U20")
        change_state[stable] = STATE_UNCHANGED
        change_state[elev_chg] = STATE_CHANGED
        change_state[new_occ] = STATE_NEWLY_OBSERVED
        change_state[rem_occ] = STATE_NO_LONGER_OBSERVED

        metadata = {
            "threshold_elevation_m": threshold,
            "status": "consecutive_comparison",
            "min_points_per_cell": min_pts,
            "comparison_type": "uniform_grid",
        }

        return TemporalChangeResult(
            frame_index=frame_index,
            previous_frame_id=pid,
            current_frame_id=cid,
            grid_shape=shape,
            resolution=res,
            bounds=bounds,
            is_initial_frame=False,
            change_state=change_state,
            elevation_difference=elev_diff,
            changed_mask=changed,
            unchanged_mask=stable,
            newly_observed_mask=new_occ,
            no_longer_observed_mask=rem_occ,
            unoccupied_mask=unoccupied,
            elevation_changed_mask=elev_chg,
            metadata=metadata,
        )

    @staticmethod
    def _validate_compatibility(grid_a: GridMap25D, grid_b: GridMap25D) -> None:
        """Validates that two grids share identical bounds, shape, and resolution."""
        if grid_a.shape != grid_b.shape:
            raise ValueError(
                f"Incompatible grid shapes for temporal comparison: {grid_a.shape} vs {grid_b.shape}"
            )
        if not np.isclose(grid_a.resolution, grid_b.resolution, atol=1e-5):
            raise ValueError(
                f"Incompatible grid resolutions: {grid_a.resolution} m vs {grid_b.resolution} m"
            )
        if not (
            np.isclose(grid_a.min_x, grid_b.min_x, atol=1e-3) and
            np.isclose(grid_a.max_x, grid_b.max_x, atol=1e-3) and
            np.isclose(grid_a.min_y, grid_b.min_y, atol=1e-3) and
            np.isclose(grid_a.max_y, grid_b.max_y, atol=1e-3)
        ):
            raise ValueError(
                f"Incompatible grid bounds: A=({grid_a.min_x}, {grid_a.max_x}, {grid_a.min_y}, {grid_a.max_y}) vs "
                f"B=({grid_b.min_x}, {grid_b.max_x}, {grid_b.min_y}, {grid_b.max_y})"
            )
