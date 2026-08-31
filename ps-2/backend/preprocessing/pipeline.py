"""
LiDAR Preprocessing Pipeline.
Executes sequence of configured filtering stages on a LidarFrame.
"""

from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np

from backend.data.frame import LidarFrame
from backend.preprocessing.config import PreprocessingConfig
from backend.preprocessing.filters import (
    remove_invalid_points,
    filter_range,
    filter_height,
    voxel_downsample,
)


class PreprocessingPipeline:
    """
    Modular preprocessing pipeline for LiDAR point cloud frames.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()

    @classmethod
    def from_config_file(cls, config_path: str | Path) -> "PreprocessingPipeline":
        """Instantiates pipeline directly from YAML configuration file."""
        cfg = PreprocessingConfig.from_yaml(config_path)
        return cls(config=cfg)

    def process(self, frame: LidarFrame) -> LidarFrame:
        """
        Applies preprocessing filters to the provided LidarFrame.
        
        Args:
            frame (LidarFrame): Raw input frame.
            
        Returns:
            LidarFrame: Cleaned point cloud frame with updated metadata.
        """
        if not self.config.enabled:
            return frame

        points = frame.points
        initial_count = len(points)
        stats: Dict[str, Any] = {
            "initial_points": initial_count,
            "stages": {},
        }

        # Stage 1: Remove Invalid Points (NaN / Inf)
        if self.config.remove_invalid:
            pts_before = len(points)
            points = remove_invalid_points(points)
            stats["stages"]["invalid_removed"] = pts_before - len(points)

        # Stage 2: Radial Range Filter
        pts_before = len(points)
        points = filter_range(
            points,
            min_range=self.config.min_range,
            max_range=self.config.max_range,
            use_2d=self.config.use_2d_range,
        )
        stats["stages"]["out_of_range_removed"] = pts_before - len(points)

        # Stage 3: Height (Z) Filter
        pts_before = len(points)
        points = filter_height(
            points,
            min_z=self.config.min_z,
            max_z=self.config.max_z,
        )
        stats["stages"]["out_of_height_removed"] = pts_before - len(points)

        # Stage 4: Optional Voxel Downsampling
        if self.config.voxel_downsample_enabled:
            pts_before = len(points)
            points = voxel_downsample(points, voxel_size=self.config.voxel_size)
            stats["stages"]["voxel_downsampled_removed"] = pts_before - len(points)

        final_count = len(points)
        stats["final_points"] = final_count
        stats["total_removed"] = initial_count - final_count

        # Preserve metadata and append preprocessing diagnostics
        updated_metadata = dict(frame.metadata)
        updated_metadata["preprocessing"] = stats

        return LidarFrame(
            points=points,
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            metadata=updated_metadata,
        )
