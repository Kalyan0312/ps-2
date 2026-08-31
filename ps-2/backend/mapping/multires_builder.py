"""
Multi-Resolution 2.5D Elevation Grid Map Builder.
Constructs a MultiResolutionMap containing GridMap25D instances across configured scales.
"""

from typing import Optional, Dict
from pathlib import Path

from backend.data.frame import LidarFrame
from backend.mapping.config import MappingConfig
from backend.mapping.grid_map import GridMap25D
from backend.mapping.uniform_builder import Uniform25DMapBuilder
from backend.mapping.multires_map import MultiResolutionMap


class MultiResolution25DMapBuilder:
    """
    Constructs multi-scale 2.5D elevation grid maps from a LiDAR point cloud.
    """

    def __init__(self, config: Optional[MappingConfig] = None):
        self.config = config or MappingConfig()

    @classmethod
    def from_config_file(cls, config_path: str | Path) -> "MultiResolution25DMapBuilder":
        """Instantiates builder directly from YAML configuration file."""
        cfg = MappingConfig.from_yaml(config_path)
        return cls(config=cfg)

    def build_multires_map(self, frame: LidarFrame) -> MultiResolutionMap:
        """
        Builds a MultiResolutionMap containing GridMap25D layers for each configured resolution.
        
        Args:
            frame (LidarFrame): Preprocessed point cloud frame.
            
        Returns:
            MultiResolutionMap: Multi-scale 2.5D elevation map container.
        """
        levels = self.config.multi_resolution_levels
        built_maps: Dict[float, GridMap25D] = {}

        for level_name, res in levels.items():
            # Build individual uniform map layer with specified resolution
            layer_config = MappingConfig(
                resolution=res,
                min_x=self.config.min_x,
                max_x=self.config.max_x,
                min_y=self.config.min_y,
                max_y=self.config.max_y,
                auto_bounds=self.config.auto_bounds,
            )
            builder = Uniform25DMapBuilder(config=layer_config)
            grid_layer = builder.build_map(frame)
            built_maps[res] = grid_layer

        metadata = {
            "source_frame_id": frame.frame_id,
            "timestamp": frame.timestamp,
            "input_points": len(frame.points),
            "num_levels": len(built_maps),
        }

        return MultiResolutionMap(
            maps=built_maps,
            levels=dict(levels),
            metadata=metadata,
        )
