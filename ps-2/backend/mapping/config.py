"""
Configuration Dataclass for 2.5D Elevation Grid Mapping.
Supports loading uniform and multi-resolution settings from dictionaries and YAML configuration files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


DEFAULT_MULTIRES_LEVELS = {
    "coarse": 0.50,
    "medium": 0.25,
    "fine": 0.10,
    "ultra_fine": 0.05,
}


@dataclass
class MappingConfig:
    """
    Configuration options for 2.5D Grid Mapping.
    
    Attributes:
        resolution (float): Default baseline uniform grid cell edge length in meters.
        min_x (float): Minimum X coordinate boundary (meters).
        max_x (float): Maximum X coordinate boundary (meters).
        min_y (float): Minimum Y coordinate boundary (meters).
        max_y (float): Maximum Y coordinate boundary (meters).
        auto_bounds (bool): If True, computes bounds dynamically from input point cloud.
        multi_resolution_levels (Dict[str, float]): Dictionary of named resolution levels in meters.
    """
    resolution: float = 0.2
    min_x: float = -25.0
    max_x: float = 25.0
    min_y: float = -25.0
    max_y: float = 25.0
    auto_bounds: bool = False
    multi_resolution_levels: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_MULTIRES_LEVELS))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MappingConfig":
        """Builds MappingConfig from a configuration dictionary."""
        map_data = data.get("mapping", data)
        
        # Parse multi-resolution levels if present
        multires_data = map_data.get("multi_resolution", {})
        if isinstance(multires_data, dict) and "levels" in multires_data:
            levels = {k: float(v) for k, v in multires_data["levels"].items()}
        else:
            levels = dict(DEFAULT_MULTIRES_LEVELS)

        return cls(
            resolution=float(map_data.get("resolution", 0.2)),
            min_x=float(map_data.get("min_x", -25.0)),
            max_x=float(map_data.get("max_x", 25.0)),
            min_y=float(map_data.get("min_y", -25.0)),
            max_y=float(map_data.get("max_y", 25.0)),
            auto_bounds=bool(map_data.get("auto_bounds", False)),
            multi_resolution_levels=levels,
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "MappingConfig":
        """Loads MappingConfig directly from a YAML file."""
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}
        return cls.from_dict(raw_data)
