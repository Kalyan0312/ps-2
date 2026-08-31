"""
Configuration Dataclass for Terrain Complexity and Roughness Analysis.
Supports loading from dictionaries and YAML configuration files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


@dataclass
class TerrainConfig:
    """
    Configuration options for terrain roughness and slope analysis.
    
    Attributes:
        enabled (bool): Toggle for terrain analysis module.
        window_size (int): Neighborhood window size in cells (odd integer >= 3, e.g. 3, 5, 7).
        calculate_roughness (bool): Compute local elevation standard deviation.
        calculate_slope (bool): Compute local spatial gradient magnitude.
        calculate_elevation_range (bool): Compute local elevation difference (max - min).
        min_valid_neighbors (int): Minimum number of valid occupied cells required in the window.
    """
    enabled: bool = True
    window_size: int = 3
    calculate_roughness: bool = True
    calculate_slope: bool = True
    calculate_elevation_range: bool = True
    min_valid_neighbors: int = 1

    def __post_init__(self):
        if self.window_size < 3 or self.window_size % 2 == 0:
            raise ValueError(f"window_size must be an odd integer >= 3, got {self.window_size}")
        if self.min_valid_neighbors < 1:
            raise ValueError(f"min_valid_neighbors must be >= 1, got {self.min_valid_neighbors}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TerrainConfig":
        """Builds TerrainConfig from a configuration dictionary."""
        terrain_data = data.get("terrain", data)
        return cls(
            enabled=bool(terrain_data.get("enabled", True)),
            window_size=int(terrain_data.get("window_size", 3)),
            calculate_roughness=bool(terrain_data.get("calculate_roughness", True)),
            calculate_slope=bool(terrain_data.get("calculate_slope", True)),
            calculate_elevation_range=bool(terrain_data.get("calculate_elevation_range", True)),
            min_valid_neighbors=int(terrain_data.get("min_valid_neighbors", 1)),
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "TerrainConfig":
        """Loads TerrainConfig directly from a YAML file."""
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}
        return cls.from_dict(raw_data)
