"""
Configuration Dataclass for Preprocessing Pipeline.
Supports loading from Python dictionaries or YAML files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


@dataclass
class PreprocessingConfig:
    """
    Configuration options for point cloud preprocessing.
    
    Attributes:
        enabled (bool): Global toggle for preprocessing pipeline.
        remove_invalid (bool): Filter NaN and Infinite values.
        min_range (float): Minimum radial distance (meters).
        max_range (float): Maximum radial distance (meters).
        use_2d_range (bool): Compute range in 2D (XY) vs 3D Euclidean.
        min_z (float): Minimum elevation cutoff (meters).
        max_z (float): Maximum elevation cutoff (meters).
        voxel_downsample_enabled (bool): Enable voxel grid downsampling.
        voxel_size (float): Voxel cell dimension in meters.
    """
    enabled: bool = True
    remove_invalid: bool = True
    min_range: float = 0.5
    max_range: float = 25.0
    use_2d_range: bool = False
    min_z: float = -2.0
    max_z: float = 4.0
    voxel_downsample_enabled: bool = False
    voxel_size: float = 0.1

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreprocessingConfig":
        """Builds PreprocessingConfig from a dictionary (e.g. from config.yaml)."""
        prep_data = data.get("preprocessing", data)
        
        # Handle nested or flat voxel settings
        voxel_cfg = prep_data.get("voxel_downsample", {})
        if isinstance(voxel_cfg, dict):
            voxel_enabled = voxel_cfg.get("enabled", False)
            voxel_sz = voxel_cfg.get("voxel_size", prep_data.get("voxel_size", 0.1))
        else:
            voxel_enabled = prep_data.get("voxel_downsample_enabled", False)
            voxel_sz = prep_data.get("voxel_size", 0.1)

        return cls(
            enabled=prep_data.get("enabled", True),
            remove_invalid=prep_data.get("remove_invalid", True),
            min_range=float(prep_data.get("min_range", 0.5)),
            max_range=float(prep_data.get("max_range", 25.0)),
            use_2d_range=bool(prep_data.get("use_2d_range", False)),
            min_z=float(prep_data.get("min_z", -2.0)),
            max_z=float(prep_data.get("max_z", 4.0)),
            voxel_downsample_enabled=bool(voxel_enabled),
            voxel_size=float(voxel_sz),
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "PreprocessingConfig":
        """Loads PreprocessingConfig directly from a YAML file."""
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}
        return cls.from_dict(raw_data)
