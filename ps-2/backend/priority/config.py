"""
Configuration Dataclass for Adaptive Resolution Selection.
Supports thresholding and resolution level definitions from dictionaries and YAML configuration files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

DEFAULT_RESOLUTION_LEVELS = {
    "coarse": 0.50,
    "medium": 0.25,
    "fine": 0.10,
    "ultra_fine": 0.05,
}

DEFAULT_COMPLEXITY_THRESHOLDS = {
    "ultra_fine": {
        "roughness": 0.25,
        "slope": 1.50,
        "elevation_range": 0.80,
    },
    "fine": {
        "roughness": 0.08,
        "slope": 0.50,
        "elevation_range": 0.30,
    },
    "medium": {
        "roughness": 0.02,
        "slope": 0.15,
        "elevation_range": 0.10,
    },
}


@dataclass
class PriorityConfig:
    """
    Configuration options for Adaptive Resolution Decision Engine.
    
    Attributes:
        enabled (bool): Toggle for adaptive resolution decision engine.
        resolution_levels (Dict[str, float]): Named mapping of spatial resolution levels in meters.
        thresholds (Dict[str, Dict[str, float]]): Complexity thresholds for resolution tier assignment.
    """
    enabled: bool = True
    resolution_levels: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_RESOLUTION_LEVELS)
    )
    thresholds: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {k: dict(v) for k, v in DEFAULT_COMPLEXITY_THRESHOLDS.items()}
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PriorityConfig":
        """Builds PriorityConfig from a configuration dictionary."""
        p_data = data.get("priority", data)
        
        # Check if resolution levels are defined in mapping or priority block
        mapping_levels = data.get("mapping", {}).get("multi_resolution", {}).get("levels")
        if mapping_levels:
            res_levels = {k: float(v) for k, v in mapping_levels.items()}
        elif "resolution_levels" in p_data:
            res_levels = {k: float(v) for k, v in p_data["resolution_levels"].items()}
        else:
            res_levels = dict(DEFAULT_RESOLUTION_LEVELS)

        # Parse thresholds
        raw_thresh = p_data.get("thresholds", {})
        if raw_thresh:
            thresholds = {}
            for tier in ["ultra_fine", "fine", "medium"]:
                if tier in raw_thresh:
                    thresholds[tier] = {
                        "roughness": float(raw_thresh[tier].get("roughness", DEFAULT_COMPLEXITY_THRESHOLDS[tier]["roughness"])),
                        "slope": float(raw_thresh[tier].get("slope", DEFAULT_COMPLEXITY_THRESHOLDS[tier]["slope"])),
                        "elevation_range": float(raw_thresh[tier].get("elevation_range", DEFAULT_COMPLEXITY_THRESHOLDS[tier]["elevation_range"])),
                    }
                else:
                    thresholds[tier] = dict(DEFAULT_COMPLEXITY_THRESHOLDS[tier])
        else:
            thresholds = {k: dict(v) for k, v in DEFAULT_COMPLEXITY_THRESHOLDS.items()}

        return cls(
            enabled=bool(p_data.get("enabled", True)),
            resolution_levels=res_levels,
            thresholds=thresholds,
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "PriorityConfig":
        """Loads PriorityConfig directly from a YAML file."""
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}
        return cls.from_dict(raw_data)
