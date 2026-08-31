"""
Temporal Processing and Change Detection Configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Union, Optional
import yaml


@dataclass
class TemporalConfig:
    """
    Configuration options for multi-frame temporal processing and change detection.

    Attributes:
        enabled: If True, temporal change detection is active.
        elevation_change_threshold: Elevation difference threshold (meters) in mean_z
            to flag a cell as changed when occupied in consecutive frames (default 0.15m).
        min_points_per_cell: Minimum point count to treat a grid cell as occupied.
        min_points_for_comparison: Alias for min_points_per_cell.
        reuse_stable_regions: If True, identifies stable regions for spatial caching.
        intensity_change_threshold: Optional threshold for reflectance/intensity changes.
    """
    enabled: bool = True
    elevation_change_threshold: float = 0.15
    min_points_per_cell: int = 1
    min_points_for_comparison: Optional[int] = None
    reuse_stable_regions: bool = True
    intensity_change_threshold: Optional[float] = None

    def __post_init__(self):
        if self.min_points_for_comparison is not None:
            self.min_points_per_cell = self.min_points_for_comparison
        else:
            self.min_points_for_comparison = self.min_points_per_cell

        if self.elevation_change_threshold < 0:
            raise ValueError(
                f"elevation_change_threshold must be non-negative, got {self.elevation_change_threshold}"
            )
        if self.min_points_per_cell < 1:
            raise ValueError(
                f"min_points_per_cell must be >= 1, got {self.min_points_per_cell}"
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemporalConfig":
        """Constructs TemporalConfig from dictionary."""
        t_data = data.get("temporal", data)
        min_pts = int(t_data.get("min_points_for_comparison", t_data.get("min_points_per_cell", 1)))
        return cls(
            enabled=bool(t_data.get("enabled", True)),
            elevation_change_threshold=float(t_data.get("elevation_change_threshold", 0.15)),
            min_points_per_cell=min_pts,
            min_points_for_comparison=min_pts,
            reuse_stable_regions=bool(t_data.get("reuse_stable_regions", True)),
            intensity_change_threshold=(
                float(t_data["intensity_change_threshold"])
                if t_data.get("intensity_change_threshold") is not None
                else None
            ),
        )

    @classmethod
    def from_yaml(cls, path: Union[Path, str]) -> "TemporalConfig":
        """Loads TemporalConfig directly from YAML file."""
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}
        return cls.from_dict(raw_data)


# Alias for Phase 13 naming convention
TemporalChangeConfig = TemporalConfig
