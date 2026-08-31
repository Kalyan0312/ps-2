"""
Multi-Resolution 2.5D Elevation Grid Map Container.
Stores and indexes multiple GridMap25D layers at different spatial resolutions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Union, Optional, Iterator
import numpy as np

from backend.mapping.grid_map import GridMap25D


@dataclass
class MultiResolutionMap:
    """
    Container storing multiple GridMap25D layers across different spatial resolutions.
    
    Attributes:
        maps (Dict[float, GridMap25D]): Mapping from resolution (float in meters) to GridMap25D.
        levels (Dict[str, float]): Mapping from level name (e.g. 'coarse', 'fine') to resolution.
        metadata (dict): Frame or generation metadata.
    """
    maps: Dict[float, GridMap25D]
    levels: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Sort maps by resolution descending (coarse -> fine)
        self.maps = dict(sorted(self.maps.items(), key=lambda item: item[0], reverse=True))

    def __len__(self) -> int:
        """Returns number of resolution levels available."""
        return len(self.maps)

    def __iter__(self) -> Iterator[float]:
        """Iterates over available resolutions."""
        return iter(self.maps.keys())

    def __getitem__(self, key: Union[float, str]) -> GridMap25D:
        """Allows indexing by float resolution or level name string."""
        return self.get_map(key)

    @property
    def available_resolutions(self) -> List[float]:
        """Returns sorted list of available resolutions in meters (coarse to fine)."""
        return list(self.maps.keys())

    @property
    def available_levels(self) -> List[str]:
        """Returns list of named resolution level aliases."""
        return list(self.levels.keys())

    def get_level_name(self, resolution: float) -> Optional[str]:
        """Finds named alias for a given resolution value if registered."""
        for name, res in self.levels.items():
            if np.isclose(res, resolution, atol=1e-5):
                return name
        return None

    def get_map(self, resolution_or_level: Union[float, str]) -> GridMap25D:
        """
        Retrieves GridMap25D by resolution value (meters) or level name string.
        
        Args:
            resolution_or_level (float | str): e.g., 0.25 or 'medium'.
            
        Returns:
            GridMap25D: The corresponding elevation grid map.
            
        Raises:
            KeyError: If the requested resolution or level name does not exist.
        """
        if isinstance(resolution_or_level, str):
            level_name = resolution_or_level.strip().lower()
            if level_name in self.levels:
                res_val = self.levels[level_name]
                return self.get_map(res_val)
            else:
                # Check if string is a numeric representation (e.g. "0.25")
                try:
                    res_val = float(level_name)
                    return self.get_map(res_val)
                except ValueError:
                    raise KeyError(
                        f"Unknown resolution level '{resolution_or_level}'. "
                        f"Available levels: {list(self.levels.keys())}, "
                        f"Available resolutions: {self.available_resolutions}"
                    )

        target_res = float(resolution_or_level)
        for res_key, grid in self.maps.items():
            if np.isclose(res_key, target_res, atol=1e-5):
                return grid

        raise KeyError(
            f"Resolution {target_res:.3f}m not found in MultiResolutionMap. "
            f"Available resolutions: {[f'{r:.3f}m' for r in self.maps.keys()]}"
        )

    def get_summary(self) -> List[Dict[str, Any]]:
        """
        Returns structural inspection summary across all stored resolution levels.
        """
        summary_list = []
        for res, grid in self.maps.items():
            level_name = self.get_level_name(res)
            summary_list.append({
                "level_name": level_name,
                "resolution": res,
                "rows": grid.rows,
                "cols": grid.cols,
                "dimensions": f"{grid.rows}x{grid.cols}",
                "total_cells": grid.total_cells,
                "occupied_cells": grid.num_occupied_cells,
                "empty_cells": grid.num_empty_cells,
                "occupancy_percentage": grid.occupancy_percentage,
                "bounds": {
                    "min_x": grid.min_x,
                    "max_x": grid.max_x,
                    "min_y": grid.min_y,
                    "max_y": grid.max_y,
                },
            })
        return summary_list
