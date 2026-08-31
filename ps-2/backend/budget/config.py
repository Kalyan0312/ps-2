"""
Compute Budgeting and Resource Allocation Configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Union, Optional
import yaml


@dataclass
class BudgetConfig:
    """
    Configuration options for resource-aware resolution budgeting.

    Attributes:
        enabled: If True, compute budgeting and resource constraints are active.
        max_memory_mb: Maximum allowed map array memory in megabytes (MiB), or None for no limit.
        max_allocated_cells: Maximum allowed total allocated grid cells, or None for no limit.
        max_compute_cost: Maximum allowed relative compute units, or None for no limit.
        preserve_high_priority_regions: If True, prioritizes downgrading low-complexity cells first.
        allow_resolution_downgrade: If True, allows optimizer to reduce resolution tiers to meet budget.
        temporal_change_weight: Multiplier weight for recently changed cells (Phase 9 integration).
    """
    enabled: bool = True
    max_memory_mb: Optional[float] = 25.0
    max_allocated_cells: Optional[int] = 500000
    max_compute_cost: Optional[float] = 100000.0
    preserve_high_priority_regions: bool = True
    allow_resolution_downgrade: bool = True
    temporal_change_weight: float = 1.5

    def __post_init__(self):
        if self.max_memory_mb is not None and self.max_memory_mb <= 0:
            raise ValueError(f"max_memory_mb must be positive, got {self.max_memory_mb}")
        if self.max_allocated_cells is not None and self.max_allocated_cells <= 0:
            raise ValueError(f"max_allocated_cells must be positive, got {self.max_allocated_cells}")
        if self.max_compute_cost is not None and self.max_compute_cost <= 0:
            raise ValueError(f"max_compute_cost must be positive, got {self.max_compute_cost}")
        if self.temporal_change_weight < 1.0:
            raise ValueError(f"temporal_change_weight must be >= 1.0, got {self.temporal_change_weight}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BudgetConfig":
        """Constructs BudgetConfig from dictionary."""
        b_data = data.get("budget", data)
        return cls(
            enabled=bool(b_data.get("enabled", True)),
            max_memory_mb=(
                float(b_data["max_memory_mb"])
                if b_data.get("max_memory_mb") is not None
                else None
            ),
            max_allocated_cells=(
                int(b_data["max_allocated_cells"])
                if b_data.get("max_allocated_cells") is not None
                else None
            ),
            max_compute_cost=(
                float(b_data["max_compute_cost"])
                if b_data.get("max_compute_cost") is not None
                else None
            ),
            preserve_high_priority_regions=bool(b_data.get("preserve_high_priority_regions", True)),
            allow_resolution_downgrade=bool(b_data.get("allow_resolution_downgrade", True)),
            temporal_change_weight=float(b_data.get("temporal_change_weight", 1.5)),
        )

    @classmethod
    def from_yaml(cls, path: Union[Path, str]) -> "BudgetConfig":
        """Loads BudgetConfig directly from YAML file."""
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}
        return cls.from_dict(raw_data)
