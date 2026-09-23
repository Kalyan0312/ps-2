"""
Configuration for Real-Time LiDAR Streaming Pipeline.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


class OverflowPolicy(Enum):
    """Defines how to handle incoming frames when the frame buffer is full."""
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    BLOCK = "block" # Provided for completeness, though stream processing typically favors dropping.


@dataclass
class RealTimePipelineConfig:
    """
    Configuration options for the Real-Time Streaming Pipeline.
    
    Attributes:
        max_buffer_size: Maximum number of frames to hold in the input buffer.
        overflow_policy: Policy for handling buffer overflow (e.g., DROP_OLDEST).
        enable_temporal_processing: Whether to track changes across frames.
        enable_incremental_updates: Whether to use incremental adaptive map updates.
        collect_timing_metrics: Whether to collect detailed processing timing for each frame.
    """
    max_buffer_size: int = 10
    overflow_policy: OverflowPolicy = OverflowPolicy.DROP_OLDEST
    enable_temporal_processing: bool = True
    enable_incremental_updates: bool = True
    collect_timing_metrics: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RealTimePipelineConfig":
        """Builds RealTimePipelineConfig from a configuration dictionary."""
        stream_data = data.get("streaming", data)
        
        policy_str = stream_data.get("overflow_policy", "drop_oldest").lower()
        try:
            policy = OverflowPolicy(policy_str)
        except ValueError:
            policy = OverflowPolicy.DROP_OLDEST

        return cls(
            max_buffer_size=int(stream_data.get("max_buffer_size", 10)),
            overflow_policy=policy,
            enable_temporal_processing=bool(stream_data.get("enable_temporal_processing", True)),
            enable_incremental_updates=bool(stream_data.get("enable_incremental_updates", True)),
            collect_timing_metrics=bool(stream_data.get("collect_timing_metrics", True)),
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "RealTimePipelineConfig":
        """Loads RealTimePipelineConfig directly from a YAML file."""
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}
        return cls.from_dict(raw_data)
