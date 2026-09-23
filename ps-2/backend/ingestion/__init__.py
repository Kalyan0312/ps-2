"""
LiDAR Ingestion Subsystem.

Provides source abstractions and boundary validation for LiDAR frame ingestion across
synthetic generators, dataset replays, and sensor streams.
"""

from backend.ingestion.base import BaseLidarSource
from backend.ingestion.validator import LidarFrameValidator, LidarFrameValidationError
from backend.ingestion.synthetic import SyntheticLidarSource
from backend.ingestion.replay import DatasetReplaySource, create_replay_source

__all__ = [
    "BaseLidarSource",
    "LidarFrameValidator",
    "LidarFrameValidationError",
    "SyntheticLidarSource",
    "DatasetReplaySource",
    "create_replay_source",
]
