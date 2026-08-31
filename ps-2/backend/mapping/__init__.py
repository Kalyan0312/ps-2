"""
Mapping module initialization.
Provides uniform, multi-resolution, and adaptive 2.5D elevation grid maps and builders.
"""

from backend.mapping.grid_map import GridMap25D
from backend.mapping.config import MappingConfig, DEFAULT_MULTIRES_LEVELS
from backend.mapping.uniform_builder import Uniform25DMapBuilder
from backend.mapping.multires_map import MultiResolutionMap
from backend.mapping.multires_builder import MultiResolution25DMapBuilder
from backend.mapping.adaptive_map import AdaptiveMap25D, AdaptiveCellInfo
from backend.mapping.adaptive_builder import AdaptiveMap25DBuilder

__all__ = [
    "GridMap25D",
    "MappingConfig",
    "DEFAULT_MULTIRES_LEVELS",
    "Uniform25DMapBuilder",
    "MultiResolutionMap",
    "MultiResolution25DMapBuilder",
    "AdaptiveMap25D",
    "AdaptiveCellInfo",
    "AdaptiveMap25DBuilder",
]
