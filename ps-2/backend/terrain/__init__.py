"""
Terrain analysis module initialization.
Provides terrain complexity, roughness, and slope feature estimation for 2.5D elevation maps.
"""

from backend.terrain.config import TerrainConfig
from backend.terrain.result import TerrainAnalysisResult
from backend.terrain.analyzer import TerrainAnalyzer

__all__ = [
    "TerrainConfig",
    "TerrainAnalysisResult",
    "TerrainAnalyzer",
]
