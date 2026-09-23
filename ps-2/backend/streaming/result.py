"""
Result structures for the Real-Time LiDAR Streaming Pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from backend.mapping.adaptive_map import AdaptiveMap25D
from backend.temporal.result import TemporalChangeResult
from backend.temporal.incremental_builder import IncrementalBuildResult


@dataclass
class StreamingPipelineResult:
    """
    Result container for a single processed frame in the streaming pipeline.
    
    Attributes:
        frame_id: Sequential or hardware identifier for the frame.
        timestamp: Time the frame was recorded or processed.
        is_initial_frame: True if this is the first frame processed.
        success: True if processing completed without unhandled errors.
        adaptive_map: The current AdaptiveMap25D produced.
        previous_map: The AdaptiveMap25D from the previous frame, if available.
        temporal_change_result: Results from temporal comparison, if any.
        incremental_build_result: Results from incremental mapping, if used.
        selected_strategy: The rebuild strategy used (FULL_BUILD, FULL_REUSE, INCREMENTAL_UPDATE, FULL_REBUILD).
        metrics: Processing timing and performance metadata.
        error_message: Any error message if success is False.
    """
    frame_id: Any
    timestamp: float
    is_initial_frame: bool
    success: bool
    adaptive_map: Optional[AdaptiveMap25D] = None
    previous_map: Optional[AdaptiveMap25D] = None
    temporal_change_result: Optional[TemporalChangeResult] = None
    incremental_build_result: Optional[IncrementalBuildResult] = None
    selected_strategy: str = "UNKNOWN"
    metrics: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None

    def get_summary(self) -> Dict[str, Any]:
        """Returns a structured diagnostic summary of the result."""
        summary = {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "success": self.success,
            "is_initial_frame": self.is_initial_frame,
            "strategy": self.selected_strategy,
        }
        
        if not self.success:
            summary["error"] = self.error_message
            return summary
            
        summary["metrics"] = self.metrics
        
        if self.incremental_build_result:
            summary["incremental_stats"] = self.incremental_build_result.get_summary()
            
        return summary
