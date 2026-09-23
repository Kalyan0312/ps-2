"""
Real-Time LiDAR Streaming Pipeline Engine.

Orchestrates continuous stream processing of LiDAR frames into adaptive 2.5D maps,
integrating preprocessing, uniform base mapping, terrain analysis, resolution
selection, temporal change detection, and incremental adaptive map construction.
"""

from __future__ import annotations

import time
from typing import Optional, Dict, Any, Union
from pathlib import Path

from backend.data.frame import LidarFrame
from backend.preprocessing.pipeline import PreprocessingPipeline
from backend.preprocessing.config import PreprocessingConfig
from backend.mapping.uniform_builder import Uniform25DMapBuilder
from backend.mapping.config import MappingConfig
from backend.terrain.analyzer import TerrainAnalyzer
from backend.terrain.config import TerrainConfig
from backend.priority.selector import AdaptiveResolutionSelector
from backend.priority.config import PriorityConfig
from backend.temporal.temporal_manager import TemporalMapManager
from backend.temporal.config import TemporalChangeConfig
from backend.temporal.incremental_builder import IncrementalAdaptiveMapBuilder

from backend.streaming.config import RealTimePipelineConfig
from backend.streaming.result import StreamingPipelineResult


class RealTimeLidarPipeline:
    """
    Central orchestration engine for processing continuous sequences of LidarFrames.
    
    The pipeline maintains state across frames, ensuring the previous adaptive map
    and temporal history are preserved for incremental updates.
    """

    def __init__(
        self,
        config: Optional[RealTimePipelineConfig] = None,
        mapping_config: Optional[MappingConfig] = None,
        preprocessing_config: Optional[PreprocessingConfig] = None,
        terrain_config: Optional[TerrainConfig] = None,
        priority_config: Optional[PriorityConfig] = None,
        temporal_config: Optional[TemporalChangeConfig] = None,
    ):
        self.config = config or RealTimePipelineConfig()
        self.mapping_config = mapping_config or MappingConfig()
        
        # Initialize component stages
        self.preprocessor = PreprocessingPipeline(preprocessing_config or PreprocessingConfig())
        self.uniform_builder = Uniform25DMapBuilder(self.mapping_config)
        self.terrain_analyzer = TerrainAnalyzer(terrain_config or TerrainConfig())
        self.resolution_selector = AdaptiveResolutionSelector(priority_config or PriorityConfig())
        
        # Initialize temporal and incremental components
        temp_cfg = temporal_config or TemporalChangeConfig()
        self.temporal_manager = TemporalMapManager(temp_cfg)
        self.incremental_builder = IncrementalAdaptiveMapBuilder(self.mapping_config, full_rebuild_threshold=0.50)
        
        # State
        self.frame_counter = 0
        
        # Metrics
        self.total_frames_received = 0
        self.total_frames_processed = 0
        self.total_processing_time_ms = 0.0
        
        self.strategy_counts = {
            "FULL_REUSE": 0,
            "INCREMENTAL_UPDATE": 0,
            "FULL_REBUILD": 0,
        }

    def reset(self) -> None:
        """Resets all pipeline state and metrics."""
        self.temporal_manager.reset()
        self.frame_counter = 0
        self.total_frames_received = 0
        self.total_frames_processed = 0
        self.total_processing_time_ms = 0.0
        for k in self.strategy_counts:
            self.strategy_counts[k] = 0

    def process_frame(self, frame: LidarFrame) -> StreamingPipelineResult:
        """
        Processes a single LidarFrame synchronously through the entire pipeline.
        
        Args:
            frame: The incoming LidarFrame.
            
        Returns:
            StreamingPipelineResult containing the new AdaptiveMap25D and metrics.
        """
        self.total_frames_received += 1
        t_start_total = time.perf_counter()
        
        # Assign sequential frame ID and timestamp if missing
        frame_id = frame.frame_id if frame.frame_id is not None else self.frame_counter
        timestamp = frame.timestamp if hasattr(frame, "timestamp") else time.time()
        
        metrics: Dict[str, float] = {}
        
        try:
            # 1. Preprocessing
            t0 = time.perf_counter()
            clean_frame = self.preprocessor.process(frame)
            metrics["preprocessing_time_ms"] = (time.perf_counter() - t0) * 1000.0
            
            # 2. Base Mapping
            t0 = time.perf_counter()
            base_grid = self.uniform_builder.build_map(clean_frame)
            metrics["base_mapping_time_ms"] = (time.perf_counter() - t0) * 1000.0
            
            # 3. Terrain Analysis
            t0 = time.perf_counter()
            terrain = self.terrain_analyzer.analyze(base_grid)
            metrics["terrain_analysis_time_ms"] = (time.perf_counter() - t0) * 1000.0
            
            # 4. Resolution Selection
            t0 = time.perf_counter()
            decision = self.resolution_selector.select_resolution(base_grid, terrain)
            metrics["resolution_selection_time_ms"] = (time.perf_counter() - t0) * 1000.0
            
            # 5. Temporal Integration & Map Update
            t0 = time.perf_counter()
            is_initial_frame = not self.temporal_manager.has_history
            previous_map = self.temporal_manager.last_map
            

            # Correct order for incremental update
            inc_result = self.incremental_builder.build(
                previous_map=previous_map,
                current_frame=clean_frame,
                current_decision=decision
            )
            
            # Now update the temporal manager with the new valid map
            temporal_result = self.temporal_manager.update(inc_result.adaptive_map, frame_id)
            
            metrics["map_update_time_ms"] = (time.perf_counter() - t0) * 1000.0
            
            strategy = inc_result.metadata.get("strategy", "UNKNOWN")
            if strategy in self.strategy_counts:
                self.strategy_counts[strategy] += 1
                
            total_latency_ms = (time.perf_counter() - t_start_total) * 1000.0
            metrics["total_latency_ms"] = total_latency_ms
            
            self.total_frames_processed += 1
            self.total_processing_time_ms += total_latency_ms
            self.frame_counter += 1
            
            return StreamingPipelineResult(
                frame_id=frame_id,
                timestamp=timestamp,
                is_initial_frame=is_initial_frame,
                success=True,
                adaptive_map=inc_result.adaptive_map,
                previous_map=previous_map,
                temporal_change_result=temporal_result,
                incremental_build_result=inc_result,
                selected_strategy=strategy,
                metrics=metrics,
            )
            
        except Exception as e:
            # Safe error handling: do not advance frame_counter or mutate TemporalMapManager state on failure.
            total_latency_ms = (time.perf_counter() - t_start_total) * 1000.0
            metrics["total_latency_ms"] = total_latency_ms
            
            return StreamingPipelineResult(
                frame_id=frame_id,
                timestamp=timestamp,
                is_initial_frame=not self.temporal_manager.has_history,
                success=False,
                error_message=str(e),
                metrics=metrics,
            )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Returns stream-level performance metrics."""
        avg_latency = (self.total_processing_time_ms / self.total_frames_processed) if self.total_frames_processed > 0 else 0.0
        avg_fps = (1000.0 / avg_latency) if avg_latency > 0 else 0.0
        
        return {
            "total_frames_received": self.total_frames_received,
            "total_frames_processed": self.total_frames_processed,
            "average_latency_ms": avg_latency,
            "average_fps": avg_fps,
            "strategy_counts": dict(self.strategy_counts),
        }
