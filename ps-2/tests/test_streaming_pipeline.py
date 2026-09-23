"""
Test Suite for Phase 14 Part A: Real-Time LiDAR Streaming Pipeline Engine.

Covers pipeline initialization, sequential frame processing, incremental map updates,
buffer mechanics, metric calculations, and error resilience.
"""

import pytest
import numpy as np
import time

from backend.data.frame import LidarFrame
from backend.data.synthetic import SyntheticLidarGenerator
from backend.mapping.config import MappingConfig
from backend.streaming.config import RealTimePipelineConfig, OverflowPolicy
from backend.streaming.frame_buffer import FrameBuffer
from backend.streaming.pipeline import RealTimeLidarPipeline


@pytest.fixture
def mapping_cfg():
    return MappingConfig(resolution=0.5, min_x=-10.0, max_x=10.0, min_y=-10.0, max_y=10.0, auto_bounds=False)


@pytest.fixture
def pipeline(mapping_cfg):
    config = RealTimePipelineConfig()
    return RealTimeLidarPipeline(config=config, mapping_config=mapping_cfg)


@pytest.fixture
def gen():
    return SyntheticLidarGenerator(seed=42)


# ---------------------------------------------------------------------------
# 1. Pipeline Initialization & First Frame
# ---------------------------------------------------------------------------
class TestPipelineInitialization:

    def test_pipeline_initialization(self, pipeline):
        assert pipeline.total_frames_received == 0
        assert pipeline.total_frames_processed == 0
        assert not pipeline.temporal_manager.has_history

    def test_first_frame_processing(self, pipeline, gen):
        f0 = gen.generate_frame(frame_id=0, num_ground_points=2000)
        res = pipeline.process_frame(f0)
        
        assert res.success
        assert res.is_initial_frame
        assert res.adaptive_map is not None
        assert res.previous_map is None
        assert res.selected_strategy in ("FULL_REBUILD", "UNKNOWN", "FULL_BUILD")  # Incremental builder returns FULL_REBUILD for initial

    def test_sequential_frame_ids_managed(self, pipeline):
        # Even without explicit frame IDs, pipeline manages them sequentially
        pts = np.array([[0,0,0,100]], dtype=np.float64)
        f0 = LidarFrame(points=pts, frame_id=None, timestamp=0.0)
        f1 = LidarFrame(points=pts, frame_id=None, timestamp=0.1)
        
        r0 = pipeline.process_frame(f0)
        r1 = pipeline.process_frame(f1)
        
        assert r0.frame_id == 0
        assert r1.frame_id == 1


# ---------------------------------------------------------------------------
# 2. Sequential Processing & Strategies
# ---------------------------------------------------------------------------
class TestSequentialProcessing:

    def test_processing_multiple_consecutive_frames(self, pipeline, gen):
        for i in range(5):
            f = gen.generate_frame(frame_id=i, num_ground_points=2000)
            res = pipeline.process_frame(f)
            assert res.success
            assert res.frame_id == i
            if i > 0:
                assert not res.is_initial_frame
                assert res.previous_map is not None

    def test_identical_consecutive_frames_full_reuse(self, pipeline, gen):
        f0 = gen.generate_frame(frame_id=0, num_ground_points=2000)
        r0 = pipeline.process_frame(f0)
        
        # Process same frame points again
        f1 = LidarFrame(points=f0.points.copy(), frame_id=1, timestamp=0.1)
        r1 = pipeline.process_frame(f1)
        
        assert r1.success
        assert r1.selected_strategy == "FULL_REUSE"
        assert pipeline.strategy_counts["FULL_REUSE"] >= 1

    def test_changed_frame_temporal_integration(self, pipeline, gen):
        f0 = gen.generate_frame(frame_id=0, num_ground_points=2000)
        pipeline.process_frame(f0)
        
        # Modify some points to trigger INCREMENTAL_UPDATE
        pts1 = f0.points.copy()
        mask = np.hypot(pts1[:, 0], pts1[:, 1]) < 2.0
        pts1[mask, 2] += 0.85
        f1 = LidarFrame(points=pts1, frame_id=1, timestamp=0.1)
        
        r1 = pipeline.process_frame(f1)
        assert r1.success
        assert r1.selected_strategy == "INCREMENTAL_UPDATE"
        assert np.any(r1.temporal_change_result.changed_mask)

    def test_tier_transition_frames(self, pipeline, gen):
        f0 = gen.generate_frame(frame_id=0, num_ground_points=2000)
        pipeline.process_frame(f0)
        
        # Add dense high-roughness obstacle to force tier escalation
        pts1 = f0.points.copy()
        cluster = np.array([
            [x, y, 2.0 + 0.4 * np.sin(x*5) * np.cos(y*5), 120.0]
            for x in np.linspace(3.0, 6.0, 10)
            for y in np.linspace(3.0, 6.0, 10)
        ], dtype=np.float64)
        pts1 = np.vstack([pts1, cluster])
        f1 = LidarFrame(points=pts1, frame_id=1, timestamp=0.1)
        
        r1 = pipeline.process_frame(f1)
        assert r1.success
        # A tier change forces rebuild, usually INCREMENTAL_UPDATE for localized
        assert r1.selected_strategy in ("INCREMENTAL_UPDATE", "FULL_REBUILD")


# ---------------------------------------------------------------------------
# 3. State Preservation & Error Resilience
# ---------------------------------------------------------------------------
class TestStateAndErrors:

    def test_pipeline_preserves_previous_map_correctness(self, pipeline, gen):
        f0 = gen.generate_frame(frame_id=0, num_ground_points=2000)
        r0 = pipeline.process_frame(f0)
        
        pts1 = f0.points.copy()
        pts1[:50, 2] += 1.50
        f1 = LidarFrame(points=pts1, frame_id=1, timestamp=0.1)
        r1 = pipeline.process_frame(f1)
        
        # Verify previous map inside TemporalMapManager was not corrupted
        assert pipeline.temporal_manager.last_map is r1.adaptive_map
        assert r1.previous_map is r0.adaptive_map

    def test_failed_frame_resilience(self, pipeline, gen):
        f0 = gen.generate_frame(frame_id=0, num_ground_points=2000)
        pipeline.process_frame(f0)
        
        # Malformed frame passed to pipeline
        class CorruptedFrame:
            frame_id = 1
            timestamp = 0.1
            points = None
            
        r_bad = pipeline.process_frame(CorruptedFrame())
        
        assert not r_bad.success
        assert r_bad.error_message is not None
        assert pipeline.total_frames_received == 2
        assert pipeline.total_frames_processed == 1 # only 1 valid
        
        # Pipeline recovers on next valid frame
        f2 = gen.generate_frame(frame_id=2, num_ground_points=2000)
        r2 = pipeline.process_frame(f2)
        assert r2.success
        assert pipeline.total_frames_processed == 2


# ---------------------------------------------------------------------------
# 4. Frame Buffer Behavior
# ---------------------------------------------------------------------------
class TestFrameBuffer:

    def test_buffer_fifo_behavior(self):
        buf = FrameBuffer(max_size=5)
        f0 = LidarFrame(points=np.zeros((1,4)), frame_id=0, timestamp=0.0)
        f1 = LidarFrame(points=np.zeros((1,4)), frame_id=1, timestamp=0.1)
        
        buf.push(f0)
        buf.push(f1)
        
        assert buf.pop().frame_id == 0
        assert buf.pop().frame_id == 1

    def test_buffer_drop_oldest(self):
        buf = FrameBuffer(max_size=2, overflow_policy=OverflowPolicy.DROP_OLDEST)
        buf.push(LidarFrame(points=np.zeros((1,4)), frame_id=0, timestamp=0.0))
        buf.push(LidarFrame(points=np.zeros((1,4)), frame_id=1, timestamp=0.1))
        buf.push(LidarFrame(points=np.zeros((1,4)), frame_id=2, timestamp=0.2)) # pushes out 0
        
        assert buf.dropped_frames == 1
        assert buf.pop().frame_id == 1
        assert buf.pop().frame_id == 2

    def test_buffer_drop_newest(self):
        buf = FrameBuffer(max_size=2, overflow_policy=OverflowPolicy.DROP_NEWEST)
        buf.push(LidarFrame(points=np.zeros((1,4)), frame_id=0, timestamp=0.0))
        buf.push(LidarFrame(points=np.zeros((1,4)), frame_id=1, timestamp=0.1))
        pushed = buf.push(LidarFrame(points=np.zeros((1,4)), frame_id=2, timestamp=0.2)) # rejected
        
        assert not pushed
        assert buf.dropped_frames == 1
        assert buf.pop().frame_id == 0
        assert buf.pop().frame_id == 1


# ---------------------------------------------------------------------------
# 5. Metrics & Reset
# ---------------------------------------------------------------------------
class TestMetricsAndReset:

    def test_streaming_metrics_calculated_correctly(self, pipeline, gen):
        for i in range(3):
            pipeline.process_frame(gen.generate_frame(frame_id=i, num_ground_points=500))
            
        m = pipeline.get_performance_metrics()
        assert m["total_frames_received"] == 3
        assert m["total_frames_processed"] == 3
        assert m["average_latency_ms"] > 0
        assert m["average_fps"] > 0
        assert sum(m["strategy_counts"].values()) == 3

    def test_timing_metrics_non_negative(self, pipeline, gen):
        f0 = gen.generate_frame(frame_id=0, num_ground_points=500)
        r0 = pipeline.process_frame(f0)
        
        assert r0.metrics["preprocessing_time_ms"] >= 0
        assert r0.metrics["map_update_time_ms"] >= 0
        assert r0.metrics["total_latency_ms"] >= 0

    def test_pipeline_reset(self, pipeline, gen):
        pipeline.process_frame(gen.generate_frame(frame_id=0, num_ground_points=500))
        assert pipeline.total_frames_processed == 1
        
        pipeline.reset()
        assert pipeline.total_frames_processed == 0
        assert pipeline.total_frames_received == 0
        assert not pipeline.temporal_manager.has_history
        assert sum(pipeline.strategy_counts.values()) == 0


# ---------------------------------------------------------------------------
# 6. Compatibility & Equivalence
# ---------------------------------------------------------------------------
class TestCompatibility:

    def test_streaming_output_world_query_valid(self, pipeline, gen):
        f0 = gen.generate_frame(frame_id=0, num_ground_points=2000)
        r0 = pipeline.process_frame(f0)
        
        # Test basic world query on resulting map
        q = r0.adaptive_map.world_query(0.0, 0.0)
        # Assuming the generated points hit the center
        assert q is not None
        assert isinstance(q.is_represented, bool)

    def test_inclusive_boundary_points(self, pipeline):
        corners = np.array([
            [-10.0, -10.0, 1.0, 100.0],
            [ 10.0, -10.0, 2.0, 100.0],
            [-10.0,  10.0, 3.0, 100.0],
            [ 10.0,  10.0, 4.0, 100.0],
        ], dtype=np.float64)
        f0 = LidarFrame(points=corners, frame_id=0, timestamp=0.0)
        
        r0 = pipeline.process_frame(f0)
        assert r0.success
        assert r0.adaptive_map.num_represented_cells > 0

    def test_existing_phase13_incremental_equivalence_intact(self, pipeline, mapping_cfg, gen):
        f0 = gen.generate_frame(frame_id=0, num_ground_points=1500)
        f1 = LidarFrame(points=f0.points.copy(), frame_id=1, timestamp=0.1)
        
        r0 = pipeline.process_frame(f0)
        r1 = pipeline.process_frame(f1)
        
        # Test exactly equal arrays
        m0 = r0.adaptive_map
        m1 = r1.adaptive_map
        for t in m0.available_tiers:
            assert np.array_equal(m0.tier_maps[t].point_count, m1.tier_maps[t].point_count)
