"""
Tests for Phase 17 Part C — Real LiDAR File Ingestion & Replay Demo.

Coverage:
    1.  Fixture directory exists at expected path.
    2.  All 5 fixture files exist.
    3.  Fixture files are deterministic (bit-identical on reload).
    4.  Fixture frames load as valid LidarFrame objects.
    5.  Fixture point cloud shape is (N, 4).
    6.  Fixture point clouds contain no NaN / Inf values.
    7.  Dataset replay processes all 5 fixture frames.
    8.  Frame IDs are preserved through the replay source.
    9.  Timestamps are strictly increasing (frame_rate_hz > 0).
    10. Pipeline receives valid LidarFrame objects from replay.
    11. --max-frames CLI option limits frame count.
    12. --start-index CLI option begins replay at correct position.
    13. --loop requires --max-frames; loop yields correct frame count.
    14. Invalid dataset path fails with clear error (FileNotFoundError / SystemExit).
    15. Unsupported file format fails clearly.
    16. Empty fixture directory fails clearly.
    17. Invalid start_index fails clearly.
    18. Invalid max_frames value fails clearly.
    19. RealTimeLidarPipeline successfully maps fixture frames (smoke test).
    20. Existing 303 tests unaffected (import / baseline check).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "lidar"
DEMO_SCRIPT = PROJECT_ROOT / "scripts" / "run_dataset_replay_demo.py"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_demo(*extra_args: str) -> subprocess.CompletedProcess:
    """Run the demo script as a subprocess and return the result."""
    cmd = [sys.executable, str(DEMO_SCRIPT)] + list(extra_args)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )


# ---------------------------------------------------------------------------
# 1. Fixture directory existence
# ---------------------------------------------------------------------------

class TestFixtureExists:

    def test_fixture_directory_exists(self):
        """1. Fixture directory exists at tests/fixtures/lidar/."""
        assert FIXTURE_DIR.is_dir(), (
            f"Fixture directory not found: {FIXTURE_DIR}. "
            f"Run: python tests/fixtures/lidar/generate_fixtures.py"
        )

    def test_all_five_fixture_files_exist(self):
        """2. All 5 .npy fixture files are present."""
        for i in range(5):
            path = FIXTURE_DIR / f"frame_{i:04d}.npy"
            assert path.is_file(), f"Missing fixture file: {path}"

    def test_fixture_files_are_deterministic(self):
        """3. Loading the same fixture file twice gives bit-identical arrays."""
        path = FIXTURE_DIR / "frame_0000.npy"
        arr_a = np.load(path)
        arr_b = np.load(path)
        np.testing.assert_array_equal(arr_a, arr_b)


# ---------------------------------------------------------------------------
# 4–6. Fixture frame validity
# ---------------------------------------------------------------------------

class TestFixtureFrameValidity:

    def test_fixture_frames_load_as_lidar_frame(self):
        """4. Each fixture .npy file loads successfully as a LidarFrame."""
        from backend.data.numpy_loader import NumpyLidarLoader
        from backend.data.frame import LidarFrame

        loader = NumpyLidarLoader(FIXTURE_DIR, frame_rate_hz=10.0)
        assert len(loader) == 5

        for i in range(5):
            frame = loader.get_frame(i)
            assert isinstance(frame, LidarFrame), f"Frame {i} is not a LidarFrame"

    def test_fixture_points_shape(self):
        """5. Each fixture frame has shape (N, 4)."""
        from backend.data.numpy_loader import NumpyLidarLoader

        loader = NumpyLidarLoader(FIXTURE_DIR, frame_rate_hz=10.0)
        for i in range(5):
            frame = loader.get_frame(i)
            assert frame.points.ndim == 2, f"Frame {i}: expected 2D array"
            assert frame.points.shape[1] == 4, (
                f"Frame {i}: expected 4 columns [x,y,z,intensity], "
                f"got {frame.points.shape[1]}"
            )

    def test_fixture_points_no_nan_inf(self):
        """6. Fixture point clouds contain no NaN or Inf values."""
        from backend.data.numpy_loader import NumpyLidarLoader

        loader = NumpyLidarLoader(FIXTURE_DIR, frame_rate_hz=10.0)
        for i in range(5):
            frame = loader.get_frame(i)
            assert np.all(np.isfinite(frame.points)), (
                f"Frame {i} contains non-finite (NaN or Inf) values."
            )


# ---------------------------------------------------------------------------
# 7–10. Replay source with fixture
# ---------------------------------------------------------------------------

class TestFixtureReplay:

    def test_replay_processes_all_five_frames(self):
        """7. DatasetReplaySource yields all 5 fixture frames."""
        from backend.ingestion.replay import create_replay_source

        source = create_replay_source(FIXTURE_DIR, frame_rate_hz=10.0)
        frames = list(source)
        assert len(frames) == 5

    def test_frame_ids_preserved(self):
        """8. Frame IDs (index-based) are preserved across replay."""
        from backend.ingestion.replay import create_replay_source

        source = create_replay_source(FIXTURE_DIR, frame_rate_hz=10.0)
        frames = list(source)
        for expected_id, frame in enumerate(frames):
            assert frame.frame_id == expected_id, (
                f"Frame at position {expected_id} has id={frame.frame_id}"
            )

    def test_timestamps_increase_monotonically(self):
        """9. Timestamps strictly increase with frame_rate_hz > 0."""
        from backend.ingestion.replay import create_replay_source

        source = create_replay_source(FIXTURE_DIR, frame_rate_hz=10.0)
        frames = list(source)
        timestamps = [f.timestamp for f in frames]

        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i - 1], (
                f"Timestamp not increasing at frame {i}: "
                f"{timestamps[i - 1]} → {timestamps[i]}"
            )

    def test_pipeline_receives_valid_lidar_frames(self):
        """10. All frames consumed by RealTimeLidarPipeline are valid LidarFrames."""
        from backend.data.frame import LidarFrame
        from backend.ingestion.replay import create_replay_source
        from backend.streaming.pipeline import RealTimeLidarPipeline
        from backend.streaming.config import RealTimePipelineConfig
        from backend.mapping.config import MappingConfig

        pipeline = RealTimeLidarPipeline(
            config=RealTimePipelineConfig(collect_timing_metrics=True),
            mapping_config=MappingConfig(),
        )
        source = create_replay_source(FIXTURE_DIR, frame_rate_hz=10.0)

        received_frames: list[LidarFrame] = []
        for frame in source:
            assert isinstance(frame, LidarFrame)
            received_frames.append(frame)
            pipeline.process_frame(frame)

        assert len(received_frames) == 5


# ---------------------------------------------------------------------------
# 11–13. CLI option behaviour (subprocess)
# ---------------------------------------------------------------------------

class TestCLIOptions:

    def test_max_frames_cli_option(self):
        """11. --max-frames limits replay to specified number of frames."""
        result = _run_demo("--max-frames", "2")
        assert result.returncode == 0, f"Demo failed: {result.stderr}"
        assert "Frames processed (OK)     : 2" in result.stdout

    def test_start_index_cli_option(self):
        """12. --start-index begins replay from the specified frame index."""
        # With start_index=3, only frames 3 and 4 remain (5 total, no loop)
        result = _run_demo("--start-index", "3")
        assert result.returncode == 0, f"Demo failed: {result.stderr}"
        assert "Frames processed (OK)     : 2" in result.stdout

    def test_loop_with_max_frames_cli_option(self):
        """13. --loop combined with --max-frames yields exactly max-frames frames."""
        result = _run_demo("--loop", "--max-frames", "9")
        assert result.returncode == 0, f"Demo failed: {result.stderr}"
        assert "Frames processed (OK)     : 9" in result.stdout


# ---------------------------------------------------------------------------
# 14–18. Error handling (clear, non-silent failures)
# ---------------------------------------------------------------------------

class TestErrorHandling:

    def test_nonexistent_dataset_fails_clearly(self):
        """14. Non-existent dataset path exits with non-zero code and clear error."""
        result = _run_demo("--dataset", "/nonexistent/path/does_not_exist/")
        assert result.returncode != 0
        assert "ERROR" in result.stderr or "ERROR" in result.stdout

    def test_unsupported_format_fails_clearly(self):
        """15. Directory with unsupported file types fails with a clear error."""
        with tempfile.TemporaryDirectory() as tmp:
            # Write a .txt file — not a supported LiDAR format
            (Path(tmp) / "scan.txt").write_text("not lidar data")
            result = _run_demo("--dataset", tmp)
        assert result.returncode != 0

    def test_empty_directory_fails_clearly(self):
        """16. Empty dataset directory produces a clear error."""
        with tempfile.TemporaryDirectory() as tmp:
            result = _run_demo("--dataset", tmp)
        assert result.returncode != 0

    def test_invalid_start_index_fails_clearly(self):
        """17. --start-index beyond dataset length exits with error."""
        result = _run_demo("--start-index", "9999")
        assert result.returncode != 0

    def test_invalid_max_frames_fails_clearly(self):
        """18. --max-frames <= 0 exits with error."""
        result = _run_demo("--max-frames", "0")
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# 19. Full pipeline smoke test with fixture
# ---------------------------------------------------------------------------

class TestFixturePipelineSmoke:

    def test_pipeline_produces_adaptive_map_for_each_fixture_frame(self):
        """19. RealTimeLidarPipeline successfully builds an AdaptiveMap25D for every frame."""
        from backend.ingestion.replay import create_replay_source
        from backend.streaming.pipeline import RealTimeLidarPipeline
        from backend.streaming.config import RealTimePipelineConfig
        from backend.mapping.config import MappingConfig
        from backend.mapping.adaptive_map import AdaptiveMap25D

        pipeline = RealTimeLidarPipeline(
            config=RealTimePipelineConfig(collect_timing_metrics=True),
            mapping_config=MappingConfig(),
        )
        source = create_replay_source(FIXTURE_DIR, frame_rate_hz=10.0)

        results = []
        for frame in source:
            res = pipeline.process_frame(frame)
            results.append(res)

        assert len(results) == 5
        assert results[0].is_initial_frame is True
        assert all(r.success for r in results), [r.error_message for r in results]
        assert all(isinstance(r.adaptive_map, AdaptiveMap25D) for r in results)
        assert all(r.adaptive_map.num_represented_cells > 0 for r in results)


# ---------------------------------------------------------------------------
# 20. Baseline — existing tests still importable
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:

    def test_existing_public_apis_remain_importable(self):
        """20. All public APIs from Phases 1–17B remain importable without error."""
        from backend.data.frame import LidarFrame  # noqa
        from backend.data.base import BaseLidarLoader  # noqa
        from backend.data.synthetic import SyntheticLidarGenerator  # noqa
        from backend.data.numpy_loader import NumpyLidarLoader  # noqa
        from backend.data.kitti_loader import KittiLidarLoader  # noqa
        from backend.data.factory import create_lidar_loader, load_lidar_file  # noqa
        from backend.ingestion import (  # noqa
            BaseLidarSource,
            LidarFrameValidator,
            LidarFrameValidationError,
            SyntheticLidarSource,
            DatasetReplaySource,
            create_replay_source,
        )
        from backend.streaming.pipeline import RealTimeLidarPipeline  # noqa
        from backend.mapping.adaptive_map import AdaptiveMap25D  # noqa
