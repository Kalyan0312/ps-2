"""
Streaming Session & State Manager for FastAPI.

Coordinates the background simulation worker, RealTimeLidarPipeline execution,
WebSocket telemetry broadcasting, and latest map query snapshots.
"""

from __future__ import annotations

import asyncio
import os
import time
import logging
from typing import Set, Optional, Dict, Any, List
from pathlib import Path
import numpy as np
from fastapi import WebSocket

from backend.data.frame import LidarFrame
from backend.data.synthetic import SyntheticLidarGenerator
from backend.ingestion.replay import DatasetReplaySource, create_replay_source
from backend.mapping.config import MappingConfig
from backend.mapping.adaptive_map import AdaptiveMap25D
from backend.streaming.config import RealTimePipelineConfig
from backend.streaming.pipeline import RealTimeLidarPipeline
from backend.streaming.result import StreamingPipelineResult

logger = logging.getLogger("backend.api.session")


class StreamSessionManager:
    """
    Singleton manager for live LiDAR stream simulation, dataset replay playback,
    and WebSocket broadcasting.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        
        # Initialize pipeline with default or file configuration
        if config_path and config_path.is_file():
            self.mapping_config = MappingConfig.from_yaml(config_path)
        else:
            self.mapping_config = MappingConfig(resolution=0.2, min_x=-25.0, max_x=25.0, min_y=-25.0, max_y=25.0)
            
        self.stream_config = RealTimePipelineConfig(max_buffer_size=15, collect_timing_metrics=True)
        self.pipeline = RealTimeLidarPipeline(config=self.stream_config, mapping_config=self.mapping_config)
        
        # State
        self.is_streaming = False
        self.playback_state: str = "STOPPED"  # STOPPED, PLAYING, PAUSED
        self.playback_mode: str = "replay"    # replay, synthetic
        self.playback_speed: float = 0.25     # 0.5, 1.0, 2.0, 4.0
        self.current_frame_idx: int = 0
        self.total_frames: int = 5
        
        self._worker_task: Optional[asyncio.Task] = None
        self._active_clients: Set[WebSocket] = set()
        self._lock = asyncio.Lock()
        
        # Latest snapshot
        self.latest_result: Optional[StreamingPipelineResult] = None
        self.latest_map: Optional[AdaptiveMap25D] = None
        self.latest_frame_points: Optional[np.ndarray] = None  # Raw XYZ points of latest frame
        self.current_fps: float = 0.0
        self.target_fps: float = 10.0

        # Error tracking: last error message from replay/pipeline startup
        self.last_error: Optional[str] = None

        # Synthetic generator fallback
        self.generator = SyntheticLidarGenerator(seed=42)

        # Attempt to load recorded LiDAR fixture dataset by default
        self.replay_source: Optional[DatasetReplaySource] = None
        self._init_replay_source()

    def _init_replay_source(self) -> None:
        """Loads recorded dataset replay source.

        Dataset search order:
        1. LIDAR_DATASET_PATH environment variable (allows swapping in real recordings).
        2. Default fixture directory: tests/fixtures/lidar/
        Falls back to synthetic generation if neither path is available.
        """
        project_root = Path(__file__).resolve().parent.parent.parent

        # Allow external dataset override without changing frontend
        env_path = os.environ.get("LIDAR_DATASET_PATH")
        if env_path:
            dataset_dir = Path(env_path)
            if not dataset_dir.exists():
                msg = f"LIDAR_DATASET_PATH={env_path} does not exist; falling back to fixture."
                logger.warning(msg)
                self.last_error = msg
                dataset_dir = project_root / "tests" / "fixtures" / "lidar"
        else:
            dataset_dir = project_root / "tests" / "fixtures" / "lidar"

        if dataset_dir.is_dir():
            try:
                self.replay_source = create_replay_source(dataset_dir, loop=True, validate_frames=True)
                self.total_frames = len(self.replay_source.loader)
                self.playback_mode = "replay"
                self.last_error = None
                logger.info(
                    "Loaded DatasetReplaySource from %s (%d frames)",
                    dataset_dir, self.total_frames,
                )
            except Exception as e:
                msg = f"Failed to load dataset from {dataset_dir}: {e}"
                logger.warning(msg)
                self.replay_source = None
                self.playback_mode = "synthetic"
                self.last_error = msg
        else:
            self.playback_mode = "synthetic"
            self.last_error = f"Dataset directory not found: {dataset_dir}"

    # ----------------------------------------------------------------------
    # WebSocket Client Management
    # ----------------------------------------------------------------------

    async def register_client(self, websocket: WebSocket):
        await websocket.accept()
        self._active_clients.add(websocket)
        # Send initial status packet immediately
        await websocket.send_json({
            "type": "connection_established",
            "is_streaming": self.is_streaming,
            "status": self.get_status(),
        })

    def unregister_client(self, websocket: WebSocket):
        self._active_clients.discard(websocket)

    async def broadcast(self, payload: Dict[str, Any]):
        """Broadcasts lightweight JSON telemetry to all connected WebSocket clients."""
        if not self._active_clients:
            return
            
        disconnected = []
        for ws in list(self._active_clients):
            try:
                await ws.send_json(payload)
            except Exception:
                disconnected.append(ws)
                
        for ws in disconnected:
            self._active_clients.discard(ws)

    # ----------------------------------------------------------------------
    # Stream Lifecycle Controls & Playback Operations
    # ----------------------------------------------------------------------

    async def start_stream(
        self,
        fps: float = 15.0,
        total_frames: Optional[int] = None,
        speed: float = 1.0,
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Starts background streaming task safely."""
        async with self._lock:
            if self.is_streaming:
                return {
                    "status": "already_running",
                    "message": "Stream is already running.",
                    "streaming": True,
                }

            self.target_fps = max(1.0, min(fps, 60.0))
            self.playback_speed = max(0.1, min(speed, 10.0))
            if mode in ("replay", "synthetic"):
                self.playback_mode = mode
            self.is_streaming = True
            self.playback_state = "PLAYING"
            self._worker_task = asyncio.create_task(self._stream_loop(self.target_fps, total_frames))
            
            logger.info("LiDAR Streaming session started at %.1f FPS (speed=%.1fx)", self.target_fps, self.playback_speed)
            return {
                "status": "started",
                "message": f"Stream started at {self.target_fps} FPS.",
                "streaming": True,
                "playback_state": self.playback_state,
                "current_frame": self.current_frame_idx,
                "total_frames": self.total_frames,
            }

    async def stop_stream(self) -> Dict[str, Any]:
        """Stops/pauses background streaming task safely."""
        async with self._lock:
            if not self.is_streaming:
                return {
                    "status": "already_stopped",
                    "message": "Stream is not active.",
                    "streaming": False,
                }

            self.is_streaming = False
            self.playback_state = "PAUSED"
            if self._worker_task and not self._worker_task.done():
                self._worker_task.cancel()
                try:
                    await asyncio.wait_for(self._worker_task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                self._worker_task = None

            # Broadcast stopped event
            await self.broadcast({
                "type": "stream_stopped",
                "status": self.get_status(),
            })

            logger.info("LiDAR Streaming session paused/stopped.")
            return {
                "status": "stopped",
                "message": "Stream paused/stopped successfully.",
                "streaming": False,
                "playback_state": self.playback_state,
                "current_frame": self.current_frame_idx,
                "total_frames": self.total_frames,
            }

    async def pause_stream(self) -> Dict[str, Any]:
        """Explicitly pauses playback."""
        res = await self.stop_stream()
        self.playback_state = "PAUSED"
        res["playback_state"] = "PAUSED"
        return res

    def reset(self) -> Dict[str, Any]:
        """Resets the pipeline state, stored maps, and replay playhead to frame 0."""
        self.pipeline.reset()
        self.latest_result = None
        self.latest_map = None
        self.latest_frame_points = None
        self.current_fps = 0.0
        self.current_frame_idx = 0
        self.playback_state = "STOPPED"
        # Reset replay source playhead without re-loading from disk
        if self.replay_source is not None:
            self.replay_source._current_index = self.replay_source.start_index
            self.replay_source._frames_yielded = 0
            self.replay_source._active = False
        return self.get_status()

    def _get_frame_at(self, idx: int) -> LidarFrame:
        """Loads a frame for index `idx` using recorded dataset replay or synthetic generator."""
        if self.playback_mode == "replay" and self.replay_source is not None:
            safe_idx = idx % max(1, self.total_frames)
            frame = self.replay_source.loader.get_frame(safe_idx)
            frame.timestamp = time.time()
            frame.frame_id = safe_idx
            return frame
        else:
            pts = self.generator.generate_frame(frame_id=idx, num_ground_points=4500).points
            return LidarFrame(points=pts, frame_id=idx, timestamp=time.time())

    def _process_frame_sync(self, frame: LidarFrame) -> StreamingPipelineResult:
        """Processes frame through pipeline and updates state synchronously."""
        # Store raw points for LiDAR Points layer rendering
        self.latest_frame_points = frame.points[:, :3].tolist() if frame.points is not None else []
        result = self.pipeline.process_frame(frame)
        self.latest_result = result
        if result.success and result.adaptive_map is not None:
            self.latest_map = result.adaptive_map
        return result

    def step_frame(self, direction: int = 1) -> Dict[str, Any]:
        """Steps forward (+1) or backward (-1) in recorded dataset frames.
        
        Backward stepping is safe: clamped at frame 0, does not wrap around.
        Forward stepping wraps using modulo when looping dataset.
        """
        if self.total_frames > 0:
            if direction >= 0:
                # Forward: wrap around to support looping
                self.current_frame_idx = (self.current_frame_idx + direction) % self.total_frames
            else:
                # Backward: clamp at 0, do not wrap
                self.current_frame_idx = max(0, self.current_frame_idx + direction)
        else:
            self.current_frame_idx = max(0, self.current_frame_idx + direction)

        frame = self._get_frame_at(self.current_frame_idx)
        result = self._process_frame_sync(frame)

        # Broadcast frame & map updates asynchronously
        telemetry = self._build_telemetry_payload(result)
        asyncio.create_task(self.broadcast(telemetry))
        asyncio.create_task(self.broadcast({
            "type": "map_update",
            "frame_id": result.frame_id,
            "timestamp": result.timestamp,
        }))
        return self.get_status()

    def seek_frame(self, frame_idx: int) -> Dict[str, Any]:
        """Seeks to specific frame index in recorded dataset."""
        if self.total_frames > 0:
            self.current_frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        else:
            self.current_frame_idx = max(0, frame_idx)

        frame = self._get_frame_at(self.current_frame_idx)
        result = self._process_frame_sync(frame)
        
        telemetry = self._build_telemetry_payload(result)
        asyncio.create_task(self.broadcast(telemetry))
        asyncio.create_task(self.broadcast({
            "type": "map_update",
            "frame_id": result.frame_id,
            "timestamp": result.timestamp,
        }))
        return self.get_status()

    def set_speed(self, speed: float) -> Dict[str, Any]:
        """Updates playback speed multiplier (0.5x, 1x, 2x, 4x)."""
        self.playback_speed = max(0.1, min(speed, 10.0))
        return self.get_status()

    # ----------------------------------------------------------------------
    # Background Processing Loop
    # ----------------------------------------------------------------------

    async def _stream_loop(self, fps: float, total_frames: Optional[int]):
        frame_idx = 0
        
        # Activate dataset replay source if available
        if self.playback_mode == "replay" and self.replay_source is not None:
            self.replay_source.start()
        
        try:
            while self.is_streaming:
                t0 = time.perf_counter()
                
                # Fetch next frame from recorded dataset replay or synthetic fallback
                if self.playback_mode == "replay" and self.replay_source is not None:
                    frame = self.replay_source.get_next_frame()
                    if frame is None:
                        break
                else:
                    pts = self.generator.generate_frame(frame_id=frame_idx, num_ground_points=4500).points
                    frame = LidarFrame(points=pts, frame_id=frame_idx, timestamp=time.time())

                # Update current frame index tracker
                if isinstance(frame.frame_id, int):
                    self.current_frame_idx = frame.frame_id
                elif isinstance(frame.frame_id, str) and frame.frame_id.startswith("frame_"):
                    try:
                        self.current_frame_idx = int(frame.frame_id.replace("frame_", ""))
                    except ValueError:
                        self.current_frame_idx = frame_idx
                else:
                    self.current_frame_idx = frame_idx

                # Process frame synchronously through existing pipeline
                result = await asyncio.to_thread(self.pipeline.process_frame, frame)

                self.latest_result = result
                if result.success and result.adaptive_map is not None:
                    self.latest_map = result.adaptive_map

                elapsed = time.perf_counter() - t0
                self.current_fps = 1.0 / elapsed if elapsed > 0 else 0.0

                # Formulate serializable telemetry payload
                telemetry = self._build_telemetry_payload(result)
                await self.broadcast(telemetry)

                # Broadcast map update notification when new adaptive map is ready
                if result.success and result.adaptive_map is not None:
                    await self.broadcast({
                        "type": "map_update",
                        "frame_id": result.frame_id,
                        "timestamp": result.timestamp,
                    })

                frame_idx += 1
                if total_frames is not None and frame_idx >= total_frames:
                    break

                interval = 1.0 / (fps * max(0.1, self.playback_speed))
                sleep_duration = max(0.001, interval - elapsed)
                await asyncio.sleep(sleep_duration)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Error in streaming loop: %s", str(e), exc_info=True)
            await self.broadcast({
                "type": "error",
                "message": str(e),
            })
        finally:
            self.is_streaming = False
            if self.playback_state == "PLAYING":
                self.playback_state = "PAUSED"

    # ----------------------------------------------------------------------
    # Diagnostics & Telemetry Formatting
    # ----------------------------------------------------------------------

    def _compute_candidate_obstacles_count(self) -> int:
        """Counts candidate obstacle cells from the latest adaptive map."""
        if self.latest_map is None:
            return 0
        cand_count = 0
        for grid in self.latest_map.tier_maps.values():
            frs, fcs = np.where(grid.point_count > 0)
            if len(frs) == 0:
                continue
            mean_zs = grid.mean_z[frs, fcs]
            min_zs = grid.min_z[frs, fcs]
            max_zs = grid.max_z[frs, fcs]
            
            z_diffs = max_zs - min_zs
            is_cand = (z_diffs > 0.25) | (mean_zs > 0.35)
            cand_count += int(np.count_nonzero(is_cand))
        return cand_count

    def _compute_terrain_risk(self, cand_count: int) -> str:
        """Determines terrain risk rating (LOW, MEDIUM, HIGH) from pipeline output."""
        if self.latest_map is None:
            return "LOW"
        if cand_count > 15:
            return "HIGH"
        elif cand_count > 5:
            return "MEDIUM"
        return "LOW"

    def _build_telemetry_payload(self, result: StreamingPipelineResult) -> Dict[str, Any]:
        """Creates a lightweight serializable dictionary from a StreamingPipelineResult."""
        metrics = self.pipeline.get_performance_metrics()
        
        # Temporal statistics
        temp_res = result.temporal_change_result
        if temp_res is not None:
            changed_cells = int(np.count_nonzero(temp_res.changed_mask))
            unchanged_cells = int(np.count_nonzero(temp_res.unchanged_mask))
            newly_observed = int(np.count_nonzero(temp_res.newly_observed_mask))
            no_longer_observed = int(np.count_nonzero(temp_res.no_longer_observed_mask))
        else:
            changed_cells = 0
            unchanged_cells = 0
            newly_observed = 0
            no_longer_observed = 0

        # Adaptive resolution breakdown
        res_stats = {}
        if result.adaptive_map is not None:
            total_rep = result.adaptive_map.num_represented_cells
            for tier, grid in result.adaptive_map.tier_maps.items():
                occ = int(np.count_nonzero(grid.point_count > 0))
                pct = (occ / total_rep * 100.0) if total_rep > 0 else 0.0
                res_stats[tier] = {
                    "resolution": result.adaptive_map.tier_resolutions.get(tier, 0.0),
                    "occupied_cells": occ,
                    "percentage": round(pct, 1),
                }

        cand_count = self._compute_candidate_obstacles_count()

        return {
            "type": "frame_update",
            "frame_id": result.frame_id,
            "timestamp": result.timestamp,
            "success": result.success,
            "is_initial_frame": result.is_initial_frame,
            "strategy": result.selected_strategy,
            "latency_ms": round(result.metrics.get("total_latency_ms", 0.0), 2),
            "current_fps": round(self.current_fps, 1),
            "average_fps": round(metrics.get("average_fps", 0.0), 1),
            "average_latency_ms": round(metrics.get("average_latency_ms", 0.0), 2),
            "frames_received": metrics.get("total_frames_received", 0),
            "frames_processed": metrics.get("total_frames_processed", 0),
            "frames_dropped": 0,
            "strategy_counts": metrics.get("strategy_counts", {}),
            "playback": {
                "state": self.playback_state,
                "mode": self.playback_mode,
                "current_frame": self.current_frame_idx,
                "total_frames": self.total_frames,
                "speed": self.playback_speed,
            },
            "environment": {
                "candidate_obstacles_count": cand_count,
                "terrain_risk": self._compute_terrain_risk(cand_count),
                "represented_cells": result.adaptive_map.num_represented_cells if result.adaptive_map else 0,
            },
            "temporal": {
                "changed_cells": changed_cells,
                "unchanged_cells": unchanged_cells,
                "newly_observed_cells": newly_observed,
                "no_longer_observed_cells": no_longer_observed,
            },
            "resolution_stats": res_stats,
            "stage_timings": {
                "preprocessing_ms": round(result.metrics.get("preprocessing_time_ms", 0.0), 2),
                "base_mapping_ms": round(result.metrics.get("base_mapping_time_ms", 0.0), 2),
                "terrain_analysis_ms": round(result.metrics.get("terrain_analysis_time_ms", 0.0), 2),
                "resolution_selection_ms": round(result.metrics.get("resolution_selection_time_ms", 0.0), 2),
                "map_update_ms": round(result.metrics.get("map_update_time_ms", 0.0), 2),
            }
        }

    def get_status(self) -> Dict[str, Any]:
        """Returns structured status dictionary."""
        metrics = self.pipeline.get_performance_metrics()
        cand_count = self._compute_candidate_obstacles_count()
        return {
            "is_streaming": self.is_streaming,
            "playback_state": self.playback_state,
            "playback_mode": self.playback_mode,
            "current_frame": self.current_frame_idx,
            "total_frames": self.total_frames,
            "playback_speed": self.playback_speed,
            "target_fps": self.target_fps,
            "current_fps": round(self.current_fps, 1),
            "average_fps": round(metrics.get("average_fps", 0.0), 1),
            "average_latency_ms": round(metrics.get("average_latency_ms", 0.0), 2),
            "frames_received": metrics.get("total_frames_received", 0),
            "frames_processed": metrics.get("total_frames_processed", 0),
            "frames_dropped": 0,
            "strategy_counts": metrics.get("strategy_counts", {}),
            "has_active_map": self.latest_map is not None,
            "candidate_obstacles_count": cand_count,
            "terrain_risk": self._compute_terrain_risk(cand_count),
            "last_error": self.last_error,
        }

    def query_map(self, x: float, y: float) -> Dict[str, Any]:
        """Performs constant-time O(1) query on latest AdaptiveMap25D."""
        if self.latest_map is None:
            return {
                "is_represented": False,
                "message": "No active AdaptiveMap25D generated yet.",
                "x": x,
                "y": y,
            }

        q = self.latest_map.world_query(x, y)
        if not q.is_represented:
            return {
                "is_represented": False,
                "x": x,
                "y": y,
                "tier": None,
                "resolution": None,
                "point_count": 0,
                "min_z": None,
                "max_z": None,
                "mean_z": None,
                "mean_intensity": None,
            }

        return {
            "is_represented": True,
            "x": x,
            "y": y,
            "tier": q.tier,
            "resolution": q.resolution,
            "point_count": q.point_count,
            "min_z": float(q.min_z) if not np.isnan(q.min_z) else None,
            "max_z": float(q.max_z) if not np.isnan(q.max_z) else None,
            "mean_z": float(q.mean_z) if not np.isnan(q.mean_z) else None,
            "mean_intensity": float(q.mean_intensity) if not np.isnan(q.mean_intensity) else None,
        }

    def get_map_snapshot(self) -> Dict[str, Any]:
        """Returns live JSON snapshot of the latest AdaptiveMap25D enriched with candidate obstacles & playback info."""
        raw_point_count = len(self.latest_frame_points) if self.latest_frame_points else 0

        if self.latest_map is None:
            return {
                "available": False,
                "frame_id": None,
                "cells": [],
                "tiers": {},
                "represented_cells": 0,
                "raw_point_count": raw_point_count,
                "candidate_obstacles_count": 0,
                "terrain_risk": "LOW",
                "last_error": self.last_error,
                "playback": {
                    "mode": self.playback_mode,
                    "state": self.playback_state,
                    "current_frame": self.current_frame_idx,
                    "total_frames": self.total_frames,
                    "speed": self.playback_speed,
                }
            }

        snapshot = self.latest_map.to_snapshot_dict()
        if snapshot.get("frame_id") is None and self.latest_result is not None:
            snapshot["frame_id"] = self.latest_result.frame_id

        # Enrich cells with candidate obstacle classification & refinement reasons
        cand_count = 0
        for cell in snapshot.get("cells", []):
            z = cell.get("z", 0.0)
            z_diff = cell.get("max_z", 0.0) - cell.get("min_z", 0.0)
            tier = cell.get("tier", "")

            is_cand = False
            reason = None
            if z_diff > 0.25:
                is_cand = True
                reason = "High terrain variation"
            elif z > 0.35:
                is_cand = True
                reason = "Candidate obstacle region"
            elif tier in ("fine", "ultra_fine", "ultrafine"):
                is_cand = True
                reason = "High spatial complexity"

            cell["is_candidate_obstacle"] = is_cand
            cell["refinement_reason"] = reason
            if is_cand:
                cand_count += 1

        snapshot["candidate_obstacles_count"] = cand_count
        snapshot["terrain_risk"] = self._compute_terrain_risk(cand_count)
        snapshot["raw_point_count"] = raw_point_count
        snapshot["last_error"] = self.last_error
        snapshot["playback"] = {
            "mode": self.playback_mode,
            "state": self.playback_state,
            "current_frame": self.current_frame_idx,
            "total_frames": self.total_frames,
            "speed": self.playback_speed,
        }

        return snapshot


# Global session instance

session_manager = StreamSessionManager(config_path=Path(__file__).resolve().parent.parent.parent / "configs" / "config.yaml")

