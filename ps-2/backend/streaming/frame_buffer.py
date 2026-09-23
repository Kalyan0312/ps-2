"""
Lightweight frame buffering mechanism for the streaming pipeline.
"""

from collections import deque
from typing import Optional, Tuple, Dict, Any
from backend.data.frame import LidarFrame
from backend.streaming.config import OverflowPolicy


class FrameBuffer:
    """
    FIFO buffer for managing incoming LidarFrames before processing.
    
    Provides configurable overflow policies to handle bursts of data
    that exceed the pipeline's processing rate.
    """
    def __init__(self, max_size: int = 10, overflow_policy: OverflowPolicy = OverflowPolicy.DROP_OLDEST):
        if max_size <= 0:
            raise ValueError("max_size must be strictly positive.")
            
        self.max_size = max_size
        self.overflow_policy = overflow_policy
        self._buffer: deque[LidarFrame] = deque()
        
        # Metrics tracking
        self.received_frames: int = 0
        self.processed_frames: int = 0
        self.dropped_frames: int = 0

    def push(self, frame: LidarFrame) -> bool:
        """
        Pushes a new frame into the buffer.
        
        Args:
            frame: The LidarFrame to buffer.
            
        Returns:
            True if the frame was added, False if it was dropped (DROP_NEWEST).
        """
        self.received_frames += 1
        
        if len(self._buffer) >= self.max_size:
            if self.overflow_policy == OverflowPolicy.DROP_OLDEST:
                self._buffer.popleft()
                self.dropped_frames += 1
            elif self.overflow_policy == OverflowPolicy.DROP_NEWEST:
                self.dropped_frames += 1
                return False
            # If BLOCK, we'd need thread coordination. For synchronous simulation, 
            # we'll treat BLOCK as raising an error or just dropping newest.
            elif self.overflow_policy == OverflowPolicy.BLOCK:
                raise RuntimeError("Buffer is full and blocking is not implemented for synchronous streams.")
                
        self._buffer.append(frame)
        return True

    def pop(self) -> Optional[LidarFrame]:
        """
        Pops the oldest frame from the buffer for processing.
        
        Returns:
            The next LidarFrame, or None if the buffer is empty.
        """
        if not self._buffer:
            return None
            
        frame = self._buffer.popleft()
        self.processed_frames += 1
        return frame

    def peek(self) -> Optional[LidarFrame]:
        """Returns the oldest frame without removing it, or None if empty."""
        return self._buffer[0] if self._buffer else None

    def clear(self) -> None:
        """Clears the buffer and resets metrics."""
        self._buffer.clear()
        self.received_frames = 0
        self.processed_frames = 0
        self.dropped_frames = 0

    @property
    def current_size(self) -> int:
        """Returns the current number of frames in the buffer."""
        return len(self._buffer)

    @property
    def is_empty(self) -> bool:
        """Returns True if the buffer is empty."""
        return len(self._buffer) == 0

    @property
    def is_full(self) -> bool:
        """Returns True if the buffer is full."""
        return len(self._buffer) >= self.max_size

    def get_metrics(self) -> Dict[str, int]:
        """Returns a summary of buffer metrics."""
        return {
            "current_size": self.current_size,
            "received_frames": self.received_frames,
            "processed_frames": self.processed_frames,
            "dropped_frames": self.dropped_frames,
        }
