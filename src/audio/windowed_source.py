# =============================================================================
# audio/windowed_source.py — Single owner of frame → queue routing
#
# Both the microphone (audio/capture.py) and the WAV replay thread
# (audio/file_capture.py) push frames through this class, so the windowing and
# enqueue logic exists exactly once. Keep it that way: two copies means every
# windowing change has to be made twice and the two sources drift apart.
#
# Routing:
#   extractor enabled  → SlidingWindowBuffer → (seq, window, new_samples) tuples
#                        on window_queue
#   extractor disabled → raw 20ms frames on frame_queue
#
# Sequence numbers are allocated ONLY on a successful enqueue.  If the queue is
# full the window is dropped and no sequence number is burned, so the consumer's
# ResequencingBuffer can never be left waiting on a number that will never
# arrive.  Drops are always logged as structured events, never swallowed.
# =============================================================================
from __future__ import annotations

import queue

import numpy as np

import config
from audio.window_buffer import SlidingWindowBuffer
from utils.logger import get_logger

_log = get_logger()

_SENTINEL_TIMEOUT_S = 5


class WindowedSource:
    """Routes captured frames into the queue the active pipeline mode reads."""

    def __init__(self, window_queue: queue.Queue, frame_queue: queue.Queue) -> None:
        self._window_queue = window_queue
        self._frame_queue = frame_queue
        self._extractor = config.EXTRACTOR_ENABLED
        self._win_buf = SlidingWindowBuffer() if self._extractor else None
        self._seq = 0
        self._dropped = 0

    @property
    def dropped(self) -> int:
        """Number of windows/frames dropped because the queue was full."""
        return self._dropped

    def push_frame(self, frame: np.ndarray) -> None:
        """
        Feed one 20ms frame from the capture thread.

        Never blocks: a full queue drops the newest item rather than stalling
        the audio thread, which would cause input underruns.
        """
        if self._win_buf is None:
            self._enqueue_frame(frame)
            return
        result = self._win_buf.push(frame)
        if result is not None:
            self._enqueue_window(*result)

    def close(self) -> None:
        """
        Flush any buffered audio and push the EOF sentinel.

        Call once after the audio source stops.  The sentinel goes to whichever
        queue is active, and is pushed with a timeout rather than put_nowait so
        that a transiently full queue does not leave the consumer hanging.
        """
        if self._win_buf is not None:
            result = self._win_buf.flush()
            if result is not None:
                self._enqueue_window(*result, blocking=True)
            self._put_sentinel(self._window_queue)
        else:
            self._put_sentinel(self._frame_queue)

    def reset(self) -> None:
        """Reset windowing state and the sequence counter — call between sessions."""
        if self._win_buf is not None:
            self._win_buf.reset()
        self._seq = 0
        self._dropped = 0

    # -- internals ----------------------------------------------------------

    def _enqueue_window(self, window: np.ndarray, new_samples: int,
                        blocking: bool = False) -> None:
        """Enqueue a window, allocating its sequence number only if it lands."""
        item = (self._seq, window, new_samples)
        try:
            if blocking:
                self._window_queue.put(item, timeout=2)
            else:
                self._window_queue.put_nowait(item)
        except queue.Full:
            # Sequence number deliberately NOT consumed — a gap here would stall
            # the resequencer permanently.
            self._dropped += 1
            _log("capture", "window_dropped", reason="window_queue_full",
                 new_samples=int(new_samples), total_dropped=self._dropped)
            return
        self._seq += 1

    def _enqueue_frame(self, frame: np.ndarray) -> None:
        try:
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            # put_nowait discards the NEWEST frame; the already-queued backlog
            # is kept. Consumers therefore see a gap at the drop instant.
            self._dropped += 1
            _log("capture", "frame_dropped", reason="frame_queue_full",
                 total_dropped=self._dropped)

    @staticmethod
    def _put_sentinel(q: queue.Queue) -> None:
        try:
            q.put(None, timeout=_SENTINEL_TIMEOUT_S)
        except queue.Full:
            _log("capture", "sentinel_drop", reason="queue_full_at_eof")
