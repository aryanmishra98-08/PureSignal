# =============================================================================
# audio/window_buffer.py — Sliding window assembly + resequencing buffer
#
# SlidingWindowBuffer:
#   Accumulates 20ms frames into overlapping 1s windows.
#   Emits (seq_no, window_array) every HOP_S seconds.
#   Overlap between windows preserves context at boundaries.
#
# ResequencingBuffer:
#   Receives (seq_no, result) from parallel workers (possibly out of order).
#   Releases results in strict sequence order.
#   Pattern: hold window 3 until window 2 arrives, then flush both in order.
# =============================================================================
from __future__ import annotations
import threading
from collections import deque
from typing import Iterator
import numpy as np
import config


class SlidingWindowBuffer:
    """Accumulates raw 20ms frames and emits overlapping 1s windows."""

    def __init__(self) -> None:
        self._window_samples = int(config.EXTRACTOR_WINDOW_S * config.SAMPLE_RATE)
        self._hop_samples = int(config.EXTRACTOR_HOP_S * config.SAMPLE_RATE)
        self._buffer: deque = deque()
        self._seq: int = 0
        self._samples_since_last_emit: int = 0

    def push(self, frame: np.ndarray) -> tuple[int, np.ndarray] | None:
        """
        Push one 20ms frame. Returns (seq_no, window) when a full hop
        boundary is crossed, otherwise returns None.
        """
        self._buffer.extend(frame.tolist())
        self._samples_since_last_emit += len(frame)
        if len(self._buffer) < self._window_samples:
            return None
        if self._samples_since_last_emit < self._hop_samples:
            return None
        buf_list = list(self._buffer)
        window = np.array(buf_list[-self._window_samples:], dtype=np.float32)
        seq = self._seq
        self._seq += 1
        self._samples_since_last_emit = 0
        # Retain only the last WINDOW_SAMPLES for overlap continuity
        while len(self._buffer) > self._window_samples:
            self._buffer.popleft()
        return seq, window

    def flush(self) -> tuple[int, np.ndarray] | None:
        """Return remaining buffer as a zero-padded final window at EOF."""
        if not self._buffer:
            return None
        buf_list = list(self._buffer)
        if len(buf_list) < self._window_samples:
            buf_list += [0.0] * (self._window_samples - len(buf_list))
        window = np.array(buf_list[-self._window_samples:], dtype=np.float32)
        seq = self._seq
        self._seq += 1
        self._buffer.clear()
        return seq, window

    def reset(self) -> None:
        """Reset buffer and sequence counter — call between sessions."""
        self._buffer.clear()
        self._seq = 0
        self._samples_since_last_emit = 0


class ResequencingBuffer:
    """
    Holds out-of-order extraction results and releases them in sequence.
    Thread-safe: put() may be called from multiple worker threads.
    drain() should be called from the main thread only.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store: dict[int, np.ndarray] = {}
        self._next_seq: int = 0

    def put(self, seq_no: int, result: np.ndarray) -> None:
        """Store an extraction result keyed by its sequence number.

        Safe to call from any thread; the internal store is lock-protected.
        """
        with self._lock:
            self._store[seq_no] = result

    def drain(self) -> Iterator[np.ndarray]:
        """Yield all contiguous in-order results from next_seq onwards."""
        with self._lock:
            while self._next_seq in self._store:
                yield self._store.pop(self._next_seq)
                self._next_seq += 1

    def pending_count(self) -> int:
        """Return the number of results waiting to be drained (useful for backpressure monitoring)."""
        with self._lock:
            return len(self._store)

    def reset(self) -> None:
        """Discard all pending results and reset the expected sequence counter."""
        with self._lock:
            self._store.clear()
            self._next_seq = 0
