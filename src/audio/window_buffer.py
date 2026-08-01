# =============================================================================
# audio/window_buffer.py — Sliding window assembly + resequencing buffer
#
# SlidingWindowBuffer:
#   Accumulates 20ms frames into overlapping windows of EXTRACTOR_WINDOW_S.
#   Emits (window, new_samples) every EXTRACTOR_HOP_S seconds.
#
#   The overlap is deliberate: the separator wants the full window as model
#   context.  But only the newest `new_samples` of each window are genuinely
#   new audio — the rest was already emitted in the previous window.  Consumers
#   MUST forward only window[-new_samples:] downstream, or every sample is
#   processed window_s / hop_s times over (4x at the shipped defaults).
#
#   Note this class does NOT allocate sequence numbers.  Sequence allocation
#   belongs to whoever enqueues the window, so that a dropped enqueue cannot
#   burn a sequence number that the resequencer will then wait on forever.
#   See audio/windowed_source.py.
#
# ResequencingBuffer:
#   Receives (seq_no, result) from parallel workers (possibly out of order).
#   Releases results in strict sequence order.
#   Pattern: hold window 3 until window 2 arrives, then flush both in order.
# =============================================================================
from __future__ import annotations

import threading
from collections import deque

import numpy as np

import config


class SlidingWindowBuffer:
    """Accumulates raw 20ms frames and emits overlapping windows."""

    def __init__(self) -> None:
        self._window_samples = int(
            config.EXTRACTOR_WINDOW_S * config.SAMPLE_RATE)
        self._hop_samples = int(config.EXTRACTOR_HOP_S * config.SAMPLE_RATE)
        self._buffer: deque = deque()
        self._samples_since_last_emit: int = 0

    @property
    def window_samples(self) -> int:
        return self._window_samples

    def push(self, frame: np.ndarray) -> tuple[np.ndarray, int] | None:
        """
        Push one 20ms frame.

        Returns:
            (window, new_samples) once a full hop boundary is crossed, else None.
            `window` is always `window_samples` long and carries the overlap as
            model context.  `new_samples` counts the trailing samples that have
            not appeared in any previous window: window[-new_samples:] is the
            audio to forward downstream, exactly once.
        """
        self._buffer.extend(frame.tolist())
        self._samples_since_last_emit += len(frame)
        if len(self._buffer) < self._window_samples:
            return None
        if self._samples_since_last_emit < self._hop_samples:
            return None

        # Capture before the reset — this is the count the caller needs.
        new_samples = min(self._samples_since_last_emit, self._window_samples)
        buf_list = list(self._buffer)
        window = np.array(buf_list[-self._window_samples:], dtype=np.float32)
        self._samples_since_last_emit = 0
        # Retain only the last window_samples for overlap continuity
        while len(self._buffer) > self._window_samples:
            self._buffer.popleft()
        return window, new_samples

    def flush(self) -> tuple[np.ndarray, int] | None:
        """
        Return the trailing audio as a final full-length window at EOF.

        Padding goes at the FRONT so that the newest audio stays at the end of
        the array and the window[-new_samples:] contract holds here too.
        Returns None when nothing new has accumulated since the last emit.
        """
        if not self._buffer:
            return None
        new_samples = min(self._samples_since_last_emit, len(self._buffer))
        if new_samples <= 0:
            return None
        buf_list = list(self._buffer)[-self._window_samples:]
        pad = self._window_samples - len(buf_list)
        if pad > 0:
            buf_list = [0.0] * pad + buf_list
        window = np.array(buf_list, dtype=np.float32)
        self._buffer.clear()
        self._samples_since_last_emit = 0
        return window, new_samples

    def reset(self) -> None:
        """Reset buffer state — call between sessions."""
        self._buffer.clear()
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

    def drain(self) -> list[np.ndarray]:
        """
        Return all contiguous in-order results from next_seq onwards.

        Returns a list rather than yielding under the lock.  A generator would
        hold the lock across every yield for the whole duration of the caller's
        loop body, which serializes every worker thread calling put() against
        the consumer's VAD/encoder/network work.
        """
        out: list[np.ndarray] = []
        with self._lock:
            while self._next_seq in self._store:
                out.append(self._store.pop(self._next_seq))
                self._next_seq += 1
        return out

    def force_advance(self) -> int:
        """
        Skip a permanently missing sequence number.

        Called when pending results have piled up behind a gap that will never
        be filled.  Advances to the lowest sequence number actually present so
        drain() can make progress again.

        Returns:
            int — how many sequence numbers were skipped.
        """
        with self._lock:
            if not self._store:
                return 0
            lowest = min(self._store)
            skipped = lowest - self._next_seq
            if skipped <= 0:
                return 0
            self._next_seq = lowest
            return skipped

    def pending_count(self) -> int:
        """Return the number of results waiting to be drained (backpressure signal)."""
        with self._lock:
            return len(self._store)

    def reset(self) -> None:
        """Discard all pending results and reset the expected sequence counter."""
        with self._lock:
            self._store.clear()
            self._next_seq = 0
