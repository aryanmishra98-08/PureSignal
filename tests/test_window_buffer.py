# =============================================================================
# tests/test_window_buffer.py — windowing and resequencing
#
# Pure numpy. No models, no hardware, no network.
#
# SlidingWindowBuffer: sample conservation, window shape, hop spacing, overlap
# content, and flush behaviour. The conservation test is the important one —
# forwarding a window's overlap downstream duplicates audio window_s / hop_s
# times over, and nothing else in the suite would notice.
# ResequencingBuffer: ordering, gaps, force_advance, and lock release.
# =============================================================================
import math
import queue
import threading
import time

import numpy as np
import pytest

import config
from audio.window_buffer import ResequencingBuffer, SlidingWindowBuffer
from audio.windowed_source import WindowedSource


def _window_samples() -> int:
    return int(config.EXTRACTOR_WINDOW_S * config.SAMPLE_RATE)


def _steady_new_samples() -> int:
    """New samples per emit once running: hop rounded up to a whole frame."""
    hop = int(config.EXTRACTOR_HOP_S * config.SAMPLE_RATE)
    frame = config.FRAME_SAMPLES
    return int(math.ceil(hop / frame) * frame)


def _push_all(buf: SlidingWindowBuffer, audio: np.ndarray) -> list[tuple[np.ndarray, int]]:
    """Feed `audio` frame by frame, returning every emitted (window, new)."""
    out = []
    for i in range(0, len(audio), config.FRAME_SAMPLES):
        frame = audio[i: i + config.FRAME_SAMPLES]
        if len(frame) < config.FRAME_SAMPLES:
            frame = np.pad(frame, (0, config.FRAME_SAMPLES - len(frame)))
        result = buf.push(frame)
        if result is not None:
            out.append(result)
    return out


# ---------------------------------------------------------------------------
# SlidingWindowBuffer
# ---------------------------------------------------------------------------

def test_no_sample_duplication():
    """
    Case 1 — the important one.

    Concatenating the new-sample tail of every emitted window, plus the flush
    tail, must reproduce the input exactly: same length, same values, same
    order. Forwarding whole overlapping windows fails this at 4x the length.
    """
    buf = SlidingWindowBuffer()
    # Unique value per sample so duplication and reordering are both visible.
    audio = np.arange(config.SAMPLE_RATE * 3, dtype=np.float32)

    recovered: list[float] = []
    for window, new_samples in _push_all(buf, audio):
        recovered.extend(window[-new_samples:].tolist())
    tail = buf.flush()
    if tail is not None:
        window, new_samples = tail
        recovered.extend(window[-new_samples:].tolist())

    assert len(recovered) == len(audio), "N seconds in must be N seconds out"
    np.testing.assert_array_equal(np.array(recovered, dtype=np.float32), audio)


def test_window_shape_is_always_full():
    """Case 2 — every emitted window is exactly window_s long."""
    buf = SlidingWindowBuffer()
    audio = np.random.default_rng(0).standard_normal(
        config.SAMPLE_RATE * 2).astype(np.float32)
    emitted = _push_all(buf, audio)
    assert emitted, "expected at least one window from 2s of audio"
    for window, _ in emitted:
        assert len(window) == _window_samples()


def test_hop_spacing_is_uniform_after_first():
    """Case 3 — every emit after the first advances by one whole hop."""
    buf = SlidingWindowBuffer()
    audio = np.arange(config.SAMPLE_RATE * 3, dtype=np.float32)
    emitted = _push_all(buf, audio)
    assert len(emitted) >= 3
    for _, new_samples in emitted[1:]:
        assert new_samples == _steady_new_samples()


def test_first_window_is_all_new_audio():
    """
    Case 4 — the first window contains window_s of genuinely new audio, not
    hop_s. An Option-A fix that hardcodes new_samples to the hop loses the
    first 0.75s of every session.
    """
    buf = SlidingWindowBuffer()
    audio = np.arange(config.SAMPLE_RATE * 2, dtype=np.float32)
    emitted = _push_all(buf, audio)
    assert emitted[0][1] == _window_samples()


def test_consecutive_windows_share_overlap():
    """Case 5 — window k+1's head equals window k's tail over the overlap."""
    buf = SlidingWindowBuffer()
    audio = np.arange(config.SAMPLE_RATE * 3, dtype=np.float32)
    emitted = _push_all(buf, audio)
    assert len(emitted) >= 2
    for (win_a, _), (win_b, new_b) in zip(emitted, emitted[1:]):
        overlap = len(win_b) - new_b
        assert overlap > 0
        np.testing.assert_array_equal(
            win_b[:overlap], win_a[len(win_a) - overlap:])


def test_flush_pads_to_full_window():
    """Case 6 — a partial buffer flushes as a full-length window."""
    buf = SlidingWindowBuffer()
    # Less than one full window, so nothing is emitted by push().
    audio = np.ones(config.FRAME_SAMPLES * 5, dtype=np.float32)
    assert _push_all(buf, audio) == []
    result = buf.flush()
    assert result is not None
    window, new_samples = result
    assert len(window) == _window_samples()
    assert new_samples == len(audio)
    # Padding goes at the front, so the real audio is the tail — which is what
    # the window[-new_samples:] contract requires.
    np.testing.assert_array_equal(window[-new_samples:], audio)
    assert np.all(window[:len(window) - new_samples] == 0.0)


def test_flush_when_empty_returns_none():
    """Case 7."""
    assert SlidingWindowBuffer().flush() is None


def test_flush_returns_none_when_nothing_new():
    """A flush straight after an emit has no new audio to contribute."""
    buf = SlidingWindowBuffer()
    audio = np.arange(_window_samples(), dtype=np.float32)
    assert len(_push_all(buf, audio)) == 1
    assert buf.flush() is None


def test_reset_clears_state():
    """Case 8 — after reset the buffer behaves like a fresh one."""
    buf = SlidingWindowBuffer()
    _push_all(buf, np.arange(config.SAMPLE_RATE * 2, dtype=np.float32))
    buf.reset()
    assert buf.flush() is None
    emitted = _push_all(buf, np.arange(
        config.SAMPLE_RATE * 2, dtype=np.float32))
    assert emitted[0][1] == _window_samples()


# ---------------------------------------------------------------------------
# WindowedSource — sequence allocation is drop-safe
# ---------------------------------------------------------------------------

def _drain(q: queue.Queue) -> list:
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


def test_windowed_source_sequences_are_contiguous():
    wq, fq = queue.Queue(maxsize=100), queue.Queue(maxsize=100)
    src = WindowedSource(wq, fq)
    if not config.EXTRACTOR_ENABLED:
        pytest.skip("extractor disabled in the active config")
    audio = np.arange(config.SAMPLE_RATE * 2, dtype=np.float32)
    for i in range(0, len(audio), config.FRAME_SAMPLES):
        src.push_frame(audio[i: i + config.FRAME_SAMPLES])
    src.close()
    items = [i for i in _drain(wq) if i is not None]
    assert [seq for seq, _, _ in items] == list(range(len(items)))


def test_dropped_window_does_not_burn_a_sequence_number():
    """
    A full queue must not consume a sequence number.

    If a drop burned a sequence number, the consumer's ResequencingBuffer would
    wait on it forever and the pipeline would stall silently.
    """
    if not config.EXTRACTOR_ENABLED:
        pytest.skip("extractor disabled in the active config")
    wq, fq = queue.Queue(maxsize=2), queue.Queue(maxsize=100)
    src = WindowedSource(wq, fq)
    audio = np.arange(config.SAMPLE_RATE * 4, dtype=np.float32)
    for i in range(0, len(audio), config.FRAME_SAMPLES):
        src.push_frame(audio[i: i + config.FRAME_SAMPLES])

    assert src.dropped > 0, "expected drops from a maxsize=2 queue"
    items = [i for i in _drain(wq) if i is not None]
    assert [seq for seq, _, _ in items] == list(range(len(items)))


# ---------------------------------------------------------------------------
# ResequencingBuffer
# ---------------------------------------------------------------------------

def _arr(v: int) -> np.ndarray:
    return np.full(4, float(v), dtype=np.float32)


def _values(results) -> list[float]:
    return [float(r[0]) for r in results]


def test_reseq_in_order_passthrough():
    """Case 1."""
    buf = ResequencingBuffer()
    for i in range(3):
        buf.put(i, _arr(i))
    assert _values(buf.drain()) == [0.0, 1.0, 2.0]


def test_reseq_out_of_order_is_held():
    """Case 2 — a later result waits for its predecessor."""
    buf = ResequencingBuffer()
    buf.put(2, _arr(2))
    buf.put(0, _arr(0))
    assert _values(buf.drain()) == [0.0]
    buf.put(1, _arr(1))
    assert _values(buf.drain()) == [1.0, 2.0]


def test_reseq_gap_blocks_and_is_counted():
    """Case 3."""
    buf = ResequencingBuffer()
    buf.put(0, _arr(0))
    buf.put(2, _arr(2))
    assert _values(buf.drain()) == [0.0]
    assert buf.pending_count() == 1


def test_reseq_force_advance_recovers():
    """Case 4 — a permanently missing sequence number can be skipped."""
    buf = ResequencingBuffer()
    buf.put(0, _arr(0))
    buf.put(2, _arr(2))
    buf.drain()
    assert buf.force_advance() == 1
    assert _values(buf.drain()) == [2.0]
    assert buf.pending_count() == 0


def test_reseq_force_advance_on_empty_store_is_noop():
    buf = ResequencingBuffer()
    assert buf.force_advance() == 0


def test_reseq_lock_released_after_drain():
    """
    Case 5 — put() from another thread must not block on the consumer.

    A generator implementation would hold the lock across every yield, blocking
    any worker calling put() for the whole duration of the consumer's loop
    body — which cancels the parallelism the buffer exists to enable.
    """
    buf = ResequencingBuffer()
    for i in range(3):
        buf.put(i, _arr(i))

    done = threading.Event()

    def _putter():
        buf.put(99, _arr(99))
        done.set()

    results = buf.drain()
    thread = threading.Thread(target=_putter, daemon=True)
    thread.start()
    for _ in results:
        time.sleep(0.02)  # simulate slow consumer work per item
    assert done.wait(
        timeout=1.0), "put() blocked while the drain result was iterated"


def test_reseq_concurrent_puts_drain_in_order():
    """Case 6 — 10 threads × 10 sequence numbers, all released in order."""
    buf = ResequencingBuffer()
    threads = [
        threading.Thread(target=lambda base=b: [buf.put(base + i, _arr(base + i))
                                                for i in range(10)])
        for b in range(0, 100, 10)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert _values(buf.drain()) == [float(i) for i in range(100)]


def test_reseq_reset():
    """Case 7."""
    buf = ResequencingBuffer()
    buf.put(5, _arr(5))
    buf.reset()
    assert buf.pending_count() == 0
    buf.put(0, _arr(0))
    assert _values(buf.drain()) == [0.0]
