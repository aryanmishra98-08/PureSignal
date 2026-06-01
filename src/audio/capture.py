# =============================================================================
# audio/capture.py — Microphone capture with dual-output routing
#
# Provides two queues that downstream consumers read from:
#
#   frame_queue  (gatekeeper mode, extractor.enabled = false)
#       Raw 20ms float32 frames pushed one at a time.
#       Consumers: vad.process_frame() in _process_loop_gatekeeper.
#
#   window_queue  (extractor mode, extractor.enabled = true)
#       Overlapping 1s windows emitted by SlidingWindowBuffer as
#       (seq_no, np.ndarray) tuples, ready for parallel SpeakerBeam workers.
#       Consumers: _process_loop_extractor via ResequencingBuffer.
#
# Changes from v1:
#   - Adds window_queue for extractor mode (SlidingWindowBuffer → (seq, window) tuples)
#   - frame_queue retained for gatekeeper mode (extractor.enabled = false)
# =============================================================================
import queue
from collections import deque
import config
import numpy as np
import sounddevice as sd

# Both queues are module-level so file_capture.py can share the same objects.
frame_queue: queue.Queue = queue.Queue(maxsize=500)   # gatekeeper mode — raw 20ms frames
window_queue: queue.Queue = queue.Queue(maxsize=100)  # extractor mode — (seq, window) tuples

# Rolling snapshot of the last WINDOW_SIZE_S seconds (used by get_ring_snapshot)
_ring_buffer = deque(maxlen=int(config.WINDOW_SIZE_S * config.SAMPLE_RATE))

# SlidingWindowBuffer instance — created at import time if extractor is enabled
_win_buf = None
if config.EXTRACTOR_ENABLED:
    from audio.window_buffer import SlidingWindowBuffer
    _win_buf = SlidingWindowBuffer()


def _capture_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
    """
    sounddevice callback — called on the audio thread for every captured block.

    Routing logic:
      - Always extend the ring buffer (for get_ring_snapshot).
      - Extractor mode: push frame into SlidingWindowBuffer; emit window when ready.
      - Gatekeeper mode: push raw frame directly to frame_queue.

    Dropped frames (full queues) are silently discarded to avoid blocking the
    audio thread, which would cause underruns.
    """
    if status:
        print(f"[capture] sounddevice status: {status}")
    frame = indata[:, 0].copy()  # mono: take first channel
    _ring_buffer.extend(frame)
    if _win_buf is not None:
        # Extractor path — buffer accumulates frames until a full window is ready
        result = _win_buf.push(frame)
        if result is not None:
            try:
                window_queue.put_nowait(result)
            except queue.Full:
                pass  # drop window rather than stall the audio thread
    else:
        # Gatekeeper path — pass raw frame straight through
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass


def flush_window_buffer() -> None:
    """
    Flush any buffered audio at end-of-stream and push a None sentinel.

    Call this after the audio source stops to signal downstream consumers
    to finalize their processing loops.  The sentinel is pushed to whichever
    queue is active (window_queue or frame_queue).
    """
    if _win_buf is not None:
        result = _win_buf.flush()
        if result is not None:
            try:
                window_queue.put(result, timeout=2)
            except queue.Full:
                pass
        try:
            window_queue.put(None, timeout=2)  # EOF sentinel
        except queue.Full:
            pass
    else:
        try:
            frame_queue.put(None, timeout=2)  # EOF sentinel
        except queue.Full:
            pass


def start_capture() -> sd.InputStream:
    """
    Open and start the default microphone input stream.

    Returns:
        sd.InputStream — the active stream; pass to stop_capture() to close it.

    Raises:
        RuntimeError if PortAudio cannot open the microphone.
    """
    try:
        stream = sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=config.FRAME_SAMPLES,
            callback=_capture_callback,
        )
        stream.start()
    except sd.PortAudioError as e:
        raise RuntimeError(f"[capture] Failed to open microphone: {e}") from e
    print(
        f"[capture] mic open — {config.SAMPLE_RATE}Hz, {config.FRAME_MS}ms frames, "
        f"extractor={'enabled' if config.EXTRACTOR_ENABLED else 'disabled'}"
    )
    return stream


def stop_capture(stream: sd.InputStream) -> None:
    """Stop and close the microphone stream opened by start_capture()."""
    stream.stop()
    stream.close()
    print("[capture] mic closed")


def get_ring_snapshot() -> np.ndarray:
    """Return a float32 copy of the ring buffer — the last WINDOW_SIZE_S seconds of audio."""
    return np.array(_ring_buffer, dtype=np.float32)
