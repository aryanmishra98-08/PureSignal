# =============================================================================
# audio/capture.py — Mic capture into a circular ring buffer
# =============================================================================

import numpy as np
import sounddevice as sd
import queue
from collections import deque
import config

# Raw frame queue — capture callback pushes here, processing loop reads
frame_queue = queue.Queue()

# Ring buffer — holds last WINDOW_SIZE_S of audio
_ring_buffer = deque(
    maxlen=int(config.WINDOW_SIZE_S * config.SAMPLE_RATE)
)

def _capture_callback(indata: np.ndarray, frames: int,
                      time_info, status) -> None:
    """sounddevice input callback — runs on audio thread, must not block."""
    if status:
        print(f"[capture] sounddevice status: {status}")
    frame = indata[:, 0].copy()   # mono, float32
    _ring_buffer.extend(frame)
    frame_queue.put(frame)

def start_capture() -> sd.InputStream:
    """Open and start the mic input stream. Returns stream handle."""
    stream = sd.InputStream(
        samplerate=config.SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=config.FRAME_SAMPLES,
        callback=_capture_callback,
    )
    stream.start()
    print(f"[capture] mic open — {config.SAMPLE_RATE}Hz, "
          f"{config.FRAME_MS}ms frames ({config.FRAME_SAMPLES} samples)")
    return stream

def stop_capture(stream: sd.InputStream) -> None:
    stream.stop()
    stream.close()
    print("[capture] mic closed")

def get_ring_snapshot() -> np.ndarray:
    """Return a copy of the current ring buffer contents."""
    return np.array(_ring_buffer, dtype=np.float32)
