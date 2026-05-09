"""
audio/file_capture.py — WAV-file audio source that mimics capture.py's interface.

Pushes 20ms float32 frames into frame_queue at real-time cadence so the rest of
the pipeline (VAD → encoder → tracker → policy) runs unmodified.

Supported input: mono, 16kHz, float32 WAV.
If the file is stereo or at a different sample rate, it is converted automatically.

Usage (from main.py --source <path>):
    from audio.file_capture import AudioFileCapture
    source = AudioFileCapture(Path("recording.wav"), frame_queue)
    thread = source.start()
    ...
    source.stop()
    thread.join()
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import config
import numpy as np


def _load_wav_as_float32_mono_16k(path: Path) -> np.ndarray:
    """
    Load a WAV file and return a mono float32 array at 16kHz.
    Converts sample rate and channel count if needed.
    """
    try:
        from scipy.io import wavfile
        from scipy.signal import resample_poly
        from math import gcd
    except ImportError as e:
        raise RuntimeError(
            "scipy is required for WAV file playback: pip install scipy"
        ) from e

    rate, data = wavfile.read(path)

    # Convert to float32
    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.float32:
        audio = data
    elif data.dtype == np.float64:
        audio = data.astype(np.float32)
    else:
        audio = data.astype(np.float32)

    # Stereo → mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Resample to 16kHz if needed
    target_rate = config.SAMPLE_RATE
    if rate != target_rate:
        g = gcd(rate, target_rate)
        audio = resample_poly(audio, target_rate // g, rate // g).astype(np.float32)

    return audio


class AudioFileCapture:
    """
    Reads a WAV file and feeds it into frame_queue at real-time 20ms cadence.
    When the file is exhausted, pushes None as a sentinel so the main loop exits.
    """

    def __init__(self, wav_path: Path, frame_queue: queue.Queue) -> None:
        self._path = Path(wav_path)
        self._queue = frame_queue
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> threading.Thread:
        """Load the WAV and begin streaming frames. Returns the background thread."""
        audio = _load_wav_as_float32_mono_16k(self._path)
        self._thread = threading.Thread(
            target=self._run,
            args=(audio,),
            daemon=True,
            name="file-capture",
        )
        self._thread.start()
        print(
            f"[file_capture] streaming '{self._path.name}' — "
            f"{len(audio) / config.SAMPLE_RATE:.1f}s @ {config.SAMPLE_RATE}Hz"
        )
        return self._thread

    def stop(self) -> None:
        self._stop_event.set()

    def _run(self, audio: np.ndarray) -> None:
        frame_size = config.FRAME_SAMPLES
        frame_duration_s = config.FRAME_MS / 1000.0
        offset = 0

        while not self._stop_event.is_set():
            chunk = audio[offset : offset + frame_size]
            if len(chunk) == 0:
                break

            # Pad the last frame if it's shorter than frame_size
            if len(chunk) < frame_size:
                chunk = np.pad(chunk, (0, frame_size - len(chunk)))

            try:
                self._queue.put_nowait(chunk)
            except queue.Full:
                pass  # drop frame if consumer is behind

            offset += frame_size
            time.sleep(frame_duration_s)  # real-time cadence

        # Sentinel — signals main loop to exit cleanly.
        # Use blocking put (not put_nowait) so the sentinel is never dropped,
        # but cap the wait at 5s to avoid hanging if the consumer has stalled.
        try:
            self._queue.put(None, timeout=5)
        except queue.Full:
            pass  # consumer stalled; main loop will time out via other means
        print("[file_capture] stream ended")
