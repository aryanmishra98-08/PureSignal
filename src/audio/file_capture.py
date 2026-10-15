"""WAV-file audio source that mirrors the microphone source's interface.

Pushes 20ms float32 frames into a frame queue at real-time cadence so the rest
of the pipeline (VAD -> encoder -> tracker -> policy) runs unmodified.

Any WAV is accepted: channel count and sample rate are converted on load.

Usage (from main.py --source <path>):
    source = AudioFileCapture(Path("recording.wav"), frame_queue)
    source.start()
    ...
    source.stop()
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import numpy as np

import config
from audio.wav_io import load_wav
from utils.logger import get_logger

_log = get_logger()

# How long stop() waits for the streaming thread to finish, in seconds.
_JOIN_TIMEOUT_S = 2.0

# How long the end-of-stream sentinel waits for a slot in the queue, in seconds.
_SENTINEL_TIMEOUT_S = 5.0


class AudioFileCapture:
    """Streams a WAV file into ``frame_queue`` at real-time 20ms cadence.

    When the file is exhausted, pushes ``None`` as a sentinel so the consumer
    knows the source has ended.
    """

    def __init__(self, wav_path: Path | str, frame_queue: queue.Queue) -> None:
        self._path = Path(wav_path)
        self._queue = frame_queue
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Load the WAV and begin streaming frames on a background thread."""
        audio = load_wav(self._path)
        self._thread = threading.Thread(
            target=self._run,
            args=(audio,),
            daemon=True,
            name="file-capture",
        )
        self._thread.start()
        _log.info(
            "file_capture",
            f"streaming '{self._path.name}' — "
            f"{len(audio) / config.SAMPLE_RATE:.1f}s @ {config.SAMPLE_RATE}Hz",
        )

    def stop(self) -> None:
        """Signal the streaming thread to stop and wait briefly for it to exit."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=_JOIN_TIMEOUT_S)
            self._thread = None

    def _run(self, audio: np.ndarray) -> None:
        frame_size = config.FRAME_SAMPLES
        frame_duration_s = config.FRAME_MS / 1000.0

        for offset in range(0, len(audio), frame_size):
            if self._stop_event.is_set():
                break

            chunk = audio[offset : offset + frame_size]
            if len(chunk) < frame_size:  # pad the final short frame
                chunk = np.pad(chunk, (0, frame_size - len(chunk)))

            try:
                self._queue.put_nowait(chunk)
            except queue.Full:
                pass  # consumer is behind; drop the frame

            time.sleep(frame_duration_s)  # real-time cadence

        # Sentinel signals end-of-stream. Use a blocking put so it is never
        # dropped, but cap the wait so a stalled consumer cannot hang shutdown.
        try:
            self._queue.put(None, timeout=_SENTINEL_TIMEOUT_S)
        except queue.Full:
            pass
        _log.info("file_capture", "stream ended")
