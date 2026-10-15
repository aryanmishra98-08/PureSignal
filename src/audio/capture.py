"""Microphone audio source.

Mirrors AudioFileCapture's interface so main.py drives either source
through the same start()/stop() calls.
"""

from __future__ import annotations

import queue

import numpy as np
import sounddevice as sd

import config
from utils.logger import get_logger

_log = get_logger()

# Raw frame queue — the capture callback pushes here, the pipeline reads.
# 500 frames is roughly 10s of 20ms audio.
frame_queue: queue.Queue = queue.Queue(maxsize=500)


class MicCapture:
    """Streams microphone input into ``frame_queue`` as 20ms float32 frames."""

    def __init__(self, target_queue: queue.Queue | None = None) -> None:
        self._queue = frame_queue if target_queue is None else target_queue
        self._stream: sd.InputStream | None = None

    def start(self) -> None:
        """Open and start the microphone input stream.

        Raises:
            RuntimeError: If the input device cannot be opened.
        """
        try:
            self._stream = sd.InputStream(
                samplerate=config.SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=config.FRAME_SAMPLES,
                callback=self._on_audio,
            )
            self._stream.start()
        except sd.PortAudioError as e:
            self._stream = None
            raise RuntimeError(
                f"Failed to open microphone: {e}\n"
                "  Check that a microphone is connected and accessible."
            ) from e

        _log.info(
            "capture",
            f"mic open — {config.SAMPLE_RATE}Hz, {config.FRAME_MS}ms frames "
            f"({config.FRAME_SAMPLES} samples)",
        )

    def stop(self) -> None:
        """Stop and close the microphone stream."""
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None
        _log.info("capture", "mic closed")

    def _on_audio(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """sounddevice callback — runs on the audio thread and must not block."""
        if status:
            _log.warning("capture", f"sounddevice status: {status}")
        try:
            self._queue.put_nowait(indata[:, 0].copy())  # mono float32
        except queue.Full:
            pass  # the pipeline is falling behind; drop the frame
