"""
audio/file_capture.py — WAV-file audio source, drop-in replacement for the mic.

Reads a WAV file from disk and feeds it into the same queues that
audio/capture.py normally fills from the microphone, so the rest of the
pipeline (VAD, encoder, etc.) is unaware of the source difference.

Routing is delegated to the shared WindowedSource owned by audio/capture.py, so
mic mode and file mode go through byte-identical windowing and enqueue logic.

Audio is replayed in real-time (time.sleep per frame) to mimic live capture.
A None sentinel is pushed after the last frame to signal end-of-stream.
"""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import numpy as np

import config
from audio.wav_io import load_wav


class AudioFileCapture:
    """
    Daemon thread that streams a WAV file into the pipeline queues.

    Usage:
        source = AudioFileCapture(Path("speech.wav"), capture.frame_queue)
        thread = source.start()
        # ... pipeline runs ...
        source.stop()
        thread.join()
    """

    def __init__(self, wav_path: Path, frame_queue: queue.Queue,
                 realtime: bool = True) -> None:
        """
        Args:
            wav_path:    Path to the source WAV file.
            frame_queue: capture.frame_queue — retained for API compatibility;
                         actual routing goes through capture's WindowedSource.
            realtime:    Sleep between frames to mimic live capture.  Tests set
                         this False to replay a file as fast as possible.
        """
        self._path = Path(wav_path)
        self._frame_queue = frame_queue
        self._realtime = realtime
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> threading.Thread:
        """
        Load the WAV file and begin streaming on a daemon thread.

        Returns:
            The started threading.Thread — join it after stop() to wait for EOF.
        """
        audio = load_wav(self._path)
        self._thread = threading.Thread(target=self._run, args=(audio,),
                                        daemon=True, name="file-capture")
        self._thread.start()
        print(
            f"[file_capture] streaming '{self._path.name}' — "
            f"{len(audio)/config.SAMPLE_RATE:.1f}s, "
            f"extractor={'enabled' if config.EXTRACTOR_ENABLED else 'disabled'}"
        )
        return self._thread

    def stop(self) -> None:
        """Signal the streaming thread to stop after the current frame."""
        self._stop_event.set()

    def _run(self, audio: np.ndarray) -> None:
        """
        Main streaming loop — runs on a daemon thread.

        Slices `audio` into FRAME_SAMPLES chunks and pushes each through the
        shared WindowedSource, sleeping FRAME_MS ms between chunks to simulate
        real-time capture.  On natural EOF or stop(), closes the source, which
        flushes buffered audio and pushes the None sentinel.
        """
        from audio import capture as _capture
        source = _capture.get_source()
        frame_size = config.FRAME_SAMPLES
        frame_dur = config.FRAME_MS / 1000.0  # seconds per frame
        offset = 0

        while not self._stop_event.is_set():
            chunk = audio[offset: offset + frame_size]
            if len(chunk) == 0:
                break  # natural EOF
            if len(chunk) < frame_size:
                # Zero-pad the final partial frame
                chunk = np.pad(chunk, (0, frame_size - len(chunk)))
            source.push_frame(chunk)
            offset += frame_size
            if self._realtime:
                time.sleep(frame_dur)  # real-time pacing

        source.close()  # flush + EOF sentinel
        print("[file_capture] stream ended")
