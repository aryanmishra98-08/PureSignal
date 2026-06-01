"""
audio/file_capture.py — WAV-file audio source, drop-in replacement for the mic.

Reads a WAV file from disk and feeds it into the same queues that
audio/capture.py normally fills from the microphone, so the rest of the
pipeline (VAD, encoder, etc.) is unaware of the source difference.

Routing (mirrors capture.py):
  Extractor enabled  → pushes (seq_no, window) tuples to capture.window_queue
  Extractor disabled → pushes raw 20ms float32 frames to capture.frame_queue

Audio is replayed in real-time (time.sleep per frame) to mimic live capture.
A None sentinel is pushed after the last frame to signal end-of-stream.
"""
from __future__ import annotations
import queue, threading, time
from pathlib import Path
import config
import numpy as np


def _load_wav(path: Path) -> np.ndarray:
    """
    Load a WAV file and return a 16kHz mono float32 array.

    Handles any integer or float dtype and resamples to config.SAMPLE_RATE
    if needed.  Stereo files are downmixed to mono by averaging channels.

    Args:
        path: Path to the .wav file.

    Returns:
        np.ndarray [N] — float32 samples in the range [-1, 1] at 16kHz.
    """
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from math import gcd
    rate, data = wavfile.read(path)
    # Normalize to float32 [-1, 1]
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
    # Downmix stereo → mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    # Resample if native rate differs from pipeline rate
    target = config.SAMPLE_RATE
    if rate != target:
        g = gcd(rate, target)
        audio = resample_poly(audio, target // g, rate // g).astype(np.float32)
    return audio


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

    def __init__(self, wav_path: Path, frame_queue: queue.Queue) -> None:
        """
        Args:
            wav_path:    Path to the source WAV file.
            frame_queue: capture.frame_queue — used in gatekeeper mode only.
                         Extractor mode writes to capture.window_queue directly.
        """
        self._path = Path(wav_path)
        self._frame_queue = frame_queue
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> threading.Thread:
        """
        Load the WAV file and begin streaming on a daemon thread.

        Returns:
            The started threading.Thread — join it after stop() to wait for EOF.
        """
        audio = _load_wav(self._path)
        self._thread = threading.Thread(target=self._run, args=(audio,), daemon=True, name="file-capture")
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

        Slices `audio` into FRAME_SAMPLES chunks and pushes each to the
        appropriate queue, sleeping FRAME_MS ms between chunks to simulate
        real-time capture.  On natural EOF or stop(), flushes any buffered
        window data and pushes a None sentinel.
        """
        from audio import capture as _capture
        frame_size = config.FRAME_SAMPLES
        frame_dur = config.FRAME_MS / 1000.0  # seconds per frame
        offset = 0
        win_buf = None
        if config.EXTRACTOR_ENABLED:
            from audio.window_buffer import SlidingWindowBuffer
            win_buf = SlidingWindowBuffer()

        while not self._stop_event.is_set():
            chunk = audio[offset: offset + frame_size]
            if len(chunk) == 0:
                break  # natural EOF
            if len(chunk) < frame_size:
                # Zero-pad the final partial frame
                chunk = np.pad(chunk, (0, frame_size - len(chunk)))
            if win_buf is not None:
                # Extractor path — accumulate until a full window is ready
                result = win_buf.push(chunk)
                if result is not None:
                    try:
                        _capture.window_queue.put_nowait(result)
                    except queue.Full:
                        pass
            else:
                # Gatekeeper path — push raw frame directly
                try:
                    self._frame_queue.put_nowait(chunk)
                except queue.Full:
                    pass
            offset += frame_size
            time.sleep(frame_dur)  # real-time pacing

        # EOF — flush remaining buffered audio and push None sentinel
        if win_buf is not None:
            result = win_buf.flush()
            if result is not None:
                try:
                    _capture.window_queue.put(result, timeout=2)
                except queue.Full:
                    pass
            try:
                _capture.window_queue.put(None, timeout=5)  # EOF sentinel
            except queue.Full:
                pass
        else:
            try:
                self._frame_queue.put(None, timeout=5)  # EOF sentinel
            except queue.Full:
                pass
        print("[file_capture] stream ended")
