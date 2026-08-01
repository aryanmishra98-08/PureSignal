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
#       Overlapping windows as (seq_no, window, new_samples) tuples, ready for
#       parallel extraction workers.
#       Consumers: _process_loop_extractor via ResequencingBuffer.
#
# Routing itself lives in audio/windowed_source.py so that the mic and the WAV
# replay thread share one implementation.  Nothing here reads config at import
# time — init() is called by the pipeline after any --set overrides are bound.
# =============================================================================
import queue

import numpy as np

import config
from audio.windowed_source import WindowedSource

# sounddevice is imported lazily inside the mic entry points. It requires
# PortAudio at import time, which file-mode and container runs do not have and
# do not need.

# Both queues are module-level so file_capture.py can share the same objects.
frame_queue: queue.Queue = queue.Queue(
    maxsize=500)   # gatekeeper mode — raw 20ms frames
window_queue: queue.Queue = queue.Queue(
    maxsize=100)  # extractor mode — window tuples

_source: WindowedSource | None = None


def get_source() -> WindowedSource:
    """
    Return the routing layer, constructing it on first use.

    Construction is deferred rather than done at import time: which queue is
    used depends on config.EXTRACTOR_ENABLED, which must be read after any
    --set overrides are applied.  Idempotent.
    """
    global _source
    if _source is None:
        _source = WindowedSource(window_queue, frame_queue)
    return _source


def _capture_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
    """
    sounddevice callback — called on the audio thread for every captured block.

    Hands the frame to WindowedSource, which routes it to the queue matching the
    active pipeline mode.  Never blocks: full queues drop rather than stall the
    audio thread, which would cause input underruns.  Drops are logged.
    """
    if status:
        print(f"[capture] sounddevice status: {status}")
    frame = indata[:, 0].copy()  # mono: take first channel
    get_source().push_frame(frame)


def flush_window_buffer() -> None:
    """
    Flush buffered audio at end-of-stream and push the None sentinel.

    Called by the pipeline shutdown path so that mic mode gets the same clean
    end-of-stream handling as file mode, instead of relying on the process
    exiting out from under the consumer loop.
    """
    get_source().close()


def start_capture():
    """
    Open and start the default microphone input stream.

    Returns:
        sd.InputStream — the active stream; pass to stop_capture() to close it.

    Raises:
        RuntimeError if PortAudio is unavailable or cannot open the microphone.
    """
    get_source()
    try:
        import sounddevice as sd
    except OSError as e:
        raise RuntimeError(
            f"[capture] PortAudio is unavailable, so mic capture is not "
            f"possible: {e}\n  Use --source path/to/file.wav instead."
        ) from e
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


def stop_capture(stream) -> None:
    """Stop and close the microphone stream opened by start_capture()."""
    stream.stop()
    stream.close()
    print("[capture] mic closed")


def reset() -> None:
    """Drop the routing layer so the next session rebuilds it from live config."""
    global _source
    if _source is not None:
        _source.reset()
    _source = None
