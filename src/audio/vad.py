# =============================================================================
# audio/vad.py — Frame-level Voice Activity Detection
#                Energy + Zero-Crossing Rate + adaptive noise floor + hangover
#
# Config is read at call time, never baked in at import, so that --set
# overrides and eval sweeps actually take effect.
# =============================================================================

import numpy as np

import config
from utils.logger import get_logger

_log = get_logger()

# Internal state — module-level, reset via reset()
_noise_floor = config.NOISE_FLOOR_INIT
_hangover_frames_left = 0
_speech_active = False
_segment_buf: np.ndarray | None = None
_segment_fill = 0


def _hangover_frames() -> int:
    """Frames of tail silence tolerated before a segment closes."""
    return int(config.HANGOVER_MS / config.FRAME_MS)


def _max_segment_samples() -> int:
    """Hard cap on segment length; the segment force-closes here."""
    return int(config.MAX_SEGMENT_S * config.SAMPLE_RATE)


def _min_noise_floor() -> float:
    """
    Lower bound on the adaptive noise floor.

    Without this the EMA converges toward zero on digitally-silent input, and
    the speech threshold (floor × multiplier) converges with it, turning the
    VAD into a hair trigger that fires on inaudible noise.
    """
    return config.NOISE_FLOOR_INIT * config.NOISE_FLOOR_MIN_RATIO


def _ensure_buf() -> np.ndarray:
    """Allocate the segment buffer lazily so MAX_SEGMENT_S stays overridable."""
    global _segment_buf
    needed = _max_segment_samples()
    if _segment_buf is None or len(_segment_buf) != needed:
        _segment_buf = np.empty(needed, dtype=np.float32)
    return _segment_buf


def _rms(frame: np.ndarray) -> float:
    """Return the root-mean-square energy of a single frame."""
    return float(np.sqrt(np.mean(frame**2)))


def _zcr(frame: np.ndarray) -> float:
    """Return the zero-crossing rate of a frame (crossings per sample).

    High ZCR is characteristic of noise or unvoiced fricatives;
    low ZCR combined with high energy signals voiced speech.
    """
    signs = np.sign(frame)
    crossings = np.sum(np.abs(np.diff(signs))) / 2
    return float(crossings / len(frame))


def _update_noise_floor(rms: float) -> None:
    """Slowly adapt noise floor upward/downward during silence, clamped below."""
    global _noise_floor
    updated = (
        config.NOISE_FLOOR_EMA_SLOW * _noise_floor + config.NOISE_FLOOR_EMA_FAST * rms
    )
    _noise_floor = max(updated, _min_noise_floor())


def _append(frame: np.ndarray) -> np.ndarray | None:
    """
    Append a frame to the open segment.

    Returns a completed segment if the length cap was reached, in which case the
    segment is force-closed and a new one is started with `frame`.  Returns None
    in the normal case.
    """
    global _segment_fill
    buf = _ensure_buf()
    cap = len(buf)
    end = _segment_fill + len(frame)
    if end <= cap:
        buf[_segment_fill:end] = frame
        _segment_fill = end
        return None

    # Cap reached — close what we have rather than silently discarding audio.
    _log("vad", "segment_cap_reached", sample_count=int(_segment_fill),
         max_segment_s=config.MAX_SEGMENT_S)
    segment = buf[:_segment_fill].copy()
    n = min(len(frame), cap)
    buf[0:n] = frame[:n]
    _segment_fill = n
    return segment


def process_frame(frame: np.ndarray) -> np.ndarray | None:
    """
    Feed one 20ms frame. Returns a complete speech segment (np.ndarray)
    when a segment closes, otherwise returns None.

    A segment closes when:
      - We were in speech, energy drops below threshold, and hangover expires
      - Or the segment reaches MAX_SEGMENT_S and is force-closed
    """
    global _noise_floor, _hangover_frames_left, _speech_active, _segment_fill

    rms = _rms(frame)
    zcr = _zcr(frame)

    is_speech_frame = (
        rms > _noise_floor * config.ENERGY_MULTIPLIER and zcr < config.ZCR_THRESHOLD
    )

    if is_speech_frame:
        _speech_active = True
        _hangover_frames_left = _hangover_frames()
        return _append(frame)

    # Not a speech frame
    if _speech_active:
        if _hangover_frames_left > 0:
            # Within hangover window — still part of segment
            _hangover_frames_left -= 1
            return _append(frame)
        else:
            # Hangover expired — close segment
            _speech_active = False
            if _segment_fill > 0:
                segment = _ensure_buf()[:_segment_fill].copy()
                _segment_fill = 0
                _update_noise_floor(rms)
                return segment
    else:
        # Pure silence — update noise floor
        _update_noise_floor(rms)

    return None


def flush() -> np.ndarray | None:
    """
    Return any buffered speech segment and reset speech state.
    Call when the audio source ends (e.g. EOF sentinel from file_capture)
    to avoid losing the final speech segment that never saw trailing silence.
    """
    global _speech_active, _segment_fill
    if _speech_active and _segment_fill > 0:
        segment = _ensure_buf()[:_segment_fill].copy()
        _speech_active = False
        _segment_fill = 0
        return segment
    return None


def reset() -> None:
    """Reset all VAD state — call between sessions."""
    global _noise_floor, _hangover_frames_left, _speech_active, _segment_fill, _segment_buf
    _noise_floor = config.NOISE_FLOOR_INIT
    _hangover_frames_left = 0
    _speech_active = False
    _segment_fill = 0
    _segment_buf = None
