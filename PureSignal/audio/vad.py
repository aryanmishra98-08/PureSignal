# =============================================================================
# audio/vad.py — Frame-level Voice Activity Detection
#                Energy + Zero-Crossing Rate + adaptive noise floor + hangover
# =============================================================================

import config
import numpy as np

# Internal state — module-level, reset via reset()
_noise_floor = config.NOISE_FLOOR_INIT
_hangover_frames_left = 0
_speech_active = False
_MAX_SEGMENT_SAMPLES = int(30 * config.SAMPLE_RATE)  # 30s hard cap on segment length
_segment_buf = np.empty(_MAX_SEGMENT_SAMPLES, dtype=np.float32)
_segment_fill = 0

# How many frames of hangover silence to tolerate before closing a segment
_HANGOVER_FRAMES = int(config.HANGOVER_MS / config.FRAME_MS)


def _rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(frame**2)))


def _zcr(frame: np.ndarray) -> float:
    signs = np.sign(frame)
    crossings = np.sum(np.abs(np.diff(signs))) / 2
    return float(crossings / len(frame))


def _update_noise_floor(rms: float) -> None:
    """Slowly adapt noise floor upward/downward during silence."""
    global _noise_floor
    _noise_floor = (
        config.NOISE_FLOOR_EMA_SLOW * _noise_floor + config.NOISE_FLOOR_EMA_FAST * rms
    )


def process_frame(frame: np.ndarray) -> np.ndarray | None:
    """
    Feed one 20ms frame. Returns a complete speech segment (np.ndarray)
    when a segment closes, otherwise returns None.

    A segment closes when:
      - We were in speech
      - Energy drops below threshold
      - Hangover counter expires
    """
    global _noise_floor, _hangover_frames_left, _speech_active, _segment_fill

    rms = _rms(frame)
    zcr = _zcr(frame)

    is_speech_frame = (
        rms > _noise_floor * config.ENERGY_MULTIPLIER and zcr < config.ZCR_THRESHOLD
    )

    if is_speech_frame:
        _speech_active = True
        _hangover_frames_left = _HANGOVER_FRAMES
        end = _segment_fill + len(frame)
        if end <= _MAX_SEGMENT_SAMPLES:
            _segment_buf[_segment_fill:end] = frame
            _segment_fill = end
        return None

    # Not a speech frame
    if _speech_active:
        if _hangover_frames_left > 0:
            # Within hangover window — still part of segment
            _hangover_frames_left -= 1
            end = _segment_fill + len(frame)
            if end <= _MAX_SEGMENT_SAMPLES:
                _segment_buf[_segment_fill:end] = frame
                _segment_fill = end
            return None
        else:
            # Hangover expired — close segment
            _speech_active = False
            if _segment_fill > 0:
                segment = _segment_buf[:_segment_fill].copy()
                _segment_fill = 0
                _update_noise_floor(rms)
                return segment
    else:
        # Pure silence — update noise floor
        _update_noise_floor(rms)

    return None


def reset() -> None:
    """Reset all VAD state — call between sessions."""
    global _noise_floor, _hangover_frames_left, _speech_active, _segment_fill
    _noise_floor = config.NOISE_FLOOR_INIT
    _hangover_frames_left = 0
    _speech_active = False
    _segment_fill = 0
