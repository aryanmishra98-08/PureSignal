# =============================================================================
# audio/vad.py — Frame-level Voice Activity Detection
#                Energy + Zero-Crossing Rate + adaptive noise floor + hangover
# =============================================================================

import numpy as np
import config

# Internal state — module-level, reset via reset()
_noise_floor          = config.NOISE_FLOOR_INIT
_hangover_frames_left = 0
_speech_active        = False
_segment_buffer       = []   # accumulates speech frames into a segment

# How many frames of hangover silence to tolerate before closing a segment
_HANGOVER_FRAMES = int(config.HANGOVER_MS / config.FRAME_MS)

def _rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(frame ** 2)))

def _zcr(frame: np.ndarray) -> float:
    signs      = np.sign(frame)
    crossings  = np.sum(np.abs(np.diff(signs))) / 2
    return float(crossings / len(frame))

def _update_noise_floor(rms: float) -> None:
    """Slowly adapt noise floor upward/downward during silence."""
    global _noise_floor
    _noise_floor = 0.95 * _noise_floor + 0.05 * rms

def process_frame(frame: np.ndarray) -> np.ndarray | None:
    """
    Feed one 20ms frame. Returns a complete speech segment (np.ndarray)
    when a segment closes, otherwise returns None.

    A segment closes when:
      - We were in speech
      - Energy drops below threshold
      - Hangover counter expires
    """
    global _noise_floor, _hangover_frames_left, _speech_active

    rms = _rms(frame)
    zcr = _zcr(frame)

    is_speech_frame = (
        rms > _noise_floor * config.ENERGY_MULTIPLIER
        and zcr < config.ZCR_THRESHOLD
    )

    if is_speech_frame:
        _speech_active        = True
        _hangover_frames_left = _HANGOVER_FRAMES
        _segment_buffer.append(frame)
        return None

    # Not a speech frame
    if _speech_active:
        if _hangover_frames_left > 0:
            # Within hangover window — still part of segment
            _hangover_frames_left -= 1
            _segment_buffer.append(frame)
            return None
        else:
            # Hangover expired — close segment
            _speech_active = False
            if len(_segment_buffer) > 0:
                segment = np.concatenate(_segment_buffer)
                _segment_buffer.clear()
                _update_noise_floor(rms)
                return segment
    else:
        # Pure silence — update noise floor
        _update_noise_floor(rms)

    return None

def reset() -> None:
    """Reset all VAD state — call between sessions."""
    global _noise_floor, _hangover_frames_left, _speech_active
    _noise_floor          = config.NOISE_FLOOR_INIT
    _hangover_frames_left = 0
    _speech_active        = False
    _segment_buffer.clear()
