# =============================================================================
# audio/resampler.py — 16kHz float32 → 48kHz int16 PCM bytes
#                      Used immediately before WebSocket send
# =============================================================================

import config
import numpy as np
from scipy.signal import resample_poly

# Resample ratio: 48000 / 16000 = 3/1
_UP = 3
_DOWN = 1


def to_48k_pcm(segment: np.ndarray) -> bytes:
    """
    Resample a 16kHz float32 segment to 48kHz int16 PCM bytes.

    Args:
        segment: float32 ndarray at 16kHz

    Returns:
        bytes — int16 little-endian PCM at 48kHz
    """
    resampled = resample_poly(segment, _UP, _DOWN).astype(np.float32)

    # Clip to [-1, 1] before int16 conversion to avoid wraparound distortion
    resampled = np.clip(resampled, -1.0, 1.0)

    # Scale to int16 range
    pcm_int16 = (resampled * 32767).astype(np.int16)
    return pcm_int16.tobytes()


def silence_frame_48k() -> bytes:
    """
    Return one 20ms silence frame at 48kHz int16.
    Used to pad between speech segments on the WebSocket stream.
    960 samples = 20ms @ 48kHz.
    """
    samples = int(config.ULTRAVOX_IN_RATE * config.ULTRAVOX_CHUNK_MS / 1000)
    return bytes(samples * 2)  # int16 = 2 bytes per sample
