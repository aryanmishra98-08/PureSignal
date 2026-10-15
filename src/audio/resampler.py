"""Conversion from pipeline audio to the PCM format Ultravox expects.

16kHz float32 -> 48kHz int16, applied immediately before each WebSocket send.
"""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly

import config

# int16 full scale, used to convert normalized floats to PCM.
_INT16_PEAK = 32767

# Resampling ratio derived from the configured rates (16k -> 48k gives 3/1).
_GCD = gcd(config.SAMPLE_RATE, config.ULTRAVOX_IN_RATE)
_UP = config.ULTRAVOX_IN_RATE // _GCD
_DOWN = config.SAMPLE_RATE // _GCD


def to_48k_pcm(segment: np.ndarray) -> bytes:
    """Resample a pipeline-rate float32 segment to Ultravox input PCM.

    Args:
        segment: float32 samples at ``config.SAMPLE_RATE``.

    Returns:
        Little-endian int16 PCM bytes at ``config.ULTRAVOX_IN_RATE``.
    """
    resampled = resample_poly(segment, _UP, _DOWN)
    # Clip before the int16 cast; resampling overshoot would otherwise wrap.
    resampled = np.clip(resampled, -1.0, 1.0)
    return (resampled * _INT16_PEAK).astype(np.int16).tobytes()


def silence_frame_48k() -> bytes:
    """Return one chunk of int16 silence at the Ultravox input rate.

    Used to keep the outbound stream continuous between speech segments.
    """
    return bytes(pcm_frame_bytes())


def pcm_frame_bytes() -> int:
    """Return the byte length of one Ultravox chunk of int16 PCM."""
    samples = int(config.ULTRAVOX_IN_RATE * config.ULTRAVOX_CHUNK_MS / 1000)
    return samples * 2  # int16 = 2 bytes per sample
