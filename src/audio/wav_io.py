"""WAV file loading — the single decoder used by the pipeline and the evaluators.

Normalises any WAV into the pipeline's canonical representation: mono, float32,
in [-1, 1], resampled to the configured sample rate.
"""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np

import config

# Full-scale divisors for the integer PCM formats scipy returns.
_INT_FULL_SCALE: dict[type, float] = {
    np.dtype(np.int16).type: 32768.0,
    np.dtype(np.int32).type: 2147483648.0,
}


def _to_float32(data: np.ndarray) -> np.ndarray:
    """Convert raw WAV samples to float32, scaling integer formats to [-1, 1]."""
    scale = _INT_FULL_SCALE.get(data.dtype.type)
    if scale is not None:
        return data.astype(np.float32) / scale
    # float32 passes through; float64 and any exotic dtype are cast as-is.
    return data.astype(np.float32, copy=False)


def load_wav(path: Path | str, target_rate: int | None = None) -> np.ndarray:
    """Load a WAV file as a mono float32 array at ``target_rate``.

    Args:
        path: Path to the WAV file.
        target_rate: Desired sample rate. Defaults to ``config.SAMPLE_RATE``.

    Returns:
        A 1-D float32 array at ``target_rate``.

    Raises:
        RuntimeError: If scipy is not installed.
    """
    try:
        from scipy.io import wavfile
        from scipy.signal import resample_poly
    except ImportError as e:
        raise RuntimeError(
            "scipy is required to read WAV files: pip install scipy"
        ) from e

    target_rate = config.SAMPLE_RATE if target_rate is None else target_rate
    rate, data = wavfile.read(path)

    audio = _to_float32(data)
    if audio.ndim == 2:  # stereo -> mono
        audio = audio.mean(axis=1)

    if rate != target_rate:
        divisor = gcd(rate, target_rate)
        audio = resample_poly(
            audio, target_rate // divisor, rate // divisor
        ).astype(np.float32)

    return audio
