"""
audio/wav_io.py — WAV loading shared by the pipeline and the eval scripts.

One implementation, so a fix to dtype handling or resampling reaches the live
file source and every offline measurement at the same time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import config


def load_wav(path: Path | str) -> np.ndarray:
    """
    Load a WAV file and return a mono float32 array at config.SAMPLE_RATE.

    Handles any integer or float dtype, downmixes stereo to mono by averaging
    channels, and resamples when the file's native rate differs from the
    pipeline rate.

    Args:
        path: Path to the .wav file.

    Returns:
        np.ndarray [N] — float32 samples in the range [-1, 1].
    """
    from math import gcd

    from scipy.io import wavfile
    from scipy.signal import resample_poly

    rate, data = wavfile.read(path)

    # Normalize to float32 [-1, 1]
    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.float32:
        audio = data
    else:
        audio = data.astype(np.float32)

    # Downmix stereo → mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Resample if the native rate differs from the pipeline rate
    target = config.SAMPLE_RATE
    if rate != target:
        g = gcd(rate, target)
        audio = resample_poly(audio, target // g, rate // g).astype(np.float32)

    return audio.astype(np.float32)
