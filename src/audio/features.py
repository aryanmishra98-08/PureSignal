"""Audio pre-processing applied before the encoder.

Amplitude normalization only — mel extraction is handled inside the encoder.
"""

from __future__ import annotations

import numpy as np

import config


def normalize(segment: np.ndarray) -> np.ndarray:
    """Peak-normalize a speech segment to [-1, 1].

    Removes microphone level variance, which otherwise destabilises the encoder.

    Args:
        segment: Audio samples at the pipeline sample rate.

    Returns:
        A float32 array at the original sample rate. Near-silent input is
        returned unscaled: the encoder yields a near-zero embedding for it,
        which the caller discards.
    """
    peak = float(np.max(np.abs(segment)))
    if peak < config.NORM_FLOOR:
        return segment.astype(np.float32)
    return (segment / peak).astype(np.float32)
