# =============================================================================
# audio/features.py — Audio pre-processing before encoder ingestion
#                     Normalization only — mel extraction handled by encoder
# =============================================================================

import numpy as np

def normalize(segment: np.ndarray) -> np.ndarray:
    """
    Amplitude-normalize a speech segment to [-1, 1].
    Prevents encoder instability from mic level variance.
    Returns float32 ndarray at original sample rate.
    """
    peak = np.max(np.abs(segment))
    if peak < 1e-6:
        # Near-silent segment — return as-is, encoder will produce
        # a near-zero embedding, tracker will discard it
        return segment.astype(np.float32)
    return (segment / peak).astype(np.float32)
