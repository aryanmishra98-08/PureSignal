# =============================================================================
# speaker/tracker.py — Online speaker tracking
#                      Cosine similarity + EMA centroid updates
#                      In-memory only — gallery resets each session
# =============================================================================

import numpy as np
import config

# Gallery: { "S1": np.ndarray [256], "S2": ... }
_gallery: dict[str, np.ndarray] = {}
_speaker_counter = 0


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))   # both already L2-normalized


def _next_speaker_id() -> str:
    global _speaker_counter
    _speaker_counter += 1
    return f"S{_speaker_counter}"


def assign(embedding: np.ndarray) -> str:
    """
    Given a new embedding, return a speaker ID.

    Flow:
      1. Compare against all gallery centroids
      2. If best match > SIMILARITY_THRESHOLD → assign + update centroid
      3. Else if gallery has room → register as new speaker
      4. Else → assign to closest (gallery full fallback)

    Returns:
        str — speaker ID e.g. "S1", "S2"
    """
    if not _gallery:
        # First speaker
        speaker_id = _next_speaker_id()
        _gallery[speaker_id] = embedding.copy()
        return speaker_id

    # Score against all centroids
    scores = {
        sid: _cosine_sim(embedding, centroid)
        for sid, centroid in _gallery.items()
    }
    best_id    = max(scores, key=scores.__getitem__)
    best_score = scores[best_id]

    if best_score >= config.SIMILARITY_THRESHOLD:
        # Known speaker — update centroid with EMA
        _gallery[best_id] = _ema_update(_gallery[best_id], embedding)
        return best_id

    if len(_gallery) < config.MAX_SPEAKERS:
        # New speaker — register
        speaker_id = _next_speaker_id()
        _gallery[speaker_id] = embedding.copy()
        return speaker_id

    # Gallery full — assign to closest regardless of threshold
    return best_id


def _ema_update(centroid: np.ndarray, new_embedding: np.ndarray) -> np.ndarray:
    """Exponential moving average update, re-normalized."""
    updated = (1 - config.EMA_ALPHA) * centroid + config.EMA_ALPHA * new_embedding
    norm = np.linalg.norm(updated)
    if norm < 1e-6:
        return centroid   # degenerate case — keep old centroid
    return (updated / norm).astype(np.float32)


def get_gallery() -> dict[str, np.ndarray]:
    """Return a snapshot of current gallery — for inspection/debugging."""
    return {sid: vec.copy() for sid, vec in _gallery.items()}


def reset() -> None:
    """Clear gallery — call between sessions if needed."""
    global _speaker_counter
    _gallery.clear()
    _speaker_counter = 0
