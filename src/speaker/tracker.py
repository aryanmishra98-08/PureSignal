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
    return float(np.dot(a, b))  # both already L2-normalized


def _next_speaker_id() -> str:
    global _speaker_counter
    _speaker_counter += 1
    return f"S{_speaker_counter}"


def assign(embedding: np.ndarray) -> str:
    """
    Given a new embedding, return a speaker ID.

    Thin wrapper over assign_with_score() for callers that do not need the
    similarity. Prefer assign_with_score() when logging — a hardcoded score
    makes the tracker impossible to tune from logs.
    """
    speaker_id, _ = assign_with_score(embedding)
    return speaker_id


def assign_with_score(embedding: np.ndarray) -> tuple[str, float]:
    """
    Given a new embedding, return (speaker_id, best_similarity).

    Flow:
      1. Compare against all gallery centroids
      2. If best match > SIMILARITY_THRESHOLD → assign + update centroid
      3. Else if gallery has room → register as new speaker
      4. Else → assign to closest (gallery full fallback)

    Returns:
        (str, float) — speaker ID e.g. "S1", and the best cosine similarity
        against the gallery as it stood *before* this embedding was added.
        The similarity is -1.0 when the gallery was empty, since there was
        nothing to compare against.
    """
    if not _gallery:
        # First speaker — nothing to score against
        speaker_id = _next_speaker_id()
        _gallery[speaker_id] = embedding.copy()
        return speaker_id, -1.0

    # Score against all centroids
    scores = {
        sid: _cosine_sim(embedding, centroid) for sid, centroid in _gallery.items()
    }
    best_id = max(scores, key=scores.__getitem__)
    best_score = scores[best_id]

    if best_score >= config.SIMILARITY_THRESHOLD:
        # Known speaker — update centroid with EMA
        _gallery[best_id] = _ema_update(_gallery[best_id], embedding)
        return best_id, best_score

    if len(_gallery) < config.MAX_SPEAKERS:
        # New speaker — register
        speaker_id = _next_speaker_id()
        _gallery[speaker_id] = embedding.copy()
        return speaker_id, best_score

    # Gallery full — assign to closest regardless of threshold
    return best_id, best_score


def _ema_update(centroid: np.ndarray, new_embedding: np.ndarray) -> np.ndarray:
    """Exponential moving average update, re-normalized."""
    updated = (1 - config.EMA_ALPHA) * centroid + \
        config.EMA_ALPHA * new_embedding
    norm = np.linalg.norm(updated)
    if norm < config.NORM_FLOOR:
        return centroid  # degenerate case — keep old centroid
    return (updated / norm).astype(np.float32)


def get_gallery() -> dict[str, np.ndarray]:
    """Return a snapshot of current gallery — for inspection/debugging."""
    return {sid: vec.copy() for sid, vec in _gallery.items()}


def reset() -> None:
    """Clear gallery — call between sessions if needed."""
    global _speaker_counter
    _gallery.clear()
    _speaker_counter = 0
