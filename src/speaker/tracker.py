"""Online speaker tracking.

Cosine similarity against a gallery of centroids, updated by exponential
moving average. Each tracker owns its gallery, which lives only as long as
the instance — so a session ends by dropping the tracker, not by resetting a
module global.
"""

from __future__ import annotations

import numpy as np

import config


class SpeakerTracker:
    """Assigns embeddings to speaker IDs, learning new speakers as they appear."""

    def __init__(
        self,
        *,
        similarity_threshold: float | None = None,
        ema_alpha: float | None = None,
        max_speakers: int | None = None,
        norm_floor: float | None = None,
    ) -> None:
        """Build a tracker. Any argument left as None is read from config."""
        self.similarity_threshold = (
            config.SIMILARITY_THRESHOLD
            if similarity_threshold is None
            else similarity_threshold
        )
        self.ema_alpha = config.EMA_ALPHA if ema_alpha is None else ema_alpha
        self.max_speakers = (
            config.MAX_SPEAKERS if max_speakers is None else max_speakers
        )
        self.norm_floor = config.NORM_FLOOR if norm_floor is None else norm_floor

        self._gallery: dict[str, np.ndarray] = {}
        self._counter = 0

    # -- internals -------------------------------------------------------

    def _next_speaker_id(self) -> str:
        self._counter += 1
        return f"S{self._counter}"

    def _register(self, embedding: np.ndarray) -> str:
        """Add a new speaker to the gallery and return its ID."""
        speaker_id = self._next_speaker_id()
        self._gallery[speaker_id] = embedding.copy()
        return speaker_id

    def _ema_update(self, centroid: np.ndarray, embedding: np.ndarray) -> np.ndarray:
        """Blend a new embedding into a centroid and re-normalize."""
        updated = (1 - self.ema_alpha) * centroid + self.ema_alpha * embedding
        norm = np.linalg.norm(updated)
        if norm < self.norm_floor:
            return centroid  # degenerate — keep the existing centroid
        return (updated / norm).astype(np.float32)

    # -- public API ------------------------------------------------------

    def assign_with_score(self, embedding: np.ndarray) -> tuple[str, float]:
        """Assign an embedding to a speaker and report the driving similarity.

        A match above ``similarity_threshold`` updates that centroid;
        otherwise a new speaker is registered while the gallery has room.
        Once the gallery is full the closest speaker wins regardless.

        Args:
            embedding: An L2-normalized embedding.

        Returns:
            ``(speaker_id, best_similarity)``. For the first speaker of a
            session there is nothing to compare against, so it is 0.0.
        """
        if not self._gallery:
            return self._register(embedding), 0.0

        # Both sides are L2-normalized, so the dot product is cosine similarity.
        scores = {sid: float(np.dot(embedding, c)) for sid, c in self._gallery.items()}
        best_id = max(scores, key=scores.__getitem__)
        best_score = scores[best_id]

        if best_score >= self.similarity_threshold:
            self._gallery[best_id] = self._ema_update(
                self._gallery[best_id], embedding
            )
            return best_id, best_score

        if len(self._gallery) < self.max_speakers:
            return self._register(embedding), best_score

        return best_id, best_score  # gallery full — fall back to the closest

    def assign(self, embedding: np.ndarray) -> str:
        """Assign an embedding to a speaker ID, e.g. "S1"."""
        speaker_id, _ = self.assign_with_score(embedding)
        return speaker_id

    def gallery(self) -> dict[str, np.ndarray]:
        """Return a deep copy of the gallery, for inspection and debugging."""
        return {sid: vec.copy() for sid, vec in self._gallery.items()}

    def speaker_ids(self) -> set[str]:
        """Return the IDs currently in the gallery, without copying vectors."""
        return set(self._gallery)

    def reset(self) -> None:
        """Clear the gallery and restart speaker numbering."""
        self._gallery.clear()
        self._counter = 0

    def __len__(self) -> int:
        """Number of speakers currently in the gallery."""
        return len(self._gallery)
