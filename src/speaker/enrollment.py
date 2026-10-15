"""Enrollment store: loads the profiles written by enroll.py and matches
incoming embeddings against them.

The store is an object rather than module state, so a failed load simply
yields no store — there is no half-populated global left behind for a gate to
consult. Quality scoring and profile writing are stateless and stay as
module-level functions.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import config
from utils.logger import get_logger

_log = get_logger()

# Entropy floor for the quality gate. Empirically, well-distributed embeddings
# score above 1.0 in the squared-magnitude sense.
_ENTROPY_THRESHOLD = 1.0

# Guard against log(0) when an embedding component is exactly zero.
_LOG_EPSILON = 1e-9


class EnrollmentStore:
    """Holds enrolled voice profiles and matches embeddings against them."""

    def __init__(
        self,
        profiles: dict[str, np.ndarray] | None = None,
        *,
        threshold: float | None = None,
        norm_floor: float | None = None,
    ) -> None:
        """Build a store. Any argument left as None is read from config."""
        self.threshold = (
            config.ENROLLMENT_THRESHOLD if threshold is None else threshold
        )
        self.norm_floor = config.NORM_FLOOR if norm_floor is None else norm_floor
        self._profiles: dict[str, np.ndarray] = dict(profiles or {})

    @classmethod
    def load(
        cls,
        usernames: list[str],
        profiles_dir: Path | None = None,
        **kwargs,
    ) -> EnrollmentStore:
        """Load the named profiles from disk.

        Args:
            usernames: Profile names to load.
            profiles_dir: Directory holding ``<username>.npy``. Defaults to
                ``config.PROFILES_DIR``.
            **kwargs: Passed through to the constructor.

        Returns:
            A store holding every requested profile.

        Raises:
            FileNotFoundError: If any requested profile is missing. No store
                is produced in that case, so a gate cannot consult a partial
                one.
        """
        directory = config.PROFILES_DIR if profiles_dir is None else profiles_dir
        store = cls(**kwargs)

        profiles: dict[str, np.ndarray] = {}
        for username in usernames:
            path = directory / f"{username}.npy"
            if not path.exists():
                raise FileNotFoundError(
                    f"No profile for '{username}' at '{path}'. "
                    "Run enroll.py first to create it."
                )

            embedding = np.load(path).astype(np.float32)
            norm = np.linalg.norm(embedding)
            if norm > store.norm_floor:
                embedding /= norm

            profiles[username] = embedding
            _log.info("enrollment", f"loaded profile '{username}' from '{path}'")

        store._profiles = profiles
        return store

    def match_with_score(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """Find the closest enrolled speaker and report the similarity.

        Args:
            embedding: An L2-normalized embedding.

        Returns:
            ``(username, similarity)`` when the best similarity clears
            ``threshold``, otherwise ``(None, similarity)``. With no profiles
            loaded the similarity is -1.0.
        """
        best_username: str | None = None
        best_sim = -1.0

        for username, enrolled in self._profiles.items():
            sim = float(np.dot(embedding, enrolled))
            if sim > best_sim:
                best_sim = sim
                best_username = username

        if best_sim >= self.threshold:
            return best_username, best_sim
        return None, best_sim

    def match(self, embedding: np.ndarray) -> str | None:
        """Return the matching enrolled username, or None below threshold."""
        username, _ = self.match_with_score(embedding)
        return username

    def names(self) -> list[str]:
        """Return the enrolled usernames, in load order."""
        return list(self._profiles)

    def __len__(self) -> int:
        return len(self._profiles)


def validate_enrollment_quality(embedding: np.ndarray) -> dict:
    """Score a candidate enrollment embedding.

    Args:
        embedding: The embedding to assess.

    Returns:
        A dict with ``norm`` (L2 norm, ~1.0 once normalized), ``entropy``
        (information entropy of the squared magnitudes) and ``quality_pass``.
    """
    norm = float(np.linalg.norm(embedding))
    squared = embedding**2
    entropy = float(-np.sum(squared * np.log(squared + _LOG_EPSILON)))
    return {
        "norm": norm,
        "entropy": entropy,
        "quality_pass": norm > config.NORM_FLOOR and entropy > _ENTROPY_THRESHOLD,
    }


def save_with_metadata(
    embedding: np.ndarray,
    username: str,
    profiles_dir: Path,
    num_samples: int,
    quality: dict,
) -> None:
    """Write ``<username>.npy`` and its companion ``<username>_meta.json``.

    Raises:
        OSError: If either file cannot be written.
    """
    profiles_dir.mkdir(parents=True, exist_ok=True)
    np.save(profiles_dir / f"{username}.npy", embedding)

    meta = {
        "username": username,
        "num_samples": num_samples,
        "timestamp": time.time(),
        "quality": quality,
    }
    meta_path = profiles_dir / f"{username}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
