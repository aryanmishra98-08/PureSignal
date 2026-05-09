# =============================================================================
# speaker/enrollment.py — Enrollment store loader and matcher
#                         Profiles are written by enroll.py and loaded at runtime
# =============================================================================

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import config
import numpy as np

# Maps username -> normalized embedding
_enrolled: dict[str, np.ndarray] = {}

# Entropy threshold for the quality gate (empirically: well-distributed
# embeddings have entropy > 1.0 in the squared-magnitude sense)
_ENTROPY_THRESHOLD = 1.0


def load_profiles(usernames: list[str]) -> None:
    """
    Load one or more profiles from profiles/<username>.npy into memory.
    Call once at startup in main.py after usernames are selected.
    Exits with an error if any requested profile is missing.
    """
    global _enrolled
    _enrolled = {}

    for username in usernames:
        path = config.PROFILES_DIR / f"{username}.npy"
        if not path.exists():
            print(
                f"[enrollment] ERROR: profile not found for '{username}' at '{path}'.\n"
                f"  Run enroll.py first to create it."
            )
            raise FileNotFoundError(f"Profile not found: {path}")

        embedding = np.load(path).astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > config.NORM_FLOOR:
            embedding /= norm

        _enrolled[username] = embedding
        print(f"[enrollment] loaded profile '{username}' from '{path}'")


def match(embedding: np.ndarray) -> Optional[str]:
    """
    Check if embedding matches any enrolled speaker.

    Returns:
        The matched username if above threshold, None otherwise.
    """
    best_username: Optional[str] = None
    best_sim = -1.0

    for username, enrolled_emb in _enrolled.items():
        sim = float(np.dot(embedding, enrolled_emb))
        if sim > best_sim:
            best_sim = sim
            best_username = username

    if best_sim >= config.ENROLLMENT_THRESHOLD:
        return best_username
    return None


def match_with_score(embedding: np.ndarray) -> tuple[Optional[str], float]:
    """Like match(), but also returns the best similarity score."""
    best_username: Optional[str] = None
    best_sim = -1.0

    for username, enrolled_emb in _enrolled.items():
        sim = float(np.dot(embedding, enrolled_emb))
        if sim > best_sim:
            best_sim = sim
            best_username = username

    if best_sim >= config.ENROLLMENT_THRESHOLD:
        return best_username, best_sim
    return None, best_sim


def validate_enrollment_quality(embedding: np.ndarray) -> dict:
    """
    Compute quality metrics for a candidate enrollment embedding.

    Returns a dict with:
        norm         — L2 norm (should be ~1.0 after normalization)
        entropy      — information entropy of squared magnitudes
        quality_pass — True if both checks pass
    """
    norm = float(np.linalg.norm(embedding))
    squared = embedding ** 2
    entropy = float(-np.sum(squared * np.log(squared + 1e-9)))
    quality_pass = norm > config.NORM_FLOOR and entropy > _ENTROPY_THRESHOLD
    return {"norm": norm, "entropy": entropy, "quality_pass": quality_pass}


def save_with_metadata(
    embedding: np.ndarray,
    username: str,
    profiles_dir: Path,
    num_samples: int,
    quality: dict,
) -> None:
    """Save .npy embedding + companion _meta.json."""
    profiles_dir.mkdir(parents=True, exist_ok=True)
    npy_path = profiles_dir / f"{username}.npy"
    meta_path = profiles_dir / f"{username}_meta.json"

    np.save(npy_path, embedding)

    meta = {
        "username": username,
        "num_samples": num_samples,
        "timestamp": time.time(),
        "quality": quality,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def is_loaded() -> bool:
    return len(_enrolled) > 0
