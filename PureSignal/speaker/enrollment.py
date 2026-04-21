# =============================================================================
# speaker/enrollment.py — Enrollment store loader and matcher
#                         Profiles are written by enroll.py and loaded at runtime
# =============================================================================

from typing import Optional

import config
import numpy as np

# Maps username -> normalized embedding
_enrolled: dict[str, np.ndarray] = {}


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


def is_loaded() -> bool:
    return len(_enrolled) > 0
