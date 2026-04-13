# =============================================================================
# speaker/enrollment.py — Enrollment store loader and matcher
#                         Profiles are written by enroll.py and loaded at runtime
# =============================================================================

import numpy as np
import config
from typing import Optional

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
        if norm > 1e-6:
            embedding /= norm

        _enrolled[username] = embedding
        print(f"[enrollment] loaded profile '{username}' from '{path}'")


def load_store() -> None:
    """
    Legacy single-user load — kept for backward compatibility.
    Loads all .npy files found in the profiles directory.
    """
    if not config.PROFILES_DIR.exists():
        print(
            f"[enrollment] WARNING: profiles directory not found at '{config.PROFILES_DIR}'.\n"
            f"  ENROLLED mode will drop all audio.\n"
            f"  Run enroll.py first if using ENROLLED mode."
        )
        return

    usernames = [p.stem for p in config.PROFILES_DIR.glob("*.npy")]
    if not usernames:
        print(
            f"[enrollment] WARNING: no profiles found in '{config.PROFILES_DIR}'.\n"
            f"  ENROLLED mode will drop all audio.\n"
            f"  Run enroll.py first if using ENROLLED mode."
        )
        return

    load_profiles(usernames)


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
