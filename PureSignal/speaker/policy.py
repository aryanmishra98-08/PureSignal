# =============================================================================
# speaker/policy.py — Speaker policy gate
#                     Decides whether a speaker's audio passes to Ultravox
#
# ENROLLED mode: passes audio only if embedding matches enrollment store
# DYNAMIC mode:  passes audio only if speaker ID matches DYNAMIC_TARGET
# =============================================================================

import config
import numpy as np

from speaker import enrollment


def should_pass(speaker_id: str, embedding: np.ndarray) -> bool:
    """
    Gate function — returns True if this speaker's audio should be
    sent to Ultravox, False if it should be dropped silently.

    Args:
        speaker_id: label from tracker e.g. "S1"
        embedding:  [256] L2-normalized embedding for this segment

    Returns:
        bool
    """
    if config.POLICY_MODE == "ENROLLED":
        matched_name = enrollment.match(embedding)
        return matched_name is not None

    if config.POLICY_MODE == "DYNAMIC":
        return speaker_id == config.DYNAMIC_TARGET

    # Unknown mode — fail loudly rather than silently passing all audio
    raise ValueError(
        f"[policy] Unknown POLICY_MODE: '{config.POLICY_MODE}'. "
        "Expected 'ENROLLED' or 'DYNAMIC'."
    )


def active_mode() -> str:
    """Return currently active policy mode — for logging."""
    return config.POLICY_MODE


def set_dynamic_target(speaker_id: str) -> None:
    """
    Switch DYNAMIC_TARGET at runtime without restarting.
    Only meaningful when POLICY_MODE = 'DYNAMIC'.
    """
    config.DYNAMIC_TARGET = speaker_id
    print(f"[policy] dynamic target updated → {speaker_id}")
