# =============================================================================
# speaker/policy.py — Speaker policy gate
#                     Decides whether a speaker's audio passes to Ultravox
#
# ENROLLED mode: passes audio only if embedding matches enrollment store
# ALL mode:      passes every speaker — the passthrough baseline for evaluation
# DYNAMIC mode:  passes audio only if speaker ID matches DYNAMIC_TARGET
# =============================================================================

import numpy as np

import config
from speaker import enrollment

VALID_MODES = ("ENROLLED", "ALL", "DYNAMIC")

# Distinguishes "caller supplied no match result" from "caller supplied None,
# meaning the embedding matched nobody".
_UNSET = object()


def validate_mode(mode: str | None = None) -> str:
    """
    Check that a policy mode is implemented.

    Called at startup rather than on the first speech segment, so a typo fails
    before models load and the Ultravox call is created — not several seconds
    into a session.

    Raises:
        ValueError if the mode is not one of VALID_MODES.
    """
    mode = config.POLICY_MODE if mode is None else mode
    if mode not in VALID_MODES:
        raise ValueError(
            f"[policy] Unknown policy.mode: '{mode}'. "
            f"Expected one of: {', '.join(VALID_MODES)}."
        )
    return mode


def should_pass(speaker_id: str, embedding: np.ndarray, matched_name=_UNSET) -> bool:
    """
    Gate function — returns True if this speaker's audio should be
    sent to Ultravox, False if it should be dropped silently.

    Args:
        speaker_id:   label from tracker e.g. "S1"
        embedding:    [256] L2-normalized embedding for this segment
        matched_name: optional, the enrollment match already computed by the
                      caller (str for a match, None for no match).  Supplying it
                      avoids running the enrolled-profile cosine loop a second
                      time per segment.  Omit it and the match is computed here.

    Returns:
        bool
    """
    mode = validate_mode()

    if mode == "ALL":
        return True

    if mode == "ENROLLED":
        if matched_name is _UNSET:
            matched_name = enrollment.match(embedding)
        return matched_name is not None

    # DYNAMIC
    return speaker_id == config.DYNAMIC_TARGET
