"""Policy gate deciding whether a speaker's audio reaches Ultravox.

ENROLLED: pass only if the embedding matches an enrolled profile.
DYNAMIC:  pass only if the tracker's speaker ID matches the dynamic target.

The gate holds the enrollment store it consults, so the dependency is
explicit rather than reached for through module state.
"""

from __future__ import annotations

import numpy as np

import config
from speaker.enrollment import EnrollmentStore

MODE_ENROLLED = "ENROLLED"
MODE_DYNAMIC = "DYNAMIC"


class PolicyGate:
    """Decides which speakers are forwarded and which are dropped."""

    def __init__(
        self,
        enrollment: EnrollmentStore,
        *,
        mode: str | None = None,
        dynamic_target: str | None = None,
    ) -> None:
        """Build a gate over an enrollment store.

        Args:
            enrollment: The store consulted in ENROLLED mode.
            mode: ENROLLED or DYNAMIC. Defaults to ``config.POLICY_MODE``.
            dynamic_target: Speaker ID to pass in DYNAMIC mode. Defaults to
                ``config.DYNAMIC_TARGET``.

        Raises:
            ValueError: If ``mode`` is not a recognised mode. Rejecting it
                here means an unknown mode cannot pass audio through later.
        """
        self.enrollment = enrollment
        self.mode = config.POLICY_MODE if mode is None else mode
        self.dynamic_target = (
            config.DYNAMIC_TARGET if dynamic_target is None else dynamic_target
        )

        if self.mode not in (MODE_ENROLLED, MODE_DYNAMIC):
            raise ValueError(
                f"[policy] Unknown POLICY_MODE: '{self.mode}'. "
                f"Expected '{MODE_ENROLLED}' or '{MODE_DYNAMIC}'."
            )

    def should_pass(self, speaker_id: str, embedding: np.ndarray) -> bool:
        """Decide whether this segment is forwarded or dropped silently.

        Args:
            speaker_id: The tracker's label for this segment, e.g. "S1".
            embedding: The L2-normalized embedding for this segment.

        Returns:
            True to forward the segment, False to drop it.
        """
        if self.mode == MODE_ENROLLED:
            return self.enrollment.match(embedding) is not None
        return speaker_id == self.dynamic_target
