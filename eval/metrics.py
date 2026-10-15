"""Pure functions for speaker verification evaluation.

No I/O. Every function takes parallel ``scores`` and ``labels`` sequences:
    score: float in [-1, 1] — cosine similarity
    label: int — 1 for a genuine (target) pair, 0 for an impostor
"""

from __future__ import annotations


def compute_far(scores: list[float], labels: list[int], threshold: float) -> float:
    """Return the False Accept Rate: impostors scoring at or above threshold."""
    impostors = [s for s, label in zip(scores, labels) if label == 0]
    if not impostors:
        return 0.0
    return sum(1 for s in impostors if s >= threshold) / len(impostors)


def compute_frr(scores: list[float], labels: list[int], threshold: float) -> float:
    """Return the False Reject Rate: genuine pairs scoring below threshold."""
    genuines = [s for s, label in zip(scores, labels) if label == 1]
    if not genuines:
        return 0.0
    return sum(1 for s in genuines if s < threshold) / len(genuines)


def compute_eer(scores: list[float], labels: list[int]) -> tuple[float, float]:
    """Return the Equal Error Rate — the point where FAR and FRR are closest.

    Returns:
        ``(eer_value, eer_threshold)``, both NaN if ``scores`` is empty.
    """
    best_diff = float("inf")
    eer_val = float("nan")
    eer_thresh = float("nan")

    for threshold in sorted(set(scores)):
        far = compute_far(scores, labels, threshold)
        frr = compute_frr(scores, labels, threshold)
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            eer_val = (far + frr) / 2
            eer_thresh = threshold

    return eer_val, eer_thresh


def det_curve(
    scores: list[float], labels: list[int], n_points: int = 200
) -> tuple[list[float], list[float]]:
    """Return a Detection Error Tradeoff curve.

    Args:
        scores: Similarity scores.
        labels: Matching genuine/impostor labels.
        n_points: Number of thresholds to sweep between the score extremes.

    Returns:
        ``(far_values, frr_values)`` ready to plot, or two empty lists if
        ``scores`` is empty.
    """
    if not scores:
        return [], []

    lo, hi = min(scores), max(scores)
    step = (hi - lo) / max(n_points - 1, 1)
    thresholds = [lo + i * step for i in range(n_points)]

    far_vals = [compute_far(scores, labels, t) for t in thresholds]
    frr_vals = [compute_frr(scores, labels, t) for t in thresholds]
    return far_vals, frr_vals
