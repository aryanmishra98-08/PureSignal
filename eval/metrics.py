"""
eval/metrics.py — Pure functions for speaker verification evaluation.

No I/O. All inputs are lists of (score, label) pairs where:
    score: float in [-1, 1] — cosine similarity
    label: int  1 = genuine (target), 0 = impostor

Functions:
    compute_far, compute_frr, compute_eer, det_curve
"""

from __future__ import annotations

import math


def compute_far(scores: list[float], labels: list[int], threshold: float) -> float:
    """False Accept Rate: fraction of impostors accepted (score >= threshold)."""
    impostors = [(s, l) for s, l in zip(scores, labels) if l == 0]
    if not impostors:
        return 0.0
    accepted = sum(1 for s, _ in impostors if s >= threshold)
    return accepted / len(impostors)


def compute_frr(scores: list[float], labels: list[int], threshold: float) -> float:
    """False Reject Rate: fraction of genuine pairs rejected (score < threshold)."""
    genuines = [(s, l) for s, l in zip(scores, labels) if l == 1]
    if not genuines:
        return 0.0
    rejected = sum(1 for s, _ in genuines if s < threshold)
    return rejected / len(genuines)


def compute_eer(
    scores: list[float], labels: list[int]
) -> tuple[float, float]:
    """
    Equal Error Rate — threshold where FAR ≈ FRR.

    Returns:
        (eer_value, eer_threshold) — both in [0, 1]
    """
    thresholds = sorted(set(scores))
    best_diff = float("inf")
    eer_val = float("nan")
    eer_thresh = float("nan")

    for t in thresholds:
        far = compute_far(scores, labels, t)
        frr = compute_frr(scores, labels, t)
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            eer_val = (far + frr) / 2
            eer_thresh = t

    return eer_val, eer_thresh


def det_curve(
    scores: list[float], labels: list[int], n_points: int = 200
) -> tuple[list[float], list[float]]:
    """
    Detection Error Tradeoff curve.

    Returns:
        (far_values, frr_values) across a sweep of n_points thresholds
        from min(scores) to max(scores), suitable for a DET plot.
    """
    if not scores:
        return [], []

    lo, hi = min(scores), max(scores)
    step = (hi - lo) / max(n_points - 1, 1)
    thresholds = [lo + i * step for i in range(n_points)]

    far_vals = [compute_far(scores, labels, t) for t in thresholds]
    frr_vals = [compute_frr(scores, labels, t) for t in thresholds]
    return far_vals, frr_vals
