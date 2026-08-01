"""
eval/vad_metrics.py — Segment-level VAD scoring.

Split out of vad_eval.py to keep both files within the project's 300-line limit.

Compares predicted (start_s, end_s) speech intervals against ground truth and
returns precision, recall, F1, over-segmentation, and under-segmentation.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Segment-level metrics
# ---------------------------------------------------------------------------


def interval_overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def compute_metrics(
    predicted: list[tuple[float, float]],
    ground_truth: list[tuple[float, float]],
    iou_threshold: float = 0.3,
) -> dict:
    """
    Segment-level precision / recall / F1.

    A predicted segment is a true positive if it overlaps with any ground-truth
    segment by >= iou_threshold * its own duration.
    """
    tp = 0
    fp = 0
    matched_gt: set[int] = set()

    for pred in predicted:
        pred_dur = pred[1] - pred[0]
        matched = False
        for i, gt in enumerate(ground_truth):
            overlap = interval_overlap(pred, gt)
            if overlap >= iou_threshold * pred_dur:
                tp += 1
                matched_gt.add(i)
                matched = True
                break
        if not matched:
            fp += 1

    fn = len(ground_truth) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision +
                                     recall) if (precision + recall) > 0 else 0.0

    # Over-segmentation: one GT interval matched by multiple predicted intervals
    gt_hit_count: dict[int, int] = {}
    for pred in predicted:
        pred_dur = pred[1] - pred[0]
        for i, gt in enumerate(ground_truth):
            if interval_overlap(pred, gt) >= iou_threshold * pred_dur:
                gt_hit_count[i] = gt_hit_count.get(i, 0) + 1
    over_seg = sum(1 for v in gt_hit_count.values()
                   if v > 1) / max(len(ground_truth), 1)

    # Under-segmentation: one predicted interval covers multiple GT intervals
    under_seg_count = 0
    for pred in predicted:
        covered = sum(
            1 for gt in ground_truth if interval_overlap(pred, gt) > 0
        )
        if covered > 1:
            under_seg_count += 1
    under_seg = under_seg_count / max(len(predicted), 1)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "over_segmentation": over_seg,
        "under_segmentation": under_seg,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_predicted": len(predicted),
        "n_ground_truth": len(ground_truth),
    }
