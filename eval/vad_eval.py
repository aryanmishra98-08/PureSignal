"""VAD parameter sweep evaluation.

Computes segment-level precision, recall, F1, over-segmentation and
under-segmentation against ground-truth labels, sweeping energy_multiplier
and hangover_ms to justify the chosen defaults empirically.

Ground truth formats:
  .lab       "start_s end_s label" per line; speech rows are labelled "speech"
  .TextGrid  Praat TextGrid with a tier named "speech" (needs the textgrid pkg)

Directory structure:
  --audio-dir    audio/<name>.wav
  --labels-dir   labels/<name>.lab  (or <name>.TextGrid)

Usage:
    python eval/vad_eval.py \\
        --audio-dir  eval/vad_audio \\
        --labels-dir eval/vad_labels \\
        --param-sweep energy_multiplier=2.0,3.0,4.0 hangover_ms=200,400,600 \\
        --out eval/results/vad
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from common import get_pyplot, load_wav

_TAG = "vad_eval"

# A predicted segment counts as a hit when it overlaps a ground-truth interval
# by at least this fraction of its own duration.
_IOU_THRESHOLD = 0.3

Interval = tuple[float, float]


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

def _load_lab(path: Path) -> list[Interval]:
    """Load speech intervals from a .lab file ("start_s end_s label" lines)."""
    intervals: list[Interval] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                start, end = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            if parts[2].lower() == "speech":
                intervals.append((start, end))
    return intervals


def _load_textgrid(path: Path) -> list[Interval]:
    """Load speech intervals from the 'speech' tier of a Praat TextGrid."""
    try:
        import textgrid as tg_lib
    except ImportError as e:
        raise RuntimeError(
            "Install 'textgrid' to read TextGrid files: pip install textgrid"
        ) from e

    grid = tg_lib.TextGrid.fromFile(str(path))
    for tier in grid.tiers:
        if tier.name.lower() == "speech":
            return [
                (interval.minTime, interval.maxTime)
                for interval in tier.intervals
                if interval.mark.lower() == "speech"
            ]
    raise ValueError(f"No 'speech' tier found in {path}")


def _load_ground_truth(path: Path) -> list[Interval]:
    """Dispatch to the loader matching the label file's extension."""
    suffix = path.suffix.lower()
    if suffix == ".lab":
        return _load_lab(path)
    if suffix == ".textgrid":
        return _load_textgrid(path)
    raise ValueError(f"Unsupported label format: {path.suffix}")


# ---------------------------------------------------------------------------
# VAD runner
# ---------------------------------------------------------------------------

def _run_vad_on_audio(
    audio: np.ndarray,
    sample_rate: int,
    energy_multiplier: float,
    hangover_ms: int,
) -> list[Interval]:
    """Run the pipeline VAD over one clip with the given parameters.

    Builds a detector configured for this sweep point. Nothing global is
    mutated, so sweep points cannot leak settings into each other.

    Returns:
        The (start_s, end_s) boundaries of each segment the VAD produced.
    """
    from audio.vad import VoiceActivityDetector

    detector = VoiceActivityDetector(
        energy_multiplier=energy_multiplier, hangover_ms=hangover_ms
    )

    frame_size = detector.frame_samples
    segments: list[Interval] = []

    for offset in range(0, len(audio), frame_size):
        frame = audio[offset : offset + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))

        segment = detector.process_frame(frame)
        if segment is not None:
            end_s = offset / sample_rate
            start_s = end_s - len(segment) / sample_rate
            segments.append((max(0.0, start_s), end_s))

    return segments


# ---------------------------------------------------------------------------
# Segment-level metrics
# ---------------------------------------------------------------------------

def _overlap(a: Interval, b: Interval) -> float:
    """Return the overlap in seconds between two intervals."""
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _hits(pred: Interval, ground_truth: list[Interval]) -> list[int]:
    """Return the indices of the ground-truth intervals ``pred`` matches."""
    threshold = _IOU_THRESHOLD * (pred[1] - pred[0])
    return [i for i, gt in enumerate(ground_truth) if _overlap(pred, gt) >= threshold]


def _compute_metrics(
    predicted: list[Interval], ground_truth: list[Interval]
) -> dict:
    """Score predicted segments against ground truth.

    Returns:
        Precision, recall and F1, plus over-/under-segmentation rates and the
        raw TP/FP/FN counts.
    """
    tp = 0
    matched_gt: set[int] = set()
    gt_hit_count: dict[int, int] = {}

    for pred in predicted:
        hits = _hits(pred, ground_truth)
        if hits:
            tp += 1
            matched_gt.add(hits[0])
        for i in hits:
            gt_hit_count[i] = gt_hit_count.get(i, 0) + 1

    fp = len(predicted) - tp
    fn = len(ground_truth) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    # Over-segmentation: one ground-truth interval matched by several segments.
    over_seg = sum(1 for c in gt_hit_count.values() if c > 1) / max(
        len(ground_truth), 1
    )
    # Under-segmentation: one segment spanning several ground-truth intervals.
    under_seg = sum(
        1
        for pred in predicted
        if sum(1 for gt in ground_truth if _overlap(pred, gt) > 0) > 1
    ) / max(len(predicted), 1)

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


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VAD parameter sweep evaluation")
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument(
        "--param-sweep",
        nargs="+",
        default=["energy_multiplier=2.0,3.0,4.0", "hangover_ms=200,400,600"],
        help="Param sweep, e.g. 'energy_multiplier=2.0,3.0,4.0' 'hangover_ms=200,400'",
    )
    parser.add_argument("--out", default="eval/results/vad")
    return parser.parse_args()


def _parse_sweep(specs: list[str]) -> dict[str, list]:
    """Parse ``name=v1,v2,v3`` specs into {name: [values]}, ints where possible."""
    sweep: dict[str, list] = {}
    for spec in specs:
        name, _, values = spec.partition("=")
        sweep[name.strip()] = [
            float(v) if "." in v else int(v) for v in values.split(",")
        ]
    return sweep


def _find_pairs(audio_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    """Pair each WAV with its label file, skipping WAVs that have none."""
    pairs: list[tuple[Path, Path]] = []
    for wav_path in sorted(audio_dir.glob("*.wav")):
        for ext in (".lab", ".TextGrid", ".textgrid"):
            label_path = labels_dir / (wav_path.stem + ext)
            if label_path.exists():
                pairs.append((wav_path, label_path))
                break
    return pairs


_MEAN_KEYS = (
    "precision",
    "recall",
    "f1",
    "over_segmentation",
    "under_segmentation",
)


def _evaluate_params(
    pairs: list[tuple[Path, Path]],
    target_rate: int,
    energy_multiplier: float,
    hangover_ms: int,
) -> dict:
    """Average the segment metrics across every file for one parameter pair."""
    per_file = [
        _compute_metrics(
            _run_vad_on_audio(
                load_wav(wav_path, target_rate),
                target_rate,
                energy_multiplier,
                hangover_ms,
            ),
            _load_ground_truth(label_path),
        )
        for wav_path, label_path in pairs
    ]

    row = {"energy_multiplier": energy_multiplier, "hangover_ms": hangover_ms}
    row.update(
        {key: float(np.mean([m[key] for m in per_file])) for key in _MEAN_KEYS}
    )
    return row


def _plot_heatmap(
    sweep_rows: list[dict],
    energy_values: list,
    hangover_values: list,
    out_path: Path,
) -> None:
    plt = get_pyplot(_TAG)
    if plt is None:
        return

    f1_by_params = {
        (r["energy_multiplier"], r["hangover_ms"]): r["f1"] for r in sweep_rows
    }
    grid = np.array([
        [f1_by_params[(em, hms)] for em in energy_values]
        for hms in hangover_values
    ])

    fig, ax = plt.subplots(figsize=(6, 4))
    image = ax.imshow(grid, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(energy_values)))
    ax.set_xticklabels([str(e) for e in energy_values])
    ax.set_yticks(range(len(hangover_values)))
    ax.set_yticklabels([str(h) for h in hangover_values])
    ax.set_xlabel("energy_multiplier")
    ax.set_ylabel("hangover_ms")
    ax.set_title("VAD F1 — parameter sweep")
    plt.colorbar(image, ax=ax, label="F1")

    for row_idx in range(len(hangover_values)):
        for col_idx in range(len(energy_values)):
            ax.text(
                col_idx,
                row_idx,
                f"{grid[row_idx, col_idx]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[{_TAG}] heatmap -> {out_path}")


def main() -> None:
    args = _parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import config

    target_rate = config.SAMPLE_RATE

    sweep = _parse_sweep(args.param_sweep)
    energy_values = sweep.get("energy_multiplier", [config.ENERGY_MULTIPLIER])
    hangover_values = sweep.get("hangover_ms", [config.HANGOVER_MS])

    pairs = _find_pairs(Path(args.audio_dir), Path(args.labels_dir))
    if not pairs:
        print(f"[{_TAG}] ERROR: no matching audio/label pairs found")
        sys.exit(1)

    print(
        f"[{_TAG}] {len(pairs)} file(s), "
        f"{len(energy_values)}x{len(hangover_values)} param combinations"
    )

    sweep_rows: list[dict] = []
    for raw_em in energy_values:
        for raw_hms in hangover_values:
            em, hms = float(raw_em), int(raw_hms)
            row = _evaluate_params(pairs, target_rate, em, hms)
            sweep_rows.append(row)
            print(
                f"  em={em:.1f} hangover={hms:3d}ms  "
                f"P={row['precision']:.3f} R={row['recall']:.3f} "
                f"F1={row['f1']:.3f}"
            )

    csv_path = out_dir / "param_sweep.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)
    print(f"\n[{_TAG}] sweep results -> {csv_path}")

    best = max(sweep_rows, key=lambda r: r["f1"])
    best_json = out_dir / "best_params.json"
    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "energy_multiplier": best["energy_multiplier"],
                "hangover_ms": best["hangover_ms"],
                "f1": best["f1"],
            },
            f,
            indent=2,
        )
    print(f"[{_TAG}] best params -> {best_json}")
    print(
        f"  Best: energy_multiplier={best['energy_multiplier']}, "
        f"hangover_ms={best['hangover_ms']}, F1={best['f1']:.3f}"
    )

    _plot_heatmap(
        sweep_rows, energy_values, hangover_values, out_dir / "heatmap.png"
    )


if __name__ == "__main__":
    main()
