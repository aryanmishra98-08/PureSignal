"""
eval/vad_eval.py — VAD parameter sweep evaluation.

Computes segment-level precision, recall, F1, over-segmentation, and
under-segmentation rates against ground-truth labels (TextGrid or .lab format).

Supports a parameter sweep over energy_multiplier and hangover_ms,
producing a heatmap that empirically justifies the chosen defaults.

Ground truth formats:
  .lab files:  "start_s end_s label" (per line), speech label = "speech"
  .TextGrid:   Praat TextGrid with a tier named "speech" (requires textgrid package)

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

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

def _load_lab(path: Path) -> list[tuple[float, float]]:
    """
    Load a .lab file. Lines: "start_s end_s label".
    Returns list of (start_s, end_s) for speech intervals.
    """
    intervals = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                start, end, label = float(parts[0]), float(parts[1]), parts[2]
            except ValueError:
                continue
            if label.lower() == "speech":
                intervals.append((start, end))
    return intervals


def _load_textgrid(path: Path) -> list[tuple[float, float]]:
    """
    Load a Praat TextGrid. Requires the 'textgrid' PyPI package.
    Looks for a tier named 'speech'.
    """
    try:
        import textgrid as tg_lib
    except ImportError:
        raise RuntimeError("Install 'textgrid' to read TextGrid files: pip install textgrid")

    tg = tg_lib.TextGrid.fromFile(str(path))
    for tier in tg.tiers:
        if tier.name.lower() == "speech":
            return [
                (interval.minTime, interval.maxTime)
                for interval in tier.intervals
                if interval.mark.lower() == "speech"
            ]
    raise ValueError(f"No 'speech' tier found in {path}")


def _load_ground_truth(path: Path) -> list[tuple[float, float]]:
    if path.suffix.lower() == ".lab":
        return _load_lab(path)
    if path.suffix.lower() == ".textgrid":
        return _load_textgrid(path)
    raise ValueError(f"Unsupported label format: {path.suffix}")


# ---------------------------------------------------------------------------
# VAD runner (stateless — resets between files)
# ---------------------------------------------------------------------------

def _run_vad_on_audio(
    audio: np.ndarray,
    sample_rate: int,
    energy_multiplier: float,
    hangover_ms: int,
) -> list[tuple[float, float]]:
    """
    Run the pipeline VAD with overridden parameters. Returns list of
    (start_s, end_s) segment boundaries produced by the VAD.
    """
    import config as _cfg

    # Temporarily patch config constants so vad.py picks them up on reset
    _cfg.ENERGY_MULTIPLIER = energy_multiplier
    _cfg.HANGOVER_MS = hangover_ms

    from audio import vad
    vad.reset()
    # Patch internal constants that were computed at module load time
    vad._HANGOVER_FRAMES = int(hangover_ms / _cfg.FRAME_MS)

    frame_size = _cfg.FRAME_SAMPLES
    segments: list[tuple[float, float]] = []
    offset = 0

    while offset < len(audio):
        frame = audio[offset: offset + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))
        segment = vad.process_frame(frame)
        if segment is not None:
            end_s = offset / sample_rate
            # Estimate start from segment length
            start_s = end_s - len(segment) / sample_rate
            segments.append((max(0.0, start_s), end_s))
        offset += frame_size

    vad.reset()
    return segments


# ---------------------------------------------------------------------------
# Segment-level metrics
# ---------------------------------------------------------------------------

def _interval_overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _compute_metrics(
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
            overlap = _interval_overlap(pred, gt)
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
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Over-segmentation: one GT interval matched by multiple predicted intervals
    gt_hit_count: dict[int, int] = {}
    for pred in predicted:
        pred_dur = pred[1] - pred[0]
        for i, gt in enumerate(ground_truth):
            if _interval_overlap(pred, gt) >= iou_threshold * pred_dur:
                gt_hit_count[i] = gt_hit_count.get(i, 0) + 1
    over_seg = sum(1 for v in gt_hit_count.values() if v > 1) / max(len(ground_truth), 1)

    # Under-segmentation: one predicted interval covers multiple GT intervals
    under_seg_count = 0
    for pred in predicted:
        covered = sum(
            1 for gt in ground_truth if _interval_overlap(pred, gt) > 0
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_wav(path: Path, target_rate: int) -> tuple[np.ndarray, int]:
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from math import gcd

    rate, data = wavfile.read(path)
    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.float64:
        audio = data.astype(np.float32)
    else:
        audio = data.astype(np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if rate != target_rate:
        g = gcd(rate, target_rate)
        audio = resample_poly(audio, target_rate // g, rate // g).astype(np.float32)
        rate = target_rate
    return audio, rate


def main() -> None:
    parser = argparse.ArgumentParser(description="VAD parameter sweep evaluation")
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument(
        "--param-sweep",
        nargs="+",
        default=["energy_multiplier=2.0,3.0,4.0", "hangover_ms=200,400,600"],
        help="Param sweep, e.g. 'energy_multiplier=2.0,3.0,4.0' 'hangover_ms=200,400,600'",
    )
    parser.add_argument("--out", default="eval/results/vad")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    labels_dir = Path(args.labels_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import config
    target_rate = config.SAMPLE_RATE

    # Parse sweep params
    sweep_params: dict[str, list] = {}
    for spec in args.param_sweep:
        name, _, vals = spec.partition("=")
        sweep_params[name.strip()] = [
            (int(v) if "." not in v else float(v)) for v in vals.split(",")
        ]

    energy_values = sweep_params.get("energy_multiplier", [config.ENERGY_MULTIPLIER])
    hangover_values = sweep_params.get("hangover_ms", [config.HANGOVER_MS])

    # Collect audio/label pairs
    pairs: list[tuple[Path, Path]] = []
    for wav_path in sorted(audio_dir.glob("*.wav")):
        for ext in (".lab", ".TextGrid", ".textgrid"):
            label_path = labels_dir / (wav_path.stem + ext)
            if label_path.exists():
                pairs.append((wav_path, label_path))
                break

    if not pairs:
        print(f"[vad_eval] ERROR: no matching audio/label pairs found")
        sys.exit(1)

    print(f"[vad_eval] {len(pairs)} file(s), "
          f"{len(energy_values)}×{len(hangover_values)} param combinations")

    sweep_rows: list[dict] = []
    best_f1 = -1.0
    best_params: dict = {}

    for em in energy_values:
        for hms in hangover_values:
            agg: dict[str, list[float]] = {
                "precision": [], "recall": [], "f1": [],
                "over_segmentation": [], "under_segmentation": [],
            }

            for wav_path, label_path in pairs:
                audio, rate = _load_wav(wav_path, target_rate)
                gt = _load_ground_truth(label_path)
                predicted = _run_vad_on_audio(audio, rate, float(em), int(hms))
                m = _compute_metrics(predicted, gt)
                for key in agg:
                    agg[key].append(m[key])

            row = {
                "energy_multiplier": em,
                "hangover_ms": hms,
                "precision": float(np.mean(agg["precision"])),
                "recall": float(np.mean(agg["recall"])),
                "f1": float(np.mean(agg["f1"])),
                "over_segmentation": float(np.mean(agg["over_segmentation"])),
                "under_segmentation": float(np.mean(agg["under_segmentation"])),
            }
            sweep_rows.append(row)

            if row["f1"] > best_f1:
                best_f1 = row["f1"]
                best_params = {"energy_multiplier": em, "hangover_ms": hms}

            print(
                f"  em={em:.1f} hangover={hms:3d}ms  "
                f"P={row['precision']:.3f} R={row['recall']:.3f} F1={row['f1']:.3f}"
            )

    # Write CSV
    csv_path = out_dir / "param_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)
    print(f"\n[vad_eval] sweep results → {csv_path}")

    # Write best params
    best_json = out_dir / "best_params.json"
    with open(best_json, "w") as f:
        json.dump({**best_params, "f1": best_f1}, f, indent=2)
    print(f"[vad_eval] best params → {best_json}")
    print(f"  Best: energy_multiplier={best_params['energy_multiplier']}, "
          f"hangover_ms={best_params['hangover_ms']}, F1={best_f1:.3f}")

    # Heatmap
    try:
        import matplotlib.pyplot as plt

        f1_grid = np.array([
            [r["f1"] for r in sweep_rows if r["hangover_ms"] == hms]
            for hms in hangover_values
        ])

        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(f1_grid, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(energy_values)))
        ax.set_xticklabels([str(e) for e in energy_values])
        ax.set_yticks(range(len(hangover_values)))
        ax.set_yticklabels([str(h) for h in hangover_values])
        ax.set_xlabel("energy_multiplier")
        ax.set_ylabel("hangover_ms")
        ax.set_title("VAD F1 — parameter sweep")
        plt.colorbar(im, ax=ax, label="F1")

        for i, hms in enumerate(hangover_values):
            for j, em in enumerate(energy_values):
                val = f1_grid[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

        fig.tight_layout()
        heatmap_path = out_dir / "heatmap.png"
        fig.savefig(heatmap_path, dpi=150)
        print(f"[vad_eval] heatmap → {heatmap_path}")
    except ImportError:
        print("[vad_eval] matplotlib not installed — skipping heatmap")


if __name__ == "__main__":
    main()
