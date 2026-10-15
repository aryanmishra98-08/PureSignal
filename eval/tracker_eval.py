"""Multi-speaker session tracker evaluation.

Feeds utterances through the encoder and tracker in a specified order, then
compares the tracker's assignments against the true speaker labels.

Directory structure:
    --session-dir  session/<speaker_id>/<utterance_N>.wav

Usage:
    python eval/tracker_eval.py \\
        --session-dir eval/session \\
        --sequence "A B A B C A C B" \\
        --out eval/results/tracker_eval
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from itertools import cycle
from pathlib import Path

from common import embed_wav, get_pyplot, load_encoder

_TAG = "tracker_eval"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tracker purity evaluation")
    parser.add_argument(
        "--session-dir", required=True, help="Directory: speaker_id/utterance_N.wav"
    )
    parser.add_argument(
        "--sequence",
        required=True,
        help="Space-separated order of speaker labels, e.g. 'A B A B C A'",
    )
    parser.add_argument(
        "--out", default="eval/results/tracker_eval", help="Output directory"
    )
    return parser.parse_args()


def _compute_purity(assignments: list[tuple[str, str]]) -> float:
    """Return tracker purity.

    For each assigned ID, take the count of its most frequent true label;
    purity is the sum of those counts over the total number of assignments.
    """
    if not assignments:
        return 0.0

    clusters: dict[str, Counter] = defaultdict(Counter)
    for true_label, assigned_id in assignments:
        clusters[assigned_id][true_label] += 1

    correct = sum(counts.most_common(1)[0][1] for counts in clusters.values())
    return correct / len(assignments)


def _build_confusion(assignments: list[tuple[str, str]]) -> dict:
    """Build a true-label by assigned-ID count matrix."""
    true_labels = sorted({t for t, _ in assignments})
    assigned_ids = sorted({a for _, a in assignments})

    matrix = {t: dict.fromkeys(assigned_ids, 0) for t in true_labels}
    for true_label, assigned_id in assignments:
        matrix[true_label][assigned_id] += 1

    return {
        "true_labels": true_labels,
        "assigned_ids": assigned_ids,
        "matrix": matrix,
    }


def _load_session(session_dir: Path, sequence: list[str]) -> dict[str, cycle]:
    """Return an endless WAV iterator per speaker in the sequence.

    Exits if any speaker named in the sequence has no directory or no WAVs.
    """
    iterators: dict[str, cycle] = {}
    for speaker_id in set(sequence):
        speaker_path = session_dir / speaker_id
        if not speaker_path.exists():
            print(f"[{_TAG}] ERROR: directory not found: {speaker_path}")
            sys.exit(1)

        wavs = sorted(speaker_path.glob("*.wav"))
        if not wavs:
            print(f"[{_TAG}] ERROR: no WAV files in {speaker_path}")
            sys.exit(1)

        # Cycle so a sequence longer than the utterance list wraps around.
        iterators[speaker_id] = cycle(wavs)
    return iterators


def _plot_timeline(
    assignments: list[tuple[str, str]], purity: float, out_path: Path
) -> None:
    plt = get_pyplot(_TAG)
    if plt is None:
        return
    import matplotlib.patches as mpatches

    label_colors = {
        label: f"C{i}"
        for i, label in enumerate(sorted({t for t, _ in assignments}))
    }
    assigned_to_int = {
        assigned: i
        for i, assigned in enumerate(sorted({a for _, a in assignments}))
    }

    fig, ax = plt.subplots(figsize=(max(8, len(assignments) * 0.4), 4))
    for step, (true_label, assigned_id) in enumerate(assignments):
        ax.scatter(
            step,
            assigned_to_int[assigned_id],
            color=label_colors[true_label],
            s=80,
            zorder=3,
        )

    ax.set_yticks(list(assigned_to_int.values()))
    ax.set_yticklabels(list(assigned_to_int.keys()))
    ax.set_xlabel("Utterance step")
    ax.set_ylabel("Tracker assignment")
    ax.set_title(f"Speaker tracker timeline  (purity={purity * 100:.1f}%)")
    ax.legend(
        handles=[
            mpatches.Patch(color=color, label=label)
            for label, color in label_colors.items()
        ],
        title="True speaker",
    )
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  Timeline plot -> {out_path}")


def main() -> None:
    args = _parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sequence = args.sequence.strip().split()
    speaker_iters = _load_session(Path(args.session_dir), sequence)

    print(f"[{_TAG}] loading encoder...")
    from speaker.tracker import SpeakerTracker

    encoder = load_encoder()
    tracker = SpeakerTracker()

    assignments: list[tuple[str, str]] = []
    timeline: list[dict] = []

    print(f"[{_TAG}] simulating {len(sequence)}-utterance session...")
    for step, true_label in enumerate(sequence):
        wav_path = next(speaker_iters[true_label])
        embedding = embed_wav(encoder, wav_path)
        if embedding is None:
            print(
                f"  [step {step}] {true_label}: {wav_path.name} too short — skipped"
            )
            continue

        assigned_id = tracker.assign(embedding)
        assignments.append((true_label, assigned_id))
        timeline.append({
            "step": step,
            "true_label": true_label,
            "assigned_id": assigned_id,
            "utterance": wav_path.name,
            "gallery_size": len(tracker),
        })
        print(f"  step {step:2d}: true={true_label}  assigned={assigned_id}")

    if not assignments:
        print(f"[{_TAG}] ERROR: no utterances could be embedded")
        sys.exit(1)

    purity = _compute_purity(assignments)
    results = {
        "purity": purity,
        "n_steps": len(assignments),
        "confusion": _build_confusion(assignments),
        "timeline": timeline,
    }

    out_json = out_dir / "tracker_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Tracker purity: {purity * 100:.1f}%  ({len(assignments)} assignments)")
    print(f"  Results -> {out_json}\n")

    _plot_timeline(assignments, purity, out_dir / "timeline.png")


if __name__ == "__main__":
    main()
