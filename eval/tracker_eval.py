"""
eval/tracker_eval.py — Multi-speaker session tracker evaluation.

Simulates a session by feeding utterances in a specified order through the
encoder + tracker, then compares assignments to ground truth speaker labels.

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
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np


def _load_wav(path: Path) -> np.ndarray:
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from math import gcd
    import config

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

    target = config.SAMPLE_RATE
    if rate != target:
        g = gcd(rate, target)
        audio = resample_poly(audio, target // g, rate // g).astype(np.float32)
    return audio


def _embed_wav(path: Path) -> np.ndarray | None:
    from audio.features import normalize
    from speaker.encoder import embed
    audio = _load_wav(path)
    return embed(normalize(audio))


def _compute_purity(assignments: list[tuple[str, str]]) -> float:
    """
    Tracker purity: for each tracker-assigned ID, find the most frequent
    true label, then sum those counts divided by total.
    """
    cluster_to_true: dict[str, list[str]] = defaultdict(list)
    for true_label, assigned_id in assignments:
        cluster_to_true[assigned_id].append(true_label)

    correct = sum(
        max(defaultdict(int, {v: vs.count(v) for v in set(vs)}).values())
        for vs in cluster_to_true.values()
    )
    return correct / len(assignments) if assignments else 0.0


def _build_confusion(assignments: list[tuple[str, str]]) -> dict:
    true_labels = sorted({t for t, _ in assignments})
    assigned_ids = sorted({a for _, a in assignments})
    matrix: dict[str, dict[str, int]] = {t: {a: 0 for a in assigned_ids} for t in true_labels}
    for true_label, assigned_id in assignments:
        matrix[true_label][assigned_id] += 1
    return {"true_labels": true_labels, "assigned_ids": assigned_ids, "matrix": matrix}


def main() -> None:
    parser = argparse.ArgumentParser(description="Tracker purity evaluation")
    parser.add_argument("--session-dir", required=True, help="Directory: speaker_id/utterance_N.wav")
    parser.add_argument(
        "--sequence",
        required=True,
        help="Space-separated order of speaker labels, e.g. 'A B A B C A'",
    )
    parser.add_argument("--out", default="eval/results/tracker_eval", help="Output directory")
    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sequence = args.sequence.strip().split()

    # Build per-speaker utterance iterators
    speaker_wavs: dict[str, list[Path]] = {}
    for speaker_id in set(sequence):
        speaker_path = session_dir / speaker_id
        if not speaker_path.exists():
            print(f"[tracker_eval] ERROR: directory not found: {speaker_path}")
            sys.exit(1)
        wavs = sorted(speaker_path.glob("*.wav"))
        if not wavs:
            print(f"[tracker_eval] ERROR: no WAV files in {speaker_path}")
            sys.exit(1)
        speaker_wavs[speaker_id] = wavs

    # Cycling iterator: if fewer utterances than sequence calls, wrap around
    speaker_iters = {sid: iter(wavs * (len(sequence) // len(wavs) + 1)) for sid, wavs in speaker_wavs.items()}

    # Load encoder and reset tracker
    print("[tracker_eval] loading encoder...")
    from speaker.encoder import load_encoder
    from speaker.tracker import assign, reset, get_gallery
    load_encoder()
    reset()

    assignments: list[tuple[str, str]] = []
    timeline: list[dict] = []

    print(f"[tracker_eval] simulating {len(sequence)}-utterance session...")
    for step, true_label in enumerate(sequence):
        wav_path = next(speaker_iters[true_label])
        emb = _embed_wav(wav_path)
        if emb is None:
            print(f"  [step {step}] {true_label}: {wav_path.name} too short — skipped")
            continue

        assigned_id = assign(emb)
        assignments.append((true_label, assigned_id))
        timeline.append({
            "step": step,
            "true_label": true_label,
            "assigned_id": assigned_id,
            "utterance": wav_path.name,
            "gallery_size": len(get_gallery()),
        })
        print(f"  step {step:2d}: true={true_label}  assigned={assigned_id}  {'OK' if assigned_id else ''}")

    purity = _compute_purity(assignments)
    confusion = _build_confusion(assignments)

    results = {
        "purity": purity,
        "n_steps": len(assignments),
        "confusion": confusion,
        "timeline": timeline,
    }

    out_json = out_dir / "tracker_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Tracker purity: {purity * 100:.1f}%  ({len(assignments)} assignments)")
    print(f"  Results → {out_json}\n")

    # Timeline plot
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        true_labels_sorted = sorted({t for t, _ in assignments})
        assigned_ids_sorted = sorted({a for _, a in assignments})
        label_colors = {l: f"C{i}" for i, l in enumerate(true_labels_sorted)}
        assigned_to_int = {a: i for i, a in enumerate(assigned_ids_sorted)}

        fig, ax = plt.subplots(figsize=(max(8, len(assignments) * 0.4), 4))
        for i, (true_label, assigned_id) in enumerate(assignments):
            color = label_colors[true_label]
            y = assigned_to_int[assigned_id]
            ax.scatter(i, y, color=color, s=80, zorder=3)

        ax.set_yticks(list(assigned_to_int.values()))
        ax.set_yticklabels(list(assigned_to_int.keys()))
        ax.set_xlabel("Utterance step")
        ax.set_ylabel("Tracker assignment")
        ax.set_title(f"Speaker tracker timeline  (purity={purity*100:.1f}%)")
        legend_handles = [mpatches.Patch(color=c, label=l) for l, c in label_colors.items()]
        ax.legend(handles=legend_handles, title="True speaker")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        plot_path = out_dir / "timeline.png"
        fig.savefig(plot_path, dpi=150)
        print(f"  Timeline plot → {plot_path}")
    except ImportError:
        print("[tracker_eval] matplotlib not installed — skipping timeline plot")


if __name__ == "__main__":
    main()
