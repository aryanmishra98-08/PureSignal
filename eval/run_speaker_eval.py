"""FAR / FRR / EER speaker verification evaluation.

Calls encoder.embed() and enrollment.match_with_score() directly — no mic,
no WebSocket.

Directory structure expected:
    --enrolled-dir   profiles/          (<username>.npy files)
    --test-dir       test_audio/<username>/<utterance_N>.wav
    --imposters-dir  imposters/<speaker_id>/<utterance_N>.wav

Usage:
    python eval/run_speaker_eval.py \\
        --enrolled-dir profiles \\
        --test-dir eval/test_audio \\
        --imposters-dir eval/imposters \\
        --threshold-sweep 0.4 0.5 0.6 0.65 0.7 0.8 \\
        --out eval/results/speaker_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from common import collect_wavs, embed_wav, get_pyplot, load_encoder
from metrics import compute_eer, compute_far, compute_frr, det_curve

_TAG = "run_speaker_eval"

_DEFAULT_THRESHOLDS = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Speaker verification FAR/FRR/EER evaluation"
    )
    parser.add_argument(
        "--enrolled-dir", required=True, help="Path to profiles/ directory"
    )
    parser.add_argument(
        "--test-dir",
        required=True,
        help="Path to test WAV directory (speaker/utterance.wav)",
    )
    parser.add_argument(
        "--imposters-dir", required=True, help="Path to impostor WAV directory"
    )
    parser.add_argument(
        "--threshold-sweep",
        nargs="+",
        type=float,
        default=_DEFAULT_THRESHOLDS,
        help="Threshold values to evaluate",
    )
    parser.add_argument(
        "--out", default="eval/results/speaker_eval", help="Output directory"
    )
    return parser.parse_args()


def _score_utterances(
    encoder,
    store,
    wavs_by_speaker: dict[str, list[Path]],
    genuine_names: set[str],
) -> list[dict]:
    """Embed each utterance and score it against the enrolled profiles.

    Args:
        encoder: A loaded SpeakerEncoder.
        store: The EnrollmentStore to score against.
        wavs_by_speaker: ``{speaker_label: [wav_path, ...]}``.
        genuine_names: Labels that count as genuine; everything else is an
            impostor. Pass an empty set to score every utterance as impostor.

    Returns:
        One row per successfully embedded utterance.
    """
    rows: list[dict] = []
    for speaker_name, wavs in wavs_by_speaker.items():
        for wav_path in wavs:
            embedding = embed_wav(encoder, wav_path)
            if embedding is None:
                print(f"  [skip] {wav_path.name} — too short")
                continue
            _, best_sim = store.match_with_score(embedding)
            rows.append({
                "utterance": str(wav_path),
                "speaker": speaker_name,
                "label": 1 if speaker_name in genuine_names else 0,
                "similarity": best_sim,
            })
    return rows


def _plot_det(
    far_vals: list[float], frr_vals: list[float], eer_val: float, out_path: Path
) -> None:
    plt = get_pyplot(_TAG)
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([f * 100 for f in far_vals], [f * 100 for f in frr_vals], color="steelblue")
    ax.plot([eer_val * 100], [eer_val * 100], "ro", label=f"EER={eer_val * 100:.1f}%")
    ax.set_xlabel("FAR (%)")
    ax.set_ylabel("FRR (%)")
    ax.set_title("Detection Error Tradeoff (DET) Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[{_TAG}] DET curve -> {out_path}")


def main() -> None:
    args = _parse_args()

    enrolled_dir = Path(args.enrolled_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{_TAG}] loading encoder...")
    from speaker.enrollment import EnrollmentStore

    encoder = load_encoder()

    print(f"[{_TAG}] loading enrolled profiles...")
    enrolled_names = [p.stem for p in sorted(enrolled_dir.glob("*.npy"))]
    if not enrolled_names:
        print(f"[{_TAG}] ERROR: no .npy files found in {enrolled_dir}")
        sys.exit(1)
    store = EnrollmentStore.load(enrolled_names, profiles_dir=enrolled_dir)

    test_by_speaker = collect_wavs(Path(args.test_dir))
    impostor_by_speaker = collect_wavs(Path(args.imposters_dir))

    n_test = sum(len(v) for v in test_by_speaker.values())
    print(f"[{_TAG}] embedding {n_test} test utterances...")
    rows = _score_utterances(
        encoder, store, test_by_speaker, set(enrolled_names)
    )

    n_impostor = sum(len(v) for v in impostor_by_speaker.values())
    print(f"[{_TAG}] embedding {n_impostor} impostor utterances...")
    rows += _score_utterances(
        encoder, store, impostor_by_speaker, genuine_names=set()
    )

    if not rows:
        print(f"[{_TAG}] ERROR: no utterances could be embedded")
        sys.exit(1)

    scores = [r["similarity"] for r in rows]
    labels = [r["label"] for r in rows]

    scores_csv = out_dir / "scores.csv"
    with open(scores_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["utterance", "speaker", "label", "similarity"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[{_TAG}] scores -> {scores_csv}")

    eer_val, eer_thresh = compute_eer(scores, labels)
    n_genuine = sum(labels)

    metrics_out = {
        "thresholds": {
            str(t): {
                "threshold": t,
                "FAR": compute_far(scores, labels, t),
                "FRR": compute_frr(scores, labels, t),
            }
            for t in args.threshold_sweep
        },
        "EER": eer_val,
        "EER_threshold": eer_thresh,
        "n_genuine": n_genuine,
        "n_impostor": len(labels) - n_genuine,
    }
    metrics_json = out_dir / "metrics.json"
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"[{_TAG}] metrics -> {metrics_json}")

    eer_txt = out_dir / "eer_summary.txt"
    eer_txt.write_text(
        f"EER: {eer_val * 100:.2f}%  (threshold={eer_thresh:.4f})\n"
        f"Genuine utterances: {n_genuine}\n"
        f"Impostor utterances: {len(labels) - n_genuine}\n",
        encoding="utf-8",
    )
    print(f"[{_TAG}] EER summary -> {eer_txt}")
    print(f"\n  EER: {eer_val * 100:.2f}%  (threshold={eer_thresh:.4f})\n")

    far_vals, frr_vals = det_curve(scores, labels)
    _plot_det(far_vals, frr_vals, eer_val, out_dir / "det_curve.png")


if __name__ == "__main__":
    main()
