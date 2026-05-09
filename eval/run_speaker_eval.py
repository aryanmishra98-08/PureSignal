"""
eval/run_speaker_eval.py — FAR / FRR / EER speaker verification evaluation.

Directly calls encoder.embed() and enrollment.match() — no mic, no WebSocket.

Directory structure expected:
    --enrolled-dir   profiles/          (<username>.npy files)
    --test-dir       test_audio/<username>/<utterance_N>.wav
    --imposters-dir  imposters/<speaker_id>/<utterance_N>.wav

Usage:
    python eval/run_speaker_eval.py \\
        --enrolled-dir profiles \\
        --test-dir eval/test_audio \\
        --imposters-dir eval/imposters \\
        --threshold-sweep 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 \\
        --out eval/results/speaker_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np
from metrics import compute_far, compute_frr, compute_eer, det_curve


def _load_wav(path: Path) -> np.ndarray:
    """Load a WAV as mono float32 at 16kHz."""
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
    normalized = normalize(audio)
    return embed(normalized)


def _collect_wavs(directory: Path) -> dict[str, list[Path]]:
    """Return {speaker_label: [wav_path, ...]} from a two-level directory."""
    result: dict[str, list[Path]] = {}
    if not directory.exists():
        return result
    for speaker_dir in sorted(directory.iterdir()):
        if not speaker_dir.is_dir():
            continue
        wavs = sorted(speaker_dir.glob("*.wav"))
        if wavs:
            result[speaker_dir.name] = wavs
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Speaker verification FAR/FRR/EER evaluation")
    parser.add_argument("--enrolled-dir", required=True, help="Path to profiles/ directory")
    parser.add_argument("--test-dir", required=True, help="Path to test WAV directory (speaker/utterance.wav)")
    parser.add_argument("--imposters-dir", required=True, help="Path to impostor WAV directory")
    parser.add_argument(
        "--threshold-sweep",
        nargs="+",
        type=float,
        default=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
        help="Threshold values to evaluate",
    )
    parser.add_argument("--out", default="eval/results/speaker_eval", help="Output directory")
    args = parser.parse_args()

    enrolled_dir = Path(args.enrolled_dir)
    test_dir = Path(args.test_dir)
    imposters_dir = Path(args.imposters_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load encoder
    print("[run_speaker_eval] loading encoder...")
    from speaker.encoder import load_encoder
    load_encoder()

    # Load enrolled profiles
    print("[run_speaker_eval] loading enrolled profiles...")
    from speaker.enrollment import load_profiles
    enrolled_names = [p.stem for p in sorted(enrolled_dir.glob("*.npy"))]
    if not enrolled_names:
        print(f"[run_speaker_eval] ERROR: no .npy files found in {enrolled_dir}")
        sys.exit(1)
    load_profiles(enrolled_names)

    # Collect test and impostor WAVs
    test_by_speaker = _collect_wavs(test_dir)
    impostor_by_speaker = _collect_wavs(imposters_dir)

    # Build (score, label) pairs
    scores: list[float] = []
    labels: list[int] = []
    rows: list[dict] = []

    print(f"[run_speaker_eval] embedding {sum(len(v) for v in test_by_speaker.values())} test utterances...")
    from speaker.enrollment import match_with_score
    for speaker_name, wavs in test_by_speaker.items():
        for wav_path in wavs:
            emb = _embed_wav(wav_path)
            if emb is None:
                print(f"  [skip] {wav_path.name} — too short")
                continue
            _, best_sim = match_with_score(emb)
            is_genuine = speaker_name in enrolled_names
            label = 1 if is_genuine else 0
            scores.append(best_sim)
            labels.append(label)
            rows.append({
                "utterance": str(wav_path),
                "speaker": speaker_name,
                "label": label,
                "similarity": best_sim,
            })

    print(f"[run_speaker_eval] embedding {sum(len(v) for v in impostor_by_speaker.values())} impostor utterances...")
    for speaker_name, wavs in impostor_by_speaker.items():
        for wav_path in wavs:
            emb = _embed_wav(wav_path)
            if emb is None:
                continue
            _, best_sim = match_with_score(emb)
            scores.append(best_sim)
            labels.append(0)
            rows.append({
                "utterance": str(wav_path),
                "speaker": speaker_name,
                "label": 0,
                "similarity": best_sim,
            })

    if not scores:
        print("[run_speaker_eval] ERROR: no utterances could be embedded")
        sys.exit(1)

    # Write raw scores CSV
    scores_csv = out_dir / "scores.csv"
    with open(scores_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["utterance", "speaker", "label", "similarity"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[run_speaker_eval] scores → {scores_csv}")

    # Threshold sweep
    metrics: dict[str, dict] = {}
    for t in args.threshold_sweep:
        far = compute_far(scores, labels, t)
        frr = compute_frr(scores, labels, t)
        metrics[str(t)] = {"threshold": t, "FAR": far, "FRR": frr}

    eer_val, eer_thresh = compute_eer(scores, labels)
    far_vals, frr_vals = det_curve(scores, labels)

    metrics_out = {
        "thresholds": metrics,
        "EER": eer_val,
        "EER_threshold": eer_thresh,
        "n_genuine": sum(labels),
        "n_impostor": len(labels) - sum(labels),
    }
    metrics_json = out_dir / "metrics.json"
    with open(metrics_json, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"[run_speaker_eval] metrics → {metrics_json}")

    # EER summary
    eer_txt = out_dir / "eer_summary.txt"
    eer_txt.write_text(
        f"EER: {eer_val * 100:.2f}%  (threshold={eer_thresh:.4f})\n"
        f"Genuine utterances: {sum(labels)}\n"
        f"Impostor utterances: {len(labels) - sum(labels)}\n"
    )
    print(f"[run_speaker_eval] EER summary → {eer_txt}")
    print(f"\n  EER: {eer_val * 100:.2f}%  (threshold={eer_thresh:.4f})\n")

    # DET curve plot
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([f * 100 for f in far_vals], [f * 100 for f in frr_vals], color="steelblue")
        ax.plot([eer_val * 100], [eer_val * 100], "ro", label=f"EER={eer_val*100:.1f}%")
        ax.set_xlabel("FAR (%)")
        ax.set_ylabel("FRR (%)")
        ax.set_title("Detection Error Tradeoff (DET) Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        det_png = out_dir / "det_curve.png"
        fig.savefig(det_png, dpi=150)
        print(f"[run_speaker_eval] DET curve → {det_png}")
    except ImportError:
        print("[run_speaker_eval] matplotlib not installed — skipping DET plot")


if __name__ == "__main__":
    main()
