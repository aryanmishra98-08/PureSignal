"""
eval/enrollment_quality_eval.py — EER vs. number of enrollment samples curve.

Enrolls the same speaker with 1, 2, 3, 5 samples and runs the speaker evaluator
for each, then plots EER on y vs. num_samples on x.

Requires the same test/impostor directory layout as run_speaker_eval.py.

Usage:
    python eval/enrollment_quality_eval.py \\
        --speaker-wavs eval/enrollment_wavs/alice  \\   # ≥5 WAVs for alice
        --enrolled-dir profiles         \\
        --test-dir     eval/test_audio             \\
        --imposters-dir eval/imposters             \\
        --username     alice                       \\
        --out          eval/results/quality_vs_samples
"""

from __future__ import annotations

import argparse
import json
import sys
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


def _enroll_with_n_samples(
    username: str,
    wavs: list[Path],
    n: int,
    profiles_dir: Path,
) -> bool:
    """Average the first n embeddings and save as username.npy. Returns False if none succeed."""
    import config
    from speaker.enrollment import validate_enrollment_quality, save_with_metadata

    selected = wavs[:n]
    embeddings = []
    for wav_path in selected:
        emb = _embed_wav(wav_path)
        if emb is not None:
            embeddings.append(emb)

    if not embeddings:
        return False

    profile = np.mean(embeddings, axis=0).astype(np.float32)
    norm = np.linalg.norm(profile)
    if norm > config.NORM_FLOOR:
        profile /= norm

    quality = validate_enrollment_quality(profile)
    save_with_metadata(profile, username, profiles_dir, num_samples=len(embeddings), quality=quality)
    return True


def _run_eval(enrolled_dir: Path, test_dir: Path, imposters_dir: Path, enrolled_names: list[str]) -> float:
    """Run speaker evaluation and return EER."""
    from speaker.enrollment import load_profiles, match_with_score
    from metrics import compute_eer

    load_profiles(enrolled_names)

    scores: list[float] = []
    labels: list[int] = []

    def _collect(directory: Path, is_genuine: bool) -> None:
        if not directory.exists():
            return
        for speaker_dir in sorted(directory.iterdir()):
            if not speaker_dir.is_dir():
                continue
            for wav_path in sorted(speaker_dir.glob("*.wav")):
                emb = _embed_wav(wav_path)
                if emb is None:
                    continue
                _, best_sim = match_with_score(emb)
                scores.append(best_sim)
                labels.append(1 if is_genuine and speaker_dir.name in enrolled_names else 0)

    _collect(test_dir, is_genuine=True)
    _collect(imposters_dir, is_genuine=False)

    if not scores:
        return float("nan")

    eer_val, _ = compute_eer(scores, labels)
    return eer_val


def main() -> None:
    parser = argparse.ArgumentParser(description="EER vs enrollment samples curve")
    parser.add_argument("--speaker-wavs", required=True, help="Directory with ≥5 WAVs for the target speaker")
    parser.add_argument("--enrolled-dir", required=True, help="profiles/ directory (will be written to)")
    parser.add_argument("--test-dir", required=True, help="test_audio/ directory")
    parser.add_argument("--imposters-dir", required=True, help="imposters/ directory")
    parser.add_argument("--username", required=True, help="Username of the target speaker")
    parser.add_argument("--sample-counts", nargs="+", type=int, default=[1, 2, 3, 5],
                        help="Number of samples to test (default: 1 2 3 5)")
    parser.add_argument("--out", default="eval/results/quality_vs_samples", help="Output directory")
    args = parser.parse_args()

    speaker_wavs_dir = Path(args.speaker_wavs)
    enrolled_dir = Path(args.enrolled_dir)
    test_dir = Path(args.test_dir)
    imposters_dir = Path(args.imposters_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(speaker_wavs_dir.glob("*.wav"))
    if not wavs:
        print(f"[enrollment_quality_eval] ERROR: no WAVs in {speaker_wavs_dir}")
        sys.exit(1)

    max_n = max(args.sample_counts)
    if len(wavs) < max_n:
        print(
            f"[enrollment_quality_eval] WARNING: only {len(wavs)} WAVs available; "
            f"max sample count {max_n} will use all available."
        )

    print("[enrollment_quality_eval] loading encoder...")
    from speaker.encoder import load_encoder
    load_encoder()

    # Also load any other enrolled profiles (to not change FAR baseline)
    other_names = [p.stem for p in sorted(enrolled_dir.glob("*.npy")) if p.stem != args.username]

    results = []
    for n in sorted(args.sample_counts):
        actual_n = min(n, len(wavs))
        print(f"\n[enrollment_quality_eval] enrolling '{args.username}' with {actual_n} sample(s)...")
        ok = _enroll_with_n_samples(args.username, wavs, actual_n, enrolled_dir)
        if not ok:
            print(f"  Skipped (no valid embeddings for n={actual_n})")
            continue

        enrolled_names = other_names + [args.username]
        eer = _run_eval(enrolled_dir, test_dir, imposters_dir, enrolled_names)
        print(f"  n={actual_n}: EER={eer*100:.2f}%")
        results.append({"num_samples": actual_n, "eer": eer})

    out_json = out_dir / "quality_vs_samples.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[enrollment_quality_eval] results → {out_json}")

    try:
        import matplotlib.pyplot as plt

        xs = [r["num_samples"] for r in results]
        ys = [r["eer"] * 100 for r in results]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, ys, "o-", color="steelblue", linewidth=2, markersize=8)
        ax.set_xlabel("Number of enrollment samples")
        ax.set_ylabel("EER (%)")
        ax.set_title(f"Enrollment quality vs. accuracy — {args.username}")
        ax.set_xticks(xs)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plot_path = out_dir / "eer_vs_samples.png"
        fig.savefig(plot_path, dpi=150)
        print(f"[enrollment_quality_eval] plot → {plot_path}")
    except ImportError:
        print("[enrollment_quality_eval] matplotlib not installed — skipping plot")


if __name__ == "__main__":
    main()
