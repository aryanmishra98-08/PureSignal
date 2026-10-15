"""EER vs. number of enrollment samples.

Re-enrolls the same speaker from 1, 2, 3, 5 samples and measures EER for each,
then plots EER against the sample count.

Expects the same test/impostor directory layout as run_speaker_eval.py.

Usage:
    python eval/enrollment_quality_eval.py \\
        --speaker-wavs  eval/enrollment_wavs/alice \\
        --enrolled-dir  profiles \\
        --test-dir      eval/test_audio \\
        --imposters-dir eval/imposters \\
        --username      alice \\
        --out           eval/results/quality_vs_samples
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from common import collect_wavs, embed_wav, get_pyplot, load_encoder
from metrics import compute_eer

_TAG = "enrollment_quality_eval"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EER vs enrollment samples curve")
    parser.add_argument(
        "--speaker-wavs",
        required=True,
        help="Directory with the target speaker's enrollment WAVs",
    )
    parser.add_argument(
        "--enrolled-dir", required=True, help="profiles/ directory (written to)"
    )
    parser.add_argument("--test-dir", required=True, help="test_audio/ directory")
    parser.add_argument(
        "--imposters-dir", required=True, help="imposters/ directory"
    )
    parser.add_argument(
        "--username", required=True, help="Username of the target speaker"
    )
    parser.add_argument(
        "--sample-counts",
        nargs="+",
        type=int,
        default=[1, 2, 3, 5],
        help="Numbers of samples to test (default: 1 2 3 5)",
    )
    parser.add_argument(
        "--out", default="eval/results/quality_vs_samples", help="Output directory"
    )
    return parser.parse_args()


def _enroll_with_n_samples(
    encoder, username: str, wavs: list[Path], n: int, profiles_dir: Path
) -> bool:
    """Average the first ``n`` embeddings into a profile and save it.

    Returns:
        False if none of the selected WAVs could be embedded.
    """
    import config
    from speaker.enrollment import save_with_metadata, validate_enrollment_quality

    embeddings = [
        e for e in (embed_wav(encoder, p) for p in wavs[:n]) if e is not None
    ]
    if not embeddings:
        return False

    profile = np.mean(embeddings, axis=0).astype(np.float32)
    norm = np.linalg.norm(profile)
    if norm > config.NORM_FLOOR:
        profile /= norm

    save_with_metadata(
        profile,
        username,
        profiles_dir,
        num_samples=len(embeddings),
        quality=validate_enrollment_quality(profile),
    )
    return True


def _measure_eer(
    encoder,
    test_dir: Path,
    imposters_dir: Path,
    enrolled_names: list[str],
    profiles_dir: Path,
) -> float:
    """Load the named profiles and return EER over the test and impostor sets."""
    from speaker.enrollment import EnrollmentStore

    store = EnrollmentStore.load(enrolled_names, profiles_dir=profiles_dir)
    genuine = set(enrolled_names)

    scores: list[float] = []
    labels: list[int] = []

    for directory, is_genuine_set in ((test_dir, True), (imposters_dir, False)):
        for speaker_name, wavs in collect_wavs(directory).items():
            for wav_path in wavs:
                embedding = embed_wav(encoder, wav_path)
                if embedding is None:
                    continue
                _, best_sim = store.match_with_score(embedding)
                scores.append(best_sim)
                labels.append(
                    1 if is_genuine_set and speaker_name in genuine else 0
                )

    if not scores:
        return float("nan")

    eer_val, _ = compute_eer(scores, labels)
    return eer_val


def _plot_curve(results: list[dict], username: str, out_path: Path) -> None:
    plt = get_pyplot(_TAG)
    if plt is None:
        return

    xs = [r["num_samples"] for r in results]
    ys = [r["eer"] * 100 for r in results]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, "o-", color="steelblue", linewidth=2, markersize=8)
    ax.set_xlabel("Number of enrollment samples")
    ax.set_ylabel("EER (%)")
    ax.set_title(f"Enrollment quality vs. accuracy — {username}")
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[{_TAG}] plot -> {out_path}")


def main() -> None:
    args = _parse_args()

    enrolled_dir = Path(args.enrolled_dir)
    test_dir = Path(args.test_dir)
    imposters_dir = Path(args.imposters_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(Path(args.speaker_wavs).glob("*.wav"))
    if not wavs:
        print(f"[{_TAG}] ERROR: no WAVs in {args.speaker_wavs}")
        sys.exit(1)

    max_n = max(args.sample_counts)
    if len(wavs) < max_n:
        print(
            f"[{_TAG}] WARNING: only {len(wavs)} WAVs available; "
            f"max sample count {max_n} will use all available."
        )

    print(f"[{_TAG}] loading encoder...")
    encoder = load_encoder()

    # Keep the other enrolled profiles loaded so the FAR baseline is unchanged.
    other_names = [
        p.stem for p in sorted(enrolled_dir.glob("*.npy")) if p.stem != args.username
    ]

    results: list[dict] = []
    for n in sorted(args.sample_counts):
        actual_n = min(n, len(wavs))
        print(f"\n[{_TAG}] enrolling '{args.username}' with {actual_n} sample(s)...")

        if not _enroll_with_n_samples(
            encoder, args.username, wavs, actual_n, enrolled_dir
        ):
            print(f"  Skipped (no valid embeddings for n={actual_n})")
            continue

        eer = _measure_eer(
            encoder,
            test_dir,
            imposters_dir,
            [*other_names, args.username],
            enrolled_dir,
        )
        print(f"  n={actual_n}: EER={eer * 100:.2f}%")
        results.append({"num_samples": actual_n, "eer": eer})

    out_json = out_dir / "quality_vs_samples.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[{_TAG}] results -> {out_json}")

    if results:
        _plot_curve(results, args.username, out_dir / "eer_vs_samples.png")


if __name__ == "__main__":
    main()
