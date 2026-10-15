"""Standalone enrollment script.

Run once per user, before main.py, to register their voice.

Usage:
    export HF_TOKEN="your_token_here"
    python enroll.py [--samples N]

Output:
    profiles/<username>.npy
    profiles/<username>_meta.json
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import NoReturn

import numpy as np
import sounddevice as sd

import config
from audio import features
from speaker.encoder import SpeakerEncoder
from speaker.enrollment import save_with_metadata, validate_enrollment_quality
from utils.logger import close as close_logger
from utils.logger import get_logger

_log = get_logger()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Speaker enrollment")
    parser.add_argument(
        "--samples",
        type=int,
        default=config.ENROLLMENT_NUM_SAMPLES,
        metavar="N",
        help=(
            "Number of recording samples to average "
            f"(default: {config.ENROLLMENT_NUM_SAMPLES})"
        ),
    )
    return parser.parse_args()


def _fatal(message: str) -> NoReturn:
    """Log an error and exit non-zero."""
    _log.error("enroll", message)
    close_logger()
    sys.exit(1)


def _confirm(prompt: str) -> bool:
    """Ask a yes/no question that defaults to yes."""
    return input(prompt).strip().lower() != "n"


def record(duration_s: int, sample_index: int, total: int) -> np.ndarray:
    """Record from the microphone for ``duration_s`` seconds.

    Returns:
        A 1-D float32 array at ``config.SAMPLE_RATE``.
    """
    print(
        f"\n[enroll] Sample {sample_index}/{total} — "
        f"Recording for {duration_s}s. Speak naturally...\n"
    )

    try:
        buffer = sd.rec(
            frames=duration_s * config.SAMPLE_RATE,
            samplerate=config.SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
    except sd.PortAudioError as e:
        _fatal(f"Could not open microphone: {e}")

    for remaining in range(duration_s, 0, -1):
        print(f"  {remaining}s remaining...", end="\r")
        time.sleep(1)

    sd.wait()
    print("\n[enroll] Recording complete.")
    return buffer[:, 0]


def prompt_username() -> str:
    """Prompt for a profile name, stripping anything unsafe for a filename."""
    while True:
        username = input("\n[enroll] Enter username for this profile: ").strip()
        if not username:
            print("  Username cannot be empty. Try again.")
            continue

        sanitized = "".join(c for c in username if c.isalnum() or c in "-_")
        if not sanitized:
            print("  Username must contain alphanumeric characters. Try again.")
            continue
        if sanitized != username:
            print(f"  Username sanitized to: '{sanitized}'")
        return sanitized


def capture_sample(
    encoder: SpeakerEncoder, index: int, total: int
) -> np.ndarray | None:
    """Record one sample and embed it, re-recording while the user asks to.

    Returns:
        The embedding, or None if the user chose to skip this sample.
    """
    while True:
        audio = record(config.ENROLLMENT_DURATION_S, sample_index=index, total=total)

        print(f"[enroll] Extracting embedding for sample {index}/{total}...")
        embedding = encoder.embed(features.normalize(audio))

        if embedding is None:
            print(
                "\n[enroll] WARNING: embedding returned None "
                "(recording too short or silent).\n"
                "  Please try again and speak clearly throughout.\n"
            )
            if _confirm("  Retry this sample? [Y/n]: "):
                continue
            print("[enroll] Skipping this sample.")
            return None

        quality = validate_enrollment_quality(embedding)
        print(
            f"  Quality — norm: {quality['norm']:.4f}, "
            f"entropy: {quality['entropy']:.4f}, "
            f"pass: {quality['quality_pass']}"
        )
        if not quality["quality_pass"]:
            print(
                "\n[enroll] WARNING: Low-quality embedding detected.\n"
                "  Possible causes: background noise, too-quiet recording, "
                "or short silence.\n"
            )
            if _confirm("  Re-record this sample? [Y/n]: "):
                continue

        return embedding


def average_embeddings(embeddings: list[np.ndarray]) -> np.ndarray:
    """Average sample embeddings into one L2-normalized profile."""
    profile = np.mean(embeddings, axis=0).astype(np.float32)
    norm = np.linalg.norm(profile)
    if norm > config.NORM_FLOOR:
        profile /= norm
    return profile


def main() -> None:
    args = _parse_args()
    num_samples = max(1, args.samples)

    print("=" * 50)
    print("  Speaker Enrollment")
    print(f"  Samples: {num_samples}")
    print("=" * 50)

    username = prompt_username()
    if (config.PROFILES_DIR / f"{username}.npy").exists():
        overwrite = input(
            f"\n  Profile '{username}' already exists. Overwrite? [y/N]: "
        ).strip().lower()
        if overwrite != "y":
            print("[enroll] Aborted.")
            close_logger()
            sys.exit(0)

    _log.info("enroll", "Loading encoder (first run downloads model)...")
    encoder = SpeakerEncoder()
    try:
        encoder.load()
    except OSError as e:
        _fatal(str(e))

    embeddings = [
        embedding
        for i in range(1, num_samples + 1)
        if (embedding := capture_sample(encoder, i, num_samples)) is not None
    ]

    if not embeddings:
        _fatal("no valid embeddings collected.")

    profile_embedding = average_embeddings(embeddings)
    final_quality = validate_enrollment_quality(profile_embedding)

    try:
        save_with_metadata(
            embedding=profile_embedding,
            username=username,
            profiles_dir=config.PROFILES_DIR,
            num_samples=len(embeddings),
            quality=final_quality,
        )
    except OSError as e:
        _fatal(f"Failed to save profile: {e}")

    print(
        f"\n[enroll] Enrollment complete.\n"
        f"  Username        : {username}\n"
        f"  Samples used    : {len(embeddings)}/{num_samples}\n"
        f"  Embedding shape : {profile_embedding.shape}\n"
        f"  Quality (final) : norm={final_quality['norm']:.4f}, "
        f"entropy={final_quality['entropy']:.4f}, "
        f"pass={final_quality['quality_pass']}\n"
        f"  Saved to        : {config.PROFILES_DIR / username}.npy\n"
        f"\n  You can now run main.py.\n"
    )
    close_logger()


if __name__ == "__main__":
    main()
