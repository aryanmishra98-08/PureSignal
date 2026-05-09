# =============================================================================
# enroll.py — Standalone enrollment script
#             Run once per user before main.py to register their voice
#
# Usage:
#   export HF_TOKEN="your_token_here"
#   python enroll.py [--samples N]
#
# Output:
#   profiles/<username>.npy
#   profiles/<username>_meta.json
# =============================================================================

import argparse
import sys
import time

import config
import numpy as np
import sounddevice as sd
from audio import features
from speaker import encoder
from speaker.enrollment import validate_enrollment_quality, save_with_metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Speaker enrollment")
    parser.add_argument(
        "--samples",
        type=int,
        default=config.ENROLLMENT_NUM_SAMPLES,
        metavar="N",
        help=f"Number of recording samples to average (default: {config.ENROLLMENT_NUM_SAMPLES})",
    )
    return parser.parse_args()


def record(duration_s: int, sample_index: int, total: int) -> np.ndarray:
    """Blocking mic record for duration_s seconds. Returns float32 array."""
    total_samples = duration_s * config.SAMPLE_RATE
    print(f"\n[enroll] Sample {sample_index}/{total} — Recording for {duration_s}s. Speak naturally...\n")

    try:
        buffer = sd.rec(
            frames=total_samples,
            samplerate=config.SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
    except sd.PortAudioError as e:
        print(f"[enroll] ERROR: Could not open microphone: {e}")
        sys.exit(1)

    for remaining in range(duration_s, 0, -1):
        print(f"  {remaining}s remaining...", end="\r")
        time.sleep(1)

    sd.wait()
    print("\n[enroll] Recording complete.")
    return buffer[:, 0]


def prompt_username() -> str:
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


def main() -> None:
    args = _parse_args()
    num_samples = max(1, args.samples)

    print("=" * 50)
    print("  Speaker Enrollment")
    print(f"  Samples: {num_samples}")
    print("=" * 50)

    username = prompt_username()
    output_path = config.PROFILES_DIR / f"{username}.npy"

    if output_path.exists():
        overwrite = (
            input(f"\n  Profile '{username}' already exists. Overwrite? [y/N]: ")
            .strip()
            .lower()
        )
        if overwrite != "y":
            print("[enroll] Aborted.")
            sys.exit(0)

    print("\n[enroll] Loading encoder (first run downloads model)...")
    try:
        encoder.load_encoder()
    except EnvironmentError as e:
        print(str(e))
        sys.exit(1)

    # Collect num_samples embeddings
    embeddings: list[np.ndarray] = []

    for i in range(1, num_samples + 1):
        while True:
            audio = record(config.ENROLLMENT_DURATION_S, sample_index=i, total=num_samples)
            normalized = features.normalize(audio)

            print(f"[enroll] Extracting embedding for sample {i}/{num_samples}...")
            emb = encoder.embed(normalized)

            if emb is None:
                print(
                    "\n[enroll] WARNING: embedding returned None (recording too short or silent).\n"
                    "  Please try again and speak clearly throughout.\n"
                )
                retry = input("  Retry this sample? [Y/n]: ").strip().lower()
                if retry != "n":
                    continue
                else:
                    print("[enroll] Skipping this sample.")
            else:
                # Quality check
                quality = validate_enrollment_quality(emb)
                print(
                    f"  Quality — norm: {quality['norm']:.4f}, "
                    f"entropy: {quality['entropy']:.4f}, "
                    f"pass: {quality['quality_pass']}"
                )
                if not quality["quality_pass"]:
                    print(
                        "\n[enroll] WARNING: Low-quality embedding detected.\n"
                        "  Possible causes: background noise, too-quiet recording, or short silence.\n"
                    )
                    retry = input("  Re-record this sample? [Y/n]: ").strip().lower()
                    if retry != "n":
                        continue
                embeddings.append(emb)
            break

    if not embeddings:
        print("\n[enroll] FAILED: no valid embeddings collected.")
        sys.exit(1)

    # Average and re-normalize
    profile_embedding = np.mean(embeddings, axis=0).astype(np.float32)
    norm = np.linalg.norm(profile_embedding)
    if norm > config.NORM_FLOOR:
        profile_embedding /= norm

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
        print(f"[enroll] ERROR: Failed to save profile: {e}")
        sys.exit(1)

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


if __name__ == "__main__":
    main()
