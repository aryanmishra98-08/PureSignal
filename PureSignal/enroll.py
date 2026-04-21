# =============================================================================
# enroll.py — Standalone enrollment script
#             Run once per user before main.py to register their voice
#
# Usage:
#   export HF_TOKEN="your_token_here"
#   python enroll.py
#
# Output:
#   profiles/<username>.npy
# =============================================================================

import sys
import time
from pathlib import Path
import config
import numpy as np
import sounddevice as sd
from audio import features
from dotenv import load_dotenv
from speaker import encoder

# Load secrets from keys/.env before any module that needs env vars
load_dotenv(Path(__file__).parent / "keys" / ".env")


def record(duration_s: int) -> np.ndarray:
    """Blocking mic record for duration_s seconds. Returns float32 array."""
    total_samples = duration_s * config.SAMPLE_RATE
    print(f"\n[enroll] Recording for {duration_s}s — speak naturally now...\n")

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

    # Print a simple countdown
    for remaining in range(duration_s, 0, -1):
        print(f"  {remaining}s remaining...", end="\r")
        time.sleep(1)

    sd.wait()
    print("\n[enroll] Recording complete.")
    return buffer[:, 0]  # flatten to 1D


def prompt_username() -> str:
    """Prompt the user for a username and return it sanitized."""
    while True:
        username = input("\n[enroll] Enter username for this profile: ").strip()
        if not username:
            print("  Username cannot be empty. Try again.")
            continue
        # Restrict to safe filename characters
        sanitized = "".join(c for c in username if c.isalnum() or c in "-_")
        if not sanitized:
            print("  Username must contain alphanumeric characters. Try again.")
            continue
        if sanitized != username:
            print(f"  Username sanitized to: '{sanitized}'")
        return sanitized


def main() -> None:
    print("=" * 50)
    print("  Speaker Enrollment")
    print("=" * 50)

    # Step 1 — Get username
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

    # Step 2 — Load encoder
    print("\n[enroll] Loading encoder (first run downloads model)...")
    try:
        encoder.load_encoder()
    except EnvironmentError as e:
        print(str(e))
        sys.exit(1)

    # Step 3 — Record
    audio = record(config.ENROLLMENT_DURATION_S)

    # Step 4 — Normalize
    normalized = features.normalize(audio)

    # Step 5 — Extract embedding
    print("[enroll] Extracting speaker embedding...")
    embedding = encoder.embed(normalized)

    if embedding is None:
        print(
            "\n[enroll] FAILED: embedding returned None.\n"
            "  Possible causes:\n"
            "  - Recording was too short or silent\n"
            "  - Mic level too low\n"
            "  Try again and speak clearly throughout the recording."
        )
        sys.exit(1)

    # Step 6 — Save to profiles/<username>.npy
    try:
        config.PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        np.save(output_path, embedding)
    except OSError as e:
        print(f"[enroll] ERROR: Failed to save profile to '{output_path}': {e}")
        sys.exit(1)
    print(
        f"\n[enroll] Enrollment complete.\n"
        f"  Username        : {username}\n"
        f"  Embedding shape : {embedding.shape}\n"
        f"  Saved to        : {output_path}\n"
        f"\n  You can now run main.py.\n"
    )


if __name__ == "__main__":
    main()
