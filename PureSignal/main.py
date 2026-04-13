# =============================================================================
# main.py — Pipeline orchestrator
#
# Usage:
#   export HF_TOKEN="your_token_here"
#   Set ULTRAVOX_JOIN_URL in config.py
#   python main.py
#
# Prerequisites:
#   - enroll.py must have been run for each user (profiles/<username>.npy)
# =============================================================================

import os
import sys
import signal
import requests
import config
from audio import capture, vad, features
from speaker import encoder, tracker, enrollment, policy
from llm import ultravox_client

_MAX_MULTI_USER = 10   # upper bound on multi-user selections


def log(msg: str) -> None:
    """Print only when DEBUG is enabled."""
    if config.DEBUG:
        print(msg)


def _create_ultravox_call() -> str:
    """
    Create a new Ultravox call via the REST API and return a fresh join URL.
    Reads API key from env var ULTRAVOX_API_KEY or config.ULTRAVOX_API_KEY.
    """
    api_key = os.environ.get("ULTRAVOX_API_KEY") or config.ULTRAVOX_API_KEY
    if not api_key:
        print(
            "\n[main] FATAL: No Ultravox API key found.\n"
            "  Set ULTRAVOX_API_KEY in config.py or export it as an env var.\n"
        )
        sys.exit(1)

    resp = requests.post(
        "https://api.ultravox.ai/api/calls",
        headers={"X-API-Key": api_key, "Content-Type": "application/json"},
        json={
            "systemPrompt": config.ULTRAVOX_SYSTEM_PROMPT,
            "medium": {
                "serverWebSocket": {
                    "inputSampleRate": config.ULTRAVOX_IN_RATE,
                    "outputSampleRate": config.ULTRAVOX_OUT_RATE,
                    "clientBufferSizeMs": 30000,
                }
            },
        },
        timeout=10,
    )
    if not resp.ok:
        print(f"\n[main] FATAL: Ultravox call creation failed: {resp.status_code} {resp.text}\n")
        sys.exit(1)

    join_url = resp.json()["joinUrl"]
    print(f"[main] Ultravox call created — {join_url[:60]}...")
    return join_url


def _list_available_profiles() -> list[str]:
    """Return all usernames that have a saved profile."""
    if not config.PROFILES_DIR.exists():
        return []
    return sorted(p.stem for p in config.PROFILES_DIR.glob("*.npy"))


def _select_users() -> list[str]:
    """
    Interactive prompt to select one or more users from the profiles directory.
    Returns a list of validated usernames.
    """
    available = _list_available_profiles()
    if not available:
        print(
            f"\n[main] No profiles found in '{config.PROFILES_DIR}'.\n"
            f"  Run enroll.py first to register at least one user.\n"
        )
        sys.exit(1)

    print("\n" + "=" * 50)
    print("  User Selection")
    print("=" * 50)
    print(f"  Available profiles: {', '.join(available)}")
    print("  Mode options:")
    print("    [1] Single-user")
    print("    [2] Multi-user")
    print("=" * 50)

    while True:
        mode = input("\n  Select mode [1/2]: ").strip()
        if mode in ("1", "2"):
            break
        print("  Invalid choice. Enter 1 or 2.")

    if mode == "1":
        while True:
            username = input("  Enter username: ").strip()
            if username in available:
                return [username]
            print(f"  Profile '{username}' not found. Available: {', '.join(available)}")
    else:
        # Multi-user: collect until "done" or _MAX_MULTI_USER limit
        selected: list[str] = []
        print(f"  Enter usernames one at a time. Type 'done' when finished (max {_MAX_MULTI_USER}).")
        while len(selected) < _MAX_MULTI_USER:
            username = input(f"  Username [{len(selected)+1}] (or 'done'): ").strip()
            if username.lower() == "done":
                if not selected:
                    print("  Must select at least one user.")
                    continue
                break
            if username not in available:
                print(f"  Profile '{username}' not found. Available: {', '.join(available)}")
                continue
            if username in selected:
                print(f"  '{username}' already added.")
                continue
            selected.append(username)
            print(f"  Added '{username}'.")
        return selected


def _validate_config() -> None:
    """Fail fast if critical config values are missing."""
    api_key = os.environ.get("ULTRAVOX_API_KEY") or config.ULTRAVOX_API_KEY
    join_url = config.ULTRAVOX_JOIN_URL
    if not api_key and not join_url:
        print(
            "\n[main] FATAL: Neither ULTRAVOX_API_KEY nor ULTRAVOX_JOIN_URL is set.\n"
            "  Set ULTRAVOX_API_KEY in config.py (or as env var) to auto-create calls.\n"
        )
        sys.exit(1)


def _startup(usernames: list[str]):
    """
    Initialise all components in dependency order.
    Returns the mic input stream handle.
    """
    print("\n" + "=" * 50)
    print("  Speaker Focus Pipeline")
    print("=" * 50)
    print(f"  Active users : {', '.join(usernames)}")
    print(f"  Policy mode  : {config.POLICY_MODE}")
    print(f"  Debug        : {config.DEBUG}")
    print("=" * 50 + "\n")

    # Enrollment store — load selected profiles before encoder
    enrollment.load_profiles(usernames)

    # Encoder — downloads model on first run, moves to MPS
    encoder.load_encoder()

    # Ultravox WebSocket client — create a fresh call if no URL is pinned
    join_url = config.ULTRAVOX_JOIN_URL or _create_ultravox_call()
    ultravox_client.start(join_url)

    # Mic capture — last, so everything downstream is ready
    stream = capture.start_capture()

    print("\n[main] pipeline ready — speak now. Press Ctrl+C to stop.\n")
    return stream


def _shutdown(stream) -> None:
    """Graceful teardown in reverse startup order."""
    print("\n[main] shutting down...")
    capture.stop_capture(stream)
    ultravox_client.stop()
    vad.reset()
    tracker.reset()
    print("[main] shutdown complete.\n")


def _process_loop(stream) -> None:
    """
    Main processing loop — runs until KeyboardInterrupt.
    Reads frames from capture.frame_queue and drives the full pipeline.
    """
    log("[main] processing loop started")

    while True:
        # Blocking read — waits for next 20ms frame from mic
        frame = capture.frame_queue.get()

        # VAD — returns None (mid-segment / silence) or complete segment
        segment = vad.process_frame(frame)
        if segment is None:
            continue

        log(f"[vad] segment ready — {len(segment)} samples "
            f"({len(segment) / config.SAMPLE_RATE:.2f}s)")

        # Normalize
        normalized = features.normalize(segment)

        # Embed
        embedding = encoder.embed(normalized)
        if embedding is None:
            log("[encoder] segment too short — skipped")
            continue

        # Track
        speaker_id = tracker.assign(embedding)
        log(f"[tracker] assigned → {speaker_id} "
            f"(gallery size: {len(tracker.get_gallery())})")

        # Policy gate
        passes = policy.should_pass(speaker_id, embedding)

        if passes:
            log(f"[policy] {speaker_id} → PASS — sending to Ultravox")
            ultravox_client.send_segment(normalized)
        else:
            log(f"[policy] {speaker_id} → DROP")


def main() -> None:
    _validate_config()

    usernames = _select_users()

    stream = _startup(usernames)

    # Register Ctrl+C handler for clean shutdown
    def _handle_sigint(sig, frame):
        _shutdown(stream)
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        _process_loop(stream)
    except Exception as e:
        print(f"\n[main] unhandled error: {e}")
        _shutdown(stream)
        sys.exit(1)


if __name__ == "__main__":
    main()
