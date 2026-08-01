# =============================================================================
# pipeline/session.py — Session lifecycle
#
# User selection, config validation, model loading, and shutdown.  Split out of
# main.py so that main.py can parse arguments and rebind config before any of
# this is imported — audio.* and speaker.* read config constants at import time.
# =============================================================================
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

import config
from audio import vad
from speaker import encoder, enrollment, policy, tracker
from utils.logger import close as close_logger
from utils.logger import get_log_path

_MAX_MULTI_USER = 10

# _shutdown() is reachable from both the normal exit path and an interrupt.
# Running it twice closes an already-closed stream and an already-closed log.
_shutdown_done = False


def list_available_profiles() -> list[str]:
    """Return a sorted list of enrolled usernames (stems of .npy files in profiles/)."""
    if not config.PROFILES_DIR.exists():
        return []
    return sorted(p.stem for p in config.PROFILES_DIR.glob("*.npy"))


def select_users() -> list[str]:
    """
    Interactively prompt the operator to choose one or more enrolled users.

    Presents a numbered menu of available profiles and supports both
    single-user and multi-user (up to _MAX_MULTI_USER) selection.

    Returns:
        list[str] — selected usernames that have profiles on disk.
    """
    available = list_available_profiles()
    if not available:
        print("\n[main] No profiles found. Run src/enroll.py first.\n")
        sys.exit(1)
    print("\n" + "="*50)
    print("  User Selection")
    print("="*50)
    print(f"  Available: {', '.join(available)}")
    print("    [1] Single-user  [2] Multi-user")
    print("="*50)
    while True:
        mode = input("\n  Select mode [1/2]: ").strip()
        if mode in ("1", "2"):
            break
        print("  Enter 1 or 2.")

    if mode == "1":
        while True:
            u = input("  Enter username: ").strip()
            if u in available:
                return [u]
            print(f"  Not found. Available: {', '.join(available)}")

    selected: list[str] = []
    print(
        f"Enter usernames one at a time. 'done' to finish (max {_MAX_MULTI_USER}).")
    while len(selected) < _MAX_MULTI_USER:
        u = input(f"  Username [{len(selected)+1}] (or 'done'): ").strip()
        if u.lower() == "done":
            if not selected:
                print("  Must select at least one.")
                continue
            break
        if u not in available:
            print("  Not found.")
            continue
        if u in selected:
            print("  Already added.")
            continue
        selected.append(u)
        print(f"  Added '{u}'.")
    return selected


def validate_config(no_ultravox: bool) -> None:
    """
    Fail fast if the configuration cannot produce a working session.

    Checks the policy mode here rather than on the first speech segment, so a
    typo surfaces before models load and the Ultravox call is created.
    """
    try:
        policy.validate_mode()
    except ValueError as e:
        print(f"\n[main] FATAL: {e}\n")
        sys.exit(1)

    if no_ultravox:
        return
    if not (config.ULTRAVOX_API_KEY or config.ULTRAVOX_JOIN_URL):
        print("\n[main] FATAL: No ULTRAVOX_API_KEY or ULTRAVOX_JOIN_URL set.\n")
        sys.exit(1)


def create_ultravox_call() -> str:
    """
    Create a new Ultravox call via the REST API and return the WebSocket joinUrl.

    Exits the process if the API key is missing or the request fails.
    """
    import requests
    api_key = config.ULTRAVOX_API_KEY
    if not api_key:
        print("\n[main] FATAL: No Ultravox API key found.\n")
        sys.exit(1)
    resp = requests.post(
        "https://api.ultravox.ai/api/calls",
        headers={"X-API-Key": api_key, "Content-Type": "application/json"},
        json={"systemPrompt": config.ULTRAVOX_SYSTEM_PROMPT,
              "medium": {"serverWebSocket": {
                  "inputSampleRate": config.ULTRAVOX_IN_RATE,
                  "outputSampleRate": config.ULTRAVOX_OUT_RATE,
                  "clientBufferSizeMs": 30000}}},
        timeout=10,
    )
    if not resp.ok:
        print(
            f"\n[main] FATAL: Ultravox call creation failed: {resp.status_code}\n")
        sys.exit(1)
    join_url = resp.json().get("joinUrl")
    if not join_url:
        print("\n[main] FATAL: Ultravox response missing 'joinUrl'.\n")
        sys.exit(1)
    print(f"[main] Ultravox call created — {join_url[:60]}...")
    return join_url


def get_target_embedding(usernames: list[str]) -> np.ndarray | None:
    """Load the first enrolled user's embedding as the extraction target."""
    path = config.PROFILES_DIR / f"{usernames[0]}.npy"
    if not path.exists():
        return None
    emb = np.load(path).astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm > config.NORM_FLOOR:
        emb /= norm
    return emb


def build_source(source_arg: str):
    """
    Construct the audio source object matching `source_arg`.

    Args:
        source_arg: "mic" or a path string to a WAV file.

    Returns:
        (source, frame_queue) — source is either the capture module (mic)
        or an AudioFileCapture instance (file).
    """
    from audio import capture
    capture.get_source()
    if source_arg == "mic":
        return capture, capture.frame_queue
    from audio.file_capture import AudioFileCapture
    wav_path = Path(source_arg)
    if not wav_path.exists():
        print(f"[main] FATAL: WAV file not found: '{wav_path}'")
        sys.exit(1)
    return AudioFileCapture(wav_path, capture.frame_queue), capture.frame_queue


def startup(usernames: list[str], source_arg: str, no_ultravox: bool):
    """
    Load all models, start the audio source, and optionally connect to Ultravox.

    Returns:
        (source, frame_queue, file_thread, ultravox_thread)
        file_thread is None for mic mode; ultravox_thread is None if --no-ultravox.
    """
    global _shutdown_done
    _shutdown_done = False
    mode_label = (f"extractor ({config.EXTRACTOR_MODEL})"
                  if config.EXTRACTOR_ENABLED else "gatekeeper")
    print("\n" + "="*50)
    print("  PureSignal v2")
    print("="*50)
    print(f"  Users    : {', '.join(usernames)}")
    print(f"  Source   : {source_arg}")
    print(f"  Pipeline : {mode_label}")
    print(f"  Policy   : {config.POLICY_MODE}")
    print(f"  Log      : {get_log_path()}")
    print("="*50 + "\n")
    enrollment.load_profiles(usernames)
    encoder.load_encoder()
    if config.EXTRACTOR_ENABLED:
        from audio.extractor import load_extractor
        load_extractor()
    ultravox_thread = None
    if not no_ultravox:
        from llm import ultravox_client
        join_url = config.ULTRAVOX_JOIN_URL
        if join_url:
            print(
                f"[main] reusing existing Ultravox call — {join_url[:60]}...")
        else:
            join_url = create_ultravox_call()
        ultravox_thread = ultravox_client.start(join_url)
    source, fq = build_source(source_arg)
    if source_arg == "mic":
        from audio import capture
        stream = capture.start_capture()
        print("\n[main] ready — speak now. Ctrl+C to stop.\n")
        return stream, fq, None, ultravox_thread
    else:
        file_thread = source.start()
        print("\n[main] ready — playing file. Ctrl+C to stop.\n")
        return source, fq, file_thread, ultravox_thread


def shutdown(source, source_arg: str, no_ultravox: bool, ultravox_thread=None) -> None:
    """
    Gracefully stop all pipeline components. Idempotent.

    Stops the audio source, signals Ultravox to disconnect (if active),
    resets per-session state (VAD, tracker, extractor, capture), and closes the
    session log file.
    """
    global _shutdown_done
    if _shutdown_done:
        return
    _shutdown_done = True

    print("\n[main] shutting down...")
    if source_arg == "mic":
        from audio import capture
        capture.stop_capture(source)
        # Mic mode never pushed an EOF sentinel of its own; give the consumer
        # loop the same clean exit that file mode gets.
        capture.flush_window_buffer()
    else:
        source.stop()
    if not no_ultravox:
        from llm import ultravox_client
        ultravox_client.stop()
        if ultravox_thread:
            ultravox_thread.join(timeout=3)
    vad.reset()
    tracker.reset()
    if config.EXTRACTOR_ENABLED:
        from audio.extractor import reset as ext_reset
        ext_reset()
    from audio import capture
    capture.reset()
    close_logger()
    print("[main] shutdown complete.\n")


def seed(value: int) -> None:
    """Seed numpy and PyTorch for reproducible runs."""
    import torch
    np.random.seed(value)
    torch.manual_seed(value)
    os.environ.setdefault("PYTHONHASHSEED", str(value))
