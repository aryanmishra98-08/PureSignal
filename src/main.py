"""Pipeline orchestrator.

Usage:
    python main.py [--source mic|path/to/file.wav]
                   [--config config/base.yaml]
                   [--set vad.hangover_ms=300]
                   [--no-ultravox]   # skip Ultravox (eval / offline mode)

This module is the composition root: it builds the pipeline components and
owns them for the life of a session. Ending a session means dropping the
Pipeline, not resetting shared state.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import queue
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

import numpy as np

import config
from audio import capture, features
from audio.file_capture import AudioFileCapture
from audio.vad import VoiceActivityDetector
from speaker.encoder import SpeakerEncoder
from speaker.enrollment import EnrollmentStore
from speaker.policy import PolicyGate
from speaker.tracker import SpeakerTracker
from utils.logger import close as close_logger
from utils.logger import get_log_path, get_logger

_MAX_MULTI_USER = 10

# How long the pipeline blocks on the frame queue before looping again.
# Bounded so Ctrl+C stays responsive on every platform.
_QUEUE_POLL_S = 0.5

# Ultravox buffers this much audio client-side before playback.
_CLIENT_BUFFER_MS = 30000
_API_TIMEOUT_S = 10

_log = get_logger()


@dataclass
class Pipeline:
    """The per-session components, owned together and discarded together."""

    vad: VoiceActivityDetector
    encoder: SpeakerEncoder
    tracker: SpeakerTracker
    enrollment: EnrollmentStore
    policy: PolicyGate


def _fatal(message: str) -> NoReturn:
    """Log a fatal error and exit non-zero."""
    _log.error("main", f"FATAL: {message}")
    sys.exit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PureSignal speaker-ID pipeline")
    parser.add_argument(
        "--source",
        default="mic",
        metavar="mic|PATH",
        help="Audio source: 'mic' (default) or path to a WAV file",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to YAML config file (default: config/base.yaml)",
    )
    parser.add_argument(
        "--set",
        dest="set_args",
        action="append",
        default=[],
        metavar="key=value",
        help="Override a config value, e.g. --set vad.hangover_ms=300",
    )
    parser.add_argument(
        "--no-ultravox",
        action="store_true",
        help="Skip Ultravox connection (useful for offline evaluation)",
    )
    return parser.parse_args()


def _seed(seed: int) -> None:
    """Seed every RNG the pipeline draws from."""
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Ultravox setup
# ---------------------------------------------------------------------------

def _validate_ultravox_config(no_ultravox: bool) -> None:
    """Exit early if Ultravox is required but not configured."""
    if no_ultravox:
        return
    if not config.ULTRAVOX_API_KEY and not config.ULTRAVOX_JOIN_URL:
        _fatal(
            "Neither ULTRAVOX_API_KEY nor ULTRAVOX_JOIN_URL is set.\n"
            "  Use --no-ultravox for offline evaluation."
        )


def _create_ultravox_call() -> str:
    """Create an Ultravox call over the REST API and return its join URL."""
    import requests

    if not config.ULTRAVOX_API_KEY:
        _fatal(
            "No Ultravox API key found.\n"
            "  Set ULTRAVOX_API_KEY in .env or export it as an env var."
        )

    try:
        resp = requests.post(
            "https://api.ultravox.ai/api/calls",
            headers={
                "X-API-Key": config.ULTRAVOX_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "systemPrompt": config.ULTRAVOX_SYSTEM_PROMPT,
                "medium": {
                    "serverWebSocket": {
                        "inputSampleRate": config.ULTRAVOX_IN_RATE,
                        "outputSampleRate": config.ULTRAVOX_OUT_RATE,
                        "clientBufferSizeMs": _CLIENT_BUFFER_MS,
                    }
                },
            },
            timeout=_API_TIMEOUT_S,
        )
    except requests.RequestException as e:
        _fatal(f"Could not reach the Ultravox API: {e}")

    if not resp.ok:
        _fatal(f"Ultravox call creation failed: {resp.status_code} {resp.text}")

    join_url = resp.json().get("joinUrl")
    if not join_url:
        _fatal(f"Ultravox response missing 'joinUrl'. Response: {resp.json()}")

    _log.info("main", f"Ultravox call created — {join_url[:60]}...")
    return join_url


# ---------------------------------------------------------------------------
# User selection
# ---------------------------------------------------------------------------

def _list_available_profiles() -> list[str]:
    """Return the names of every enrolled profile, sorted."""
    if not config.PROFILES_DIR.exists():
        return []
    return sorted(p.stem for p in config.PROFILES_DIR.glob("*.npy"))


def _prompt_single_user(available: list[str]) -> list[str]:
    while True:
        username = input("  Enter username: ").strip()
        if username in available:
            return [username]
        print(f"  Profile '{username}' not found. Available: {', '.join(available)}")


def _prompt_multi_user(available: list[str]) -> list[str]:
    selected: list[str] = []
    print(
        f"Enter usernames one at a time. Type 'done' when finished "
        f"(max {_MAX_MULTI_USER})."
    )
    while len(selected) < _MAX_MULTI_USER:
        username = input(f"  Username [{len(selected) + 1}] (or 'done'): ").strip()
        if username.lower() == "done":
            if selected:
                break
            print("  Must select at least one user.")
        elif username not in available:
            print(
                f"  Profile '{username}' not found. "
                f"Available: {', '.join(available)}"
            )
        elif username in selected:
            print(f"  '{username}' already added.")
        else:
            selected.append(username)
            print(f"  Added '{username}'.")
    return selected


def _select_users() -> list[str]:
    """Prompt for the profiles to activate this session."""
    available = _list_available_profiles()
    if not available:
        _fatal(
            f"No profiles found in '{config.PROFILES_DIR}'.\n"
            "  Run enroll.py first to register at least one user."
        )

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
        return _prompt_single_user(available)
    return _prompt_multi_user(available)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def _build_source(source_arg: str):
    """Return the audio source named by ``source_arg``.

    Both sources expose start()/stop() and feed ``capture.frame_queue``.
    """
    if source_arg == "mic":
        return capture.MicCapture()

    wav_path = Path(source_arg)
    if not wav_path.exists():
        _fatal(f"WAV file not found: '{wav_path}'")
    return AudioFileCapture(wav_path, capture.frame_queue)


def _build_pipeline(usernames: list[str]) -> Pipeline:
    """Construct the session's components, reading current config.

    Each component captures its tunables on construction, so --config and
    --set overrides applied earlier are picked up here.
    """
    try:
        enrollment = EnrollmentStore.load(usernames)
    except FileNotFoundError as e:
        _fatal(str(e))

    try:
        policy = PolicyGate(enrollment)
    except ValueError as e:
        _fatal(str(e))

    encoder = SpeakerEncoder()
    try:
        encoder.load()
    except OSError as e:
        _fatal(str(e))

    return Pipeline(
        vad=VoiceActivityDetector(),
        encoder=encoder,
        tracker=SpeakerTracker(),
        enrollment=enrollment,
        policy=policy,
    )


def _startup(usernames: list[str], source_arg: str, no_ultravox: bool):
    """Build the pipeline, connect Ultravox, and start the audio source.

    Returns:
        ``(source, pipeline)``.
    """
    print("\n" + "=" * 50)
    print("  Speaker Focus Pipeline")
    print("=" * 50)
    print(f"  Active users : {', '.join(usernames)}")
    print(f"  Source       : {source_arg}")
    print(f"  Policy mode  : {config.POLICY_MODE}")
    print(f"  Debug        : {config.DEBUG}")
    print(f"  Log          : {get_log_path()}")
    print("=" * 50 + "\n")

    pipeline = _build_pipeline(usernames)

    if not no_ultravox:
        from llm import ultravox_client

        join_url = config.ULTRAVOX_JOIN_URL or _create_ultravox_call()
        ultravox_client.start(join_url)

    source = _build_source(source_arg)
    try:
        source.start()
    except RuntimeError as e:
        _fatal(str(e))

    ready = "speak now" if source_arg == "mic" else "playing file"
    print(f"\n[main] pipeline ready — {ready}. Press Ctrl+C to stop.\n")
    return source, pipeline


def _shutdown(source, no_ultravox: bool) -> None:
    """Stop the source, close Ultravox, and flush the session log.

    The pipeline components need no teardown: the caller drops them.
    """
    _log.info("main", "shutting down...")
    source.stop()

    if not no_ultravox:
        from llm import ultravox_client

        ultravox_client.stop()  # signals, joins its thread, closes playback

    _log.info("main", "shutdown complete.")
    close_logger()


# ---------------------------------------------------------------------------
# Processing loop
# ---------------------------------------------------------------------------

@dataclass
class _PendingSegment:
    """A segment whose embedding is being computed on the encoder thread."""

    future: concurrent.futures.Future
    audio: np.ndarray
    vad_close_ts: float
    encoder_start_ts: float


def _submit(
    pool: concurrent.futures.Executor,
    pipeline: Pipeline,
    segment: np.ndarray,
    vad_close_ts: float,
) -> _PendingSegment:
    """Normalize a segment and queue it for embedding."""
    normalized = features.normalize(segment)
    return _PendingSegment(
        future=pool.submit(pipeline.encoder.embed, normalized),
        audio=normalized,
        vad_close_ts=vad_close_ts,
        encoder_start_ts=time.monotonic(),
    )


def _handle_pending(
    pipeline: Pipeline, pending: _PendingSegment, no_ultravox: bool
) -> None:
    """Identify a finished segment's speaker and forward it if policy allows."""
    encoder_done_ts = time.monotonic()
    embedding = pending.future.result()
    seg_duration_s = len(pending.audio) / pipeline.vad.sample_rate

    if embedding is None:
        _log.encoder_segment_too_short(segment_duration_s=seg_duration_s)
        return

    _log.encoder_embed_complete(
        latency_ms=(encoder_done_ts - pending.encoder_start_ts) * 1000,
        segment_duration_s=seg_duration_s,
    )

    known_before = pipeline.tracker.speaker_ids()
    speaker_id, best_sim = pipeline.tracker.assign_with_score(embedding)
    matched_name = pipeline.enrollment.match(embedding)
    _log.tracker_speaker_assigned(
        speaker_id=speaker_id,
        best_sim=best_sim,
        gallery_size=len(pipeline.tracker),
        is_new=speaker_id not in known_before,
    )

    policy_ts = time.monotonic()
    passes = pipeline.policy.should_pass(speaker_id, embedding)
    _log.policy_decision(
        speaker_id=speaker_id,
        matched_name=matched_name,
        decision="PASS" if passes else "DROP",
        mode=pipeline.policy.mode,
    )

    send_ts = time.monotonic()
    if passes and not no_ultravox:
        from llm import ultravox_client

        pcm_bytes = ultravox_client.send_segment(pending.audio)
        _log.ultravox_segment_sent(
            pcm_bytes=pcm_bytes,
            queue_depth=ultravox_client.audio_send_queue.qsize(),
        )
    elif passes:
        label = matched_name if matched_name is not None else speaker_id
        _log.debug("policy", f"{label} -> PASS (no-ultravox mode)")

    _log.latency_record({
        "vad_close_ts": pending.vad_close_ts,
        "encoder_start_ts": pending.encoder_start_ts,
        "encoder_done_ts": encoder_done_ts,
        "policy_ts": policy_ts,
        "send_ts": send_ts,
        "segment_duration_s": seg_duration_s,
    })


def _log_segment(segment: np.ndarray, sample_rate: int) -> None:
    _log.vad_segment_closed(
        duration_s=len(segment) / sample_rate,
        sample_count=len(segment),
    )


def _process_loop(
    frame_queue: queue.Queue, pipeline: Pipeline, no_ultravox: bool
) -> None:
    """Drive frames through VAD, the encoder, the tracker and the policy gate.

    One segment is embedded at a time; segments that arrive while the encoder
    is busy are dropped. At end-of-stream the in-flight segment and any
    segment still open inside the VAD are both processed before returning.
    """
    _log.debug("main", "processing loop started")

    sample_rate = pipeline.vad.sample_rate

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="encoder"
    ) as pool:
        pending: _PendingSegment | None = None
        leftover: np.ndarray | None = None

        while True:
            try:
                frame = frame_queue.get(timeout=_QUEUE_POLL_S)
            except queue.Empty:
                continue

            # Sentinel from AudioFileCapture signals end of file.
            if frame is None:
                leftover = pipeline.vad.flush()
                break

            vad_close_ts = time.monotonic()
            segment = pipeline.vad.process_frame(frame)
            if segment is None:
                continue

            _log_segment(segment, sample_rate)

            if pending is not None:
                if not pending.future.done():
                    _log.debug("main", "encoder busy — dropping segment")
                    continue
                _handle_pending(pipeline, pending, no_ultravox)

            pending = _submit(pool, pipeline, segment, vad_close_ts)

        # End of stream: finish the in-flight segment, then the flushed tail.
        if pending is not None:
            _handle_pending(pipeline, pending, no_ultravox)

        if leftover is not None:
            _log_segment(leftover, sample_rate)
            _handle_pending(
                pipeline,
                _submit(pool, pipeline, leftover, time.monotonic()),
                no_ultravox,
            )


def main() -> None:
    args = _parse_args()

    if args.config or args.set_args:
        config.reload(path=args.config, set_args=args.set_args)

    _seed(config.RANDOM_SEED)
    _validate_ultravox_config(args.no_ultravox)

    usernames = _select_users()
    source, pipeline = _startup(usernames, args.source, args.no_ultravox)

    try:
        _process_loop(capture.frame_queue, pipeline, args.no_ultravox)
    except KeyboardInterrupt:
        pass
    finally:
        _shutdown(source, args.no_ultravox)


if __name__ == "__main__":
    main()
