# =============================================================================
# main.py — Pipeline orchestrator
#
# Usage:
#   python main.py [--source mic|path/to/file.wav]
#                  [--config config/base.yaml]
#                  [--set vad.hangover_ms=300]
#                  [--no-ultravox]   # skip Ultravox (eval / offline mode)
# =============================================================================

import argparse
import concurrent.futures
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

import config
from audio import features, vad
from speaker import encoder, enrollment, policy, tracker
from utils.logger import get_logger, get_log_path, close as close_logger

_MAX_MULTI_USER = 10
_log = get_logger()


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
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)


def _create_ultravox_call() -> str:
    import requests

    api_key = os.environ.get("ULTRAVOX_API_KEY") or config.ULTRAVOX_API_KEY
    if not api_key:
        print(
            "\n[main] FATAL: No Ultravox API key found.\n"
            "  Set ULTRAVOX_API_KEY in keys/.env or export it as an env var.\n"
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
        print(
            f"\n[main] FATAL: Ultravox call creation failed: "
            f"{resp.status_code} {resp.text}\n"
        )
        sys.exit(1)

    data = resp.json()
    join_url = data.get("joinUrl")
    if not join_url:
        print(f"\n[main] FATAL: Ultravox response missing 'joinUrl'. Response: {data}\n")
        sys.exit(1)
    print(f"[main] Ultravox call created — {join_url[:60]}...")
    return join_url


def _list_available_profiles() -> list[str]:
    if not config.PROFILES_DIR.exists():
        return []
    return sorted(p.stem for p in config.PROFILES_DIR.glob("*.npy"))


def _select_users() -> list[str]:
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
        selected: list[str] = []
        print(
            f"Enter usernames one at a time. Type 'done' when finished "
            f"(max {_MAX_MULTI_USER})."
        )
        while len(selected) < _MAX_MULTI_USER:
            username = input(f"  Username [{len(selected) + 1}] (or 'done'): ").strip()
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


def _validate_config(no_ultravox: bool) -> None:
    if no_ultravox:
        return
    api_key = os.environ.get("ULTRAVOX_API_KEY") or config.ULTRAVOX_API_KEY
    join_url = config.ULTRAVOX_JOIN_URL
    if not api_key and not join_url:
        print(
            "\n[main] FATAL: Neither ULTRAVOX_API_KEY nor ULTRAVOX_JOIN_URL is set.\n"
            "  Use --no-ultravox for offline evaluation.\n"
        )
        sys.exit(1)


def _build_source(source_arg: str):
    """Return a (capture_module_or_object, frame_queue) pair."""
    if source_arg == "mic":
        from audio import capture
        return capture, capture.frame_queue
    else:
        from audio import capture
        from audio.file_capture import AudioFileCapture
        wav_path = Path(source_arg)
        if not wav_path.exists():
            print(f"[main] FATAL: WAV file not found: '{wav_path}'")
            sys.exit(1)
        file_src = AudioFileCapture(wav_path, capture.frame_queue)
        return file_src, capture.frame_queue


def _startup(usernames: list[str], source_arg: str, no_ultravox: bool):
    print("\n" + "=" * 50)
    print("  Speaker Focus Pipeline")
    print("=" * 50)
    print(f"  Active users : {', '.join(usernames)}")
    print(f"  Source       : {source_arg}")
    print(f"  Policy mode  : {config.POLICY_MODE}")
    print(f"  Debug        : {config.DEBUG}")
    print(f"  Log          : {get_log_path()}")
    print("=" * 50 + "\n")

    enrollment.load_profiles(usernames)
    encoder.load_encoder()

    ultravox_thread = None
    if not no_ultravox:
        from llm import ultravox_client
        join_url = config.ULTRAVOX_JOIN_URL or _create_ultravox_call()
        ultravox_thread = ultravox_client.start(join_url)

    source, fq = _build_source(source_arg)

    if source_arg == "mic":
        from audio import capture
        stream = capture.start_capture()
        print("\n[main] pipeline ready — speak now. Press Ctrl+C to stop.\n")
        return stream, fq, None, ultravox_thread
    else:
        file_thread = source.start()
        print("\n[main] pipeline ready — playing file. Press Ctrl+C to stop.\n")
        return source, fq, file_thread, ultravox_thread


def _shutdown(
    source, source_arg: str, no_ultravox: bool, ultravox_thread=None
) -> None:
    print("\n[main] shutting down...")
    if source_arg == "mic":
        from audio import capture
        capture.stop_capture(source)
    else:
        source.stop()

    if not no_ultravox:
        from llm import ultravox_client
        ultravox_client.stop()
        if ultravox_thread is not None:
            ultravox_thread.join(timeout=3)

    vad.reset()
    tracker.reset()
    close_logger()
    print("[main] shutdown complete.\n")


def _process_loop(frame_queue, no_ultravox: bool) -> None:
    if config.DEBUG:
        print("[main] processing loop started")

    if not no_ultravox:
        from llm import ultravox_client

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="encoder"
    ) as pool:
        pending: concurrent.futures.Future | None = None
        pending_normalized: np.ndarray | None = None
        pending_timing: dict | None = None

        while True:
            frame = frame_queue.get()

            # Sentinel from AudioFileCapture signals end of file
            if frame is None:
                # Flush any buffered speech that never saw trailing silence
                leftover = vad.flush()
                if leftover is not None:
                    duration_s = len(leftover) / config.SAMPLE_RATE
                    _log.vad_segment_closed(duration_s=duration_s, sample_count=len(leftover))
                break

            vad_close_ts = time.monotonic()
            segment = vad.process_frame(frame)
            if segment is None:
                continue

            duration_s = len(segment) / config.SAMPLE_RATE
            _log.vad_segment_closed(duration_s=duration_s, sample_count=len(segment))

            if pending is not None and not pending.done():
                if config.DEBUG:
                    print("[main] encoder busy — dropping segment")
                continue

            if pending is not None and pending.done():
                encoder_done_ts = time.monotonic()
                embedding = pending.result()
                seg_duration_s = len(pending_normalized) / config.SAMPLE_RATE

                if embedding is not None:
                    _log.encoder_embed_complete(
                        latency_ms=(encoder_done_ts - pending_timing["encoder_start_ts"]) * 1000,
                        segment_duration_s=seg_duration_s,
                    )

                    gallery_before = tracker.get_gallery()
                    speaker_id = tracker.assign(embedding)
                    gallery = tracker.get_gallery()
                    matched_name = enrollment.match(embedding)
                    display_label = matched_name if matched_name is not None else speaker_id

                    _log.tracker_speaker_assigned(
                        speaker_id=speaker_id,
                        best_sim=0.0,  # tracker.assign doesn't expose best_sim externally
                        gallery_size=len(gallery),
                        is_new=speaker_id not in gallery_before,
                    )

                    policy_ts = time.monotonic()
                    passes = policy.should_pass(speaker_id, embedding)
                    decision = "PASS" if passes else "DROP"
                    _log.policy_decision(
                        speaker_id=speaker_id,
                        matched_name=matched_name,
                        decision=decision,
                        mode=config.POLICY_MODE,
                    )

                    send_ts = time.monotonic()
                    if passes and not no_ultravox:
                        ultravox_client.send_segment(pending_normalized)
                        # 16kHz float32 → 48kHz int16: 3× samples, 2 bytes each
                        _log.ultravox_segment_sent(
                            pcm_bytes=len(pending_normalized) * 3 * 2,
                            queue_depth=ultravox_client.audio_send_queue.qsize(),
                        )
                    elif passes:
                        if config.DEBUG:
                            print(f"[policy] {display_label} → PASS (no-ultravox mode)")

                    # Write latency record
                    _log.latency_record({
                        "vad_close_ts": pending_timing["vad_close_ts"],
                        "encoder_start_ts": pending_timing["encoder_start_ts"],
                        "encoder_done_ts": encoder_done_ts,
                        "policy_ts": policy_ts,
                        "send_ts": send_ts,
                        "segment_duration_s": seg_duration_s,
                    })
                else:
                    _log.encoder_segment_too_short(segment_duration_s=seg_duration_s)

                pending = None
                pending_normalized = None
                pending_timing = None

            normalized = features.normalize(segment)
            pending_normalized = normalized
            encoder_start_ts = time.monotonic()
            pending_timing = {
                "vad_close_ts": vad_close_ts,
                "encoder_start_ts": encoder_start_ts,
            }
            pending = pool.submit(encoder.embed, normalized)


def main() -> None:
    args = _parse_args()

    # Re-load config if custom path or --set overrides were provided
    if args.config or args.set_args:
        import importlib.util as _ilu
        _loader_path = Path(__file__).parent.parent / "config" / "loader.py"
        _spec = _ilu.spec_from_file_location("_config_loader", _loader_path)
        _loader_mod = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
        _spec.loader.exec_module(_loader_mod)  # type: ignore[union-attr]
        new_cfg = _loader_mod.load_config(path=args.config, set_args=args.set_args)
        # Patch the singleton on the already-imported shim so sub-modules see overrides
        config.cfg = new_cfg

    _seed(config.RANDOM_SEED)
    _validate_config(args.no_ultravox)

    usernames = _select_users()

    stream_holder: list = [None]
    source_arg = args.source
    no_ultravox = args.no_ultravox

    ultravox_thread_holder: list = [None]

    def _handle_sigint(sig, frame):
        if stream_holder[0] is not None:
            _shutdown(stream_holder[0], source_arg, no_ultravox, ultravox_thread_holder[0])
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_sigint)

    source, frame_queue, file_thread, ultravox_thread = _startup(
        usernames, source_arg, no_ultravox
    )
    stream_holder[0] = source
    ultravox_thread_holder[0] = ultravox_thread

    try:
        _process_loop(frame_queue, no_ultravox)
    except KeyboardInterrupt:
        pass
    finally:
        _shutdown(source, source_arg, no_ultravox, ultravox_thread)
        if file_thread is not None:
            file_thread.join(timeout=2)
        sys.exit(0)


if __name__ == "__main__":
    main()
