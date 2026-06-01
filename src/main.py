# =============================================================================
# main.py — Pipeline orchestrator v2
#
# Usage:
#   python src/main.py [--source mic|path/to/file.wav]
#                      [--config config/base.yaml]
#                      [--set extractor.enabled=false]   # revert to gatekeeper
#                      [--set vad.hangover_ms=300]
#                      [--no-ultravox]
#
# EXTRACTOR MODE (extractor.enabled = true, default):
#   mic → SlidingWindowBuffer → window_queue
#       → parallel SpeakerBeam (EXTRACTOR_MAX_WORKERS)
#       → ResequencingBuffer (restore order)
#       → cleaned frames → VAD → encoder → tracker → policy → Ultravox
#
# GATEKEEPER MODE (extractor.enabled = false):
#   mic → frame_queue → VAD → encoder → tracker → policy → Ultravox
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


def _parse_args():
    """Parse and return CLI arguments for the pipeline."""
    p = argparse.ArgumentParser(description="PureSignal speaker-ID pipeline v2")
    p.add_argument("--source", default="mic", metavar="mic|PATH")
    p.add_argument("--config", default=None, metavar="PATH")
    p.add_argument("--set", dest="set_args", action="append", default=[], metavar="key=value")
    p.add_argument("--no-ultravox", action="store_true")
    return p.parse_args()


def _seed(seed):
    """Seed numpy and PyTorch for reproducible runs."""
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)


def _create_ultravox_call():
    """
    Create a new Ultravox call via the REST API and return the WebSocket joinUrl.

    Exits the process if the API key is missing or the request fails.
    """
    import requests
    api_key = os.environ.get("ULTRAVOX_API_KEY") or config.ULTRAVOX_API_KEY
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
        print(f"\n[main] FATAL: Ultravox call creation failed: {resp.status_code}\n")
        sys.exit(1)
    join_url = resp.json().get("joinUrl")
    if not join_url:
        print("\n[main] FATAL: Ultravox response missing 'joinUrl'.\n")
        sys.exit(1)
    print(f"[main] Ultravox call created — {join_url[:60]}...")
    return join_url


def _list_available_profiles():
    """Return a sorted list of enrolled usernames (stems of .npy files in profiles/)."""
    if not config.PROFILES_DIR.exists():
        return []
    return sorted(p.stem for p in config.PROFILES_DIR.glob("*.npy"))


def _select_users():
    """
    Interactively prompt the operator to choose one or more enrolled users.

    Presents a numbered menu of available profiles and supports both
    single-user and multi-user (up to _MAX_MULTI_USER) selection.

    Returns:
        list[str] — selected usernames that have profiles on disk.
    """
    available = _list_available_profiles()
    if not available:
        print(f"\n[main] No profiles found. Run src/enroll.py first.\n")
        sys.exit(1)
    print("\n" + "="*50)
    print("  User Selection")
    print("="*50)
    print(f"  Available: {', '.join(available)}")
    print("    [1] Single-user  [2] Multi-user")
    print("="*50)
    while True:
        mode = input("\n  Select mode [1/2]: ").strip()
        if mode in ("1","2"): break
        print("  Enter 1 or 2.")
    if mode == "1":
        while True:
            u = input("  Enter username: ").strip()
            if u in available: return [u]
            print(f"  Not found. Available: {', '.join(available)}")
    else:
        selected = []
        print(f"Enter usernames one at a time. 'done' to finish (max {_MAX_MULTI_USER}).")
        while len(selected) < _MAX_MULTI_USER:
            u = input(f"  Username [{len(selected)+1}] (or 'done'): ").strip()
            if u.lower() == "done":
                if not selected: print("  Must select at least one."); continue
                break
            if u not in available: print(f"  Not found."); continue
            if u in selected: print(f"  Already added."); continue
            selected.append(u); print(f"  Added '{u}'.")
        return selected


def _validate_config(no_ultravox):
    """
    Fail fast if required configuration is missing.

    Exits if Ultravox mode is active but neither an API key nor a joinUrl
    has been provided.
    """
    if no_ultravox: return
    if not (os.environ.get("ULTRAVOX_API_KEY") or config.ULTRAVOX_API_KEY or config.ULTRAVOX_JOIN_URL):
        print("\n[main] FATAL: No ULTRAVOX_API_KEY or ULTRAVOX_JOIN_URL set.\n")
        sys.exit(1)


def _get_target_embedding(usernames):
    """Load the first enrolled user's embedding as SpeakerBeam fingerprint."""
    path = config.PROFILES_DIR / f"{usernames[0]}.npy"
    if not path.exists(): return None
    emb = np.load(path).astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm > config.NORM_FLOOR: emb /= norm
    return emb


def _build_source(source_arg):
    """
    Construct the audio source object matching `source_arg`.

    Args:
        source_arg: "mic" or a path string to a WAV file.

    Returns:
        (source, frame_queue) — source is either the capture module (mic)
        or an AudioFileCapture instance (file).  frame_queue is always
        capture.frame_queue regardless of mode.
    """
    from audio import capture
    if source_arg == "mic":
        return capture, capture.frame_queue
    from audio.file_capture import AudioFileCapture
    wav_path = Path(source_arg)
    if not wav_path.exists():
        print(f"[main] FATAL: WAV file not found: '{wav_path}'"); sys.exit(1)
    return AudioFileCapture(wav_path, capture.frame_queue), capture.frame_queue


def _startup(usernames, source_arg, no_ultravox):
    """
    Load all models, start the audio source, and optionally connect to Ultravox.

    Prints a startup banner, loads enrolled profiles and the speaker encoder,
    conditionally loads the extractor, then starts either the mic stream or
    the WAV file replay thread.

    Returns:
        (source, frame_queue, file_thread, ultravox_thread)
        file_thread is None for mic mode; ultravox_thread is None if --no-ultravox.
    """
    mode_label = "extractor (SpeakerBeam)" if config.EXTRACTOR_ENABLED else "gatekeeper"
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
        join_url = config.ULTRAVOX_JOIN_URL or _create_ultravox_call()
        ultravox_thread = ultravox_client.start(join_url)
    source, fq = _build_source(source_arg)
    if source_arg == "mic":
        from audio import capture
        stream = capture.start_capture()
        print("\n[main] ready — speak now. Ctrl+C to stop.\n")
        return stream, fq, None, ultravox_thread
    else:
        file_thread = source.start()
        print("\n[main] ready — playing file. Ctrl+C to stop.\n")
        return source, fq, file_thread, ultravox_thread


def _shutdown(source, source_arg, no_ultravox, ultravox_thread=None):
    """
    Gracefully stop all pipeline components.

    Stops the audio source, signals Ultravox to disconnect (if active),
    resets per-session state (VAD, tracker, extractor), and closes the
    session log file.
    """
    print("\n[main] shutting down...")
    if source_arg == "mic":
        from audio import capture
        capture.stop_capture(source)
    else:
        source.stop()
    if not no_ultravox:
        from llm import ultravox_client
        ultravox_client.stop()
        if ultravox_thread: ultravox_thread.join(timeout=3)
    vad.reset()
    tracker.reset()
    if config.EXTRACTOR_ENABLED:
        from audio.extractor import reset as ext_reset
        ext_reset()
    close_logger()
    print("[main] shutdown complete.\n")


# ---------------------------------------------------------------------------
# Shared helper: feed cleaned audio into VAD and submit encoder when segment closes
# ---------------------------------------------------------------------------

def _vad_and_encode(
    cleaned_window, frame_size, enc_pool, no_ultravox,
    pending_state,   # dict: {future, normalized, timing}
):
    """
    Slice cleaned_window into 20ms frames, feed into VAD.
    When a segment closes, submit to encoder pool.
    Updates pending_state in place.
    """
    if not no_ultravox:
        from llm import ultravox_client

    offset = 0
    while offset < len(cleaned_window):
        frame = cleaned_window[offset: offset + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))

        vad_close_ts = time.monotonic()
        segment = vad.process_frame(frame)
        offset += frame_size
        if segment is None:
            continue

        dur = len(segment) / config.SAMPLE_RATE
        _log.vad_segment_closed(duration_s=dur, sample_count=len(segment))

        # Drain completed encoder future before submitting new one
        fut = pending_state.get("future")
        if fut is not None and fut.done():
            _drain_encoder(fut, pending_state, no_ultravox)
            pending_state.clear()

        if pending_state.get("future") is not None and not pending_state["future"].done():
            if config.DEBUG: print("[main] encoder busy — dropping segment")
            continue

        normalized = features.normalize(segment)
        enc_start = time.monotonic()
        pending_state["future"] = enc_pool.submit(encoder.embed, normalized)
        pending_state["normalized"] = normalized
        pending_state["timing"] = {"vad_close_ts": vad_close_ts, "encoder_start_ts": enc_start}


def _drain_encoder(fut, pending_state, no_ultravox):
    if not no_ultravox:
        from llm import ultravox_client
    enc_done_ts = time.monotonic()
    embedding = fut.result()
    normalized = pending_state["normalized"]
    timing = pending_state["timing"]
    seg_dur = len(normalized) / config.SAMPLE_RATE

    if embedding is None:
        _log.encoder_segment_too_short(segment_duration_s=seg_dur)
        return

    _log.encoder_embed_complete(
        latency_ms=(enc_done_ts - timing["encoder_start_ts"]) * 1000,
        segment_duration_s=seg_dur,
    )
    gallery_before = tracker.get_gallery()
    speaker_id = tracker.assign(embedding)
    gallery = tracker.get_gallery()
    matched_name = enrollment.match(embedding)
    _log.tracker_speaker_assigned(
        speaker_id=speaker_id, best_sim=0.0,
        gallery_size=len(gallery), is_new=speaker_id not in gallery_before,
    )
    policy_ts = time.monotonic()
    passes = policy.should_pass(speaker_id, embedding)
    decision = "PASS" if passes else "DROP"
    _log.policy_decision(speaker_id=speaker_id, matched_name=matched_name,
                         decision=decision, mode=config.POLICY_MODE)
    send_ts = time.monotonic()
    if passes and not no_ultravox:
        ultravox_client.send_segment(normalized)
        _log.ultravox_segment_sent(
            pcm_bytes=len(normalized)*3*2,
            queue_depth=ultravox_client.audio_send_queue.qsize(),
        )
    elif passes and config.DEBUG:
        print(f"[policy] {matched_name or speaker_id} → PASS (no-ultravox)")
    _log.latency_record({
        "vad_close_ts": timing["vad_close_ts"],
        "encoder_start_ts": timing["encoder_start_ts"],
        "encoder_done_ts": enc_done_ts,
        "policy_ts": policy_ts,
        "send_ts": send_ts,
        "segment_duration_s": seg_dur,
    })


# ---------------------------------------------------------------------------
# EXTRACTOR MODE
# ---------------------------------------------------------------------------

def _process_loop_extractor(usernames, no_ultravox):
    """
    Extractor-mode pipeline loop.

    Reads (seq_no, window) tuples from capture.window_queue, submits each
    window to a SpeakerBeam thread pool, collects results through a
    ResequencingBuffer to restore original order, then feeds cleaned audio
    into _vad_and_encode().  Runs until a None sentinel arrives.
    """
    if config.DEBUG: print("[main] extractor pipeline started")
    from audio import capture as _capture
    from audio.extractor import extract
    from audio.window_buffer import ResequencingBuffer

    target_emb = _get_target_embedding(usernames)
    if target_emb is None:
        print("[main] WARNING: no target embedding — using blind separation")

    reseq = ResequencingBuffer()
    frame_size = config.FRAME_SAMPLES
    pending_state: dict = {}

    def _run_extraction(seq_no, window):
        cleaned = extract(window, target_emb)
        reseq.put(seq_no, cleaned)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=config.EXTRACTOR_MAX_WORKERS, thread_name_prefix="extractor"
    ) as ext_pool, concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="encoder"
    ) as enc_pool:

        while True:
            item = _capture.window_queue.get()

            if item is None:  # EOF sentinel
                for cleaned in reseq.drain():
                    _vad_and_encode(cleaned, frame_size, enc_pool, no_ultravox, pending_state)
                leftover = vad.flush()
                if leftover is not None:
                    _log.vad_segment_closed(
                        duration_s=len(leftover)/config.SAMPLE_RATE,
                        sample_count=len(leftover))
                # Drain final encoder future
                if pending_state.get("future") and pending_state["future"].done():
                    _drain_encoder(pending_state["future"], pending_state, no_ultravox)
                break

            seq_no, window = item
            ext_pool.submit(_run_extraction, seq_no, window)

            for cleaned in reseq.drain():
                _vad_and_encode(cleaned, frame_size, enc_pool, no_ultravox, pending_state)

            # Drain completed encoder if ready
            fut = pending_state.get("future")
            if fut is not None and fut.done():
                _drain_encoder(fut, pending_state, no_ultravox)
                pending_state.clear()


# ---------------------------------------------------------------------------
# GATEKEEPER MODE (original behaviour)
# ---------------------------------------------------------------------------

def _process_loop_gatekeeper(no_ultravox):
    """
    Gatekeeper-mode pipeline loop (original v1 behaviour).

    Reads raw 20ms frames from capture.frame_queue, passes each through the
    VAD, and submits closed speech segments to the encoder pool.  Completed
    embeddings are matched against enrolled profiles and gated by policy
    before forwarding to Ultravox.  Runs until a None sentinel arrives.
    """
    if config.DEBUG: print("[main] gatekeeper pipeline started")
    from audio import capture as _capture
    if not no_ultravox:
        from llm import ultravox_client

    with concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="encoder") as pool:
        pending = None
        pending_normalized = None
        pending_timing = None

        while True:
            frame = _capture.frame_queue.get()
            if frame is None:
                leftover = vad.flush()
                if leftover is not None:
                    _log.vad_segment_closed(
                        duration_s=len(leftover)/config.SAMPLE_RATE,
                        sample_count=len(leftover))
                break

            vad_close_ts = time.monotonic()
            segment = vad.process_frame(frame)
            if segment is None: continue

            dur = len(segment)/config.SAMPLE_RATE
            _log.vad_segment_closed(duration_s=dur, sample_count=len(segment))

            if pending is not None and not pending.done():
                if config.DEBUG: print("[main] encoder busy — dropping segment")
                continue

            if pending is not None and pending.done():
                enc_done_ts = time.monotonic()
                embedding = pending.result()
                seg_dur = len(pending_normalized)/config.SAMPLE_RATE
                if embedding is not None:
                    _log.encoder_embed_complete(
                        latency_ms=(enc_done_ts-pending_timing["encoder_start_ts"])*1000,
                        segment_duration_s=seg_dur)
                    gallery_before = tracker.get_gallery()
                    speaker_id = tracker.assign(embedding)
                    gallery = tracker.get_gallery()
                    matched_name = enrollment.match(embedding)
                    _log.tracker_speaker_assigned(speaker_id=speaker_id, best_sim=0.0,
                        gallery_size=len(gallery), is_new=speaker_id not in gallery_before)
                    policy_ts = time.monotonic()
                    passes = policy.should_pass(speaker_id, embedding)
                    _log.policy_decision(speaker_id=speaker_id, matched_name=matched_name,
                        decision="PASS" if passes else "DROP", mode=config.POLICY_MODE)
                    send_ts = time.monotonic()
                    if passes and not no_ultravox:
                        ultravox_client.send_segment(pending_normalized)
                        _log.ultravox_segment_sent(pcm_bytes=len(pending_normalized)*3*2,
                            queue_depth=ultravox_client.audio_send_queue.qsize())
                    elif passes and config.DEBUG:
                        print(f"[policy] {matched_name or speaker_id} → PASS (no-ultravox)")
                    _log.latency_record({
                        "vad_close_ts": pending_timing["vad_close_ts"],
                        "encoder_start_ts": pending_timing["encoder_start_ts"],
                        "encoder_done_ts": enc_done_ts,
                        "policy_ts": policy_ts, "send_ts": send_ts,
                        "segment_duration_s": seg_dur})
                else:
                    _log.encoder_segment_too_short(segment_duration_s=seg_dur)
                pending = pending_normalized = pending_timing = None

            normalized = features.normalize(segment)
            pending_normalized = normalized
            enc_start = time.monotonic()
            pending_timing = {"vad_close_ts": vad_close_ts, "encoder_start_ts": enc_start}
            pending = pool.submit(encoder.embed, normalized)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """
    Entry point — wire together config, signal handling, and the pipeline loop.

    Flow:
      1. Parse CLI args; reload config if --config or --set supplied.
      2. Seed RNG, validate config, select enrolled users.
      3. Register SIGINT handler for clean shutdown.
      4. Run _startup() to load models and start audio.
      5. Dispatch to _process_loop_extractor or _process_loop_gatekeeper.
      6. On exit (KeyboardInterrupt or EOF), call _shutdown().
    """
    args = _parse_args()
    if args.config or args.set_args:
        import importlib.util as _ilu
        _loader_path = Path(__file__).parent.parent / "config" / "loader.py"
        _spec = _ilu.spec_from_file_location("_config_loader", _loader_path)
        _loader_mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_loader_mod)
        config.cfg = _loader_mod.load_config(path=args.config, set_args=args.set_args)

    _seed(config.RANDOM_SEED)
    _validate_config(args.no_ultravox)
    usernames = _select_users()

    stream_holder, ult_holder = [None], [None]
    source_arg, no_ultravox = args.source, args.no_ultravox

    def _sigint(sig, frame):
        if stream_holder[0]: _shutdown(stream_holder[0], source_arg, no_ultravox, ult_holder[0])
        sys.exit(0)
    signal.signal(signal.SIGINT, _sigint)

    source, frame_queue, file_thread, ultravox_thread = _startup(usernames, source_arg, no_ultravox)
    stream_holder[0], ult_holder[0] = source, ultravox_thread

    try:
        if config.EXTRACTOR_ENABLED:
            _process_loop_extractor(usernames, no_ultravox)
        else:
            _process_loop_gatekeeper(no_ultravox)
    except KeyboardInterrupt:
        pass
    finally:
        _shutdown(source, source_arg, no_ultravox, ultravox_thread)
        if file_thread: file_thread.join(timeout=2)
        sys.exit(0)


if __name__ == "__main__":
    main()
