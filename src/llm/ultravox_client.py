# =============================================================================
# llm/ultravox_client.py — Fixie.ai Ultravox WebSocket client
#
# Responsibilities:
#   1. Connect to the Ultravox joinUrl
#   2. Send 20ms PCM chunks (speech segments + silence padding)
#   3. Receive audio frames from Ultravox and play via sounddevice
#
# Threading model:
#   - _send_loop runs as an async task, reads from audio_send_queue
#   - _receive_loop runs as an async task; it only ENQUEUES playback audio
#   - _playback_worker runs on its own thread and does the blocking device write
#   - main.py feeds audio_send_queue after the policy gate passes
#
# Playback must not happen on the event loop: sd.RawOutputStream.write blocks
# until the device has buffer space, which stalls _send_loop too, degrading
# uplink pacing exactly when the AI is talking.
# =============================================================================

import asyncio
import json
import queue
import threading

import numpy as np
import sounddevice as sd
import websockets
import websockets.exceptions

import config
from audio.resampler import silence_frame_48k, to_48k_pcm
from utils.logger import get_logger

_log = get_logger()

# Queue fed by main.py — holds resampled PCM bytes (full segments)
audio_send_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=50)

# Queue fed by _receive_loop, consumed by the playback thread
_playback_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=200)

# Retry constants for WebSocket connection
_MAX_RETRIES = 3
_RETRY_DELAY_S = 2.0

# How long the receive loop waits for a message before re-checking _running
_RECV_POLL_S = 0.5

# Thread safety for shared state
_lock = threading.Lock()

# Playback output stream — opened once, written to continuously
_output_stream: sd.RawOutputStream | None = None
_playback_thread: threading.Thread | None = None

# Internal flag to signal shutdown
_running = False

# 20ms frame size at 48kHz in bytes (960 samples * 2 bytes per int16 sample)
_FRAME_BYTES = int(config.ULTRAVOX_IN_RATE *
                   config.ULTRAVOX_CHUNK_MS / 1000) * 2


def _open_output_stream() -> sd.RawOutputStream:
    """Open sounddevice output stream for Ultravox audio playback."""
    stream = sd.RawOutputStream(
        samplerate=config.ULTRAVOX_OUT_RATE,
        channels=1,
        dtype="int16",
        blocksize=int(config.ULTRAVOX_OUT_RATE *
                      config.ULTRAVOX_CHUNK_MS / 1000),
    )
    stream.start()
    print(f"[ultravox] playback stream open — {config.ULTRAVOX_OUT_RATE}Hz")
    return stream


def _playback_worker() -> None:
    """
    Drain _playback_queue to the output device.

    Runs on its own thread so the blocking device write never touches the
    asyncio event loop.  Exits on the None sentinel.
    """
    while True:
        chunk = _playback_queue.get()
        if chunk is None:
            break
        stream = _output_stream
        if stream is None:
            continue
        try:
            stream.write(chunk)
        except sd.PortAudioError as e:
            print(f"[ultravox] playback error: {e}")


def _clear_playback() -> None:
    """Discard queued playback audio in response to playbackClearBuffer."""
    dropped = 0
    while True:
        try:
            _playback_queue.get_nowait()
            dropped += 1
        except queue.Empty:
            break
    if dropped:
        _log("ultravox", "playback_cleared", frames_discarded=dropped)


async def _send_loop(ws) -> None:
    """
    Continuously sends 20ms PCM frames to Ultravox WebSocket.
    Reads complete segments from audio_send_queue.
    Sends silence frames between segments to maintain stream continuity.
    """
    silence = silence_frame_48k()

    while _running:
        try:
            # Non-blocking check for a new segment
            segment_bytes = audio_send_queue.get_nowait()

            if segment_bytes is None:
                # Shutdown signal
                break

            # Slice segment into 20ms frames and send sequentially
            offset = 0
            while offset < len(segment_bytes):
                frame = segment_bytes[offset: offset + _FRAME_BYTES]

                # Pad last frame if shorter than 20ms
                if len(frame) < _FRAME_BYTES:
                    frame = frame + bytes(_FRAME_BYTES - len(frame))

                await ws.send(frame)
                offset += _FRAME_BYTES

                # Maintain 20ms cadence
                await asyncio.sleep(config.ULTRAVOX_CHUNK_MS / 1000)

            # The pacing above streams a segment at 1x realtime, so this is the
            # moment the segment has actually left the process. Without it the
            # latency record stops at enqueue and hides the send cost.
            _log("ultravox", "segment_flushed",
                 pcm_bytes=len(segment_bytes),
                 flush_ts=asyncio.get_running_loop().time())

        except queue.Empty:
            # No segment ready — send silence to keep stream alive
            await ws.send(silence)
            await asyncio.sleep(config.ULTRAVOX_CHUNK_MS / 1000)


async def _receive_loop(ws) -> None:
    """
    Receives audio frames from Ultravox and hands them to the playback thread.
    Binary messages are raw PCM audio.
    Text messages are JSON data messages (transcript, state, playbackClearBuffer).

    Uses a polled recv with timeout rather than `async for` so the _running flag
    is re-checked periodically.  `async for` blocks until the server sends
    something, which made shutdown wait for the full join timeout.
    """
    while _running:
        try:
            message = await asyncio.wait_for(ws.recv(), timeout=_RECV_POLL_S)
        except asyncio.TimeoutError:
            continue
        except websockets.exceptions.ConnectionClosed:
            break

        if isinstance(message, bytes):
            try:
                _playback_queue.put_nowait(message)
            except queue.Full:
                _log("ultravox", "playback_dropped",
                     reason="playback_queue_full")
        else:
            # JSON data message — check for playbackClearBuffer
            try:
                data = json.loads(message)
                if data.get("type") == "playbackClearBuffer":
                    _clear_playback()
            except (json.JSONDecodeError, KeyError):
                pass  # non-critical text messages — safe to ignore


async def _run(join_url: str) -> None:
    """Main async entry — opens WebSocket and runs send + receive concurrently."""
    global _running

    print(f"[ultravox] connecting to {join_url[:60]}...")

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            async with websockets.connect(join_url, open_timeout=10) as ws:
                print("[ultravox] connected")
                with _lock:
                    _running = True

                await asyncio.gather(
                    _send_loop(ws),
                    _receive_loop(ws),
                )
            print("[ultravox] WebSocket closed")
            return
        except (websockets.exceptions.WebSocketException, OSError) as e:
            print(
                f"[ultravox] connection attempt {attempt}/{_MAX_RETRIES} failed: {e}")
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_RETRY_DELAY_S)

    print("[ultravox] FATAL: all connection attempts failed — giving up")


def start(join_url: str) -> threading.Thread:
    """
    Start the Ultravox client in a background thread.
    Returns the thread handle so main.py can join on shutdown.
    """
    global _output_stream, _playback_thread
    _output_stream = _open_output_stream()

    _playback_thread = threading.Thread(
        target=_playback_worker, daemon=True, name="ultravox-playback"
    )
    _playback_thread.start()

    def _thread_target():
        asyncio.run(_run(join_url))

    thread = threading.Thread(
        target=_thread_target, daemon=True, name="ultravox-client"
    )
    thread.start()
    print("[ultravox] client thread started")
    return thread


def send_segment(segment: np.ndarray) -> None:
    """
    Called by main.py after policy gate passes.
    Resamples segment to 48kHz PCM and enqueues for send_loop.

    Args:
        segment: float32 ndarray at 16kHz
    """
    pcm_bytes = to_48k_pcm(segment)
    try:
        audio_send_queue.put_nowait(pcm_bytes)
    except queue.Full:
        print("[ultravox] send queue full — dropping segment")
        _log("ultravox", "send_dropped", reason="send_queue_full",
             pcm_bytes=len(pcm_bytes))


def stop() -> None:
    """Signal the loops to exit and close the playback stream."""
    global _running, _output_stream, _playback_thread
    with _lock:
        _running = False

    # Bounded put: the send loop may already have exited via _running, in which
    # case nothing will ever drain a full queue and a blocking put would hang.
    try:
        audio_send_queue.put(None, timeout=1)
    except queue.Full:
        pass

    try:
        _playback_queue.put(None, timeout=1)
    except queue.Full:
        pass
    if _playback_thread is not None:
        _playback_thread.join(timeout=2)
        _playback_thread = None

    with _lock:
        if _output_stream is not None:
            _output_stream.stop()
            _output_stream.close()
            _output_stream = None
    print("[ultravox] playback stream closed")
