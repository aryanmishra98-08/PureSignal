# =============================================================================
# llm/ultravox_client.py — Fixie.ai Ultravox WebSocket client
#
# Responsibilities:
#   1. Connect to ULTRAVOX_JOIN_URL
#   2. Send 20ms PCM chunks (speech segments + silence padding)
#   3. Receive audio frames from Ultravox and play via sounddevice
#
# Threading model:
#   - send_loop runs in its own async task, reads from audio_send_queue
#   - receive_loop runs in its own async task, writes to sounddevice output
#   - main.py feeds audio_send_queue after policy gate passes
# =============================================================================

import asyncio
import json
import queue
import threading
import config
import sounddevice as sd
import websockets
import websockets.exceptions
from audio.resampler import silence_frame_48k, to_48k_pcm

# Queue fed by main.py — holds resampled PCM bytes (full segments)
audio_send_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=50)

# Retry constants for WebSocket connection
_MAX_RETRIES = 3
_RETRY_DELAY_S = 2.0

# Thread safety for shared state
_lock = threading.Lock()

# Playback output stream — opened once, written to continuously
_output_stream: sd.RawOutputStream | None = None

# Internal flag to signal shutdown
_running = False

# 20ms frame size at 48kHz in bytes (960 samples * 2 bytes per int16 sample)
_FRAME_BYTES = int(config.ULTRAVOX_IN_RATE * config.ULTRAVOX_CHUNK_MS / 1000) * 2


def _open_output_stream() -> sd.RawOutputStream:
    """Open sounddevice output stream for Ultravox audio playback."""
    stream = sd.RawOutputStream(
        samplerate=config.ULTRAVOX_OUT_RATE,
        channels=1,
        dtype="int16",
        blocksize=int(config.ULTRAVOX_OUT_RATE * config.ULTRAVOX_CHUNK_MS / 1000),
    )
    stream.start()
    print(f"[ultravox] playback stream open — {config.ULTRAVOX_OUT_RATE}Hz")
    return stream


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
                frame = segment_bytes[offset : offset + _FRAME_BYTES]

                # Pad last frame if shorter than 20ms
                if len(frame) < _FRAME_BYTES:
                    frame = frame + bytes(_FRAME_BYTES - len(frame))

                await ws.send(frame)
                offset += _FRAME_BYTES

                # Maintain 20ms cadence
                await asyncio.sleep(config.ULTRAVOX_CHUNK_MS / 1000)

        except queue.Empty:
            # No segment ready — send silence to keep stream alive
            await ws.send(silence)
            await asyncio.sleep(config.ULTRAVOX_CHUNK_MS / 1000)


async def _receive_loop(ws) -> None:
    """
    Receives audio frames from Ultravox and writes to sounddevice output.
    Binary messages are raw PCM audio.
    Text messages are JSON data messages (transcript, state, playbackClearBuffer, etc.).
    """
    global _output_stream

    async for message in ws:
        if not _running:
            break
        if isinstance(message, bytes):
            if _output_stream is not None:
                try:
                    _output_stream.write(message)
                except sd.PortAudioError as e:
                    print(f"[ultravox] playback error: {e}")
        else:
            # JSON data message — check for playbackClearBuffer
            try:
                data = json.loads(message)
                if (
                    data.get("type") == "playbackClearBuffer"
                    and _output_stream is not None
                ):
                    # Abort current buffer — stop and restart stream to flush
                    _output_stream.stop()
                    _output_stream.start()
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
            print(f"[ultravox] connection attempt {attempt}/{_MAX_RETRIES} failed: {e}")
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_RETRY_DELAY_S)

    print("[ultravox] FATAL: all connection attempts failed — giving up")


def start(join_url: str) -> threading.Thread:
    """
    Start the Ultravox client in a background thread.
    Returns the thread handle so main.py can join on shutdown.
    """
    global _output_stream
    _output_stream = _open_output_stream()

    def _thread_target():
        asyncio.run(_run(join_url))

    thread = threading.Thread(
        target=_thread_target, daemon=True, name="ultravox-client"
    )
    thread.start()
    print("[ultravox] client thread started")
    return thread


def send_segment(audio_bytes: bytes) -> None:
    """
    Called by main.py after policy gate passes.
    Resamples segment to 48kHz PCM and enqueues for send_loop.

    Args:
        audio_bytes: float32 ndarray at 16kHz — already normalized
    """
    pcm_bytes = to_48k_pcm(audio_bytes)
    try:
        audio_send_queue.put_nowait(pcm_bytes)
    except queue.Full:
        print("[ultravox] send queue full — dropping segment")


def stop() -> None:
    """Signal send_loop to exit and close playback stream."""
    global _running, _output_stream
    with _lock:
        _running = False
    audio_send_queue.put(None)  # unblock send_loop if waiting

    with _lock:
        if _output_stream is not None:
            _output_stream.stop()
            _output_stream.close()
            _output_stream = None
    print("[ultravox] playback stream closed")
