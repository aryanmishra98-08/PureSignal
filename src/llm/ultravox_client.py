"""Fixie.ai Ultravox WebSocket client.

Responsibilities:
  1. Connect to the Ultravox join URL.
  2. Send 20ms PCM chunks — speech segments, with silence padding between them
     so the stream stays continuous.
  3. Receive audio frames from Ultravox and play them through sounddevice.

Threading model:
  - start() runs an asyncio event loop on a background thread.
  - _send_loop drains audio_send_queue; _receive_loop writes to the speaker.
  - main.py feeds audio_send_queue once the policy gate passes a segment.

Shutdown:
  stop() is called from the main thread while the loops are parked on another
  thread's event loop. A plain flag cannot wake a coroutine blocked in recv(),
  so stop() reaches into the loop with call_soon_threadsafe and cancels the
  tasks; cancellation interrupts recv() and closes the socket. stop() then
  joins the thread before closing the playback stream, so the receive loop can
  never write to a stream that is already closed.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import queue
import threading

import numpy as np
import sounddevice as sd
import websockets
import websockets.exceptions

import config
from audio.resampler import pcm_frame_bytes, silence_frame_48k, to_48k_pcm
from utils.logger import get_logger

_log = get_logger()

# Fed by main.py — holds resampled PCM bytes for whole segments.
audio_send_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=50)

_MAX_RETRIES = 3
_RETRY_DELAY_S = 2.0
_CONNECT_TIMEOUT_S = 10

# How long stop() waits for the client thread to exit.
_SHUTDOWN_TIMEOUT_S = 5.0

# Bytes in one outbound chunk, and the cadence at which they are sent.
_FRAME_BYTES = pcm_frame_bytes()
_CHUNK_INTERVAL_S = config.ULTRAVOX_CHUNK_MS / 1000

# Guards the module state shared between the main thread and the loop thread.
_lock = threading.Lock()

_output_stream: sd.RawOutputStream | None = None
_thread: threading.Thread | None = None
_loop: asyncio.AbstractEventLoop | None = None
_running = False

# Authoritative "shutdown wanted" flag. A threading.Event because stop() may
# arrive before the event loop exists -- racing a start() that has not yet
# created its tasks, which is exactly what Ctrl+C during startup looks like.
_stop_requested = threading.Event()

# Owned by the loop thread; touched there or via call_soon_threadsafe.
# _shutdown mirrors _stop_requested inside the loop so awaits wake promptly.
_tasks: list[asyncio.Task] = []
_shutdown: asyncio.Event | None = None


def _open_output_stream() -> sd.RawOutputStream:
    """Open the sounddevice stream used for Ultravox playback."""
    stream = sd.RawOutputStream(
        samplerate=config.ULTRAVOX_OUT_RATE,
        channels=1,
        dtype="int16",
        blocksize=int(config.ULTRAVOX_OUT_RATE * config.ULTRAVOX_CHUNK_MS / 1000),
    )
    stream.start()
    _log.info("ultravox", f"playback stream open — {config.ULTRAVOX_OUT_RATE}Hz")
    return stream


# ---------------------------------------------------------------------------
# Loops — run on the client thread's event loop
# ---------------------------------------------------------------------------

async def _send_segment_frames(ws, segment_bytes: bytes) -> None:
    """Send one segment as a sequence of fixed-size chunks at real-time cadence."""
    for offset in range(0, len(segment_bytes), _FRAME_BYTES):
        frame = segment_bytes[offset : offset + _FRAME_BYTES]
        if len(frame) < _FRAME_BYTES:  # pad the final short chunk
            frame += bytes(_FRAME_BYTES - len(frame))
        await ws.send(frame)
        await asyncio.sleep(_CHUNK_INTERVAL_S)


async def _send_loop(ws) -> None:
    """Forward queued segments, padding the gaps with silence."""
    silence = silence_frame_48k()

    while _running:
        try:
            segment_bytes = audio_send_queue.get_nowait()
        except queue.Empty:
            # Nothing queued — keep the stream alive.
            await ws.send(silence)
            await asyncio.sleep(_CHUNK_INTERVAL_S)
            continue

        if segment_bytes is None:  # shutdown sentinel
            break
        await _send_segment_frames(ws, segment_bytes)


def _play(message: bytes) -> None:
    """Write inbound PCM to the playback stream, if one is open."""
    stream = _output_stream
    if stream is None:
        return
    try:
        stream.write(message)
    except sd.PortAudioError as e:
        _log.warning("ultravox", f"playback error: {e}")


def _handle_control(message: str) -> None:
    """Act on a JSON control message. Malformed or unknown text is ignored."""
    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        return

    stream = _output_stream
    if data.get("type") == "playbackClearBuffer" and stream is not None:
        # Barge-in: restart the stream to drop what is already buffered.
        stream.stop()
        stream.start()


async def _receive_loop(ws) -> None:
    """Play inbound audio and handle control messages until cancelled."""
    async for message in ws:
        if isinstance(message, bytes):
            _play(message)
        else:
            _handle_control(message)


async def _serve_connection(ws) -> None:
    """Run the send and receive loops until one finishes or stop() cancels."""
    global _running, _tasks

    with _lock:
        _running = True

    _tasks = [
        asyncio.create_task(_send_loop(ws), name="ultravox-send"),
        asyncio.create_task(_receive_loop(ws), name="ultravox-receive"),
    ]
    # stop() may have fired while these tasks were being created, in which
    # case _request_stop found nothing to cancel. Catch that here.
    if _stop_requested.is_set():
        for task in _tasks:
            task.cancel()

    try:
        await asyncio.gather(*_tasks)
    except asyncio.CancelledError:
        pass  # expected path: stop() cancelled the tasks
    finally:
        with _lock:
            _running = False
        for task in _tasks:
            task.cancel()
        await asyncio.gather(*_tasks, return_exceptions=True)
        _tasks = []


async def _run(join_url: str) -> None:
    """Connect with retries, serve the connection, then tear down."""
    global _loop, _shutdown

    with _lock:
        _loop = asyncio.get_running_loop()
    _shutdown = asyncio.Event()

    _log.info("ultravox", f"connecting to {join_url[:60]}...")

    for attempt in range(1, _MAX_RETRIES + 1):
        if _stop_requested.is_set():
            return

        try:
            async with websockets.connect(
                join_url, open_timeout=_CONNECT_TIMEOUT_S
            ) as ws:
                if _stop_requested.is_set():
                    return  # the async with closes the socket on the way out
                _log.info("ultravox", "connected")
                await _serve_connection(ws)
            _log.info("ultravox", "WebSocket closed")
            return

        except (websockets.exceptions.WebSocketException, OSError) as e:
            _log.warning(
                "ultravox", f"connection attempt {attempt}/{_MAX_RETRIES} failed: {e}"
            )
            if attempt >= _MAX_RETRIES:
                break
            # Back off, but abandon the wait the moment stop() arrives.
            try:
                await asyncio.wait_for(_shutdown.wait(), timeout=_RETRY_DELAY_S)
                return
            except asyncio.TimeoutError:
                pass

    if not _stop_requested.is_set():
        _log.error("ultravox", "all connection attempts failed — giving up")


def _request_stop() -> None:
    """Wake the loops. Runs on the client thread via call_soon_threadsafe."""
    if _shutdown is not None:
        _shutdown.set()
    for task in _tasks:
        if not task.done():
            task.cancel()


# ---------------------------------------------------------------------------
# Public API — called from the main thread
# ---------------------------------------------------------------------------

def start(join_url: str) -> threading.Thread:
    """Start the client on a background thread.

    Returns:
        The thread handle. stop() joins it, so callers need not.

    Raises:
        RuntimeError: If the client is already running.
    """
    global _output_stream, _thread

    if _thread is not None and _thread.is_alive():
        raise RuntimeError("[ultravox] client already started — call stop() first")

    _stop_requested.clear()

    # A sentinel or stale audio left by a previous session would otherwise be
    # the first thing the new send loop reads.
    _drain_queue()

    _output_stream = _open_output_stream()
    _thread = threading.Thread(
        target=lambda: asyncio.run(_run(join_url)),
        daemon=True,
        name="ultravox-client",
    )
    _thread.start()
    _log.info("ultravox", "client thread started")
    return _thread


def send_segment(segment: np.ndarray) -> int:
    """Resample an approved segment and queue it for sending.

    Args:
        segment: Normalized float32 samples at ``config.SAMPLE_RATE``.

    Returns:
        The number of PCM bytes queued, or 0 if the queue was full.
    """
    pcm_bytes = to_48k_pcm(segment)
    try:
        audio_send_queue.put_nowait(pcm_bytes)
    except queue.Full:
        _log.warning("ultravox", "send queue full — dropping segment")
        return 0
    return len(pcm_bytes)


def _drain_queue() -> None:
    """Discard anything left in the send queue, sentinel included."""
    while True:
        try:
            audio_send_queue.get_nowait()
        except queue.Empty:
            return


def stop(timeout: float = _SHUTDOWN_TIMEOUT_S) -> None:
    """Shut the client down and release the playback device.

    Safe to call when the client was never started, and safe to call twice.

    Args:
        timeout: Seconds to wait for the client thread to exit.
    """
    global _running, _output_stream, _thread, _loop

    # Set first: this is what a start() still in flight will observe.
    _stop_requested.set()

    with _lock:
        _running = False
        loop, thread = _loop, _thread

    # Wake the loops: a bool cannot interrupt a coroutine parked in recv().
    if loop is not None and not loop.is_closed():
        with contextlib.suppress(RuntimeError):  # loop already finished
            loop.call_soon_threadsafe(_request_stop)

    with contextlib.suppress(queue.Full):
        audio_send_queue.put_nowait(None)  # unblock a waiting send loop

    if thread is not None:
        thread.join(timeout=timeout)
        if thread.is_alive():
            _log.warning(
                "ultravox", f"client thread still alive after {timeout}s"
            )

    # Only now is it safe to close the device: the receive loop has stopped,
    # so it cannot write to a stream that is already closed.
    with _lock:
        stream, _output_stream = _output_stream, None
        _thread = None
        _loop = None

    if stream is not None:
        stream.stop()
        stream.close()
        _log.info("ultravox", "playback stream closed")

    _drain_queue()


def is_running() -> bool:
    """Return True while the client is connected and its loops are live."""
    return _running


def client_thread() -> threading.Thread | None:
    """Return the background client thread, or None if it is not running."""
    return _thread
