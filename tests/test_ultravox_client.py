"""Tests for the Ultravox WebSocket client.

These run against a real loopback WebSocket server, so the client's threading
and shutdown behaviour is exercised for real rather than mocked. The
sounddevice playback stream is the only part stubbed out, since CI has no
audio device.
"""

import asyncio
import json
import queue
import threading
import time

import numpy as np
import pytest
import websockets
from websockets.asyncio.server import serve

import config
from llm import ultravox_client

# How long a correct shutdown may take before we call it a hang.
SHUTDOWN_BUDGET_S = 5.0


class FakeOutputStream:
    """Stand-in for sounddevice.RawOutputStream."""

    def __init__(self) -> None:
        self.written: list[bytes] = []
        self.start_calls = 0
        self.stop_calls = 0
        self.closed = False

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1

    def close(self) -> None:
        self.closed = True

    def write(self, data) -> None:
        self.written.append(bytes(data))


class StubUltravoxServer:
    """A loopback WebSocket server that records frames and can push messages.

    It never closes the connection on its own, so a client that relies on the
    server hanging up in order to shut down will hang here.
    """

    def __init__(self) -> None:
        self.received: list[bytes] = []
        self.client_connected = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._conn = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._shutdown: asyncio.Future | None = None
        self.port: int | None = None

    # -- lifecycle -------------------------------------------------------

    def start(self) -> str:
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        assert self._ready.wait(10), "stub server failed to start"
        return f"ws://127.0.0.1:{self.port}"

    def _serve(self) -> None:
        asyncio.run(self._main())

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._shutdown = self._loop.create_future()
        async with serve(self._handler, "127.0.0.1", 0) as server:
            self.port = server.sockets[0].getsockname()[1]
            self._ready.set()
            await self._shutdown

    async def _handler(self, websocket) -> None:
        self._conn = websocket
        self.client_connected.set()
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    self.received.append(message)
        except websockets.exceptions.ConnectionClosed:
            pass

    def stop(self) -> None:
        if self._loop is not None and self._shutdown is not None:
            self._loop.call_soon_threadsafe(
                lambda: self._shutdown.done() or self._shutdown.set_result(None)
            )
        if self._thread is not None:
            self._thread.join(timeout=5)

    # -- pushing messages to the client ----------------------------------

    def push(self, message) -> None:
        """Send a message to the connected client from the server side."""
        assert self._conn is not None, "no client connected"
        future = asyncio.run_coroutine_threadsafe(self._conn.send(message), self._loop)
        future.result(timeout=5)

    def wait_for_frames(self, count: int, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if len(self.received) >= count:
                return True
            time.sleep(0.02)
        return False


@pytest.fixture
def fake_stream(monkeypatch):
    """Replace the sounddevice playback stream with a recorder.

    Mirrors the real _open_output_stream, which start()s the device before
    returning it.
    """
    stream = FakeOutputStream()

    def _open():
        stream.start()
        return stream

    monkeypatch.setattr(ultravox_client, "_open_output_stream", _open)
    return stream


@pytest.fixture
def server():
    """A stub server that always outlives the client it is serving.

    The client must be stopped before the server goes away, otherwise the
    server-side close looks like a dropped connection and the client burns
    its reconnect attempts during teardown.
    """
    stub = StubUltravoxServer()
    yield stub
    ultravox_client.stop(timeout=SHUTDOWN_BUDGET_S)
    stub.stop()


def _wait_until(predicate, timeout: float = 10.0) -> bool:
    """Poll until predicate() is true or the timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _connect(server: StubUltravoxServer) -> None:
    """Start the client and wait until its loops are actually running.

    The server's connected event fires at the handshake, which is slightly
    before the client has created its send/receive tasks, so tests that need
    an established session must wait for is_running() rather than the
    handshake alone.
    """
    ultravox_client.start(server.start())
    assert server.client_connected.wait(10), "client never connected"
    assert _wait_until(ultravox_client.is_running), "client loops never started"


# ---------------------------------------------------------------------------
# Shutdown — the behaviour this module previously got wrong
# ---------------------------------------------------------------------------

def test_stop_terminates_thread_when_server_is_silent(server, fake_stream):
    """stop() must tear the client down even if the server never speaks.

    The receive loop parks on recv(); without a cross-thread wakeup it stays
    parked forever and the client thread leaks.
    """
    _connect(server)
    thread = ultravox_client.client_thread()
    assert thread is not None and thread.is_alive()

    started = time.monotonic()
    ultravox_client.stop(timeout=SHUTDOWN_BUDGET_S)
    elapsed = time.monotonic() - started

    assert not thread.is_alive(), "client thread survived stop() — shutdown hangs"
    assert elapsed < SHUTDOWN_BUDGET_S, f"stop() took {elapsed:.1f}s"
    assert not ultravox_client.is_running()


def test_stop_during_startup_window_does_not_leak(
    server, fake_stream, monkeypatch
):
    """stop() landing before the tasks exist must still terminate the client.

    This is Ctrl+C during startup. Cancellation alone cannot cover it: there
    is nothing to cancel yet, so the shutdown request has to be durable and
    observed once the tasks are finally created.

    The window is microseconds in production, so the test holds it open with
    an explicit gate and drives stop() from a third thread. That fixes the
    ordering -- stop() completes its signalling strictly before the tasks are
    created -- instead of relying on scheduling luck.
    """
    established = ultravox_client._serve_connection
    release = threading.Event()

    async def gated_serve(ws):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, release.wait)
        await established(ws)

    monkeypatch.setattr(ultravox_client, "_serve_connection", gated_serve)

    thread = ultravox_client.start(server.start())
    assert server.client_connected.wait(10), "client never connected"

    # stop() blocks joining the client, so it needs a thread of its own.
    stopper = threading.Thread(
        target=lambda: ultravox_client.stop(timeout=SHUTDOWN_BUDGET_S)
    )
    stopper.start()
    assert _wait_until(ultravox_client._stop_requested.is_set, timeout=5), (
        "stop() never registered its request"
    )

    release.set()  # only now may the client create its tasks
    stopper.join(timeout=SHUTDOWN_BUDGET_S + 5)

    assert not stopper.is_alive(), "stop() itself never returned"
    assert not thread.is_alive(), "client thread leaked when stopped during startup"
    assert not ultravox_client.is_running()


def test_stop_closes_playback_stream(server, fake_stream):
    _connect(server)
    ultravox_client.stop(timeout=SHUTDOWN_BUDGET_S)

    assert fake_stream.closed, "playback stream was not closed"


def test_stop_is_idempotent(server, fake_stream):
    _connect(server)
    ultravox_client.stop(timeout=SHUTDOWN_BUDGET_S)
    ultravox_client.stop(timeout=SHUTDOWN_BUDGET_S)  # must not raise

    assert not ultravox_client.is_running()


def test_stop_without_start_is_safe():
    ultravox_client.stop(timeout=SHUTDOWN_BUDGET_S)  # must not raise
    assert not ultravox_client.is_running()


def test_client_can_restart_after_stop(server, fake_stream):
    """A stop() must leave the module reusable, not permanently dead."""
    _connect(server)
    ultravox_client.stop(timeout=SHUTDOWN_BUDGET_S)

    second = StubUltravoxServer()
    try:
        ultravox_client.start(second.start())
        assert second.client_connected.wait(10), "client did not reconnect"
        assert _wait_until(ultravox_client.is_running), "restarted client is idle"
    finally:
        ultravox_client.stop(timeout=SHUTDOWN_BUDGET_S)
        second.stop()


# ---------------------------------------------------------------------------
# Sending
# ---------------------------------------------------------------------------

def test_send_segment_reaches_the_server(server, fake_stream):
    _connect(server)

    # A quarter-second tone; resampled to 48kHz this is many 20ms chunks.
    samples = np.zeros(config.SAMPLE_RATE // 4, dtype=np.float32)
    queued = ultravox_client.send_segment(samples)

    assert queued > 0
    assert server.wait_for_frames(3), "segment frames never arrived at the server"

    frame_bytes = ultravox_client._FRAME_BYTES
    assert all(len(f) == frame_bytes for f in server.received[:3])


def test_send_segment_reports_zero_when_queue_is_full(monkeypatch):
    """A full queue must be reported, not silently swallowed."""
    monkeypatch.setattr(
        ultravox_client.audio_send_queue,
        "put_nowait",
        lambda _: (_ for _ in ()).throw(queue.Full()),
    )
    assert ultravox_client.send_segment(np.zeros(1600, dtype=np.float32)) == 0


# ---------------------------------------------------------------------------
# Receiving
# ---------------------------------------------------------------------------

def test_binary_message_is_played(server, fake_stream):
    _connect(server)
    payload = b"\x01\x02" * 480

    server.push(payload)

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not fake_stream.written:
        time.sleep(0.02)
    assert payload in fake_stream.written


def test_playback_clear_buffer_restarts_stream(server, fake_stream):
    """A barge-in message must flush buffered playback."""
    _connect(server)
    before = fake_stream.stop_calls

    server.push(json.dumps({"type": "playbackClearBuffer"}))

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and fake_stream.stop_calls == before:
        time.sleep(0.02)
    assert fake_stream.stop_calls > before
    assert fake_stream.start_calls >= 2  # opened once, restarted once


def test_malformed_text_message_is_ignored(server, fake_stream):
    """Bad JSON must not kill the receive loop."""
    _connect(server)

    server.push("this is not json")
    time.sleep(0.2)
    payload = b"\x03\x04" * 480
    server.push(payload)

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and payload not in fake_stream.written:
        time.sleep(0.02)
    assert payload in fake_stream.written, "receive loop died on malformed JSON"
