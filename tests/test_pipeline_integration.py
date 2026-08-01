# =============================================================================
# tests/test_pipeline_integration.py — end-to-end pipeline behaviour
#
# Drives the real consumer loops with the encoder and separator stubbed out.
# No models, no network, no hardware.
#
# Every case runs against BOTH process_loop_extractor and
# process_loop_gatekeeper, which is what stops the two paths diverging.
# =============================================================================
import threading
import time

import numpy as np
import pytest

import config
import utils.logger as logger_mod
from audio import capture, vad
from speaker import enrollment, tracker

SPEAKER_A_HZ = 200      # enrolled speaker
SPEAKER_B_HZ = 1200     # impostor — far enough apart for the stub to separate
_ZCR_SPLIT = 0.08       # midpoint between the two tones' zero-crossing rates


# ---------------------------------------------------------------------------
# Synthetic audio
# ---------------------------------------------------------------------------

def _tone(duration_s: float, hz: int, amp: float = 0.5) -> np.ndarray:
    n = int(duration_s * config.SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / config.SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _silence(duration_s: float) -> np.ndarray:
    return np.zeros(int(duration_s * config.SAMPLE_RATE), dtype=np.float32)


def _utterances(specs, gap_s: float = 0.8, trailing_silence: bool = True):
    """Build a track of (duration, hz) utterances separated by silence."""
    parts = [_silence(0.3)]
    for i, (dur, hz) in enumerate(specs):
        parts.append(_tone(dur, hz))
        if i < len(specs) - 1 or trailing_silence:
            parts.append(_silence(gap_s))
    return np.concatenate(parts)


def _zcr(x: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(np.sign(x)))) / 2 / len(x))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def events(monkeypatch):
    """Capture structured log events instead of writing a session file."""
    collected: list[dict] = []

    def _fake_write(stage, event, data):
        collected.append({"stage": stage, "event": event, "data": data})

    monkeypatch.setattr(logger_mod, "_write", _fake_write)
    return collected


@pytest.fixture
def stub_encoder(monkeypatch):
    """
    Deterministic embedding keyed by a marker carried in the audio itself.

    Zero-crossing rate survives peak normalization, so the stub can tell the two
    synthetic speakers apart the same way a real encoder would tell two voices
    apart — without running a model.
    """
    calls = {"n": 0, "delay_s": 0.0}

    def _fake_embed(segment):
        calls["n"] += 1
        if calls["delay_s"]:
            time.sleep(calls["delay_s"])
        vec = np.zeros(256, dtype=np.float32)
        vec[0 if _zcr(segment) < _ZCR_SPLIT else 1] = 1.0
        return vec

    from speaker import encoder
    monkeypatch.setattr(encoder, "embed", _fake_embed)
    return calls


@pytest.fixture
def enrolled_alice(monkeypatch):
    """Enroll a single speaker whose profile matches the SPEAKER_A_HZ tone."""
    profile = np.zeros(256, dtype=np.float32)
    profile[0] = 1.0
    monkeypatch.setattr(enrollment, "_enrolled", {"alice": profile})


@pytest.fixture(autouse=True)
def clean_state():
    vad.reset()
    tracker.reset()
    capture.reset()
    _drain_queues()
    yield
    vad.reset()
    tracker.reset()
    capture.reset()
    _drain_queues()


def _drain_queues():
    for q in (capture.frame_queue, capture.window_queue):
        while not q.empty():
            try:
                q.get_nowait()
            except Exception:  # noqa: BLE001
                break


# ---------------------------------------------------------------------------
# Driver — feeds audio through the real capture path and runs the real loop
# ---------------------------------------------------------------------------

def _run_pipeline(audio: np.ndarray, mode: str, monkeypatch, no_ultravox=True):
    """
    Push `audio` through the production windowing path and run the loop for
    `mode` ("extractor" or "gatekeeper") until the EOF sentinel.
    """
    monkeypatch.setattr(config, "EXTRACTOR_ENABLED", mode == "extractor")
    capture.reset()

    if mode == "extractor":
        # Passthrough separator: one source, identical to the input. Keeps the
        # test on the real windowing/resequencing path without a model.
        from audio import extractor
        monkeypatch.setattr(extractor, "separate",
                            lambda w: w[np.newaxis, :].astype(np.float32))

    source = capture.get_source()
    frame_size = config.FRAME_SAMPLES
    for i in range(0, len(audio), frame_size):
        frame = audio[i: i + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))
        source.push_frame(frame)
    source.close()

    from pipeline.loops import process_loop_extractor, process_loop_gatekeeper
    if mode == "extractor":
        process_loop_extractor(None, no_ultravox)
    else:
        process_loop_gatekeeper(no_ultravox)


def _of(events, stage, event):
    return [e["data"] for e in events if e["stage"] == stage and e["event"] == event]


BOTH_MODES = pytest.mark.parametrize("mode", ["extractor", "gatekeeper"])


# ---------------------------------------------------------------------------
# Case 1 — three utterances produce three decisions
# ---------------------------------------------------------------------------

@BOTH_MODES
def test_three_utterances_three_decisions(mode, monkeypatch, events, stub_encoder,
                                          enrolled_alice):
    audio = _utterances([(1.0, SPEAKER_A_HZ)] * 3)
    _run_pipeline(audio, mode, monkeypatch)
    assert len(_of(events, "policy", "decision")) == 3


# ---------------------------------------------------------------------------
# Case 2 — a file ending mid-utterance still delivers the final segment
# ---------------------------------------------------------------------------

@BOTH_MODES
def test_final_segment_delivered_without_trailing_silence(mode, monkeypatch, events,
                                                          stub_encoder, enrolled_alice):
    """
    The VAD's end-of-stream flush must go through the same gate-and-forward
    path as any other segment. Counting it as closed without delivering it is
    worse than losing it outright — the log claims it was handled.
    """
    audio = _utterances([(1.0, SPEAKER_A_HZ)] * 2, trailing_silence=False)
    _run_pipeline(audio, mode, monkeypatch)
    closed = _of(events, "vad", "segment_closed")
    decisions = _of(events, "policy", "decision")
    assert len(closed) == 2
    assert len(decisions) == len(
        closed), "every closed segment must reach the gate"


# ---------------------------------------------------------------------------
# Case 3 — a slow encoder must not cost us the last segment
# ---------------------------------------------------------------------------

@BOTH_MODES
def test_slow_encoder_still_delivers_final_segment(mode, monkeypatch, events,
                                                   stub_encoder, enrolled_alice):
    """The end-of-stream drain must block. Skipping futures that are not yet
    done silently discards whatever the encoder was still working on."""
    stub_encoder["delay_s"] = 0.2
    audio = _utterances([(1.0, SPEAKER_A_HZ)] * 2)
    _run_pipeline(audio, mode, monkeypatch)
    assert len(_of(events, "policy", "decision")) == 2


# ---------------------------------------------------------------------------
# Case 4 — no audio duplication
# ---------------------------------------------------------------------------

@BOTH_MODES
def test_no_audio_duplication(mode, monkeypatch, events, stub_encoder, enrolled_alice):
    """
    Total audio accounted for by the VAD must match the speech actually present
    (plus one hangover tail per utterance). Forwarding each window's overlap
    would inflate this by window_s / hop_s — 4x at the shipped defaults.
    """
    n_utt, utt_s = 3, 1.0
    audio = _utterances([(utt_s, SPEAKER_A_HZ)] * n_utt)
    _run_pipeline(audio, mode, monkeypatch)

    total = sum(d["sample_count"]
                for d in _of(events, "vad", "segment_closed"))
    speech = n_utt * utt_s * config.SAMPLE_RATE
    hangover = n_utt * (config.HANGOVER_MS / 1000) * config.SAMPLE_RATE
    slack = n_utt * 3 * config.FRAME_SAMPLES
    assert speech - slack <= total <= speech + hangover + slack, (
        f"accounted {total} samples for {speech} samples of speech")


def test_both_modes_account_for_the_same_audio(monkeypatch, events, stub_encoder,
                                               enrolled_alice):
    """
    With a passthrough separator the extractor path must be sample-identical to
    the gatekeeper path. Any windowing inflation shows up as a mismatch.
    """
    audio = _utterances([(1.0, SPEAKER_A_HZ)] * 2)
    totals = {}
    for mode in ("gatekeeper", "extractor"):
        events.clear()
        vad.reset()
        tracker.reset()
        capture.reset()
        _drain_queues()
        with monkeypatch.context() as m:
            _run_pipeline(audio, mode, m)
        totals[mode] = sum(d["sample_count"]
                           for d in _of(events, "vad", "segment_closed"))
    assert totals["extractor"] == totals["gatekeeper"]


# ---------------------------------------------------------------------------
# Case 5 — enrolled passes, impostor drops
# ---------------------------------------------------------------------------

@BOTH_MODES
def test_enrolled_passes_impostor_drops(mode, monkeypatch, events, stub_encoder,
                                        enrolled_alice):
    audio = _utterances([(1.0, SPEAKER_A_HZ), (1.0, SPEAKER_B_HZ)])
    _run_pipeline(audio, mode, monkeypatch)
    decisions = [d["decision"] for d in _of(events, "policy", "decision")]
    assert decisions == ["PASS", "DROP"]


@BOTH_MODES
def test_policy_all_mode_passes_everyone(mode, monkeypatch, events, stub_encoder,
                                         enrolled_alice):
    """ALL is the passthrough baseline used to compare against gating."""
    monkeypatch.setattr(config, "POLICY_MODE", "ALL")
    audio = _utterances([(1.0, SPEAKER_A_HZ), (1.0, SPEAKER_B_HZ)])
    _run_pipeline(audio, mode, monkeypatch)
    decisions = [d["decision"] for d in _of(events, "policy", "decision")]
    assert decisions == ["PASS", "PASS"]


# ---------------------------------------------------------------------------
# Case 6 — drops are visible
# ---------------------------------------------------------------------------

@BOTH_MODES
def test_encoder_pressure_drops_are_logged(mode, monkeypatch, events, stub_encoder,
                                           enrolled_alice):
    """
    Segments dropped under encoder pressure must reach the structured log.
    A console-only, debug-gated notice makes lost utterances unaccountable and
    leaves the latency report survivorship-biased.
    """
    monkeypatch.setattr(config, "ENCODER_MAX_PENDING", 1)
    stub_encoder["delay_s"] = 0.25
    # Dense speech: short gaps, just past the hangover, so segments close
    # back to back while the encoder is still busy.
    audio = _utterances([(0.6, SPEAKER_A_HZ)] * 6, gap_s=0.45)
    _run_pipeline(audio, mode, monkeypatch)

    dropped = _of(events, "main", "segment_dropped")
    assert dropped, "expected at least one logged drop under encoder pressure"
    assert all(d["reason"] == "encoder_busy" for d in dropped)
    assert all(d["sample_count"] > 0 for d in dropped)


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------

@BOTH_MODES
def test_similarity_scores_are_real(mode, monkeypatch, events, stub_encoder,
                                    enrolled_alice):
    """Similarity scores must be the real values, not placeholders — they are
    what makes the thresholds tunable from a session log."""
    audio = _utterances([(1.0, SPEAKER_A_HZ)] * 3)
    _run_pipeline(audio, mode, monkeypatch)

    matches = _of(events, "enrollment", "match_result")
    assert matches, "enrollment.match_result must be emitted"
    assert any(m["best_sim"] > 0.5 for m in matches)

    assigned = _of(events, "tracker", "speaker_assigned")
    assert len(assigned) >= 2
    # The first assignment has an empty gallery (-1.0 sentinel); later ones must
    # carry a genuine similarity against the existing centroid.
    assert any(a["best_sim"] > 0.5 for a in assigned[1:])


@BOTH_MODES
def test_latency_record_measures_speech_onset(mode, monkeypatch, events, stub_encoder,
                                              enrolled_alice):
    """The recorded span must start at speech onset, not at encoder enqueue."""
    audio = _utterances([(1.0, SPEAKER_A_HZ)] * 2)
    _run_pipeline(audio, mode, monkeypatch)

    records = _of(events, "pipeline", "latency_record")
    assert records
    for r in records:
        assert "speech_onset_to_send_ms" in r
        assert r["segment_start_ts"] < r["vad_close_ts"] <= r["encoder_start_ts"]
        # Onset-to-send must exceed the encoder-only span it used to report.
        encoder_ms = (r["encoder_done_ts"] - r["encoder_start_ts"]) * 1000
        assert r["speech_onset_to_send_ms"] > encoder_ms


# ---------------------------------------------------------------------------
# A failing separator degrades instead of deadlocking
# ---------------------------------------------------------------------------

def test_extractor_failure_logs_and_continues(monkeypatch, events, stub_encoder,
                                              enrolled_alice):
    """
    A raise inside a worker must not leave a hole in the sequence. If the
    future is discarded the exception goes unseen, the resequencer waits on a
    number that never arrives, and the pipeline stalls with no log line.
    """
    monkeypatch.setattr(config, "EXTRACTOR_ENABLED", True)
    capture.reset()
    from audio import extractor

    state = {"n": 0}

    def _flaky(window):
        state["n"] += 1
        if state["n"] == 3:
            raise RuntimeError("separator exploded")
        return window[np.newaxis, :].astype(np.float32)

    monkeypatch.setattr(extractor, "separate", _flaky)

    audio = _utterances([(1.0, SPEAKER_A_HZ)] * 2)
    source = capture.get_source()
    for i in range(0, len(audio), config.FRAME_SAMPLES):
        frame = audio[i: i + config.FRAME_SAMPLES]
        if len(frame) < config.FRAME_SAMPLES:
            frame = np.pad(frame, (0, config.FRAME_SAMPLES - len(frame)))
        source.push_frame(frame)
    source.close()

    from pipeline.loops import process_loop_extractor

    done = threading.Event()

    def _run():
        process_loop_extractor(None, True)
        done.set()

    threading.Thread(target=_run, daemon=True).start()
    assert done.wait(
        timeout=30), "pipeline deadlocked after a separator failure"

    assert _of(events, "extractor",
               "extract_failed"), "the failure must be logged"
    assert _of(events, "policy",
               "decision"), "the pipeline must keep producing output"


# ---------------------------------------------------------------------------
# Ultravox receives unnormalized audio
# ---------------------------------------------------------------------------

def test_ultravox_receives_raw_not_peak_normalized_audio(monkeypatch, events,
                                                         stub_encoder, enrolled_alice):
    """
    Peak-normalizing per segment is right for the encoder and wrong for
    playback: a whisper and a shout arrive at Ultravox at identical amplitude.
    """
    from llm import ultravox_client
    sent: list[np.ndarray] = []
    monkeypatch.setattr(ultravox_client, "send_segment",
                        lambda seg: sent.append(seg))

    quiet = 0.05
    audio = np.concatenate([_silence(0.3), _tone(1.0, SPEAKER_A_HZ, amp=quiet),
                            _silence(0.8)])
    _run_pipeline(audio, "gatekeeper", monkeypatch, no_ultravox=False)

    assert sent, "a passing segment must be forwarded"
    peak = float(np.max(np.abs(sent[0])))
    assert peak < 0.5, f"forwarded audio was peak-normalized (peak={peak:.3f})"
    assert peak == pytest.approx(quiet, abs=0.02)
