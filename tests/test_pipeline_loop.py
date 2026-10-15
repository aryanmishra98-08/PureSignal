"""Tests for the main processing loop's end-of-stream handling.

Frames are pushed straight into the queue rather than streamed in real time,
so these run fast and deterministically. The pipeline is assembled from real
components apart from the encoder, which is stubbed to avoid loading a model.
"""

import queue

import numpy as np
import pytest

import config
import main as pipeline_module
from audio.vad import VoiceActivityDetector
from speaker.enrollment import EnrollmentStore
from speaker.policy import PolicyGate
from speaker.tracker import SpeakerTracker


class StubEncoder:
    """Encoder stand-in that records what it was asked to embed."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.durations: list[float] = []

    def embed(self, segment):
        self.durations.append(len(segment) / self.sample_rate)
        vector = np.zeros(256, dtype=np.float32)
        vector[0] = 1.0
        return vector


@pytest.fixture
def pipeline():
    """A full Pipeline with a stub encoder and a gate that drops everything.

    Dropping at the gate keeps Ultravox out of these tests; the loop itself
    is what is under test.
    """
    vad = VoiceActivityDetector()
    encoder = StubEncoder(vad.sample_rate)
    store = EnrollmentStore()
    return pipeline_module.Pipeline(
        vad=vad,
        encoder=encoder,
        tracker=SpeakerTracker(),
        enrollment=store,
        policy=PolicyGate(store, mode="DYNAMIC", dynamic_target="__none__"),
    )


def _frames(audio: np.ndarray, frame_samples: int) -> list[np.ndarray]:
    """Split audio into fixed-size frames, padding the final short one."""
    out = []
    for offset in range(0, len(audio), frame_samples):
        frame = audio[offset : offset + frame_samples]
        if len(frame) < frame_samples:
            frame = np.pad(frame, (0, frame_samples - len(frame)))
        out.append(frame)
    return out


def _speech(seconds: float, amplitude: float = 0.4, freq: float = 200.0):
    t = np.arange(int(seconds * config.SAMPLE_RATE)) / config.SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(seconds: float):
    return np.zeros(int(seconds * config.SAMPLE_RATE), dtype=np.float32)


def _run(pipeline, audio: np.ndarray) -> None:
    """Feed one clip through the processing loop, then the EOF sentinel."""
    q: queue.Queue = queue.Queue()
    for frame in _frames(audio, pipeline.vad.frame_samples):
        q.put(frame)
    q.put(None)
    pipeline_module._process_loop(q, pipeline, no_ultravox=True)


def test_final_segment_without_trailing_silence_is_processed(pipeline):
    """Speech running right up to EOF must still be embedded, not dropped.

    The VAD never closes this segment itself, so it can only reach the
    encoder through the end-of-stream flush.
    """
    _run(pipeline, np.concatenate([_silence(0.5), _speech(1.0)]))

    assert len(pipeline.encoder.durations) == 1
    assert pipeline.encoder.durations[0] == pytest.approx(1.0, abs=0.2)


def test_segment_closed_by_trailing_silence_is_processed(pipeline):
    """A segment closed normally by the hangover window is still embedded once."""
    hangover_s = pipeline.vad.hangover_ms / 1000
    _run(
        pipeline,
        np.concatenate([_silence(0.5), _speech(1.0), _silence(hangover_s + 0.5)]),
    )

    assert len(pipeline.encoder.durations) == 1


def test_silence_only_stream_embeds_nothing(pipeline):
    _run(pipeline, _silence(2.0))
    assert pipeline.encoder.durations == []


def test_loop_stops_on_eof_sentinel(pipeline):
    """The sentinel must end the loop rather than block forever."""
    q: queue.Queue = queue.Queue()
    q.put(None)
    pipeline_module._process_loop(q, pipeline, no_ultravox=True)
    assert pipeline.encoder.durations == []


def test_speaker_is_tracked_across_segments(pipeline):
    """The loop must feed embeddings into the tracker it was handed."""
    hangover_s = pipeline.vad.hangover_ms / 1000
    _run(
        pipeline,
        np.concatenate([
            _silence(0.3),
            _speech(0.8),
            _silence(hangover_s + 0.3),
            _speech(0.8),
        ]),
    )

    assert len(pipeline.encoder.durations) == 2
    # The stub returns one constant embedding, so both land on one speaker.
    assert pipeline.tracker.speaker_ids() == {"S1"}


def test_pipelines_do_not_share_state(pipeline):
    """A second pipeline starts clean — sessions are fully independent."""
    _run(pipeline, np.concatenate([_silence(0.3), _speech(1.0)]))
    assert len(pipeline.tracker) == 1

    vad = VoiceActivityDetector()
    store = EnrollmentStore()
    fresh = pipeline_module.Pipeline(
        vad=vad,
        encoder=StubEncoder(vad.sample_rate),
        tracker=SpeakerTracker(),
        enrollment=store,
        policy=PolicyGate(store, mode="DYNAMIC", dynamic_target="__none__"),
    )

    assert len(fresh.tracker) == 0
    assert fresh.encoder.durations == []
    assert fresh.vad.flush() is None
