"""Tests for the audio stages: VAD, normalization and resampling."""

import struct

import numpy as np
import pytest

import config
from audio import features, resampler
from audio.vad import VoiceActivityDetector


@pytest.fixture
def vad():
    """A detector built from the current config."""
    return VoiceActivityDetector()


def _speech_frame(detector, level: float = 0.5) -> np.ndarray:
    return np.full(detector.frame_samples, level, dtype=np.float32)


def _silence_frame(detector) -> np.ndarray:
    return np.zeros(detector.frame_samples, dtype=np.float32)


def _close_segment(detector, n_silence_frames: int | None = None):
    """Feed silence until the detector closes a segment; return it, or None."""
    if n_silence_frames is None:
        n_silence_frames = detector.hangover_frames + 2

    segment = None
    for _ in range(n_silence_frames):
        result = detector.process_frame(_silence_frame(detector))
        if result is not None:
            segment = result
    return segment


# ---------------------------------------------------------------------------
# audio/vad.py
# ---------------------------------------------------------------------------

def test_vad_silence_returns_none(vad):
    for _ in range(100):
        assert vad.process_frame(_silence_frame(vad)) is None


def test_vad_speech_segment_returned(vad):
    n_speech = 10
    for _ in range(n_speech):
        vad.process_frame(_speech_frame(vad))

    segment = _close_segment(vad)

    assert segment is not None
    assert isinstance(segment, np.ndarray)
    assert segment.dtype == np.float32
    # The segment holds the speech frames plus the hangover frames.
    min_expected = (n_speech + vad.hangover_frames) * vad.frame_samples
    assert len(segment) >= min_expected


def test_vad_segment_content_matches_input(vad):
    """The returned segment must contain the frames that were fed in."""
    speech = _speech_frame(vad)
    for _ in range(5):
        vad.process_frame(speech)

    segment = _close_segment(vad)

    assert segment is not None
    np.testing.assert_array_equal(segment[: vad.frame_samples], speech)


def test_vad_flush_after_closed_segment_returns_none(vad):
    """flush() finds nothing once process_frame has already closed the segment."""
    for _ in range(5):
        vad.process_frame(_speech_frame(vad))
    _close_segment(vad)

    assert vad.flush() is None


def test_vad_buffer_no_overflow(vad):
    """Feeding more than 30s of speech must not raise."""
    frames_30s = int(30 * vad.sample_rate / vad.frame_samples) + 10
    for _ in range(frames_30s):
        vad.process_frame(_speech_frame(vad))


def test_vad_flush_returns_active_segment(vad):
    """flush() returns buffered speech when no trailing silence ever arrives."""
    for _ in range(10):
        vad.process_frame(_speech_frame(vad))

    segment = vad.flush()

    assert segment is not None
    assert isinstance(segment, np.ndarray)
    assert len(segment) > 0


def test_vad_flush_returns_none_when_silent(vad):
    for _ in range(5):
        vad.process_frame(_silence_frame(vad))
    assert vad.flush() is None


def test_vad_honours_constructor_hangover():
    """Tunables passed to the constructor override config."""
    detector = VoiceActivityDetector(hangover_ms=100)
    assert detector.hangover_frames == int(100 / detector.frame_ms)

    for _ in range(5):
        detector.process_frame(_speech_frame(detector))

    # The segment must close within the shortened hangover window.
    assert _close_segment(detector, detector.hangover_frames + 2) is not None


def test_vad_instances_are_independent():
    """Two detectors must not share state — the point of dropping globals."""
    a = VoiceActivityDetector()
    b = VoiceActivityDetector()

    for _ in range(10):
        a.process_frame(_speech_frame(a))

    # b never saw speech, so it has nothing to flush.
    assert b.flush() is None
    assert a.flush() is not None


def test_vad_reset_clears_state_but_keeps_settings():
    detector = VoiceActivityDetector(hangover_ms=100)
    for _ in range(10):
        detector.process_frame(_speech_frame(detector))

    detector.reset()

    assert detector.flush() is None
    assert detector.hangover_frames == int(100 / detector.frame_ms)


# ---------------------------------------------------------------------------
# audio/features.py
# ---------------------------------------------------------------------------

def test_features_normalize_peak():
    signal = np.array([0.0, 0.25, 0.5, -0.5, 0.1], dtype=np.float32)
    out = features.normalize(signal)
    assert np.isclose(np.max(np.abs(out)), 1.0)
    assert out.dtype == np.float32


def test_features_normalize_near_silent():
    signal = np.full(100, 1e-9, dtype=np.float32)
    out = features.normalize(signal)
    assert out.dtype == np.float32
    assert not np.any(np.isnan(out))
    assert not np.any(np.isinf(out))


# ---------------------------------------------------------------------------
# audio/resampler.py
# ---------------------------------------------------------------------------

def test_resampler_output_length():
    one_second = np.zeros(config.SAMPLE_RATE, dtype=np.float32)
    result = resampler.to_48k_pcm(one_second)
    assert isinstance(result, bytes)
    assert len(result) == config.ULTRAVOX_IN_RATE * 2  # int16 at the output rate


def test_resampler_silence_frame():
    frame = resampler.silence_frame_48k()
    expected = int(config.ULTRAVOX_IN_RATE * config.ULTRAVOX_CHUNK_MS / 1000) * 2
    assert isinstance(frame, bytes)
    assert len(frame) == expected
    assert all(b == 0 for b in frame)


def test_resampler_frame_bytes_matches_silence_frame():
    assert resampler.pcm_frame_bytes() == len(resampler.silence_frame_48k())


def test_resampler_clipping_prevents_wraparound():
    """Values outside [-1, 1] must be clipped, not wrapped, in the int16 output."""
    loud = np.array([2.0, -3.0, 1.5, -1.5], dtype=np.float32)
    result = resampler.to_48k_pcm(loud)

    samples = struct.unpack(f"{len(result) // 2}h", result)
    assert all(-32768 <= s <= 32767 for s in samples)
    # Some samples must sit at the clip ceiling or floor.
    assert max(samples) == 32767 or min(samples) == -32768
