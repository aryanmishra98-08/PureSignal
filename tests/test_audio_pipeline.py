import numpy as np
import pytest
import config


# ---------------------------------------------------------------------------
# audio/vad.py tests
# ---------------------------------------------------------------------------

def _reset_vad():
    """Import vad fresh each test to reset module-level state."""
    from audio import vad
    vad.reset()
    return vad


def test_vad_silence_returns_none():
    vad = _reset_vad()
    frame = np.zeros(config.FRAME_SAMPLES, dtype=np.float32)
    for _ in range(100):
        assert vad.process_frame(frame) is None


def test_vad_speech_segment_returned():
    vad = _reset_vad()
    # Loud speech-like frame (high energy, low ZCR)
    speech_frame = np.ones(config.FRAME_SAMPLES, dtype=np.float32) * 0.5
    silence_frame = np.zeros(config.FRAME_SAMPLES, dtype=np.float32)

    # Feed enough speech frames to open the segment
    for _ in range(10):
        vad.process_frame(speech_frame)

    # Feed silence past the hangover window to close the segment
    hangover_frames = int(config.HANGOVER_MS / config.FRAME_MS)
    segment = None
    for _ in range(hangover_frames + 2):
        result = vad.process_frame(silence_frame)
        if result is not None:
            segment = result

    assert segment is not None
    assert isinstance(segment, np.ndarray)
    assert segment.dtype == np.float32


def test_vad_buffer_no_overflow():
    """Feeding more than 30s of speech must not raise IndexError."""
    vad = _reset_vad()
    speech_frame = np.ones(config.FRAME_SAMPLES, dtype=np.float32) * 0.5
    frames_30s = int(30 * config.SAMPLE_RATE / config.FRAME_SAMPLES) + 10
    for _ in range(frames_30s):
        vad.process_frame(speech_frame)  # should not raise


# ---------------------------------------------------------------------------
# audio/features.py tests
# ---------------------------------------------------------------------------

from audio import features


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
# audio/resampler.py tests
# ---------------------------------------------------------------------------

from audio import resampler


def test_resampler_output_length():
    one_second = np.zeros(config.SAMPLE_RATE, dtype=np.float32)
    result = resampler.to_48k_pcm(one_second)
    expected_bytes = 48000 * 2  # 48kHz, int16
    assert isinstance(result, bytes)
    assert len(result) == expected_bytes


def test_resampler_silence_frame():
    frame = resampler.silence_frame_48k()
    expected_bytes = int(config.ULTRAVOX_IN_RATE * config.ULTRAVOX_CHUNK_MS / 1000) * 2
    assert isinstance(frame, bytes)
    assert len(frame) == expected_bytes
    assert all(b == 0 for b in frame)
