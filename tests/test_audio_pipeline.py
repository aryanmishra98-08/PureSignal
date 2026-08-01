# =============================================================================
# tests/test_audio_pipeline.py — Unit tests for the audio processing stage
#
# Covers: audio/vad.py, audio/features.py, audio/resampler.py
# No hardware required — all tests use synthetic numpy arrays.
# Run with: pytest tests/test_audio_pipeline.py -v
# =============================================================================
import numpy as np

import config
from audio import features, resampler

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

    n_speech = 10
    for _ in range(n_speech):
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
    # Segment must contain at least the speech frames plus the hangover frames
    min_expected = (n_speech + hangover_frames) * config.FRAME_SAMPLES
    assert len(segment) >= min_expected


def test_vad_segment_content_matches_input():
    """Segment returned by process_frame must contain the frames that were fed in."""
    vad = _reset_vad()
    speech_frame = np.full(config.FRAME_SAMPLES, 0.5, dtype=np.float32)
    silence_frame = np.zeros(config.FRAME_SAMPLES, dtype=np.float32)

    for _ in range(5):
        vad.process_frame(speech_frame)

    hangover_frames = int(config.HANGOVER_MS / config.FRAME_MS)
    segment = None
    for _ in range(hangover_frames + 2):
        result = vad.process_frame(silence_frame)
        if result is not None:
            segment = result

    assert segment is not None
    # The first FRAME_SAMPLES values must equal the speech frames
    np.testing.assert_array_equal(segment[:config.FRAME_SAMPLES], speech_frame)


def test_vad_flush_after_closed_segment_returns_none():
    """flush() must return None when the last segment was already closed by process_frame."""
    vad = _reset_vad()
    speech_frame = np.ones(config.FRAME_SAMPLES, dtype=np.float32) * 0.5
    silence_frame = np.zeros(config.FRAME_SAMPLES, dtype=np.float32)

    for _ in range(5):
        vad.process_frame(speech_frame)

    hangover_frames = int(config.HANGOVER_MS / config.FRAME_MS)
    for _ in range(hangover_frames + 2):
        vad.process_frame(silence_frame)

    # Segment was already returned by process_frame — flush should find nothing
    assert vad.flush() is None


def test_vad_buffer_no_overflow():
    """Feeding more than 30s of speech must not raise IndexError."""
    vad = _reset_vad()
    speech_frame = np.ones(config.FRAME_SAMPLES, dtype=np.float32) * 0.5
    frames_30s = int(30 * config.SAMPLE_RATE / config.FRAME_SAMPLES) + 10
    for _ in range(frames_30s):
        vad.process_frame(speech_frame)  # should not raise


def test_vad_flush_returns_active_segment():
    """flush() must return buffered speech when speech is active and no trailing silence arrives."""
    vad = _reset_vad()
    speech_frame = np.ones(config.FRAME_SAMPLES, dtype=np.float32) * 0.5
    for _ in range(10):
        vad.process_frame(speech_frame)
    segment = vad.flush()
    assert segment is not None
    assert isinstance(segment, np.ndarray)
    assert len(segment) > 0


def test_vad_flush_returns_none_when_silent():
    """flush() must return None if no speech has been buffered."""
    vad = _reset_vad()
    silence_frame = np.zeros(config.FRAME_SAMPLES, dtype=np.float32)
    for _ in range(5):
        vad.process_frame(silence_frame)
    assert vad.flush() is None


# ---------------------------------------------------------------------------
# audio/features.py tests
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
# audio/resampler.py tests
# ---------------------------------------------------------------------------


def test_resampler_output_length():
    one_second = np.zeros(config.SAMPLE_RATE, dtype=np.float32)
    result = resampler.to_48k_pcm(one_second)
    expected_bytes = 48000 * 2  # 48kHz, int16
    assert isinstance(result, bytes)
    assert len(result) == expected_bytes


def test_resampler_silence_frame():
    frame = resampler.silence_frame_48k()
    expected_bytes = int(config.ULTRAVOX_IN_RATE *
                         config.ULTRAVOX_CHUNK_MS / 1000) * 2
    assert isinstance(frame, bytes)
    assert len(frame) == expected_bytes
    assert all(b == 0 for b in frame)


def test_resampler_clipping_prevents_wraparound():
    """Values outside [-1, 1] must be clipped, not wrapped, in the int16 output."""
    import struct
    # Signal with values well beyond [-1, 1]
    loud = np.array([2.0, -3.0, 1.5, -1.5], dtype=np.float32)
    result = resampler.to_48k_pcm(loud)
    samples = struct.unpack(f"{len(result) // 2}h", result)
    # All samples must be within int16 range without sign flip from wraparound
    assert all(-32768 <= s <= 32767 for s in samples)
    # At least some samples must be at the clip ceiling/floor
    assert max(samples) == 32767 or min(samples) == -32768
