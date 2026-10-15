"""Tests for the config surface and the shared WAV loader."""

import numpy as np
import pytest
from scipy.io import wavfile

import config
from audio.wav_io import load_wav


@pytest.fixture(autouse=True)
def restore_config():
    """Reset config back to base.yaml after tests that override it."""
    yield
    config.reload()


# ---------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------

def test_config_constants_derived_from_yaml():
    assert config.SAMPLE_RATE == config.cfg.audio.sample_rate
    assert config.HANGOVER_MS == config.cfg.vad.hangover_ms
    assert config.FRAME_SAMPLES == int(
        config.SAMPLE_RATE * config.FRAME_MS / 1000
    )


def test_config_reload_applies_set_overrides():
    """--set overrides must reach the module-level constants, not just cfg."""
    config.reload(set_args=["vad.hangover_ms=123"])

    assert config.HANGOVER_MS == 123
    assert config.cfg.vad.hangover_ms == 123


def test_config_reload_recomputes_derived_constants():
    """Derived constants must be re-derived, not left stale."""
    config.reload(set_args=["audio.sample_rate=8000"])

    assert config.SAMPLE_RATE == 8000
    assert config.FRAME_SAMPLES == int(8000 * config.FRAME_MS / 1000)


def test_config_reload_restores_defaults():
    config.reload(set_args=["vad.hangover_ms=999"])
    assert config.HANGOVER_MS == 999

    config.reload()
    assert config.HANGOVER_MS == 400


def test_config_reload_rejects_malformed_set_arg():
    with pytest.raises(ValueError):
        config.reload(set_args=["missing_equals_sign"])


# ---------------------------------------------------------------------------
# audio/wav_io.py
# ---------------------------------------------------------------------------

def _write_wav(path, data, rate):
    wavfile.write(path, rate, data)
    return path


def test_load_wav_int16_scaled_to_unit_range(tmp_path):
    data = np.array([0, 16384, -16384, 32767], dtype=np.int16)
    path = _write_wav(tmp_path / "int16.wav", data, config.SAMPLE_RATE)

    audio = load_wav(path)

    assert audio.dtype == np.float32
    assert np.max(np.abs(audio)) <= 1.0
    np.testing.assert_allclose(audio[1], 0.5, atol=1e-4)


def test_load_wav_float32_passthrough(tmp_path):
    data = np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32)
    path = _write_wav(tmp_path / "float32.wav", data, config.SAMPLE_RATE)

    np.testing.assert_allclose(load_wav(path), data, atol=1e-6)


def test_load_wav_stereo_downmixed_to_mono(tmp_path):
    stereo = np.array([[0.5, -0.5], [1.0, 0.0]], dtype=np.float32)
    path = _write_wav(tmp_path / "stereo.wav", stereo, config.SAMPLE_RATE)

    audio = load_wav(path)

    assert audio.ndim == 1
    np.testing.assert_allclose(audio, [0.0, 0.5], atol=1e-6)


def test_load_wav_resamples_to_target_rate(tmp_path):
    one_second_at_8k = np.zeros(8000, dtype=np.float32)
    path = _write_wav(tmp_path / "8k.wav", one_second_at_8k, 8000)

    audio = load_wav(path, target_rate=16000)

    assert len(audio) == pytest.approx(16000, abs=2)
    assert audio.dtype == np.float32


def test_load_wav_defaults_to_config_sample_rate(tmp_path):
    half_second_at_8k = np.zeros(4000, dtype=np.float32)
    path = _write_wav(tmp_path / "default.wav", half_second_at_8k, 8000)

    audio = load_wav(path)

    assert len(audio) == pytest.approx(config.SAMPLE_RATE // 2, abs=2)
