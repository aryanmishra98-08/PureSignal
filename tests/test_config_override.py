# =============================================================================
# tests/test_config_override.py — config override propagation
#
# --config and --set must reach the flat constants that every pipeline module
# actually reads, not just the nested Config object. A regression here is
# invisible at runtime: the override appears to apply while changing nothing.
# =============================================================================
import importlib.util
import sys
from pathlib import Path

import pytest

import config

_LOADER_PATH = Path(__file__).parent.parent / "config" / "loader.py"


def _load_loader():
    spec = importlib.util.spec_from_file_location(
        "_test_config_loader", _LOADER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


loader = _load_loader()


@pytest.fixture
def restore_config():
    """Snapshot the live config and put it back after the test."""
    original = config.cfg
    yield
    config._bind(original)
    config.cfg = original


@pytest.fixture
def unimported_pipeline(monkeypatch):
    """
    Hide already-imported pipeline modules from rebind()'s ordering guard.

    In a real run they genuinely are not imported yet at rebind time; in the
    test session other modules have pulled them in.
    """
    for name in config._PIPELINE_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)


# ---------------------------------------------------------------------------
# Case 1 — the loader applies the override
# ---------------------------------------------------------------------------

def test_set_arg_reaches_nested_config():
    cfg = loader.load_config(set_args=["vad.hangover_ms=300"])
    assert cfg.vad.hangover_ms == 300


# ---------------------------------------------------------------------------
# Case 2 — the override reaches the flat constant. This is what failed before.
# ---------------------------------------------------------------------------

def test_rebind_updates_flat_constant(restore_config, unimported_pipeline):
    assert config.HANGOVER_MS != 300, "pick a value that differs from the default"
    config.rebind(loader.load_config(set_args=["vad.hangover_ms=300"]))
    assert config.HANGOVER_MS == 300


def test_rebind_updates_derived_constant(restore_config, unimported_pipeline):
    """FRAME_SAMPLES is derived from two other keys — it must be re-derived too."""
    config.rebind(loader.load_config(set_args=["audio.frame_ms=10"]))
    assert config.FRAME_MS == 10
    assert config.FRAME_SAMPLES == int(config.SAMPLE_RATE * 10 / 1000)


def test_rebind_reaches_gatekeeper_mode(restore_config, unimported_pipeline):
    """
    --set extractor.enabled=false is the documented way to reach gatekeeper
    mode and skip installing asteroid, so it has to genuinely take effect.
    """
    config.rebind(loader.load_config(set_args=["extractor.enabled=false"]))
    assert config.EXTRACTOR_ENABLED is False


def test_vad_reads_rebound_hangover(restore_config, unimported_pipeline):
    """The VAD must pick up the new value, not one baked in at import time."""
    config.rebind(loader.load_config(set_args=["vad.hangover_ms=300"]))
    from audio import vad
    assert vad._hangover_frames() == int(300 / config.FRAME_MS)


# ---------------------------------------------------------------------------
# Case 3 — type coercion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected,kind", [
    ("true", True, bool),
    ("false", False, bool),
    ("300", 300, int),
    ("0.5", 0.5, float),
    ("S1", "S1", str),
])
def test_set_arg_type_coercion(raw, expected, kind):
    cfg = loader.load_config(set_args=[f"policy.probe={raw}"])
    value = cfg.policy.probe
    assert value == expected
    assert isinstance(value, kind)


# ---------------------------------------------------------------------------
# Case 4 — unknown nested keys are created, not rejected
# ---------------------------------------------------------------------------

def test_set_arg_creates_missing_nested_key():
    cfg = loader.load_config(set_args=["brand_new.nested.key=7"])
    assert cfg.brand_new.nested.key == 7


# ---------------------------------------------------------------------------
# Case 5 — malformed --set
# ---------------------------------------------------------------------------

def test_malformed_set_arg_raises():
    with pytest.raises(ValueError):
        loader.load_config(set_args=["vad.hangover_ms"])


# ---------------------------------------------------------------------------
# Case 6 — the ordering guard
# ---------------------------------------------------------------------------

def test_rebind_after_pipeline_import_raises(restore_config):
    """
    Rebinding after a pipeline module has imported cannot work: the module has
    already derived its state from the old values. Fail loudly instead.
    """
    import audio.vad  # noqa: F401 — ensure the module is present in sys.modules
    with pytest.raises(RuntimeError, match="rebind"):
        config.rebind(loader.load_config(set_args=["vad.hangover_ms=300"]))
