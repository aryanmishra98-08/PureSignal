import numpy as np
import pytest
from unittest.mock import patch
import config


def _unit_vec(dim: int = 256, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _orthogonal_vec(v: np.ndarray) -> np.ndarray:
    """Return a vector orthogonal to v (cosine sim == 0)."""
    rng = np.random.default_rng(42)
    candidate = rng.standard_normal(len(v)).astype(np.float32)
    candidate -= np.dot(candidate, v) * v
    return candidate / np.linalg.norm(candidate)


# ---------------------------------------------------------------------------
# speaker/enrollment.py tests
# ---------------------------------------------------------------------------

from speaker import enrollment


def test_enrollment_load_match(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    emb = _unit_vec(seed=1)
    np.save(tmp_path / "alice.npy", emb)

    enrollment.load_profiles(["alice"])
    assert enrollment.match(emb) == "alice"


def test_enrollment_match_orthogonal(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    emb = _unit_vec(seed=2)
    np.save(tmp_path / "bob.npy", emb)

    enrollment.load_profiles(["bob"])
    other = _orthogonal_vec(emb)
    assert enrollment.match(other) is None


def test_enrollment_load_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        enrollment.load_profiles(["ghost"])


# ---------------------------------------------------------------------------
# speaker/tracker.py tests
# ---------------------------------------------------------------------------

from speaker import tracker


def test_tracker_first_speaker_is_S1():
    tracker.reset()
    sid = tracker.assign(_unit_vec(seed=10))
    assert sid == "S1"


def test_tracker_same_speaker_stable():
    tracker.reset()
    emb = _unit_vec(seed=11)
    sid1 = tracker.assign(emb)
    # Slightly perturbed version — should still be same speaker
    perturbed = emb + np.random.default_rng(0).standard_normal(256).astype(np.float32) * 0.01
    perturbed /= np.linalg.norm(perturbed)
    sid2 = tracker.assign(perturbed)
    assert sid1 == sid2 == "S1"


def test_tracker_different_speaker_new_id():
    tracker.reset()
    emb_a = _unit_vec(seed=12)
    emb_b = _orthogonal_vec(emb_a)
    sid1 = tracker.assign(emb_a)
    sid2 = tracker.assign(emb_b)
    assert sid1 == "S1"
    assert sid2 == "S2"


def test_tracker_gallery_full_assigns_closest():
    tracker.reset()
    vecs = [_unit_vec(seed=i) for i in range(config.MAX_SPEAKERS + 1)]
    # Orthogonalise each against previous to force new speaker registrations
    ortho = [vecs[0]]
    for v in vecs[1:]:
        for basis in ortho:
            v = v - np.dot(v, basis) * basis
            n = np.linalg.norm(v)
            if n > 1e-6:
                v /= n
        ortho.append(v)

    ids = [tracker.assign(v) for v in ortho]
    # Last assignment must reuse an existing ID (gallery full)
    assert ids[-1] in ids[:-1]


# ---------------------------------------------------------------------------
# speaker/policy.py tests
# ---------------------------------------------------------------------------

from speaker import policy


def test_policy_enrolled_pass(monkeypatch):
    monkeypatch.setattr(config, "POLICY_MODE", "ENROLLED")
    with patch("speaker.enrollment.match", return_value="alice"):
        assert policy.should_pass("S1", _unit_vec()) is True


def test_policy_enrolled_drop(monkeypatch):
    monkeypatch.setattr(config, "POLICY_MODE", "ENROLLED")
    with patch("speaker.enrollment.match", return_value=None):
        assert policy.should_pass("S1", _unit_vec()) is False


def test_policy_dynamic_pass(monkeypatch):
    monkeypatch.setattr(config, "POLICY_MODE", "DYNAMIC")
    monkeypatch.setattr(config, "DYNAMIC_TARGET", "S1")
    assert policy.should_pass("S1", _unit_vec()) is True


def test_policy_dynamic_drop(monkeypatch):
    monkeypatch.setattr(config, "POLICY_MODE", "DYNAMIC")
    monkeypatch.setattr(config, "DYNAMIC_TARGET", "S1")
    assert policy.should_pass("S2", _unit_vec()) is False


def test_policy_invalid_mode_raises(monkeypatch):
    monkeypatch.setattr(config, "POLICY_MODE", "INVALID")
    with pytest.raises(ValueError):
        policy.should_pass("S1", _unit_vec())


# ---------------------------------------------------------------------------
# speaker/encoder.py tests (token validation only — no model load)
# ---------------------------------------------------------------------------

from speaker import encoder


def test_encoder_hf_token_missing_raises(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(EnvironmentError):
        encoder._check_hf_token()
