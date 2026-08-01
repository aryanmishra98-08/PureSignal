# =============================================================================
# tests/test_speaker_pipeline.py — Unit tests for the speaker ID stage
#
# Covers: speaker/enrollment.py, speaker/tracker.py, speaker/policy.py,
#         speaker/encoder.py (token validation only — no model load)
# No hardware or HuggingFace download required.
# Run with: pytest tests/test_speaker_pipeline.py -v
# =============================================================================
from unittest.mock import patch

import numpy as np
import pytest

import config
from speaker import encoder, enrollment, policy, tracker


def _unit_vec(dim: int = 256, seed: int = 0) -> np.ndarray:
    """Return a reproducible L2-normalized random vector of length `dim`."""
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


def test_enrollment_match_with_score_returns_name_and_sim(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    emb = _unit_vec(seed=3)
    np.save(tmp_path / "carol.npy", emb)
    enrollment.load_profiles(["carol"])

    name, sim = enrollment.match_with_score(emb)
    assert name == "carol"
    assert pytest.approx(sim, abs=1e-5) == 1.0


def test_enrollment_match_with_score_no_match_returns_none_and_sim(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    emb = _unit_vec(seed=4)
    np.save(tmp_path / "dave.npy", emb)
    enrollment.load_profiles(["dave"])

    other = _orthogonal_vec(emb)
    name, sim = enrollment.match_with_score(other)
    assert name is None
    assert abs(sim) < 1e-4  # orthogonal → near-zero cosine sim


def test_enrollment_validate_quality_normalized_vector():
    """A well-formed L2-normalized embedding should pass the quality gate."""
    emb = _unit_vec(seed=5)
    result = enrollment.validate_enrollment_quality(emb)
    assert result["quality_pass"] is True
    assert pytest.approx(result["norm"], abs=1e-5) == 1.0
    assert result["entropy"] > 0.0


def test_enrollment_validate_quality_zero_vector_fails():
    """A near-zero embedding must fail the quality gate."""
    emb = np.zeros(256, dtype=np.float32)
    result = enrollment.validate_enrollment_quality(emb)
    assert result["quality_pass"] is False


def test_enrollment_save_with_metadata_writes_npy_and_json(tmp_path):
    emb = _unit_vec(seed=6)
    quality = enrollment.validate_enrollment_quality(emb)
    enrollment.save_with_metadata(
        emb, "eve", tmp_path, num_samples=3, quality=quality)

    npy_path = tmp_path / "eve.npy"
    meta_path = tmp_path / "eve_meta.json"
    assert npy_path.exists()
    assert meta_path.exists()

    saved = np.load(npy_path)
    np.testing.assert_array_almost_equal(saved, emb)

    import json
    meta = json.loads(meta_path.read_text())
    assert meta["username"] == "eve"
    assert meta["num_samples"] == 3
    assert "timestamp" in meta
    assert "quality" in meta


# ---------------------------------------------------------------------------
# speaker/tracker.py tests
# ---------------------------------------------------------------------------


def test_tracker_first_speaker_is_S1():
    tracker.reset()
    sid = tracker.assign(_unit_vec(seed=10))
    assert sid == "S1"


def test_tracker_same_speaker_stable():
    tracker.reset()
    emb = _unit_vec(seed=11)
    sid1 = tracker.assign(emb)
    # Slightly perturbed version — should still be same speaker
    perturbed = emb + \
        np.random.default_rng(0).standard_normal(256).astype(np.float32) * 0.01
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


def test_tracker_get_gallery_returns_copy():
    """Mutating the dict returned by get_gallery must not corrupt internal state."""
    tracker.reset()
    emb = _unit_vec(seed=20)
    sid = tracker.assign(emb)

    gallery = tracker.get_gallery()
    gallery[sid] = np.zeros(256, dtype=np.float32)  # corrupt the copy

    # Internal state should be unchanged
    gallery2 = tracker.get_gallery()
    np.testing.assert_array_almost_equal(gallery2[sid], emb)


def test_tracker_ema_centroid_shifts_toward_new_embedding():
    """Centroid must drift when the same speaker is updated with a consistent perturbation."""
    tracker.reset()
    emb_a = _unit_vec(seed=10)
    tracker.assign(emb_a)  # registers S1, centroid = emb_a

    # slerp at t=0.15 — stays similar enough to emb_a to match (sim ≈ 0.985 > threshold)
    # but far enough that EMA updates produce a measurable shift
    emb_c = _unit_vec(seed=99)
    emb_b = 0.85 * emb_a + 0.15 * emb_c
    emb_b /= np.linalg.norm(emb_b)

    sim_before = float(np.dot(tracker.get_gallery()["S1"], emb_b))

    for _ in range(20):
        tracker.assign(emb_b)

    sim_after = float(np.dot(tracker.get_gallery()["S1"], emb_b))
    assert sim_after > sim_before


# ---------------------------------------------------------------------------
# speaker/policy.py tests
# ---------------------------------------------------------------------------


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


def test_encoder_hf_token_missing_raises(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(EnvironmentError):
        encoder._check_hf_token()
