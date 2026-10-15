"""Tests for the speaker stages: enrollment, tracker, policy and encoder."""

import json

import numpy as np
import pytest

import config
from speaker import encoder as encoder_module
from speaker.encoder import SpeakerEncoder, check_hf_token
from speaker.enrollment import (
    EnrollmentStore,
    save_with_metadata,
    validate_enrollment_quality,
)
from speaker.policy import PolicyGate
from speaker.tracker import SpeakerTracker


def _unit_vec(dim: int = 256, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _orthogonal_vec(v: np.ndarray) -> np.ndarray:
    """Return a vector orthogonal to v (cosine similarity 0)."""
    rng = np.random.default_rng(42)
    candidate = rng.standard_normal(len(v)).astype(np.float32)
    candidate -= np.dot(candidate, v) * v
    return candidate / np.linalg.norm(candidate)


def _write_profile(tmp_path, name: str, seed: int) -> np.ndarray:
    emb = _unit_vec(seed=seed)
    np.save(tmp_path / f"{name}.npy", emb)
    return emb


# ---------------------------------------------------------------------------
# speaker/enrollment.py
# ---------------------------------------------------------------------------

def test_enrollment_load_match(tmp_path):
    emb = _write_profile(tmp_path, "alice", seed=1)
    store = EnrollmentStore.load(["alice"], profiles_dir=tmp_path)
    assert store.match(emb) == "alice"


def test_enrollment_match_orthogonal(tmp_path):
    emb = _write_profile(tmp_path, "bob", seed=2)
    store = EnrollmentStore.load(["bob"], profiles_dir=tmp_path)
    assert store.match(_orthogonal_vec(emb)) is None


def test_enrollment_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        EnrollmentStore.load(["ghost"], profiles_dir=tmp_path)


def test_enrollment_failed_load_yields_no_store(tmp_path):
    """A failed load must produce no store at all, so nothing can consult it."""
    _write_profile(tmp_path, "alice", seed=1)

    with pytest.raises(FileNotFoundError):
        EnrollmentStore.load(["alice", "ghost"], profiles_dir=tmp_path)


def test_enrollment_defaults_to_config_profiles_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    emb = _write_profile(tmp_path, "frank", seed=7)

    store = EnrollmentStore.load(["frank"])

    assert store.match(emb) == "frank"


def test_enrollment_match_with_score_returns_name_and_sim(tmp_path):
    emb = _write_profile(tmp_path, "carol", seed=3)
    store = EnrollmentStore.load(["carol"], profiles_dir=tmp_path)

    name, sim = store.match_with_score(emb)

    assert name == "carol"
    assert pytest.approx(sim, abs=1e-5) == 1.0


def test_enrollment_match_with_score_no_match_returns_none_and_sim(tmp_path):
    emb = _write_profile(tmp_path, "dave", seed=4)
    store = EnrollmentStore.load(["dave"], profiles_dir=tmp_path)

    name, sim = store.match_with_score(_orthogonal_vec(emb))

    assert name is None
    assert abs(sim) < 1e-4  # orthogonal -> near-zero cosine similarity


def test_enrollment_empty_store_matches_nothing():
    store = EnrollmentStore()
    name, sim = store.match_with_score(_unit_vec())
    assert name is None
    assert sim == -1.0
    assert len(store) == 0


def test_enrollment_stores_are_independent(tmp_path):
    """Two stores must not share profiles."""
    emb_a = _write_profile(tmp_path, "alice", seed=1)
    emb_b = _write_profile(tmp_path, "bob", seed=2)

    store_a = EnrollmentStore.load(["alice"], profiles_dir=tmp_path)
    store_b = EnrollmentStore.load(["bob"], profiles_dir=tmp_path)

    assert store_a.match(emb_a) == "alice"
    assert store_a.match(emb_b) is None
    assert store_b.match(emb_b) == "bob"
    assert store_b.names() == ["bob"]


def test_enrollment_validate_quality_normalized_vector():
    """A well-formed L2-normalized embedding passes the quality gate."""
    result = validate_enrollment_quality(_unit_vec(seed=5))
    assert result["quality_pass"] is True
    assert pytest.approx(result["norm"], abs=1e-5) == 1.0
    assert result["entropy"] > 0.0


def test_enrollment_validate_quality_zero_vector_fails():
    result = validate_enrollment_quality(np.zeros(256, dtype=np.float32))
    assert result["quality_pass"] is False


def test_enrollment_save_with_metadata_writes_npy_and_json(tmp_path):
    emb = _unit_vec(seed=6)
    quality = validate_enrollment_quality(emb)

    save_with_metadata(emb, "eve", tmp_path, num_samples=3, quality=quality)

    npy_path = tmp_path / "eve.npy"
    meta_path = tmp_path / "eve_meta.json"
    assert npy_path.exists()
    assert meta_path.exists()
    np.testing.assert_array_almost_equal(np.load(npy_path), emb)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["username"] == "eve"
    assert meta["num_samples"] == 3
    assert "timestamp" in meta
    assert "quality" in meta


# ---------------------------------------------------------------------------
# speaker/tracker.py
# ---------------------------------------------------------------------------

def test_tracker_first_speaker_is_s1():
    assert SpeakerTracker().assign(_unit_vec(seed=10)) == "S1"


def test_tracker_same_speaker_stable():
    tracker = SpeakerTracker()
    emb = _unit_vec(seed=11)
    sid1 = tracker.assign(emb)

    perturbed = emb + _unit_vec(seed=77) * 0.01
    perturbed /= np.linalg.norm(perturbed)
    sid2 = tracker.assign(perturbed)

    assert sid1 == sid2 == "S1"


def test_tracker_different_speaker_new_id():
    tracker = SpeakerTracker()
    emb_a = _unit_vec(seed=12)
    assert tracker.assign(emb_a) == "S1"
    assert tracker.assign(_orthogonal_vec(emb_a)) == "S2"


def test_tracker_gallery_full_assigns_closest():
    tracker = SpeakerTracker()
    vecs = [_unit_vec(seed=i) for i in range(tracker.max_speakers + 1)]

    # Orthogonalise each against the previous ones to force new registrations.
    ortho = [vecs[0]]
    for v in vecs[1:]:
        for basis in ortho:
            v = v - np.dot(v, basis) * basis
            norm = np.linalg.norm(v)
            if norm > 1e-6:
                v /= norm
        ortho.append(v)

    ids = [tracker.assign(v) for v in ortho]
    assert ids[-1] in ids[:-1]  # gallery full -> reuses an existing ID
    assert len(tracker) == tracker.max_speakers


def test_tracker_gallery_returns_copy():
    """Mutating the dict from gallery() must not corrupt internal state."""
    tracker = SpeakerTracker()
    emb = _unit_vec(seed=20)
    sid = tracker.assign(emb)

    gallery = tracker.gallery()
    gallery[sid] = np.zeros(256, dtype=np.float32)

    np.testing.assert_array_almost_equal(tracker.gallery()[sid], emb)


def test_tracker_ema_centroid_shifts_toward_new_embedding():
    """The centroid drifts when a speaker is updated with a consistent offset."""
    tracker = SpeakerTracker()
    emb_a = _unit_vec(seed=10)
    tracker.assign(emb_a)  # registers S1 with centroid emb_a

    # Close enough to emb_a to keep matching, far enough to shift the centroid.
    emb_b = 0.85 * emb_a + 0.15 * _unit_vec(seed=99)
    emb_b /= np.linalg.norm(emb_b)

    sim_before = float(np.dot(tracker.gallery()["S1"], emb_b))
    for _ in range(20):
        tracker.assign(emb_b)
    sim_after = float(np.dot(tracker.gallery()["S1"], emb_b))

    assert sim_after > sim_before


def test_tracker_assign_with_score_reports_similarity():
    """assign_with_score must report the similarity that drove the assignment."""
    tracker = SpeakerTracker()
    emb = _unit_vec(seed=30)

    first_id, first_sim = tracker.assign_with_score(emb)
    assert first_id == "S1"
    assert first_sim == 0.0  # nothing to compare the first speaker against

    same_id, same_sim = tracker.assign_with_score(emb)
    assert same_id == "S1"
    assert pytest.approx(same_sim, abs=1e-5) == 1.0


def test_tracker_speaker_ids_tracks_gallery():
    tracker = SpeakerTracker()
    assert tracker.speaker_ids() == set()

    emb_a = _unit_vec(seed=12)
    tracker.assign(emb_a)
    tracker.assign(_orthogonal_vec(emb_a))

    assert tracker.speaker_ids() == {"S1", "S2"}
    assert len(tracker) == 2


def test_tracker_instances_are_independent():
    """Two trackers must not share a gallery — the point of dropping globals."""
    a = SpeakerTracker()
    b = SpeakerTracker()

    a.assign(_unit_vec(seed=40))
    a.assign(_orthogonal_vec(_unit_vec(seed=40)))

    assert len(a) == 2
    assert len(b) == 0
    assert b.assign(_unit_vec(seed=41)) == "S1"  # b starts its own numbering


def test_tracker_reset_restarts_numbering():
    tracker = SpeakerTracker()
    tracker.assign(_unit_vec(seed=50))
    tracker.reset()

    assert len(tracker) == 0
    assert tracker.assign(_unit_vec(seed=51)) == "S1"


def test_tracker_honours_constructor_overrides():
    tracker = SpeakerTracker(max_speakers=1, similarity_threshold=0.99)
    emb_a = _unit_vec(seed=60)

    assert tracker.assign(emb_a) == "S1"
    # Gallery is full at one speaker, so the orthogonal vector reuses S1.
    assert tracker.assign(_orthogonal_vec(emb_a)) == "S1"
    assert len(tracker) == 1


# ---------------------------------------------------------------------------
# speaker/policy.py
# ---------------------------------------------------------------------------

class _FakeStore:
    """Minimal EnrollmentStore stand-in for gate tests."""

    def __init__(self, result):
        self._result = result

    def match(self, embedding):
        return self._result


def test_policy_enrolled_pass():
    gate = PolicyGate(_FakeStore("alice"), mode="ENROLLED")
    assert gate.should_pass("S1", _unit_vec()) is True


def test_policy_enrolled_drop():
    gate = PolicyGate(_FakeStore(None), mode="ENROLLED")
    assert gate.should_pass("S1", _unit_vec()) is False


def test_policy_dynamic_pass():
    gate = PolicyGate(_FakeStore(None), mode="DYNAMIC", dynamic_target="S1")
    assert gate.should_pass("S1", _unit_vec()) is True


def test_policy_dynamic_drop():
    gate = PolicyGate(_FakeStore(None), mode="DYNAMIC", dynamic_target="S1")
    assert gate.should_pass("S2", _unit_vec()) is False


def test_policy_invalid_mode_raises_at_construction():
    """An unknown mode must fail at startup, not on the first utterance."""
    with pytest.raises(ValueError):
        PolicyGate(_FakeStore(None), mode="INVALID")


def test_policy_defaults_come_from_config(monkeypatch):
    monkeypatch.setattr(config, "POLICY_MODE", "DYNAMIC")
    monkeypatch.setattr(config, "DYNAMIC_TARGET", "S3")

    gate = PolicyGate(_FakeStore(None))

    assert gate.mode == "DYNAMIC"
    assert gate.should_pass("S3", _unit_vec()) is True


# ---------------------------------------------------------------------------
# speaker/encoder.py — input validation only; no model load
# ---------------------------------------------------------------------------

def test_encoder_hf_token_missing_raises(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(OSError):
        check_hf_token()


def test_encoder_hf_token_present_returned(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    assert check_hf_token() == "test-token"


def test_encoder_embed_before_load_raises():
    encoder = SpeakerEncoder()
    assert encoder.is_loaded is False
    with pytest.raises(RuntimeError):
        encoder.embed(np.zeros(config.SAMPLE_RATE, dtype=np.float32))


def test_encoder_reads_settings_from_config():
    encoder = SpeakerEncoder()
    assert encoder.model_name == config.ENCODER_MODEL
    assert encoder.sample_rate == config.SAMPLE_RATE


def test_encoder_honours_constructor_overrides():
    encoder = SpeakerEncoder(model_name="some/other-model", min_segment_s=2.0)
    assert encoder.model_name == "some/other-model"
    assert encoder.min_segment_s == 2.0


def test_encoder_short_segment_returns_none(monkeypatch):
    """A clip below min_segment_s is rejected before the model is consulted."""
    encoder = SpeakerEncoder(min_segment_s=1.0)
    # Pretend the model is loaded; a short clip must not reach it.
    monkeypatch.setattr(
        encoder, "_inference", lambda *a, **k: pytest.fail("model was consulted")
    )
    assert encoder.embed(np.zeros(100, dtype=np.float32)) is None


def test_encoder_module_exposes_check_hf_token():
    """The token check is public: callers validate before paying to load."""
    assert callable(encoder_module.check_hf_token)
