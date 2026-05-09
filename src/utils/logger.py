"""
utils/logger.py — Structured dual-output logger.

Every call writes:
  1. Console — human-readable, gated by cfg.debug
  2. logs/<session_id>.jsonl — machine-readable, always written

Each JSONL line:
    {"ts": <unix_float>, "stage": "<module>", "event": "<name>", "data": {...}}

Usage:
    from utils.logger import get_logger
    log = get_logger()
    log("vad", "segment_closed", duration_s=1.2, sample_count=19200)
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import config

_LOGS_DIR = Path(__file__).parent.parent.parent / "logs"
_session_id: str = ""
_log_file_path: Path | None = None
_log_fh = None  # open file handle, kept for the session lifetime


def _ensure_session() -> None:
    global _session_id, _log_file_path, _log_fh
    if _log_fh is not None:
        return
    _session_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    _log_file_path = _LOGS_DIR / f"{_session_id}.jsonl"
    _log_fh = open(_log_file_path, "a", buffering=1)  # line-buffered


def _write(stage: str, event: str, data: dict[str, Any]) -> None:
    _ensure_session()
    record = {"ts": time.time(), "stage": stage, "event": event, "data": data}
    if config.DEBUG:
        print(f"[{stage}] {event} {data}")
    assert _log_fh is not None
    _log_fh.write(json.dumps(record) + "\n")


class Logger:
    """Callable logger bound to the current session."""

    def __call__(self, stage: str, event: str, **data: Any) -> None:
        _write(stage, event, data)

    # Convenience methods matching the plan's per-stage events

    def vad_segment_closed(self, duration_s: float, sample_count: int) -> None:
        _write("vad", "segment_closed", {"duration_s": duration_s, "sample_count": sample_count})

    def encoder_embed_complete(self, latency_ms: float, segment_duration_s: float) -> None:
        _write("encoder", "embed_complete", {"latency_ms": latency_ms, "segment_duration_s": segment_duration_s})

    def encoder_segment_too_short(self, segment_duration_s: float) -> None:
        _write("encoder", "segment_too_short", {"segment_duration_s": segment_duration_s})

    def tracker_speaker_assigned(
        self, speaker_id: str, best_sim: float, gallery_size: int, is_new: bool
    ) -> None:
        _write(
            "tracker",
            "speaker_assigned",
            {"speaker_id": speaker_id, "best_sim": best_sim, "gallery_size": gallery_size, "is_new": is_new},
        )

    def enrollment_match_result(self, matched_name: str | None, best_sim: float) -> None:
        _write("enrollment", "match_result", {"matched_name": matched_name, "best_sim": best_sim})

    def policy_decision(
        self, speaker_id: str, matched_name: str | None, decision: str, mode: str
    ) -> None:
        _write(
            "policy",
            "decision",
            {"speaker_id": speaker_id, "matched_name": matched_name, "decision": decision, "mode": mode},
        )

    def ultravox_segment_sent(self, pcm_bytes: int, queue_depth: int) -> None:
        _write("ultravox", "segment_sent", {"pcm_bytes": pcm_bytes, "queue_depth": queue_depth})

    def latency_record(self, timing: dict[str, float]) -> None:
        _write("pipeline", "latency_record", timing)

    def enrollment_quality(self, username: str, norm: float, entropy: float, quality_pass: bool) -> None:
        _write("enrollment", "quality_check", {"username": username, "norm": norm, "entropy": entropy, "quality_pass": quality_pass})


def get_logger() -> Logger:
    """Return the module-level Logger singleton."""
    return _logger


def get_session_id() -> str:
    _ensure_session()
    return _session_id


def get_log_path() -> Path | None:
    _ensure_session()
    return _log_file_path


def close() -> None:
    """Flush and close the session log file."""
    global _log_fh
    if _log_fh is not None:
        _log_fh.flush()
        _log_fh.close()
        _log_fh = None


_logger = Logger()
