"""Structured dual-output logger.

Every call writes two places:
  1. Console — human-readable, see the level rules below.
  2. logs/<session_id>.jsonl — machine-readable, always written at every level.

Levels decide only what reaches the console:
  DEBUG             shown only when ``config.DEBUG`` is set
  INFO/WARNING/ERROR always shown

This is the diagnostic channel. Interactive output — menus, prompts, banners
and the results a command was run to produce — stays on plain print(), so it
is never suppressed by a log level or mixed into the JSONL.

Each JSONL line:
    {"ts": <unix_float>, "level": "<LEVEL>", "stage": "<module>",
     "event": "<name>", "data": {...}}

``event`` is a snake_case slug for the typed pipeline events below, and the
human-readable message for the free-form debug/info/warning/error calls.

Usage:
    from utils.logger import get_logger
    log = get_logger()
    log.info("capture", f"mic open — {rate}Hz")
    log.vad_segment_closed(duration_s=1.2, sample_count=19200)
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import config

DEBUG = "DEBUG"
INFO = "INFO"
WARNING = "WARNING"
ERROR = "ERROR"

_LOGS_DIR = Path(__file__).parent.parent.parent / "logs"
_log_file_path: Path | None = None
_log_fh: TextIO | None = None


def _ensure_session() -> TextIO:
    """Open this session's log file on first use and return the handle."""
    global _log_file_path, _log_fh
    if _log_fh is None:
        session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        _LOGS_DIR.mkdir(parents=True, exist_ok=True)
        _log_file_path = _LOGS_DIR / f"{session_id}.jsonl"
        _log_fh = open(_log_file_path, "a", buffering=1, encoding="utf-8")
    return _log_fh


def _write(
    stage: str, event: str, data: dict[str, Any], level: str = DEBUG
) -> None:
    """Write one record to the console (level permitting) and the JSONL file."""
    handle = _ensure_session()

    if level != DEBUG or config.DEBUG:
        suffix = f" {data}" if data else ""
        print(f"[{stage}] {event}{suffix}")

    handle.write(json.dumps({
        "ts": time.time(),
        "level": level,
        "stage": stage,
        "event": event,
        "data": data,
    }) + "\n")


class Logger:
    """Callable logger bound to the current session."""

    def __call__(self, stage: str, event: str, **data: Any) -> None:
        _write(stage, event, data)

    # -- level-based diagnostics ----------------------------------------

    def debug(self, stage: str, message: str, **data: Any) -> None:
        """Console only when config.DEBUG is set; always written to the log."""
        _write(stage, message, data, DEBUG)

    def info(self, stage: str, message: str, **data: Any) -> None:
        """Normal operational status — always shown."""
        _write(stage, message, data, INFO)

    def warning(self, stage: str, message: str, **data: Any) -> None:
        """Something recoverable that the operator should know about."""
        _write(stage, message, data, WARNING)

    def error(self, stage: str, message: str, **data: Any) -> None:
        """A failure. Printing is the caller's exit path, not a decision here."""
        _write(stage, message, data, ERROR)

    # -- typed pipeline events (DEBUG level) ----------------------------

    def vad_segment_closed(self, duration_s: float, sample_count: int) -> None:
        _write("vad", "segment_closed", {
            "duration_s": duration_s,
            "sample_count": sample_count,
        })

    def encoder_embed_complete(
        self, latency_ms: float, segment_duration_s: float
    ) -> None:
        _write("encoder", "embed_complete", {
            "latency_ms": latency_ms,
            "segment_duration_s": segment_duration_s,
        })

    def encoder_segment_too_short(self, segment_duration_s: float) -> None:
        _write("encoder", "segment_too_short", {
            "segment_duration_s": segment_duration_s,
        })

    def tracker_speaker_assigned(
        self, speaker_id: str, best_sim: float, gallery_size: int, is_new: bool
    ) -> None:
        _write("tracker", "speaker_assigned", {
            "speaker_id": speaker_id,
            "best_sim": best_sim,
            "gallery_size": gallery_size,
            "is_new": is_new,
        })

    def policy_decision(
        self, speaker_id: str, matched_name: str | None, decision: str, mode: str
    ) -> None:
        _write("policy", "decision", {
            "speaker_id": speaker_id,
            "matched_name": matched_name,
            "decision": decision,
            "mode": mode,
        })

    def ultravox_segment_sent(self, pcm_bytes: int, queue_depth: int) -> None:
        _write("ultravox", "segment_sent", {
            "pcm_bytes": pcm_bytes,
            "queue_depth": queue_depth,
        })

    def latency_record(self, timing: dict[str, float]) -> None:
        _write("pipeline", "latency_record", timing)


_logger = Logger()


def get_logger() -> Logger:
    """Return the module-level Logger singleton."""
    return _logger


def get_log_path() -> Path | None:
    """Return the path of this session's log file, opening it if needed."""
    _ensure_session()
    return _log_file_path


def close() -> None:
    """Flush and close the session log file."""
    global _log_fh
    if _log_fh is not None:
        _log_fh.flush()
        _log_fh.close()
        _log_fh = None
