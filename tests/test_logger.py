"""Tests for the level rules that split diagnostics from interactive output."""

import json

import pytest

import config
from utils import logger as logger_module
from utils.logger import get_logger


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    """Point the logger at a temp directory and give it a fresh handle."""
    logger_module.close()
    monkeypatch.setattr(logger_module, "_LOGS_DIR", tmp_path)
    yield tmp_path
    logger_module.close()


def _records(log_dir) -> list[dict]:
    """Read back everything written this session."""
    logger_module.close()  # flush
    files = list(log_dir.glob("*.jsonl"))
    assert len(files) == 1, f"expected one log file, got {files}"
    return [
        json.loads(line)
        for line in files[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# Console rules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("level_method", ["info", "warning", "error"])
def test_info_and_above_print_even_with_debug_off(
    log_dir, capsys, monkeypatch, level_method
):
    """Operational status must stay visible when debug is off."""
    monkeypatch.setattr(config, "DEBUG", False)

    getattr(get_logger(), level_method)("capture", "mic open")

    assert "[capture] mic open" in capsys.readouterr().out


def test_debug_is_suppressed_when_debug_off(log_dir, capsys, monkeypatch):
    monkeypatch.setattr(config, "DEBUG", False)

    get_logger().debug("main", "encoder busy")

    assert capsys.readouterr().out == ""


def test_debug_prints_when_debug_on(log_dir, capsys, monkeypatch):
    monkeypatch.setattr(config, "DEBUG", True)

    get_logger().debug("main", "encoder busy")

    assert "[main] encoder busy" in capsys.readouterr().out


def test_data_is_appended_to_the_console_line(log_dir, capsys, monkeypatch):
    monkeypatch.setattr(config, "DEBUG", False)

    get_logger().info("ultravox", "connected", attempt=2)

    out = capsys.readouterr().out
    assert "[ultravox] connected" in out
    assert "'attempt': 2" in out


def test_no_trailing_braces_when_there_is_no_data(log_dir, capsys, monkeypatch):
    monkeypatch.setattr(config, "DEBUG", False)

    get_logger().info("capture", "mic closed")

    assert capsys.readouterr().out.strip() == "[capture] mic closed"


# ---------------------------------------------------------------------------
# File rules — every level is written regardless of console visibility
# ---------------------------------------------------------------------------

def test_suppressed_debug_is_still_written_to_the_log(log_dir, monkeypatch):
    """Console gating must not lose diagnostics from the log file."""
    monkeypatch.setattr(config, "DEBUG", False)

    get_logger().debug("main", "encoder busy")

    records = _records(log_dir)
    assert len(records) == 1
    assert records[0]["level"] == "DEBUG"
    assert records[0]["event"] == "encoder busy"


def test_every_level_is_recorded_with_its_level(log_dir, monkeypatch):
    monkeypatch.setattr(config, "DEBUG", False)
    log = get_logger()

    log.debug("a", "d")
    log.info("b", "i")
    log.warning("c", "w")
    log.error("d", "e")

    assert [r["level"] for r in _records(log_dir)] == [
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
    ]


def test_record_shape_is_stable(log_dir, monkeypatch):
    """The JSONL contract other tools parse must not drift."""
    monkeypatch.setattr(config, "DEBUG", False)

    get_logger().info("capture", "mic open", rate=16000)

    record = _records(log_dir)[0]
    assert set(record) == {"ts", "level", "stage", "event", "data"}
    assert record["stage"] == "capture"
    assert record["data"] == {"rate": 16000}
    assert isinstance(record["ts"], float)


# ---------------------------------------------------------------------------
# Typed pipeline events stay DEBUG, as they were before levels existed
# ---------------------------------------------------------------------------

def test_typed_events_are_debug_level(log_dir, capsys, monkeypatch):
    monkeypatch.setattr(config, "DEBUG", False)

    get_logger().vad_segment_closed(duration_s=1.2, sample_count=19200)

    assert capsys.readouterr().out == ""  # gated, as before
    record = _records(log_dir)[0]
    assert record["level"] == "DEBUG"
    assert record["stage"] == "vad"
    assert record["event"] == "segment_closed"


def test_latency_record_still_parses_for_the_report(log_dir, monkeypatch):
    """eval/latency_report.py keys off stage and event — keep them intact."""
    monkeypatch.setattr(config, "DEBUG", False)
    timing = {"vad_close_ts": 1.0, "send_ts": 2.0}

    get_logger().latency_record(timing)

    record = _records(log_dir)[0]
    assert record["stage"] == "pipeline"
    assert record["event"] == "latency_record"
    assert record["data"] == timing


def test_callable_form_still_works(log_dir, monkeypatch):
    monkeypatch.setattr(config, "DEBUG", False)

    get_logger()("custom", "thing_happened", count=3)

    record = _records(log_dir)[0]
    assert record["stage"] == "custom"
    assert record["event"] == "thing_happened"
    assert record["data"] == {"count": 3}
