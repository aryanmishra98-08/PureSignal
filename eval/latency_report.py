"""Per-stage and end-to-end latency analysis.

Reads a session JSONL log (logs/<session>.jsonl), computes P50/P95/P99 per
stage plus the end-to-end distribution, and optionally writes a histogram.

Usage:
    python eval/latency_report.py --log logs/20240101T120000Z.jsonl
    python eval/latency_report.py --log logs/20240101T120000Z.jsonl --plot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple

from common import get_pyplot

_TAG = "latency_report"


class StageStats(NamedTuple):
    """Latency percentiles for one pipeline stage, in milliseconds."""

    name: str
    p50: float
    p95: float
    p99: float
    mean: float
    count: int


def _percentile(values: list[float], p: float) -> float:
    """Return the ``p``-th percentile by linear interpolation."""
    if not values:
        return float("nan")

    ordered = sorted(values)
    idx = (len(ordered) - 1) * p / 100
    lo = int(idx)
    hi = lo + 1
    if hi >= len(ordered):
        return ordered[lo]

    frac = idx - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def _stats(name: str, values: list[float]) -> StageStats:
    """Summarise one stage's latency samples."""
    if not values:
        nan = float("nan")
        return StageStats(name, nan, nan, nan, nan, 0)

    return StageStats(
        name=name,
        p50=_percentile(values, 50),
        p95=_percentile(values, 95),
        p99=_percentile(values, 99),
        mean=sum(values) / len(values),
        count=len(values),
    )


def _elapsed_ms(records: list[dict], start_key: str, end_key: str) -> list[float]:
    """Return the millisecond gap between two timestamps, skipping partial rows."""
    spans = []
    for record in records:
        start, end = record.get(start_key), record.get(end_key)
        if start and end:
            spans.append((end - start) * 1000)
    return spans


def load_latency_records(log_path: Path) -> list[dict]:
    """Return the payload of every pipeline latency_record entry in a log."""
    records = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if (
                entry.get("stage") == "pipeline"
                and entry.get("event") == "latency_record"
            ):
                records.append(entry["data"])
    return records


def compute_report(records: list[dict]) -> list[StageStats]:
    """Summarise per-stage and end-to-end latency across every record."""
    return [
        _stats("encoder", _elapsed_ms(records, "encoder_start_ts", "encoder_done_ts")),
        _stats("policy", _elapsed_ms(records, "encoder_done_ts", "policy_ts")),
        _stats("send", _elapsed_ms(records, "policy_ts", "send_ts")),
        _stats("end_to_end", _elapsed_ms(records, "vad_close_ts", "send_ts")),
    ]


def print_report(stats: list[StageStats]) -> None:
    """Print the stage summary as a fixed-width table."""
    header = (
        f"\n{'Stage':<14} {'P50 ms':>8} {'P95 ms':>8} "
        f"{'P99 ms':>8} {'Mean ms':>9} {'N':>6}"
    )
    print(header)
    print("-" * 58)
    for s in stats:
        print(
            f"{s.name:<14} {s.p50:>8.1f} {s.p95:>8.1f} "
            f"{s.p99:>8.1f} {s.mean:>9.1f} {s.count:>6}"
        )
    print()


def plot_histogram(records: list[dict], output_path: Path) -> None:
    """Write a histogram of end-to-end latency, if matplotlib is available."""
    plt = get_pyplot(_TAG)
    if plt is None:
        return

    e2e_ms = _elapsed_ms(records, "vad_close_ts", "send_ts")
    if not e2e_ms:
        print(f"[{_TAG}] no end-to-end records to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(e2e_ms, bins=30, edgecolor="white", color="steelblue")
    ax.axvline(_percentile(e2e_ms, 50), color="orange", linestyle="--", label="P50")
    ax.axvline(_percentile(e2e_ms, 95), color="red", linestyle="--", label="P95")
    ax.set_xlabel("End-to-end latency (ms)")
    ax.set_ylabel("Segments")
    ax.set_title("Pipeline end-to-end latency distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"[{_TAG}] histogram saved -> {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Latency report from session JSONL log"
    )
    parser.add_argument(
        "--log", required=True, metavar="PATH", help="Path to session .jsonl log"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save histogram PNG alongside the report",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"[{_TAG}] ERROR: log file not found: {log_path}")
        sys.exit(1)

    records = load_latency_records(log_path)
    if not records:
        print(f"[{_TAG}] no latency_record entries found in log")
        sys.exit(1)

    print(f"[{_TAG}] {len(records)} segments from '{log_path.name}'")
    print_report(compute_report(records))

    if args.plot:
        plot_histogram(records, log_path.with_suffix(".latency.png"))


if __name__ == "__main__":
    main()
