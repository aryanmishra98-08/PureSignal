"""
eval/latency_report.py — Per-stage and end-to-end latency analysis.

Reads a session JSONL log (logs/<session>.jsonl) and computes
P50 / P95 / P99 latency per stage plus an end-to-end distribution.
Optionally writes a histogram PNG.

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


class StageStats(NamedTuple):
    name: str
    p50: float
    p95: float
    p99: float
    mean: float
    count: int


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    sorted_v = sorted(values)
    idx = (len(sorted_v) - 1) * p / 100
    lo = int(idx)
    hi = lo + 1
    frac = idx - lo
    if hi >= len(sorted_v):
        return sorted_v[lo]
    return sorted_v[lo] * (1 - frac) + sorted_v[hi] * frac


def _stats(name: str, values: list[float]) -> StageStats:
    if not values:
        return StageStats(name, float("nan"), float("nan"), float("nan"), float("nan"), 0)
    return StageStats(
        name=name,
        p50=_percentile(values, 50),
        p95=_percentile(values, 95),
        p99=_percentile(values, 99),
        mean=sum(values) / len(values),
        count=len(values),
    )


def load_latency_records(log_path: Path) -> list[dict]:
    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("stage") == "pipeline" and entry.get("event") == "latency_record":
                records.append(entry["data"])
    return records


def load_drop_counts(log_path: Path) -> dict[str, int]:
    """
    Count work that never produced a latency record.

    A percentile table computed only over delivered segments is survivorship-
    biased: the slowest work is exactly what gets dropped. These counts are what
    make the table honest.
    """
    counts = {"segment_dropped": 0, "window_dropped": 0, "frame_dropped": 0,
              "send_dropped": 0, "segment_too_short": 0, "sequence_skipped": 0}
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            event = entry.get("event")
            if event in counts:
                if event == "sequence_skipped":
                    counts[event] += int(entry.get("data", {}).get("count", 1))
                else:
                    counts[event] += 1
    return counts


def compute_report(records: list[dict]) -> list[StageStats]:
    encoder_ms: list[float] = []
    policy_ms: list[float] = []
    send_ms: list[float] = []
    e2e_ms: list[float] = []
    onset_ms: list[float] = []

    for r in records:
        enc_start = r.get("encoder_start_ts")
        enc_done = r.get("encoder_done_ts")
        policy_ts = r.get("policy_ts")
        send_ts = r.get("send_ts")
        vad_close = r.get("vad_close_ts")
        seg_start = r.get("segment_start_ts")

        if enc_start and enc_done:
            encoder_ms.append((enc_done - enc_start) * 1000)
        if enc_done and policy_ts:
            policy_ms.append((policy_ts - enc_done) * 1000)
        if policy_ts and send_ts:
            send_ms.append((send_ts - policy_ts) * 1000)
        if vad_close and send_ts:
            e2e_ms.append((send_ts - vad_close) * 1000)
        if seg_start and send_ts:
            onset_ms.append((send_ts - seg_start) * 1000)

    return [
        _stats("encoder", encoder_ms),
        _stats("policy", policy_ms),
        _stats("send", send_ms),
        _stats("vad_close_to_send", e2e_ms),
        _stats("speech_onset_to_send", onset_ms),
    ]


def print_report(stats: list[StageStats]) -> None:
    print(
        f"\n{'Stage':<22} {'P50 ms':>8} {'P95 ms':>8} {'P99 ms':>8} {'Mean ms':>9} {'N':>6}")
    print("-" * 66)
    for s in stats:
        print(
            f"{s.name:<22} {s.p50:>8.1f} {s.p95:>8.1f} {s.p99:>8.1f} {s.mean:>9.1f} {s.count:>6}"
        )
    print("\nCovers delivered segments only. speech_onset_to_send is the "
          "user-perceived\nfigure; vad_close_to_send excludes the segment's own "
          "duration.")


def print_drops(counts: dict[str, int], delivered: int) -> None:
    """Report work that never reached the percentile table."""
    total = sum(counts.values())
    print(f"\n{'Dropped / unmeasured':<22} {'count':>8}")
    print("-" * 32)
    for name, n in counts.items():
        print(f"{name:<22} {n:>8}")
    accounted = delivered + total
    pct = (total / accounted * 100) if accounted else 0.0
    print("-" * 32)
    print(f"{'total dropped':<22} {total:>8}  ({pct:.1f}% of all work)")
    if total:
        print("\nWARNING: dropped work contributes no latency record, so the "
              "percentiles\nabove are biased optimistic.")
    print()


def plot_histogram(records: list[dict], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[latency_report] matplotlib not installed — skipping plot")
        return

    e2e_ms = []
    for r in records:
        # Prefer the onset-based span — it is what the user actually waits.
        start = r.get("segment_start_ts") or r.get("vad_close_ts")
        send_ts = r.get("send_ts")
        if start and send_ts:
            e2e_ms.append((send_ts - start) * 1000)

    if not e2e_ms:
        print("[latency_report] no end-to-end records to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(e2e_ms, bins=30, edgecolor="white", color="steelblue")
    ax.axvline(_percentile(e2e_ms, 50), color="orange",
               linestyle="--", label="P50")
    ax.axvline(_percentile(e2e_ms, 95), color="red",
               linestyle="--", label="P95")
    ax.set_xlabel("Speech onset → send (ms)")
    ax.set_ylabel("Segments")
    ax.set_title("Pipeline latency distribution (delivered segments)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"[latency_report] histogram saved → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Latency report from session JSONL log")
    parser.add_argument("--log", required=True, metavar="PATH",
                        help="Path to session .jsonl log")
    parser.add_argument("--plot", action="store_true",
                        help="Save histogram PNG alongside the report")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"[latency_report] ERROR: log file not found: {log_path}")
        sys.exit(1)

    records = load_latency_records(log_path)
    if not records:
        print("[latency_report] no latency_record entries found in log")
        sys.exit(1)

    print(
        f"[latency_report] {len(records)} delivered segments from '{log_path.name}'")
    stats = compute_report(records)
    print_report(stats)
    print_drops(load_drop_counts(log_path), delivered=len(records))

    if args.plot:
        plot_path = log_path.with_suffix(".latency.png")
        plot_histogram(records, plot_path)


if __name__ == "__main__":
    main()
