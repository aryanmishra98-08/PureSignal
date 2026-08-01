"""
eval/vad_labels.py — Ground-truth label loading for the VAD evaluation.

Split out of vad_eval.py to keep both files within the project's 300-line limit.

Formats:
  .lab files:  "start_s end_s label" (per line), speech label = "speech"
  .TextGrid:   Praat TextGrid with a tier named "speech" (requires the optional
               `textgrid` package)

Every loader returns a list of (start_s, end_s) speech intervals.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------


def load_lab(path: Path) -> list[tuple[float, float]]:
    """
    Load a .lab file. Lines: "start_s end_s label".
    Returns list of (start_s, end_s) for speech intervals.
    """
    intervals = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                start, end, label = float(parts[0]), float(parts[1]), parts[2]
            except ValueError:
                continue
            if label.lower() == "speech":
                intervals.append((start, end))
    return intervals


def load_textgrid(path: Path) -> list[tuple[float, float]]:
    """
    Load a Praat TextGrid. Requires the 'textgrid' PyPI package.
    Looks for a tier named 'speech'.
    """
    try:
        import textgrid as tg_lib
    except ImportError:
        raise RuntimeError(
            "Install 'textgrid' to read TextGrid files: pip install textgrid")

    tg = tg_lib.TextGrid.fromFile(str(path))
    for tier in tg.tiers:
        if tier.name.lower() == "speech":
            return [
                (interval.minTime, interval.maxTime)
                for interval in tier.intervals
                if interval.mark.lower() == "speech"
            ]
    raise ValueError(f"No 'speech' tier found in {path}")


def load_ground_truth(path: Path) -> list[tuple[float, float]]:
    if path.suffix.lower() == ".lab":
        return load_lab(path)
    if path.suffix.lower() == ".textgrid":
        return load_textgrid(path)
    raise ValueError(f"Unsupported label format: {path.suffix}")
