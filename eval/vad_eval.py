"""
eval/vad_eval.py — VAD parameter sweep evaluation.

Computes segment-level precision, recall, F1, over-segmentation, and
under-segmentation rates against ground-truth labels (TextGrid or .lab format).

Supports a parameter sweep over energy_multiplier, hangover_ms, zcr_threshold,
and noise_floor_min_ratio, producing a heatmap that empirically justifies the
chosen defaults.  Parameters not named in --param-sweep are held at their
config/base.yaml values.

Ground truth formats:
  .lab files:  "start_s end_s label" (per line), speech label = "speech"
  .TextGrid:   Praat TextGrid with a tier named "speech" (requires textgrid package)

Directory structure:
  --audio-dir    audio/<name>.wav
  --labels-dir   labels/<name>.lab  (or <name>.TextGrid)

Usage:
    python eval/vad_eval.py \\
        --audio-dir  eval/vad_audio \\
        --labels-dir eval/vad_labels \\
        --param-sweep energy_multiplier=2.0,3.0,4.0 hangover_ms=200,400,600 \\
        --out eval/results/vad
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from vad_labels import load_ground_truth
from vad_metrics import compute_metrics

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))  # sibling eval modules

from audio.wav_io import load_wav  # noqa: E402 — needs sys.path above

# Sweepable parameter -> the config constant holding its default.
_SWEEPABLE = {
    "energy_multiplier": "ENERGY_MULTIPLIER",
    "hangover_ms": "HANGOVER_MS",
    "zcr_threshold": "ZCR_THRESHOLD",
    "noise_floor_min_ratio": "NOISE_FLOOR_MIN_RATIO",
}


# ---------------------------------------------------------------------------
# VAD runner (stateless — resets between files)
# ---------------------------------------------------------------------------

def _run_vad_on_audio(
    audio: np.ndarray,
    sample_rate: int,
    params: dict,
) -> list[tuple[float, float]]:
    """
    Run the pipeline VAD with overridden parameters. Returns list of
    (start_s, end_s) segment boundaries produced by the VAD.

    Args:
        params: any of energy_multiplier, hangover_ms, zcr_threshold,
                noise_floor_min_ratio.
    """
    import config as _cfg

    # vad.py reads these at call time, so setting them here is enough — no
    # module-internal constants to patch.
    _cfg.ENERGY_MULTIPLIER = float(params["energy_multiplier"])
    _cfg.HANGOVER_MS = int(params["hangover_ms"])
    _cfg.ZCR_THRESHOLD = float(params["zcr_threshold"])
    _cfg.NOISE_FLOOR_MIN_RATIO = float(params["noise_floor_min_ratio"])

    from audio import vad
    vad.reset()

    frame_size = _cfg.FRAME_SAMPLES
    segments: list[tuple[float, float]] = []
    offset = 0

    while offset < len(audio):
        frame = audio[offset: offset + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))
        segment = vad.process_frame(frame)
        if segment is not None:
            end_s = offset / sample_rate
            # Estimate start from segment length
            start_s = end_s - len(segment) / sample_rate
            segments.append((max(0.0, start_s), end_s))
        offset += frame_size

    vad.reset()
    return segments


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VAD parameter sweep evaluation")
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument(
        "--param-sweep",
        nargs="+",
        default=["energy_multiplier=2.0,3.0,4.0", "hangover_ms=200,400,600"],
        help="Param sweep. Sweepable: energy_multiplier, hangover_ms, "
             "zcr_threshold, noise_floor_min_ratio. "
             "e.g. 'energy_multiplier=2.0,3.0,4.0' 'zcr_threshold=0.3,0.4,0.5'",
    )
    parser.add_argument("--out", default="eval/results/vad")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    labels_dir = Path(args.labels_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import config
    target_rate = config.SAMPLE_RATE

    # Parse sweep params
    sweep_params: dict[str, list] = {}
    for spec in args.param_sweep:
        name, _, vals = spec.partition("=")
        sweep_params[name.strip()] = [
            (int(v) if "." not in v else float(v)) for v in vals.split(",")
        ]

    unknown = set(sweep_params) - set(_SWEEPABLE)
    if unknown:
        print(
            f"[vad_eval] ERROR: unsweepable parameter(s): {', '.join(sorted(unknown))}")
        sys.exit(1)

    # Defaults come from the live config, so a one-parameter sweep holds the
    # rest at their shipped values.
    grid = {
        name: sweep_params.get(name, [getattr(config, const)])
        for name, const in _SWEEPABLE.items()
    }
    energy_values = grid["energy_multiplier"]
    hangover_values = grid["hangover_ms"]

    # Collect audio/label pairs
    pairs: list[tuple[Path, Path]] = []
    for wav_path in sorted(audio_dir.glob("*.wav")):
        for ext in (".lab", ".TextGrid", ".textgrid"):
            label_path = labels_dir / (wav_path.stem + ext)
            if label_path.exists():
                pairs.append((wav_path, label_path))
                break

    if not pairs:
        print("[vad_eval] ERROR: no matching audio/label pairs found")
        sys.exit(1)

    names = list(_SWEEPABLE)
    combos = list(itertools.product(*(grid[n] for n in names)))
    print(f"[vad_eval] {len(pairs)} file(s), {len(combos)} param combinations")

    sweep_rows: list[dict] = []
    best_f1 = -1.0
    best_params: dict = {}

    for combo in combos:
        params = dict(zip(names, combo))
        agg: dict[str, list[float]] = {
            "precision": [], "recall": [], "f1": [],
            "over_segmentation": [], "under_segmentation": [],
        }

        for wav_path, label_path in pairs:
            audio = load_wav(wav_path)
            rate = target_rate
            gt = load_ground_truth(label_path)
            predicted = _run_vad_on_audio(audio, rate, params)
            m = compute_metrics(predicted, gt)
            for key in agg:
                agg[key].append(m[key])

        row = {
            **params,
            "precision": float(np.mean(agg["precision"])),
            "recall": float(np.mean(agg["recall"])),
            "f1": float(np.mean(agg["f1"])),
            "over_segmentation": float(np.mean(agg["over_segmentation"])),
            "under_segmentation": float(np.mean(agg["under_segmentation"])),
        }
        sweep_rows.append(row)

        if row["f1"] > best_f1:
            best_f1 = row["f1"]
            best_params = dict(params)

        print(
            f"  em={params['energy_multiplier']:.1f} "
            f"hangover={int(params['hangover_ms']):3d}ms "
            f"zcr={params['zcr_threshold']:.2f} "
            f"nfmin={params['noise_floor_min_ratio']:.2f}  "
            f"P={row['precision']:.3f} R={row['recall']:.3f} F1={row['f1']:.3f}"
        )

    # Write CSV
    csv_path = out_dir / "param_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)
    print(f"\n[vad_eval] sweep results → {csv_path}")

    # Write best params
    best_json = out_dir / "best_params.json"
    with open(best_json, "w") as f:
        json.dump({**best_params, "f1": best_f1}, f, indent=2)
    print(f"[vad_eval] best params → {best_json}")
    print("  Best: " + ", ".join(f"{k}={v}" for k, v in best_params.items())
          + f", F1={best_f1:.3f}")

    # Heatmap — energy_multiplier × hangover_ms, sliced at the best value of
    # the remaining parameters so the surface is comparable cell to cell.
    try:
        import matplotlib.pyplot as plt

        held = {k: best_params[k]
                for k in ("zcr_threshold", "noise_floor_min_ratio")}

        def _slice(hms):
            return [r["f1"] for r in sweep_rows
                    if r["hangover_ms"] == hms
                    and all(r[k] == v for k, v in held.items())]

        f1_grid = np.array([_slice(hms) for hms in hangover_values])

        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(f1_grid, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(energy_values)))
        ax.set_xticklabels([str(e) for e in energy_values])
        ax.set_yticks(range(len(hangover_values)))
        ax.set_yticklabels([str(h) for h in hangover_values])
        ax.set_xlabel("energy_multiplier")
        ax.set_ylabel("hangover_ms")
        ax.set_title("VAD F1 — parameter sweep\n"
                     + ", ".join(f"{k}={v}" for k, v in held.items()))
        plt.colorbar(im, ax=ax, label="F1")

        for i, hms in enumerate(hangover_values):
            for j, em in enumerate(energy_values):
                val = f1_grid[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center",
                        va="center", fontsize=8)

        fig.tight_layout()
        heatmap_path = out_dir / "heatmap.png"
        fig.savefig(heatmap_path, dpi=150)
        print(f"[vad_eval] heatmap → {heatmap_path}")
    except ImportError:
        print("[vad_eval] matplotlib not installed — skipping heatmap")


if __name__ == "__main__":
    main()
