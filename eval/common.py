"""Shared helpers for the evaluation scripts.

Importing this module puts src/ on sys.path, so the eval scripts can import
the pipeline modules while still being run directly
(``python eval/vad_eval.py``).

Pipeline imports are deferred into the functions that need them: importing
this module must stay cheap and must not pull in the model stack.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

_SRC = Path(__file__).parent.parent / "src"

# Only src/ goes on the path. Adding the repo root too would make `import
# config` ambiguous between src/config.py and the config/ data directory.
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def load_wav(path: Path, target_rate: int | None = None) -> np.ndarray:
    """Load a WAV as mono float32 at the pipeline sample rate.

    Thin wrapper over the pipeline's own loader, so the evaluators decode
    audio exactly the way the runtime does.
    """
    from audio.wav_io import load_wav as _load_wav

    return _load_wav(path, target_rate)


def embed_wav(encoder, path: Path) -> np.ndarray | None:
    """Load, normalize and embed one WAV.

    Args:
        encoder: A loaded SpeakerEncoder.
        path: The WAV to embed.

    Returns:
        The embedding, or None if the clip is too short to embed.
    """
    from audio.features import normalize

    return encoder.embed(normalize(load_wav(path)))


def load_encoder():
    """Build and load a SpeakerEncoder, the way every evaluator needs one."""
    from speaker.encoder import SpeakerEncoder

    encoder = SpeakerEncoder()
    encoder.load()
    return encoder


def collect_wavs(directory: Path) -> dict[str, list[Path]]:
    """Map each immediate subdirectory to the WAV files it contains.

    Args:
        directory: A two-level tree of ``<speaker>/<utterance>.wav``.

    Returns:
        ``{speaker_label: [wav_path, ...]}``, empty if the directory is absent.
    """
    result: dict[str, list[Path]] = {}
    if not directory.exists():
        return result

    for speaker_dir in sorted(directory.iterdir()):
        if not speaker_dir.is_dir():
            continue
        wavs = sorted(speaker_dir.glob("*.wav"))
        if wavs:
            result[speaker_dir.name] = wavs
    return result


def get_pyplot(tag: str) -> Any | None:
    """Return ``matplotlib.pyplot``, or None if matplotlib is not installed.

    Args:
        tag: Script name used in the notice printed when plotting is skipped.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[{tag}] matplotlib not installed — skipping plot")
        return None
    return plt
