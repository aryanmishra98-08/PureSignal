# =============================================================================
# speaker/encoder.py — Speaker embedding extractor
#                      HuggingFace wespeaker-voxceleb-resnet34-LM, MPS backend
#
# Prerequisites:
#   1. Accept model terms at hf.co/pyannote/wespeaker-voxceleb-resnet34-LM
#   2. Set environment variable: export HF_TOKEN="your_token_here"
# =============================================================================

import os
import config
import numpy as np
import torch
from pyannote.audio import Inference, Model

_model = None
_inference = None
_device: str = ""


def _resolve_device() -> str:
    """Return the configured device, falling back to 'cpu' if MPS is unavailable."""
    requested = config.ENCODER_DEVICE
    if requested == "mps" and not torch.backends.mps.is_available():
        print("[encoder] WARNING: MPS requested but unavailable — falling back to CPU.")
        return "cpu"
    return requested


def _check_hf_token() -> str:
    """Fail fast with a clear message if HF_TOKEN is not set."""
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise EnvironmentError(
            "\n[encoder] HF_TOKEN environment variable is not set.\n"
            "Steps to fix:\n"
            "  1. Accept model terms at: "
            "https://hf.co/pyannote/wespeaker-voxceleb-resnet34-LM\n"
            "  2. Generate a token at: https://huggingface.co/settings/tokens\n"
            "  3. Run: export HF_TOKEN='your_token_here'\n"
            "  4. Re-run the script.\n"
        )
    return token


def load_encoder() -> None:
    """
    Load ResNet34-LM from HuggingFace and move to the resolved device.
    Call once at startup before any embed() calls.
    """
    global _model, _inference, _device

    token = _check_hf_token()

    _device = _resolve_device()
    print(f"[encoder] loading {config.ENCODER_MODEL} ...")
    _model = Model.from_pretrained(config.ENCODER_MODEL, use_auth_token=token)
    _model = _model.to(torch.device(_device))
    _model.eval()

    _inference = Inference(_model, window="whole")
    print(f"[encoder] ready — {config.ENCODER_MODEL} on {_device}")


def embed(segment: np.ndarray) -> np.ndarray | None:
    """
    Convert a normalized speech segment → 256-dim L2-normalized embedding.

    Args:
        segment: float32 ndarray, 16kHz, variable length (min ~0.5s recommended)

    Returns:
        np.ndarray [256] normalized embedding, or None if segment too short
    """
    if _inference is None:
        raise RuntimeError("[encoder] call load_encoder() before embed()")

    min_samples = int(config.MIN_SEGMENT_S * config.SAMPLE_RATE)
    if len(segment) < min_samples:
        return None  # too short for a reliable embedding

    # pyannote Inference expects a dict with waveform tensor + sample_rate
    waveform = torch.tensor(segment).unsqueeze(0)  # [1, T] — (channel, time)
    input_dict = {
        "waveform": waveform.to(torch.device(_device)),
        "sample_rate": config.SAMPLE_RATE,
    }

    with torch.no_grad():
        embedding = _inference(input_dict)  # np.ndarray [256]

    # L2 normalize
    norm = np.linalg.norm(embedding)
    if norm < config.NORM_FLOOR:
        return None
    return (embedding / norm).astype(np.float32)
