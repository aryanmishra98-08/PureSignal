# =============================================================================
# speaker/encoder.py — Speaker embedding extractor
#                      HuggingFace wespeaker-voxceleb-resnet34-LM, MPS backend
#
# Prerequisites:
#   1. Accept model terms at hf.co/pyannote/wespeaker-voxceleb-resnet34-LM
#   2. Set environment variable: export HF_TOKEN="your_token_here"
# =============================================================================

import os
import numpy as np
import torch
from pyannote.audio import Model, Inference
import config

_model     = None
_inference = None


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
    Load ResNet34-LM from HuggingFace and move to MPS.
    Call once at startup before any embed() calls.
    """
    global _model, _inference

    token = _check_hf_token()

    print(f"[encoder] loading {config.ENCODER_MODEL} ...")
    _model = Model.from_pretrained(config.ENCODER_MODEL, use_auth_token=token)
    _model = _model.to(torch.device(config.ENCODER_DEVICE))
    _model.eval()

    # Verify MPS is actually being used — fail loudly if not
    device_in_use = next(_model.parameters()).device
    assert device_in_use.type == torch.device(config.ENCODER_DEVICE).type, (
        f"[encoder] FATAL: expected {config.ENCODER_DEVICE}, "
        f"got {device_in_use}. Check PyTorch MPS availability.\n"
        f"Run: python3 -c \"import torch; print(torch.backends.mps.is_available())\""
    )

    _inference = Inference(_model, window="whole")
    print(f"[encoder] ready — {config.ENCODER_MODEL} on {config.ENCODER_DEVICE}")


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

    min_samples = int(0.5 * config.SAMPLE_RATE)
    if len(segment) < min_samples:
        return None   # too short for a reliable embedding

    # pyannote Inference expects a dict with waveform tensor + sample_rate
    waveform = torch.tensor(segment).unsqueeze(0)  # [1, T] — (channel, time)
    input_dict = {
        "waveform": waveform.to(torch.device(config.ENCODER_DEVICE)),
        "sample_rate": config.SAMPLE_RATE,
    }

    with torch.no_grad():
        embedding = _inference(input_dict)   # np.ndarray [256]

    # L2 normalize
    norm = np.linalg.norm(embedding)
    if norm < 1e-6:
        return None
    return (embedding / norm).astype(np.float32)
