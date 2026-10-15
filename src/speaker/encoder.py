"""Speaker embedding extractor.

Wraps the HuggingFace wespeaker ResNet34-LM model, preferring the MPS backend
on Apple Silicon and falling back to CPU elsewhere.

Prerequisites:
  1. Accept the model terms at hf.co/pyannote/wespeaker-voxceleb-resnet34-LM
  2. Set the HF_TOKEN environment variable.

pyannote is imported inside load() rather than at module scope so this module
can be imported — and its input validation tested — without the model stack
installed.
"""

from __future__ import annotations

import os

import numpy as np
import torch

import config
from utils.logger import get_logger

_log = get_logger()


def check_hf_token() -> str:
    """Return HF_TOKEN, or fail with instructions if it is not set.

    Raises:
        OSError: If HF_TOKEN is missing from the environment.
    """
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise OSError(
            "\n[encoder] HF_TOKEN environment variable is not set.\n"
            "Steps to fix:\n"
            "  1. Accept model terms at: "
            "https://hf.co/pyannote/wespeaker-voxceleb-resnet34-LM\n"
            "  2. Generate a token at: https://huggingface.co/settings/tokens\n"
            "  3. Run: export HF_TOKEN='your_token_here'\n"
            "  4. Re-run the script.\n"
        )
    return token


class SpeakerEncoder:
    """Turns speech segments into L2-normalized speaker embeddings."""

    def __init__(
        self,
        *,
        model_name: str | None = None,
        device: str | None = None,
        sample_rate: int | None = None,
        min_segment_s: float | None = None,
        norm_floor: float | None = None,
    ) -> None:
        """Build an encoder. Any argument left as None is read from config.

        The model is not loaded here; call load() first.
        """
        self.model_name = config.ENCODER_MODEL if model_name is None else model_name
        self.requested_device = config.ENCODER_DEVICE if device is None else device
        self.sample_rate = config.SAMPLE_RATE if sample_rate is None else sample_rate
        self.min_segment_s = (
            config.MIN_SEGMENT_S if min_segment_s is None else min_segment_s
        )
        self.norm_floor = config.NORM_FLOOR if norm_floor is None else norm_floor

        self._inference = None
        self._device = ""

    def _resolve_device(self) -> str:
        """Return the requested device, falling back to CPU if MPS is absent."""
        if self.requested_device == "mps" and not torch.backends.mps.is_available():
            _log.warning(
                "encoder", "MPS requested but unavailable — falling back to CPU."
            )
            return "cpu"
        return self.requested_device

    def load(self) -> None:
        """Load the model onto the resolved device.

        Raises:
            OSError: If HF_TOKEN is not set.
        """
        from pyannote.audio import Inference, Model

        token = check_hf_token()
        self._device = self._resolve_device()

        _log.info("encoder", f"loading {self.model_name} ...")
        model = Model.from_pretrained(self.model_name, token=token)
        model = model.to(torch.device(self._device))
        model.eval()

        self._inference = Inference(model, window="whole")
        _log.info("encoder", f"ready — {self.model_name} on {self._device}")

    def embed(self, segment: np.ndarray) -> np.ndarray | None:
        """Convert a normalized speech segment into a speaker embedding.

        Args:
            segment: float32 samples at ``sample_rate``, variable length.

        Returns:
            An L2-normalized embedding, or None if the segment is too short
            or yields a degenerate (near-zero) vector.

        Raises:
            RuntimeError: If load() has not been called.
        """
        if self._inference is None:
            raise RuntimeError("[encoder] call load() before embed()")

        if len(segment) < int(self.min_segment_s * self.sample_rate):
            return None

        # pyannote expects a (channel, time) waveform tensor alongside its rate.
        waveform = torch.tensor(segment).unsqueeze(0)
        with torch.no_grad():
            embedding = self._inference(
                {
                    "waveform": waveform.to(torch.device(self._device)),
                    "sample_rate": self.sample_rate,
                }
            )

        norm = np.linalg.norm(embedding)
        if norm < self.norm_floor:
            return None
        return (embedding / norm).astype(np.float32)

    @property
    def is_loaded(self) -> bool:
        """True once load() has completed successfully."""
        return self._inference is not None

    @property
    def device(self) -> str:
        """The device the model was actually placed on, or "" before load()."""
        return self._device
