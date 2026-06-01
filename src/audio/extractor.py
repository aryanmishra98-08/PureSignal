# =============================================================================
# audio/extractor.py — Sliding-window target speaker extraction
#
# Wraps SpeakerBeam (Asteroid toolkit) to clean a mixed audio signal so only
# the enrolled target speaker's voice remains before the VAD sees it.
#
# If extractor.enabled = false, extract() is a no-op passthrough.
# Prerequisites: pip install asteroid
# =============================================================================
from __future__ import annotations
import warnings
import numpy as np
import config

_model = None
_device: str = ""
_model_type: str = ""


def _resolve_device() -> str:
    import torch
    requested = config.EXTRACTOR_DEVICE
    if requested == "mps" and not torch.backends.mps.is_available():
        print("[extractor] WARNING: MPS requested but unavailable — falling back to CPU.")
        return "cpu"
    return requested


def load_extractor() -> None:
    """Load SpeakerBeam or Conv-TasNet from Asteroid. Call once at startup."""
    global _model, _device, _model_type
    if not config.EXTRACTOR_ENABLED:
        print("[extractor] disabled in config — passthrough mode active")
        return
    try:
        import torch
        from asteroid.models import SpeakerBeam, ConvTasNet
    except ImportError:
        raise RuntimeError(
            "\n[extractor] Asteroid toolkit not installed.\n"
            "  Run: pip install asteroid\n"
        )
    _device = _resolve_device()
    _model_type = config.EXTRACTOR_MODEL
    print(f"[extractor] loading {_model_type} on {_device} ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if _model_type == "speakerbeam":
            from asteroid.models import SpeakerBeam
            _model = SpeakerBeam.from_pretrained(
                "JorisCos/SpeakerBeam_Libri2Mix_noise-reverb",
                sample_rate=config.SAMPLE_RATE,
            )
        elif _model_type == "conv_tasnet":
            from asteroid.models import ConvTasNet
            _model = ConvTasNet.from_pretrained(
                "JorisCos/ConvTasNet_Libri2Mix_sepclean_8k",
                sample_rate=config.SAMPLE_RATE,
            )
        else:
            raise ValueError(f"[extractor] Unknown model '{_model_type}'. Use 'speakerbeam' or 'conv_tasnet'.")
    _model = _model.to(_device)
    _model.eval()
    print(f"[extractor] ready — {_model_type} on {_device}")


def extract(window: np.ndarray, target_embedding: np.ndarray | None = None) -> np.ndarray:
    """
    Extract the target speaker's voice from a mixed audio window.

    Args:
        window:           float32 ndarray at 16kHz — the raw mixed audio window
        target_embedding: 256-dim L2-normalized enrollment embedding (SpeakerBeam fingerprint)

    Returns:
        float32 ndarray at 16kHz — cleaned audio.
        Returns window unchanged if extractor disabled or model not loaded.
    """
    if not config.EXTRACTOR_ENABLED or _model is None:
        return window
    import torch
    original_len = len(window)
    min_samples = int(config.EXTRACTOR_WINDOW_S * config.SAMPLE_RATE)
    if len(window) < min_samples:
        window = np.pad(window, (0, min_samples - len(window)))
    # Shape: [1, 1, T]
    waveform = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(_device)
    with torch.no_grad():
        if _model_type == "speakerbeam" and target_embedding is not None:
            enroll_tensor = torch.tensor(target_embedding, dtype=torch.float32).unsqueeze(0).to(_device)
            sources = _model(waveform, enroll_tensor)  # [1, 1, T]
            cleaned = sources[0, 0].cpu().numpy()
        elif _model_type == "conv_tasnet":
            sources = _model(waveform)  # [1, 2, T]
            sources_np = sources[0].cpu().numpy()  # [2, T]
            if target_embedding is not None:
                from audio.features import normalize
                from speaker.encoder import embed
                best_idx, best_sim = 0, -1.0
                for i in range(sources_np.shape[0]):
                    src_emb = embed(normalize(sources_np[i]))
                    if src_emb is not None:
                        sim = float(np.dot(src_emb, target_embedding))
                        if sim > best_sim:
                            best_sim, best_idx = sim, i
                cleaned = sources_np[best_idx]
            else:
                cleaned = sources_np[np.argmax(np.abs(sources_np).mean(axis=1))]
        else:
            return window[:original_len]
    return cleaned[:original_len].astype(np.float32)


def is_loaded() -> bool:
    return _model is not None


def reset() -> None:
    global _model
    _model = None
