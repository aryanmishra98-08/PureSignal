# =============================================================================
# audio/extractor.py — Sliding-window target speaker separation
#
# Wraps Conv-TasNet (Asteroid toolkit) to separate a mixed audio window into its
# constituent sources, so only the enrolled target speaker's voice reaches the
# VAD.
#
# Split of responsibilities — this matters for thread safety:
#   separate()       runs the separator.  Called from extractor worker threads.
#                    Stateless: model forward only, no encoder, no history.
#   SourceSelector   lives in audio/source_select.py. It picks which separated
#                    source is the target, and is called from the consumer
#                    thread AFTER resequencing, so it sees windows in order and
#                    holds the only encoder reference.
#
# "speakerbeam" was removed: asteroid.models.SpeakerBeam does not exist in any
# released Asteroid version, and the checkpoint the code named does not exist on
# HuggingFace either.  See PHASE0_FINDINGS.md.
#
# If extractor.enabled = false, separate() is a no-op passthrough.
# Prerequisites: pip install asteroid
# =============================================================================
from __future__ import annotations

import time
import warnings

import numpy as np

import config

_model = None
_device: str = ""
_model_type: str = ""

_SUPPORTED_MODELS = ("conv_tasnet",)


def _resolve_device() -> str:
    import torch
    requested = config.EXTRACTOR_DEVICE
    if requested == "mps" and not torch.backends.mps.is_available():
        print(
            "[extractor] WARNING: MPS requested but unavailable — falling back to CPU.")
        return "cpu"
    return requested


def _import_model_class():
    """
    Import ConvTasNet, naming the actual missing piece on failure.

    Imports are split so a missing symbol inside an installed Asteroid is not
    misreported as "Asteroid is not installed", which is what the previous
    blanket `except ImportError` did.
    """
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            f"\n[extractor] PyTorch is not importable: {e}\n") from e
    try:
        import asteroid  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "\n[extractor] Asteroid toolkit not installed.\n"
            "  Run: pip install asteroid\n"
            "  Or skip separation entirely: --set extractor.enabled=false\n"
        ) from e
    try:
        from asteroid.models import ConvTasNet
    except ImportError as e:
        raise RuntimeError(
            f"\n[extractor] Asteroid is installed but 'ConvTasNet' could not be "
            f"imported from asteroid.models: {e}\n"
        ) from e
    return ConvTasNet


def _assert_native_rate(model) -> None:
    """
    Fail loudly if the checkpoint's native rate differs from the pipeline rate.

    from_pretrained(sample_rate=...) only sets a metadata attribute.  It does
    not resample the input and it does not rescale the learned filterbank,
    whose receptive field is calibrated in samples at the training rate.  An 8k
    checkpoint fed 16k audio silently halves the effective temporal context.
    """
    native = getattr(model, "sample_rate", None)
    if native is None:
        print("[extractor] WARNING: checkpoint exposes no sample_rate — cannot verify.")
        return
    if int(native) != int(config.SAMPLE_RATE):
        raise RuntimeError(
            f"\n[extractor] Checkpoint sample-rate mismatch.\n"
            f"  checkpoint '{config.EXTRACTOR_CHECKPOINT}' is natively {int(native)}Hz\n"
            f"  pipeline runs at {config.SAMPLE_RATE}Hz\n"
            f"  Passing sample_rate= to from_pretrained does NOT resample.\n"
            f"  Use a {config.SAMPLE_RATE}Hz checkpoint, e.g.\n"
            f"    --set extractor.checkpoint=JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k\n"
        )


def load_extractor() -> None:
    """Load the separator from Asteroid. Call once at startup."""
    global _model, _device, _model_type
    if not config.EXTRACTOR_ENABLED:
        print("[extractor] disabled in config — passthrough mode active")
        return

    _model_type = config.EXTRACTOR_MODEL
    if _model_type not in _SUPPORTED_MODELS:
        raise ValueError(
            f"[extractor] Unknown model '{_model_type}'. "
            f"Supported: {', '.join(_SUPPORTED_MODELS)}."
        )

    ConvTasNet = _import_model_class()
    _device = _resolve_device()
    print(
        f"[extractor] loading {_model_type} '{config.EXTRACTOR_CHECKPOINT}' on {_device} ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _model = ConvTasNet.from_pretrained(config.EXTRACTOR_CHECKPOINT)
    _assert_native_rate(_model)
    _model = _model.to(_device)
    _model.eval()
    print(f"[extractor] ready — {_model_type} on {_device}")
    if config.EXTRACTOR_STARTUP_CHECK:
        realtime_check()


def realtime_check(trials: int = 10) -> float:
    """
    Time the separator on synthetic audio and compare against the hop budget.

    Windows arrive every hop_s seconds and each needs one forward pass, so with
    N workers the pipeline keeps up only if a pass completes within
    hop_s * max_workers.  Warns loudly when it does not, turning an
    unpredictable mid-run stall into a startup diagnostic.

    Returns:
        float — measured mean seconds per forward pass (0.0 if not applicable).
    """
    if _model is None:
        return 0.0
    window = np.zeros(int(config.EXTRACTOR_WINDOW_S *
                      config.SAMPLE_RATE), dtype=np.float32)
    separate(window)  # warm-up: first pass includes lazy device init
    start = time.monotonic()
    for _ in range(trials):
        separate(window)
    per_pass = (time.monotonic() - start) / trials
    budget = config.EXTRACTOR_HOP_S * config.EXTRACTOR_MAX_WORKERS
    ratio = per_pass / budget if budget > 0 else float("inf")
    msg = (f"[extractor] realtime check — {per_pass*1000:.0f}ms/window, "
           f"budget {budget*1000:.0f}ms ({config.EXTRACTOR_MAX_WORKERS} workers "
           f"× {config.EXTRACTOR_HOP_S}s hop), load factor {ratio:.2f}x")
    if ratio > 1.0:
        print(f"{msg}\n[extractor] WARNING: cannot meet realtime budget — expect "
              f"window drops. Raise extractor.max_workers, raise extractor.hop_s, "
              f"or run gatekeeper mode with --set extractor.enabled=false.")
    else:
        print(msg)
    return per_pass


def separate(window: np.ndarray) -> np.ndarray:
    """
    Separate a mixed audio window into its constituent sources.

    Model forward only — safe to call from extraction worker threads.  Does no
    encoder work and keeps no cross-window state.

    Args:
        window: float32 ndarray at 16kHz — the raw mixed audio window

    Returns:
        float32 ndarray [n_src, T] at 16kHz.  When the extractor is disabled or
        the model is not loaded, returns the input unchanged as a single source
        of shape [1, T].
    """
    if not config.EXTRACTOR_ENABLED or _model is None:
        return window[np.newaxis, :].astype(np.float32)
    import torch
    original_len = len(window)
    min_samples = int(config.EXTRACTOR_WINDOW_S * config.SAMPLE_RATE)
    padded = window
    if len(padded) < min_samples:
        padded = np.pad(padded, (0, min_samples - len(padded)))
    waveform = torch.tensor(padded, dtype=torch.float32).unsqueeze(
        0).unsqueeze(0).to(_device)
    with torch.no_grad():
        sources = _model(waveform)  # [1, n_src, T]
    return sources[0].cpu().numpy()[:, :original_len].astype(np.float32)


def reset() -> None:
    global _model
    _model = None
