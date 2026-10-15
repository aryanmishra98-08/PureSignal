"""Flat constant surface over config/base.yaml.

Every tunable lives in config/base.yaml; this module re-exports each one as a
module-level constant so pipeline code reads ``config.SAMPLE_RATE`` rather
than walking the nested config tree. Parsing lives in config_loader.

Call ``reload(path, set_args)`` to swap in a different YAML file or apply
``--set key=value`` overrides. It re-derives every constant below, so modules
that already imported this one see the new values. Pipeline components read
these constants when they are constructed, so build them after any reload.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from config_loader import Config, load_config

_REPO_ROOT = Path(__file__).parent.parent

# Load secrets from .env at the repo root.
load_dotenv(dotenv_path=_REPO_ROOT / ".env")

# --- Audio capture ---
SAMPLE_RATE: int
FRAME_MS: int
FRAME_SAMPLES: int
WINDOW_SIZE_S: float
HOP_SIZE_S: float

# --- VAD ---
ENERGY_MULTIPLIER: float
ZCR_THRESHOLD: float
HANGOVER_MS: int
NOISE_FLOOR_INIT: float

# --- Speaker encoder ---
ENCODER_MODEL: str
ENCODER_DEVICE: str
EMBEDDING_DIM: int
MIN_SEGMENT_S: float

# --- Speaker tracker ---
SIMILARITY_THRESHOLD: float
EMA_ALPHA: float
MAX_SPEAKERS: int

# --- Enrollment ---
ENROLLMENT_THRESHOLD: float
ENROLLMENT_DURATION_S: int
ENROLLMENT_NUM_SAMPLES: int
PROFILES_DIR: Path = _REPO_ROOT / "profiles"

# --- Policy ---
POLICY_MODE: str
DYNAMIC_TARGET: str

# --- Ultravox / Fixie.ai ---
ULTRAVOX_API_KEY: str
ULTRAVOX_JOIN_URL: str
ULTRAVOX_SYSTEM_PROMPT: str = "You are a helpful assistant."
ULTRAVOX_IN_RATE: int
ULTRAVOX_OUT_RATE: int
ULTRAVOX_CHUNK_MS: int

# --- Internal numeric constants ---
NORM_FLOOR: float
NOISE_FLOOR_EMA_FAST: float
NOISE_FLOOR_EMA_SLOW: float

# --- Misc ---
RANDOM_SEED: int
DEBUG: bool


def _apply(cfg: Config) -> None:
    """Derive every module-level constant from a loaded Config."""
    sample_rate = cfg.audio.sample_rate
    globals().update(
        {
            # Audio capture
            "SAMPLE_RATE": sample_rate,
            "FRAME_MS": cfg.audio.frame_ms,
            "FRAME_SAMPLES": int(sample_rate * cfg.audio.frame_ms / 1000),
            "WINDOW_SIZE_S": cfg.audio.window_size_s,
            "HOP_SIZE_S": cfg.audio.hop_size_s,
            # VAD
            "ENERGY_MULTIPLIER": cfg.vad.energy_multiplier,
            "ZCR_THRESHOLD": cfg.vad.zcr_threshold,
            "HANGOVER_MS": cfg.vad.hangover_ms,
            "NOISE_FLOOR_INIT": cfg.vad.noise_floor_init,
            # Speaker encoder
            "ENCODER_MODEL": cfg.encoder.model,
            "ENCODER_DEVICE": cfg.encoder.device,
            "EMBEDDING_DIM": cfg.encoder.embedding_dim,
            "MIN_SEGMENT_S": cfg.encoder.min_segment_s,
            # Speaker tracker
            "SIMILARITY_THRESHOLD": cfg.tracker.similarity_threshold,
            "EMA_ALPHA": cfg.tracker.ema_alpha,
            "MAX_SPEAKERS": cfg.tracker.max_speakers,
            # Enrollment
            "ENROLLMENT_THRESHOLD": cfg.enrollment.threshold,
            "ENROLLMENT_DURATION_S": cfg.enrollment.duration_s,
            "ENROLLMENT_NUM_SAMPLES": cfg.enrollment.num_samples,
            # Policy
            "POLICY_MODE": cfg.policy.mode,
            "DYNAMIC_TARGET": cfg.policy.dynamic_target,
            # Ultravox — credentials come from the environment, not YAML
            "ULTRAVOX_API_KEY": os.getenv("ULTRAVOX_API_KEY", ""),
            "ULTRAVOX_JOIN_URL": os.getenv("ULTRAVOX_JOIN_URL", ""),
            "ULTRAVOX_IN_RATE": cfg.ultravox.in_rate,
            "ULTRAVOX_OUT_RATE": cfg.ultravox.out_rate,
            "ULTRAVOX_CHUNK_MS": cfg.ultravox.chunk_ms,
            # Internal
            "NORM_FLOOR": cfg.internal.norm_floor,
            "NOISE_FLOOR_EMA_FAST": cfg.internal.noise_floor_ema_fast,
            "NOISE_FLOOR_EMA_SLOW": cfg.internal.noise_floor_ema_slow,
            # Misc
            "RANDOM_SEED": cfg.random_seed,
            "DEBUG": cfg.debug,
        }
    )


def reload(
    path: str | Path | None = None, set_args: list[str] | None = None
) -> None:
    """Re-load the YAML config and re-derive every constant in this module.

    Args:
        path: YAML file to load. Defaults to config/base.yaml.
        set_args: ``key.subkey=value`` overrides from the --set CLI flag.
    """
    global cfg
    cfg = load_config(path=path, set_args=set_args)
    _apply(cfg)


cfg: Config = load_config()
_apply(cfg)
