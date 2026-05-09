# =============================================================================
# config.py — Backward-compatibility shim
#
# All settings now live in config/base.yaml (project root).
# This file re-exports every constant so existing imports keep working:
#   import config; config.SAMPLE_RATE  →  still works
# =============================================================================
import importlib.util
import os
from pathlib import Path

from dotenv import load_dotenv

# Load secrets from .env at repo root
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)

# Load config.loader from the project root (avoids shadowing by this file)
_loader_path = Path(__file__).parent.parent / "config" / "loader.py"
_spec = importlib.util.spec_from_file_location("_config_loader", _loader_path)
_loader_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_loader_mod)  # type: ignore[union-attr]

cfg = _loader_mod.cfg

# --- Audio Capture ---
SAMPLE_RATE: int = cfg.audio.sample_rate
FRAME_MS: int = cfg.audio.frame_ms
FRAME_SAMPLES: int = int(SAMPLE_RATE * FRAME_MS / 1000)
WINDOW_SIZE_S: float = cfg.audio.window_size_s
HOP_SIZE_S: float = cfg.audio.hop_size_s

# --- VAD ---
ENERGY_MULTIPLIER: float = cfg.vad.energy_multiplier
ZCR_THRESHOLD: float = cfg.vad.zcr_threshold
HANGOVER_MS: int = cfg.vad.hangover_ms
NOISE_FLOOR_INIT: float = cfg.vad.noise_floor_init

# --- Speaker Encoder ---
ENCODER_MODEL: str = cfg.encoder.model
ENCODER_DEVICE: str = cfg.encoder.device
EMBEDDING_DIM: int = cfg.encoder.embedding_dim

# --- Speaker Tracker ---
SIMILARITY_THRESHOLD: float = cfg.tracker.similarity_threshold
EMA_ALPHA: float = cfg.tracker.ema_alpha
MAX_SPEAKERS: int = cfg.tracker.max_speakers

# --- Enrollment ---
ENROLLMENT_THRESHOLD: float = cfg.enrollment.threshold
PROFILES_DIR: Path = Path(__file__).parent.parent / "profiles"
ENROLLMENT_DURATION_S: int = cfg.enrollment.duration_s
ENROLLMENT_NUM_SAMPLES: int = cfg.enrollment.num_samples

# --- Policy ---
POLICY_MODE: str = cfg.policy.mode
DYNAMIC_TARGET: str = cfg.policy.dynamic_target

# --- Debug ---
DEBUG: bool = cfg.debug

# --- Ultravox / Fixie.ai ---
ULTRAVOX_API_KEY: str = os.getenv("ULTRAVOX_API_KEY", "")
ULTRAVOX_JOIN_URL: str = ""
ULTRAVOX_SYSTEM_PROMPT: str = "You are a helpful assistant."
ULTRAVOX_IN_RATE: int = cfg.ultravox.in_rate
ULTRAVOX_OUT_RATE: int = cfg.ultravox.out_rate
ULTRAVOX_CHUNK_MS: int = cfg.ultravox.chunk_ms

# --- Internal numeric constants ---
NORM_FLOOR: float = cfg.internal.norm_floor
NOISE_FLOOR_EMA_FAST: float = cfg.internal.noise_floor_ema_fast
NOISE_FLOOR_EMA_SLOW: float = cfg.internal.noise_floor_ema_slow
MIN_SEGMENT_S: float = cfg.encoder.min_segment_s

# --- Reproducibility ---
RANDOM_SEED: int = cfg.random_seed
