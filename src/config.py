# =============================================================================
# config.py — Backward-compatibility shim
#
# Loads config/base.yaml via config/loader.py and re-exports every setting
# as a flat module-level constant.  All sub-modules import from here so that
# the rest of the codebase never references the YAML structure directly.
#
# Override at runtime:
#   python src/main.py --set vad.hangover_ms=300
# =============================================================================
import importlib.util
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env so ULTRAVOX_API_KEY / HF_TOKEN are available before any import
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)

# Allow PyTorch to fall back from MPS to CPU for unsupported ops
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Dynamically load config/loader.py to avoid a hard package dependency
_loader_path = Path(__file__).parent.parent / "config" / "loader.py"
_spec = importlib.util.spec_from_file_location("_config_loader", _loader_path)
_loader_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_loader_mod)

# cfg is the live Config object; main.py may replace it via --config / --set
cfg = _loader_mod.cfg

# ---------------------------------------------------------------------------
# Audio capture
# ---------------------------------------------------------------------------
SAMPLE_RATE: int = cfg.audio.sample_rate          # Hz — 16000
FRAME_MS: int = cfg.audio.frame_ms                # VAD frame length in ms
FRAME_SAMPLES: int = int(SAMPLE_RATE * FRAME_MS / 1000)  # pre-computed samples per frame
WINDOW_SIZE_S: float = cfg.audio.window_size_s    # extractor window length (s)
HOP_SIZE_S: float = cfg.audio.hop_size_s          # extractor hop stride (s)

# ---------------------------------------------------------------------------
# Voice Activity Detection
# ---------------------------------------------------------------------------
ENERGY_MULTIPLIER: float = cfg.vad.energy_multiplier  # threshold = noise_floor × multiplier
ZCR_THRESHOLD: float = cfg.vad.zcr_threshold          # zero-crossing rate gate
HANGOVER_MS: int = cfg.vad.hangover_ms                # tail silence before segment close (ms)
NOISE_FLOOR_INIT: float = cfg.vad.noise_floor_init    # initial noise floor estimate

# ---------------------------------------------------------------------------
# Extractor (SpeakerBeam / Conv-TasNet) — only active when enabled=true
# ---------------------------------------------------------------------------
EXTRACTOR_ENABLED: bool = cfg.extractor.enabled          # toggle whole extractor stage
EXTRACTOR_WINDOW_S: float = cfg.extractor.window_s       # window fed to model (s)
EXTRACTOR_HOP_S: float = cfg.extractor.hop_s             # hop between windows (s)
EXTRACTOR_MODEL: str = cfg.extractor.model               # "speakerbeam" or "conv_tasnet"
EXTRACTOR_DEVICE: str = cfg.extractor.device             # "mps" | "cpu" | "cuda"
EXTRACTOR_MAX_WORKERS: int = cfg.extractor.max_workers   # parallel extraction threads

# ---------------------------------------------------------------------------
# Speaker encoder (ResNet34-LM via pyannote)
# ---------------------------------------------------------------------------
ENCODER_MODEL: str = cfg.encoder.model          # HuggingFace model ID
ENCODER_DEVICE: str = cfg.encoder.device        # "mps" | "cpu" | "cuda"
EMBEDDING_DIM: int = cfg.encoder.embedding_dim  # output dimension (256)

# ---------------------------------------------------------------------------
# Online speaker tracker
# ---------------------------------------------------------------------------
SIMILARITY_THRESHOLD: float = cfg.tracker.similarity_threshold  # cosine sim to accept match
EMA_ALPHA: float = cfg.tracker.ema_alpha                        # centroid update rate
MAX_SPEAKERS: int = cfg.tracker.max_speakers                    # gallery capacity

# ---------------------------------------------------------------------------
# Enrollment store
# ---------------------------------------------------------------------------
ENROLLMENT_THRESHOLD: float = cfg.enrollment.threshold          # cosine sim for name match
PROFILES_DIR: Path = Path(__file__).parent.parent / "profiles"  # .npy files live here
ENROLLMENT_DURATION_S: int = cfg.enrollment.duration_s          # seconds of speech to record
ENROLLMENT_NUM_SAMPLES: int = cfg.enrollment.num_samples        # embeddings averaged per profile

# ---------------------------------------------------------------------------
# Gating policy
# ---------------------------------------------------------------------------
POLICY_MODE: str = cfg.policy.mode              # "ENROLLED" | "ALL" | "DYNAMIC"
DYNAMIC_TARGET: str = cfg.policy.dynamic_target # speaker ID used in DYNAMIC mode

# ---------------------------------------------------------------------------
# Debug flag
# ---------------------------------------------------------------------------
DEBUG: bool = cfg.debug  # enables verbose console prints across all modules

# ---------------------------------------------------------------------------
# Ultravox API
# ---------------------------------------------------------------------------
ULTRAVOX_API_KEY: str = os.getenv("ULTRAVOX_API_KEY", "")  # sourced from .env
ULTRAVOX_JOIN_URL: str = ""  # pre-existing call URL; if empty, main.py creates one
ULTRAVOX_SYSTEM_PROMPT: str = "You are a helpful assistant."
ULTRAVOX_IN_RATE: int = cfg.ultravox.in_rate    # PCM sample rate sent to Ultravox
ULTRAVOX_OUT_RATE: int = cfg.ultravox.out_rate  # PCM sample rate received from Ultravox
ULTRAVOX_CHUNK_MS: int = cfg.ultravox.chunk_ms  # audio chunk size for streaming (ms)

# ---------------------------------------------------------------------------
# Internal numerical constants
# ---------------------------------------------------------------------------
NORM_FLOOR: float = cfg.internal.norm_floor                  # minimum L2 norm (avoid div-by-zero)
NOISE_FLOOR_EMA_FAST: float = cfg.internal.noise_floor_ema_fast  # fast EMA coefficient (speech)
NOISE_FLOOR_EMA_SLOW: float = cfg.internal.noise_floor_ema_slow  # slow EMA coefficient (silence)
MIN_SEGMENT_S: float = cfg.encoder.min_segment_s             # shortest segment the encoder accepts

RANDOM_SEED: int = cfg.random_seed  # seeds numpy + torch for reproducibility
