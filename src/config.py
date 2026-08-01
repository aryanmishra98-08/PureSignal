# =============================================================================
# config.py — Backward-compatibility shim
#
# Loads config/base.yaml via config/loader.py and re-exports every setting
# as a flat module-level constant.  All sub-modules import from here so that
# the rest of the codebase never references the YAML structure directly.
#
# Override at runtime:
#   python src/main.py --set vad.hangover_ms=300
#
# The constants below are assigned by _bind(), not at module scope, so that
# rebind() can replace all of them atomically when --config / --set is used.
# Nothing here may be read at import time by a pipeline module: rebind() must
# be able to change it.  See rebind()'s import guard.
# =============================================================================
import importlib.util
import os
import sys
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

# Pipeline modules that read constants from this shim.  rebind() refuses to run
# once any of them is in sys.modules, because their import-time state would be
# derived from the pre-override values.
_PIPELINE_MODULES = (
    "audio.vad", "audio.capture", "audio.file_capture", "audio.windowed_source",
    "audio.window_buffer", "audio.extractor", "audio.features", "audio.resampler",
    "speaker.encoder", "speaker.enrollment", "speaker.policy", "speaker.tracker",
    "llm.ultravox_client",
)


def _bind(c) -> None:
    """Derive every flat constant from Config object `c` into module globals."""
    g = globals()

    # -----------------------------------------------------------------------
    # Audio capture
    # -----------------------------------------------------------------------
    g["SAMPLE_RATE"] = c.audio.sample_rate                          # Hz — 16000
    # VAD frame length in ms
    g["FRAME_MS"] = c.audio.frame_ms
    g["FRAME_SAMPLES"] = int(c.audio.sample_rate * c.audio.frame_ms / 1000)

    # -----------------------------------------------------------------------
    # Voice Activity Detection
    # -----------------------------------------------------------------------
    # threshold = noise_floor × multiplier
    g["ENERGY_MULTIPLIER"] = c.vad.energy_multiplier
    # zero-crossing rate gate
    g["ZCR_THRESHOLD"] = c.vad.zcr_threshold
    # tail silence before close (ms)
    g["HANGOVER_MS"] = c.vad.hangover_ms
    # initial noise floor estimate
    g["NOISE_FLOOR_INIT"] = c.vad.noise_floor_init
    # clamp as fraction of init
    g["NOISE_FLOOR_MIN_RATIO"] = c.vad.noise_floor_min_ratio
    g["MAX_SEGMENT_S"] = c.vad.max_segment_s              # force-close cap (s)

    # -----------------------------------------------------------------------
    # Extractor (Conv-TasNet) — only active when enabled=true
    # -----------------------------------------------------------------------
    # toggle whole extractor stage
    g["EXTRACTOR_ENABLED"] = c.extractor.enabled
    # window fed to model (s)
    g["EXTRACTOR_WINDOW_S"] = c.extractor.window_s
    # hop between windows (s)
    g["EXTRACTOR_HOP_S"] = c.extractor.hop_s
    g["EXTRACTOR_MODEL"] = c.extractor.model                # "conv_tasnet"
    # HuggingFace checkpoint ID
    g["EXTRACTOR_CHECKPOINT"] = c.extractor.checkpoint
    # "mps" | "cpu" | "cuda"
    g["EXTRACTOR_DEVICE"] = c.extractor.device
    # parallel extraction threads
    g["EXTRACTOR_MAX_WORKERS"] = c.extractor.max_workers
    g["EXTRACTOR_REANCHOR_EVERY"] = c.extractor.reanchor_every  # re-embed cadence
    # backpressure log cadence
    g["EXTRACTOR_LOG_EVERY"] = c.extractor.log_every
    # realtime-factor self-check
    g["EXTRACTOR_STARTUP_CHECK"] = c.extractor.startup_check

    # -----------------------------------------------------------------------
    # Speaker encoder (ResNet34-LM via pyannote)
    # -----------------------------------------------------------------------
    g["ENCODER_MODEL"] = c.encoder.model        # HuggingFace model ID
    g["ENCODER_DEVICE"] = c.encoder.device      # "mps" | "cpu" | "cuda"
    # shortest segment the encoder accepts
    g["MIN_SEGMENT_S"] = c.encoder.min_segment_s
    g["ENCODER_MAX_PENDING"] = c.encoder.max_pending_segments  # segment queue depth

    # -----------------------------------------------------------------------
    # Online speaker tracker
    # -----------------------------------------------------------------------
    # cosine sim to accept match
    g["SIMILARITY_THRESHOLD"] = c.tracker.similarity_threshold
    # centroid update rate
    g["EMA_ALPHA"] = c.tracker.ema_alpha
    g["MAX_SPEAKERS"] = c.tracker.max_speakers                  # gallery capacity

    # -----------------------------------------------------------------------
    # Enrollment store
    # -----------------------------------------------------------------------
    # cosine sim for name match
    g["ENROLLMENT_THRESHOLD"] = c.enrollment.threshold
    g["PROFILES_DIR"] = Path(__file__).parent.parent / \
        "profiles"  # .npy files live here
    # seconds of speech to record
    g["ENROLLMENT_DURATION_S"] = c.enrollment.duration_s
    # embeddings averaged per profile
    g["ENROLLMENT_NUM_SAMPLES"] = c.enrollment.num_samples

    # -----------------------------------------------------------------------
    # Gating policy
    # -----------------------------------------------------------------------
    # "ENROLLED" | "ALL" | "DYNAMIC"
    g["POLICY_MODE"] = c.policy.mode
    # speaker ID used in DYNAMIC mode
    g["DYNAMIC_TARGET"] = c.policy.dynamic_target

    # -----------------------------------------------------------------------
    # Debug flag
    # -----------------------------------------------------------------------
    g["DEBUG"] = c.debug  # enables verbose console prints across all modules

    # -----------------------------------------------------------------------
    # Ultravox API — env var wins over YAML for both URL and key
    # -----------------------------------------------------------------------
    g["ULTRAVOX_API_KEY"] = os.getenv("ULTRAVOX_API_KEY", "")
    g["ULTRAVOX_JOIN_URL"] = os.getenv(
        "ULTRAVOX_JOIN_URL", "") or c.ultravox.get("join_url", "")
    g["ULTRAVOX_SYSTEM_PROMPT"] = (
        os.getenv("ULTRAVOX_SYSTEM_PROMPT", "")
        or c.ultravox.get("system_prompt", "You are a helpful assistant.")
    )
    # PCM sample rate sent to Ultravox
    g["ULTRAVOX_IN_RATE"] = c.ultravox.in_rate
    # PCM sample rate received from Ultravox
    g["ULTRAVOX_OUT_RATE"] = c.ultravox.out_rate
    # audio chunk size for streaming (ms)
    g["ULTRAVOX_CHUNK_MS"] = c.ultravox.chunk_ms

    # -----------------------------------------------------------------------
    # Internal numerical constants
    # -----------------------------------------------------------------------
    # min L2 norm (avoid div-by-zero)
    g["NORM_FLOOR"] = c.internal.norm_floor
    # fast EMA coefficient
    g["NOISE_FLOOR_EMA_FAST"] = c.internal.noise_floor_ema_fast
    # slow EMA coefficient
    g["NOISE_FLOOR_EMA_SLOW"] = c.internal.noise_floor_ema_slow

    g["RANDOM_SEED"] = c.random_seed  # seeds numpy + torch for reproducibility


def rebind(new_cfg) -> None:
    """
    Replace the active config and re-derive every flat constant.

    Must be called before any pipeline module is imported.  Several modules
    read constants at import time to size buffers, so rebinding afterwards
    would leave them holding pre-override values, so the override would appear
    to work while changing nothing.  The guard below makes that failure loud
    instead of silent.

    Raises:
        RuntimeError if a pipeline module has already been imported.
    """
    global cfg
    already = [m for m in _PIPELINE_MODULES if m in sys.modules]
    if already:
        raise RuntimeError(
            "[config] rebind() called after pipeline modules were already "
            f"imported: {', '.join(already)}.\n"
            "  Parse --config / --set and rebind before importing audio.*, "
            "speaker.*, or llm.* — otherwise the override silently does nothing."
        )
    cfg = new_cfg
    _bind(cfg)


def load_and_rebind(path=None, set_args=None) -> None:
    """Load a config from disk with CLI overrides applied, then rebind()."""
    rebind(_loader_mod.load_config(path=path, set_args=set_args))


# cfg is the live Config object; main.py may replace it via --config / --set
cfg = _loader_mod.cfg
_bind(cfg)
