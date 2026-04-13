# =============================================================================
# config.py — Single source of truth for all pipeline configuration
# =============================================================================
import os
from pathlib import Path
from dotenv import load_dotenv

# Load secrets from keys/.env (relative to this file's directory)
_env_path = Path(__file__).parent / "keys" / ".env"
load_dotenv(dotenv_path=_env_path)

# --- Audio Capture ---
SAMPLE_RATE           = 16000     # internal pipeline sample rate (Hz)
FRAME_MS              = 20        # VAD frame size in milliseconds
FRAME_SAMPLES         = int(SAMPLE_RATE * FRAME_MS / 1000)  # 320 samples
WINDOW_SIZE_S         = 1.5       # ring buffer window length (seconds)
HOP_SIZE_S            = 0.25      # ring buffer hop length (seconds)

# --- VAD ---
ENERGY_MULTIPLIER     = 3.0       # speech energy must exceed noise_floor * this
ZCR_THRESHOLD         = 0.3       # zero-crossing rate ceiling for speech frames
HANGOVER_MS           = 400       # keep speech flag active N ms after energy drops (longer = bigger segments)
NOISE_FLOOR_INIT      = 0.01      # initial noise floor estimate

# --- Speaker Encoder ---
# ResNet34 LM — lightweight, MPS-compatible, 256-dim embeddings
ENCODER_MODEL         = "pyannote/wespeaker-voxceleb-resnet34-LM"
ENCODER_DEVICE        = "mps"     # Apple Silicon M4
EMBEDDING_DIM         = 256

# --- Speaker Tracker ---
SIMILARITY_THRESHOLD  = 0.65      # cosine sim floor to assign existing speaker
EMA_ALPHA             = 0.25      # centroid update rate (higher = adapts faster to same-speaker variation)
MAX_SPEAKERS          = 4

# --- Enrollment ---
ENROLLMENT_THRESHOLD  = 0.65      # cosine sim floor to match enrolled speaker
PROFILES_DIR          = Path(__file__).parent.parent / "profiles"  # centralized profiles directory
ENROLLMENT_DURATION_S = 12        # recording length during enroll.py

# --- Policy ---
POLICY_MODE           = "ENROLLED"   # "ENROLLED" | "DYNAMIC"
DYNAMIC_TARGET        = "S1"         # active only when POLICY_MODE = "DYNAMIC"

# --- Debug ---
DEBUG                 = True    # set False to silence runtime logs

# --- Ultravox / Fixie.ai ---
# Join URLs are single-use and expire — set ULTRAVOX_API_KEY and leave
# ULTRAVOX_JOIN_URL empty to auto-create a new call each run.
ULTRAVOX_API_KEY      = os.getenv("ULTRAVOX_API_KEY", "")   # loaded from keys/.env
ULTRAVOX_JOIN_URL     = ""        # leave empty to auto-create; or paste a fresh URL to override
ULTRAVOX_SYSTEM_PROMPT = "You are a helpful assistant."  # system prompt for the call
ULTRAVOX_IN_RATE      = 48000     # WebSocket expects 48kHz PCM input
ULTRAVOX_OUT_RATE     = 48000     # playback sample rate
ULTRAVOX_CHUNK_MS     = 20        # send one chunk every 20ms
