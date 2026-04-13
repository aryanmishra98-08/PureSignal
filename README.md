# PureSignal

A real-time speaker-focused audio pipeline for Apple Silicon. PureSignal listens to your microphone, identifies enrolled speakers using voice embeddings, and streams only their speech to an [Ultravox](https://ultravox.ai) AI voice agent — silently dropping everyone else.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Setup and Installation](#setup-and-installation)
- [Running the Application](#running-the-application)
- [Verifying the Setup](#verifying-the-setup)
- [Configuration Reference](#configuration-reference)
- [License](#license)

---

## Features

- **Voice enrollment** — record a voice sample per user and save it as a profile
- **Real-time speaker identification** — cosine similarity against enrolled embeddings on every speech segment
- **Multi-user support** — enroll and select multiple users at startup; unknown speakers are labelled automatically (`S1`, `S2`, …)
- **Policy gating** — two modes: `ENROLLED` (pass only matched users) and `DYNAMIC` (pass a specific tracker label)
- **Ultravox integration** — passes approved audio over a WebSocket to a live Ultravox AI call and plays back the response
- **Apple Silicon optimised** — encoder runs on MPS (Metal Performance Shaders) for low-latency inference
- **Adaptive VAD** — energy + zero-crossing rate detector with configurable hangover to produce clean speech segments

---

## Project Structure

```
PureSignal/
├── profiles/                  # Enrolled voice profiles (<username>.npy)
├── PureSignal/
│   ├── config.py              # Single source of truth for all settings
│   ├── enroll.py              # Standalone enrollment script
│   ├── main.py                # Pipeline orchestrator
│   ├── audio/
│   │   ├── capture.py         # Mic input → ring buffer + frame queue
│   │   ├── features.py        # Audio normalization utilities
│   │   ├── resampler.py       # 16 kHz → 48 kHz PCM conversion
│   │   └── vad.py             # Frame-level voice activity detection
│   ├── speaker/
│   │   ├── encoder.py         # ResNet34-LM speaker embedding extractor
│   │   ├── enrollment.py      # Profile loader and cosine matcher
│   │   ├── policy.py          # Pass/drop gate (ENROLLED / DYNAMIC)
│   │   └── tracker.py         # Online speaker tracking with EMA centroids
│   ├── llm/
│   │   └── ultravox_client.py # Ultravox WebSocket send/receive client
│   └── keys/
│       └── .env               # API keys (not committed)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## How It Works

```
Microphone
    │
    ▼
[capture.py] — 20ms frames → frame_queue
    │
    ▼
[vad.py] — energy + ZCR + hangover → complete speech segments
    │
    ▼
[features.py] — L2 normalization
    │
    ▼
[encoder.py] — ResNet34-LM → 256-dim embedding (MPS)
    │
    ├──▶ [tracker.py] — assign / register speaker ID (S1, S2, …)
    │
    └──▶ [enrollment.py] — cosine match against profiles/*.npy
              │
              ▼
         [policy.py] — ENROLLED: pass if matched user
                                 drop if unknown
              │
              ▼
         [ultravox_client.py] — resample → WebSocket → Ultravox AI
              │
              ▼
         Speakers (AI response playback)
```

1. **Enroll** — `enroll.py` records a voice sample, extracts an embedding, and saves it to `profiles/<username>.npy`.
2. **Select users** — `main.py` prompts you to choose enrolled users (single or multi) at startup.
3. **Capture** — the mic streams 20ms frames continuously.
4. **VAD** — frames are accumulated into speech segments using energy and zero-crossing rate thresholds with a configurable hangover window.
5. **Embed** — each segment is passed through a HuggingFace `pyannote/wespeaker-voxceleb-resnet34-LM` model running on MPS.
6. **Identify** — the embedding is compared against all enrolled profiles. Matched = username label; unmatched = tracker ID.
7. **Gate** — segments from non-enrolled speakers are silently dropped.
8. **Stream** — approved segments are resampled to 48 kHz and sent over a WebSocket to Ultravox, which responds in real time.

---

## Setup and Installation

### Prerequisites

- Python 3.10+
- macOS with Apple Silicon (M1/M2/M3/M4) for MPS acceleration
- A [HuggingFace](https://huggingface.co) account with access to [`pyannote/wespeaker-voxceleb-resnet34-LM`](https://hf.co/pyannote/wespeaker-voxceleb-resnet34-LM)
- An [Ultravox](https://ultravox.ai) API key

### 1. Clone the repository

```bash
git clone https://github.com/your-username/PureSignal.git
cd PureSignal
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

Edit `PureSignal/keys/.env`:

```env
ULTRAVOX_API_KEY=your_ultravox_api_key_here
HF_TOKEN=your_huggingface_token_here
```

> Accept the model terms at [hf.co/pyannote/wespeaker-voxceleb-resnet34-LM](https://hf.co/pyannote/wespeaker-voxceleb-resnet34-LM) before generating your HF token.

### 5. Enroll at least one user

```bash
cd PureSignal
python3 enroll.py
```

Follow the prompts — enter a username and speak clearly for 12 seconds. The profile is saved to `PureSignal/profiles/<username>.npy`.

---

## Running the Application

```bash
cd PureSignal
python3 main.py
```

On startup you will be prompted to:

1. Choose a mode — **[1] Single-user** or **[2] Multi-user**
2. Enter one or more enrolled usernames
3. The pipeline initialises and begins listening

Press `Ctrl+C` to stop cleanly.

---

## Verifying the Setup

| Check | Command / Action |
|---|---|
| Dependencies installed | `pip show pyannote.audio torch sounddevice` |
| HF token valid | `python3 -c "from huggingface_hub import whoami; print(whoami())"` |
| Profiles exist | `ls PureSignal/profiles/` |
| Encoder loads | Run `enroll.py` — watch for `[encoder] loading …` without errors |
| VAD fires | Run `main.py` with `DEBUG = True` in `config.py` and speak — watch for `[vad] segment ready` |
| Policy passing | Check logs for `→ PASS — sending to Ultravox` when the enrolled user speaks |

---

## Configuration Reference

All settings live in [`PureSignal/config.py`](PureSignal/config.py).

| Key | Default | Description |
|---|---|---|
| `SAMPLE_RATE` | `16000` | Internal pipeline sample rate (Hz) |
| `FRAME_MS` | `20` | VAD frame size (ms) |
| `WINDOW_SIZE_S` | `1.5` | Ring buffer length (s) |
| `ENERGY_MULTIPLIER` | `3.0` | Speech energy must exceed `noise_floor × this` |
| `ZCR_THRESHOLD` | `0.3` | Zero-crossing rate ceiling for speech frames |
| `HANGOVER_MS` | `400` | Keep speech flag active N ms after energy drops |
| `ENCODER_MODEL` | `pyannote/wespeaker-voxceleb-resnet34-LM` | HuggingFace model ID |
| `ENCODER_DEVICE` | `mps` | Inference device (`mps` / `cpu` / `cuda`) |
| `EMBEDDING_DIM` | `256` | Speaker embedding dimensionality |
| `SIMILARITY_THRESHOLD` | `0.65` | Cosine similarity floor for tracker assignment |
| `ENROLLMENT_THRESHOLD` | `0.65` | Cosine similarity floor for enrolled user match |
| `EMA_ALPHA` | `0.25` | Centroid update rate for tracker |
| `MAX_SPEAKERS` | `4` | Maximum simultaneous tracked speakers |
| `PROFILES_DIR` | `PureSignal/profiles/` | Directory for enrolled voice profiles |
| `ENROLLMENT_DURATION_S` | `12` | Recording length during enrollment (s) |
| `POLICY_MODE` | `ENROLLED` | `ENROLLED` passes matched users; `DYNAMIC` passes a fixed tracker label |
| `DYNAMIC_TARGET` | `S1` | Tracker label passed in `DYNAMIC` mode |
| `ULTRAVOX_IN_RATE` | `48000` | WebSocket input sample rate (Hz) |
| `ULTRAVOX_OUT_RATE` | `48000` | Playback sample rate (Hz) |
| `DEBUG` | `True` | Enable/disable runtime logs |

---

## License

This project is licensed under the [MIT License](LICENSE).