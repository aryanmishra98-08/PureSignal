# PureSignal

A real-time speaker-focused audio pipeline for Apple Silicon. PureSignal listens to your microphone, identifies enrolled speakers using voice embeddings, and streams only their speech to an [Ultravox](https://ultravox.ai) AI voice agent — silently dropping everyone else.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Setup and Installation](#setup-and-installation)
- [Running the Application](#running-the-application)
- [Testing](#testing)
- [Offline Evaluation](#offline-evaluation)
- [Verifying the Setup](#verifying-the-setup)
- [Configuration Reference](#configuration-reference)
- [License](#license)

---

## Features

- **Voice enrollment** — record multiple voice samples per user and average their embeddings into a single profile with a quality gate
- **Real-time speaker identification** — cosine similarity against enrolled embeddings on every speech segment
- **Multi-user support** — enroll and select up to 10 users at startup; unknown speakers are labelled automatically (`S1`, `S2`, …)
- **Policy gating** — two modes: `ENROLLED` (pass only matched users) and `DYNAMIC` (pass a specific tracker label)
- **Ultravox integration** — auto-creates a call via the Ultravox REST API and streams approved audio over a WebSocket; plays back the AI response in real time
- **Apple Silicon optimised** — encoder runs on MPS (Metal Performance Shaders) for low-latency inference, with automatic CPU fallback
- **Adaptive VAD** — energy + zero-crossing rate detector with configurable hangover; end-of-file flush ensures the last speech segment is never silently discarded
- **File source mode** — run the full pipeline offline against a WAV file (`--source path/to/file.wav`) without a microphone
- **Robust WebSocket client** — retries connection up to 3 times; silence padding maintains stream continuity between speech segments
- **Structured session logging** — every diagnostic is written to `logs/<session>.jsonl` at every level, whatever the console shows, so a session log is complete for latency analysis and debugging

---

## Project Structure

```
PureSignal/
├── src/                          # Pipeline source code
│   ├── config.py                 # Flat constant surface over config/base.yaml
│   ├── config_loader.py          # load_config() → Config; supports --set CLI overrides
│   ├── main.py                   # Pipeline orchestrator
│   ├── enroll.py                 # Standalone enrollment script
│   ├── audio/
│   │   ├── capture.py            # MicCapture — mic input → frame queue
│   │   ├── file_capture.py       # AudioFileCapture — WAV source, same interface
│   │   ├── features.py           # L2 peak normalization
│   │   ├── resampler.py          # 16 kHz float32 → 48 kHz int16 PCM conversion
│   │   ├── vad.py                # VoiceActivityDetector — frame-level VAD, flush() for EOF
│   │   └── wav_io.py             # Shared WAV decoder (mono float32 @ 16 kHz)
│   ├── speaker/
│   │   ├── encoder.py            # SpeakerEncoder — ResNet34-LM embedding extractor
│   │   ├── enrollment.py         # EnrollmentStore + quality gate and metadata writer
│   │   ├── policy.py             # PolicyGate — pass/drop (ENROLLED / DYNAMIC)
│   │   └── tracker.py            # SpeakerTracker — EMA centroids, per-session gallery
│   ├── llm/
│   │   └── ultravox_client.py    # Ultravox WebSocket send/receive client
│   └── utils/
│       └── logger.py             # Levelled logger: console + logs/<session>.jsonl
├── config/
│   └── base.yaml                 # All tunable settings — single source of truth (data only)
├── eval/
│   ├── common.py                 # Shared path bootstrap, WAV/embed helpers, plot guard
│   ├── metrics.py                # Pure FAR/FRR/EER/DET functions
│   ├── run_speaker_eval.py       # FAR/FRR/EER sweep over test + impostor WAVs
│   ├── tracker_eval.py           # Tracker purity from a sequenced multi-speaker session
│   ├── latency_report.py         # Per-stage P50/P95/P99 from session JSONL logs
│   ├── vad_eval.py               # VAD precision/recall/F1 parameter sweep
│   └── enrollment_quality_eval.py# EER vs. number of enrollment samples curve
├── tests/
│   ├── conftest.py
│   ├── test_audio_pipeline.py    # VAD, normalization, resampler, and flush tests
│   ├── test_config_and_wav_io.py # Config reload/overrides and WAV decoding
│   ├── test_logger.py            # Log level rules and the JSONL record contract
│   ├── test_pipeline_loop.py     # Processing loop and end-of-stream handling
│   ├── test_speaker_pipeline.py  # Encoder, enrollment, tracker, and policy tests
│   └── test_ultravox_client.py   # WebSocket send/receive and shutdown, vs a stub server
├── profiles/                     # Enrolled voice profiles (*.npy — gitignored)
├── logs/                         # Per-session JSONL logs (gitignored)
├── .env                          # API keys: ULTRAVOX_API_KEY, HF_TOKEN (gitignored)
├── Dockerfile
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## How It Works

```
[mic | WAV file]
       │
       ▼
[MicCapture / AudioFileCapture] — 20ms float32 frames → frame_queue
       │
       ▼
[vad.py] — energy + ZCR + hangover → complete speech segments
           flush() called on EOF to recover final segment
       │
       ▼
[features.py] — L2 peak normalization
       │
       ▼
[encoder.py] — ResNet34-LM → 256-dim L2-normalized embedding (MPS, worker thread)
       │
       ├──▶ [tracker.py] — assign / register speaker ID (S1, S2, …); EMA centroid updates
       │
       └──▶ [enrollment.py] — cosine match against profiles/*.npy
                  │
                  ▼
             [policy.py] — ENROLLED: pass if matched user, drop if unknown
                           DYNAMIC:  pass if speaker ID == DYNAMIC_TARGET
                  │
                  ▼
             [ultravox_client.py] — resample to 48kHz int16 → WebSocket → Ultravox AI
                  │
                  ▼
             Speakers (AI response playback)
                  │
             [logger.py] — logs/<session>.jsonl (always) + console (if DEBUG=true)
```

1. **Enroll** — `enroll.py` records N voice samples, quality-gates each embedding (norm + entropy), averages them, and saves to `profiles/<username>.npy` + `profiles/<username>_meta.json`.
2. **Select users** — `main.py` prompts you to choose enrolled users (single or multi-user, up to 10) at startup.
3. **Capture** — the mic streams 20ms frames continuously into a bounded queue (500 frames, ~10s).
4. **VAD** — frames are accumulated into speech segments using energy and zero-crossing rate thresholds with a configurable hangover window. When a WAV file source reaches EOF, `vad.flush()` recovers any buffered segment that never saw trailing silence.
5. **Embed** — each segment is normalized and passed through `pyannote/wespeaker-voxceleb-resnet34-LM` in a background thread to keep the main loop responsive.
6. **Identify** — the embedding is compared against all enrolled profiles. Matched = username label; unmatched = tracker ID (`S1`, `S2`, …).
7. **Gate** — segments from non-enrolled (or non-target) speakers are silently dropped.
8. **Stream** — approved segments are resampled to 48 kHz int16 PCM and sent over a WebSocket to Ultravox, which responds in real time. Silence frames are sent between segments to keep the stream alive.

### Component ownership

The VAD, encoder, tracker, enrollment store and policy gate are objects, not
module singletons. `main.py` is the composition root: it builds them in
`_build_pipeline()` and holds them in a `Pipeline` for the life of a session.

Two consequences worth knowing:

- **Ending a session means dropping the `Pipeline`**, not resetting shared
  state. There is no global to leak between runs, and several pipelines can
  exist in one process.
- **Tunables are captured at construction.** Each component reads `config`
  when it is built, so `--config` and `--set` overrides are picked up simply
  by building the pipeline after applying them. Callers that need something
  other than the configured value pass it directly — `vad_eval.py` builds a
  `VoiceActivityDetector(energy_multiplier=..., hangover_ms=...)` per sweep
  point rather than mutating global config.

`llm/ultravox_client.py` stays a module-level singleton on purpose: it owns
the process's audio output device and its one WebSocket, so a second instance
would contend for the same hardware.

### Output: diagnostics vs. interactive UI

There are two separate output channels, and the distinction is what decides
which one a given line uses.

**The logger** (`utils/logger.py`) carries diagnostics — anything describing
what the pipeline is doing. Every call writes to `logs/<session>.jsonl`
regardless of level; the level only decides what also reaches the console:

| Level | Console | Log file |
|---|---|---|
| `DEBUG` | only when `debug: true` | always |
| `INFO` / `WARNING` / `ERROR` | always | always |

So turning `debug` off quietens the terminal without losing anything from the
session log — per-segment VAD, encoder, tracker and policy events are still
recorded for `eval/latency_report.py` to read back.

**`print()`** carries interactive UI — the startup banner, the user-selection
menu, enrollment prompts and countdowns, and the result a command was run to
produce. This output is never suppressed by a log level and never lands in
the JSONL, because it is a conversation with the operator rather than a
record of what happened.

The `eval/` scripts keep using `print()` throughout: their stdout *is* the
report they were run to produce.

---

## Setup and Installation

### Prerequisites

- Python 3.10–3.11 (3.12+ not yet supported by `pyannote.audio`)
- macOS with Apple Silicon (M1/M2/M3/M4) for MPS acceleration; CPU fallback is automatic
- A [HuggingFace](https://huggingface.co) account with accepted terms for [`pyannote/wespeaker-voxceleb-resnet34-LM`](https://hf.co/pyannote/wespeaker-voxceleb-resnet34-LM)
- An [Ultravox](https://ultravox.ai) API key (not required for offline / `--no-ultravox` mode)

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

Create `.env` at the repo root:

```env
ULTRAVOX_API_KEY=your_ultravox_api_key_here
HF_TOKEN=your_huggingface_token_here
```

> Accept the model terms at [hf.co/pyannote/wespeaker-voxceleb-resnet34-LM](https://hf.co/pyannote/wespeaker-voxceleb-resnet34-LM) before generating your HF token.

`main.py` automatically creates a new Ultravox call each run using your API key. To reuse a specific call URL, set `ULTRAVOX_JOIN_URL` in `.env` or as an environment variable.

### 5. Enroll at least one user

```bash
python3 src/enroll.py
```

Follow the prompts — enter a username and speak clearly for 12 seconds per sample. By default 3 samples are recorded and averaged. The profile is saved to `profiles/<username>.npy`. Usernames are restricted to alphanumeric characters, `-`, and `_`.

To record more samples for a higher-quality profile:

```bash
python3 src/enroll.py --samples 5
```

---

## Running the Application

```bash
python3 src/main.py
```

On startup you will be prompted to choose a mode — **[1] Single-user** or **[2] Multi-user** (up to 10) — and enter enrolled usernames. The pipeline then initialises, connects to Ultravox, and begins listening.

Press `Ctrl+C` to stop cleanly. The pipeline shuts down in order: mic → WebSocket → VAD state → tracker state → log file.

### Useful flags

```bash
# Run against a WAV file instead of the mic (no hardware required)
python3 src/main.py --source path/to/file.wav --no-ultravox

# Skip Ultravox entirely (offline / evaluation mode)
python3 src/main.py --no-ultravox

# Override a config value at runtime
python3 src/main.py --set vad.hangover_ms=300 --set debug=false

# Point to a different config file
python3 src/main.py --config config/base.yaml
```

---

## Testing

Tests run without any hardware (no microphone, GPU, or API keys required). The
Ultravox tests spin up a loopback WebSocket server on localhost and stub only
the audio device, so the client's real threading and shutdown behaviour is
exercised:

```bash
pytest tests/ -v
```

The test suite covers 92 cases across all core modules:

| Module | Tests |
|---|---|
| `audio/vad.py` | Silence returns `None`; speech segment returned after hangover; segment content matches input; 30s buffer never overflows; `flush()` recovers active segment; `flush()` returns `None` when silent; `flush()` returns `None` after a segment already closed by `process_frame`; constructor overrides beat config; two detectors keep independent state; `reset()` clears state but keeps settings |
| `audio/features.py` | Peak normalization to `[-1, 1]`; near-silent input returns finite float32 without NaN/Inf |
| `audio/resampler.py` | 1s at 16 kHz → correct byte length at 48 kHz; silence frame correct length and all-zero; `pcm_frame_bytes()` agrees with the silence frame; clipping prevents int16 wraparound on loud signals |
| `audio/wav_io.py` | int16 scaled to unit range; float32 passes through; stereo downmixed to mono; resampled to an explicit target rate; defaults to `config.SAMPLE_RATE` |
| `config.py` | Constants derived from YAML; `reload()` applies `--set` overrides to the module constants; derived constants recomputed; defaults restored; malformed `--set` raises `ValueError` |
| `main.py` | Speech running to EOF is embedded via the end-of-stream flush; a segment closed by trailing silence is embedded once; a silence-only stream embeds nothing; the EOF sentinel ends the loop; embeddings reach the tracker the loop was handed; two pipelines share no state |
| `speaker/encoder.py` | Missing `HF_TOKEN` raises `OSError` immediately; a present token is returned; `embed()` before `load()` raises `RuntimeError`; settings come from config; constructor overrides beat config; a too-short clip is rejected without consulting the model |
| `speaker/enrollment.py` | Load + match exact vector; orthogonal vector returns `None`; missing profile raises `FileNotFoundError`; a failed load yields no store at all (fails closed); profiles directory defaults to config; `match_with_score()` returns name and similarity; an empty store matches nothing; two stores stay independent; zero-norm embedding fails quality gate; normalized embedding passes quality gate; `save_with_metadata()` writes both `.npy` and `_meta.json` |
| `speaker/tracker.py` | First speaker is `S1`; same speaker remains stable under small perturbation; orthogonal speakers get distinct IDs; gallery-full fallback assigns closest; `gallery()` returns an isolated copy; EMA centroid drifts toward repeated updates; `assign_with_score()` reports the driving similarity; `speaker_ids()` tracks the gallery; two trackers keep independent galleries; `reset()` restarts numbering; constructor overrides beat config |
| `speaker/policy.py` | `ENROLLED` mode pass/drop; `DYNAMIC` mode pass/drop; an unknown mode raises `ValueError` at construction, not mid-session; defaults come from config |
| `utils/logger.py` | INFO/WARNING/ERROR print even with debug off; DEBUG is suppressed with debug off and shown with it on; suppressed DEBUG is still written to the log; each level is recorded with its level; the JSONL record shape is stable; typed pipeline events stay DEBUG; `latency_record` still parses for the report; the callable form still works |
| `llm/ultravox_client.py` | Run against a real loopback WebSocket server: `stop()` terminates the client thread even when the server is silent; `stop()` during the startup window does not leak the thread; `stop()` is idempotent and safe without `start()`; the client restarts after `stop()`; segments reach the server as fixed-size frames; a full queue reports 0; inbound PCM is played; `playbackClearBuffer` restarts the stream; malformed JSON does not kill the receive loop |

Lint (the whole repo is clean under the configured ruleset):

```bash
ruff check .
```

---

## Offline Evaluation

The `eval/` directory contains scripts for measuring pipeline quality without a live mic or Ultravox connection. All scripts write results to `eval/results/`.

| Script | What it measures |
|---|---|
| `eval/run_speaker_eval.py` | FAR, FRR, EER over labelled test + impostor WAV directories; outputs `scores.csv`, `metrics.json`, `det_curve.png` |
| `eval/tracker_eval.py` | Tracker purity from a sequenced multi-speaker session; outputs `tracker_results.json`, `timeline.png` |
| `eval/latency_report.py` | Per-stage P50/P95/P99 latency from a session JSONL log; outputs a histogram PNG with `--plot` |
| `eval/vad_eval.py` | Precision/recall/F1 sweep over `energy_multiplier` × `hangover_ms`; outputs `param_sweep.csv`, `heatmap.png` |
| `eval/enrollment_quality_eval.py` | EER vs. number of enrollment samples; outputs `eer_vs_samples.png` |

Example — generate a latency report from a recorded session:

```bash
python3 eval/latency_report.py --log logs/<session>.jsonl --plot
```

---

## Verifying the Setup

| Check | Command / Action |
|---|---|
| Dependencies installed | `pip show pyannote.audio torch sounddevice` |
| HF token valid | `python3 -c "from huggingface_hub import whoami; print(whoami())"` |
| Profiles exist | `ls profiles/` |
| Encoder loads | Run `src/enroll.py` — watch for `[encoder] loading …` without errors |
| VAD fires | Run `src/main.py` with `debug: true` in `config/base.yaml` and speak — watch for `[vad] segment_closed` in the console |
| Policy passing | Check `logs/<session>.jsonl` for `"event": "decision", "data": {"decision": "PASS"}` when the enrolled user speaks |
| Ultravox connected | Watch for `[ultravox] connected` in the console after startup |

---

## Configuration Reference

All settings live in [`config/base.yaml`](config/base.yaml). Override any value at runtime with `--set key.subkey=value` or by editing the YAML directly. `src/config.py` re-exports every setting as a flat constant (`config.SAMPLE_RATE`), so pipeline code never walks the nested tree; `src/config_loader.py` does the YAML parsing and override merging.

| Key | Default | Description |
|---|---|---|
| `audio.sample_rate` | `16000` | Internal pipeline sample rate (Hz) |
| `audio.frame_ms` | `20` | VAD frame size (ms) |
| `audio.window_size_s` | `1.5` | Ring buffer length (s) |
| `audio.hop_size_s` | `0.25` | Ring buffer hop length (s) |
| `vad.energy_multiplier` | `3.0` | Speech energy must exceed `noise_floor × this` |
| `vad.zcr_threshold` | `0.3` | Zero-crossing rate ceiling for speech frames |
| `vad.hangover_ms` | `400` | Keep speech flag active N ms after energy drops |
| `vad.noise_floor_init` | `0.01` | Initial noise floor estimate |
| `encoder.model` | `pyannote/wespeaker-voxceleb-resnet34-LM` | HuggingFace model ID |
| `encoder.device` | `mps` | Inference device (`mps` / `cpu`); falls back to CPU automatically |
| `encoder.embedding_dim` | `256` | Speaker embedding dimensionality |
| `encoder.min_segment_s` | `0.5` | Minimum segment length accepted by the encoder (s) |
| `tracker.similarity_threshold` | `0.65` | Cosine similarity floor for tracker assignment |
| `tracker.ema_alpha` | `0.25` | Centroid update rate (higher = faster adaptation) |
| `tracker.max_speakers` | `4` | Maximum simultaneous tracked speakers per session |
| `enrollment.threshold` | `0.65` | Cosine similarity floor for enrolled user match |
| `enrollment.duration_s` | `12` | Recording length per sample during enrollment (s) |
| `enrollment.num_samples` | `3` | Default number of samples to average (`--samples N` overrides) |
| `policy.mode` | `ENROLLED` | `ENROLLED` passes matched users; `DYNAMIC` passes a fixed tracker label |
| `policy.dynamic_target` | `S1` | Tracker label passed in `DYNAMIC` mode |
| `ultravox.in_rate` | `48000` | WebSocket input sample rate (Hz) |
| `ultravox.out_rate` | `48000` | Playback sample rate (Hz) |
| `ultravox.chunk_ms` | `20` | PCM chunk size sent per WebSocket frame (ms) |
| `internal.norm_floor` | `1e-6` | Minimum L2 norm before treating a vector as zero |
| `internal.noise_floor_ema_fast` | `0.05` | Fast EMA weight for noise floor adaptation |
| `internal.noise_floor_ema_slow` | `0.95` | Slow EMA weight for noise floor adaptation |
| `random_seed` | `42` | Seed for NumPy and PyTorch (reproducibility) |
| `debug` | `true` | Print per-event console logs alongside JSONL output |

---

## License

This project is licensed under the [MIT License](LICENSE).
