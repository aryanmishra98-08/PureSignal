# PureSignal

A real-time speaker-focused audio pipeline for Apple Silicon. PureSignal listens to your microphone, isolates the enrolled target speaker's voice using neural source separation, and streams only their speech to an [Ultravox](https://ultravox.ai) AI voice agent — silently dropping everyone else.

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

- **Target speaker extraction** — Conv-TasNet separates the mixed signal into its constituent sources before the VAD, and the source matching the enrolled speaker is selected and forwarded. This suppresses competing voices rather than merely gating whole segments.
- **Voice enrollment** — record multiple voice samples per user and average their embeddings into a single profile with a quality gate
- **Real-time speaker identification** — cosine similarity against enrolled embeddings on every speech segment
- **Multi-user support** — enroll and select up to 10 users at startup; unknown speakers are labelled automatically (`S1`, `S2`, …)
- **Policy gating** — three modes: `ENROLLED` (pass only matched users), `ALL` (pass everyone — the baseline arm for evaluation), and `DYNAMIC` (pass a specific tracker label)
- **Ultravox integration** — auto-creates a call via the Ultravox REST API, or rejoins an existing one via `ULTRAVOX_JOIN_URL`; streams approved audio over a WebSocket and plays the AI response on a dedicated playback thread
- **Runs on MPS** — encoder and separator are placed on Metal Performance Shaders when available, with automatic CPU fallback. Throughput has not been benchmarked; `extractor.startup_check` times the separator at startup and warns if it cannot meet its realtime budget.
- **Adaptive VAD** — energy + zero-crossing rate detector with configurable hangover, a clamped noise floor, and a force-close segment cap. End-of-file flush routes the final speech segment through the full gate-and-forward path.
- **File source mode** — run the full pipeline offline against a WAV file (`--source path/to/file.wav`) without a microphone
- **Gatekeeper fallback** — disable extraction with `--set extractor.enabled=false` to revert to a lightweight accept/drop pipeline that never imports `asteroid`
- **Robust WebSocket client** — retries connection up to 3 times; silence padding maintains stream continuity between speech segments
- **Structured session logging** — every pipeline event, _including dropped work_, is written to `logs/<session>.jsonl` for latency analysis and debugging

---

## Project Structure

```
PureSignal/
├── src/                          # Pipeline source code
│   ├── config.py                 # Flat-constant shim over config/base.yaml; rebindable
│   ├── main.py                   # Thin entry point: parse args → rebind config → run
│   ├── enroll.py                 # Standalone enrollment script
│   ├── pipeline/
│   │   ├── session.py            # User selection, validation, startup, shutdown
│   │   ├── stages.py             # SegmentPipeline: encode → track → gate → forward
│   │   └── loops.py              # Extractor-mode and gatekeeper-mode consumer loops
│   ├── audio/
│   │   ├── capture.py            # Mic input → frame/window queues
│   │   ├── file_capture.py       # WAV file source; same queue interface as capture.py
│   │   ├── windowed_source.py    # Shared frame routing + drop-safe sequence allocation
│   │   ├── extractor.py          # Conv-TasNet separation (model forward only)
│   │   ├── source_select.py      # Picks the target speaker's separated source
│   │   ├── window_buffer.py      # SlidingWindowBuffer + ResequencingBuffer
│   │   ├── features.py           # L2 peak normalization
│   │   ├── wav_io.py             # Shared WAV loader (pipeline + eval scripts)
│   │   ├── resampler.py          # 16 kHz float32 → 48 kHz int16 PCM conversion
│   │   └── vad.py                # Frame-level VAD with flush() for EOF
│   ├── speaker/
│   │   ├── encoder.py            # ResNet34-LM speaker embedding extractor
│   │   ├── enrollment.py         # Profile loader, cosine matcher, quality gate, metadata writer
│   │   ├── policy.py             # Pass/drop gate (ENROLLED / ALL / DYNAMIC)
│   │   └── tracker.py            # Online speaker tracking with EMA centroids
│   ├── llm/
│   │   └── ultravox_client.py    # Ultravox WebSocket send/receive client
│   └── utils/
│       └── logger.py             # Dual-output logger: console + logs/<session>.jsonl
├── config/
│   ├── base.yaml                 # All tunable settings — single source of truth
│   ├── container.yaml            # CPU / file-mode overrides used by the Docker image
│   └── loader.py                 # load_config() → Config; supports --set CLI overrides
├── eval/
│   ├── README.md                 # Corpus layout, how to rebuild it, how to run each script
│   ├── metrics.py                # Pure FAR/FRR/EER/DET functions
│   ├── run_speaker_eval.py       # FAR/FRR/EER sweep over test + impostor WAVs
│   ├── tracker_eval.py           # Tracker purity from a sequenced multi-speaker session
│   ├── latency_report.py         # Per-stage percentiles + drop counts from session logs
│   ├── vad_eval.py               # VAD precision/recall/F1 parameter sweep
│   ├── vad_labels.py             # Ground-truth label loading (.lab / TextGrid)
│   ├── vad_metrics.py            # Segment-level VAD scoring
│   └── enrollment_quality_eval.py# EER vs. number of enrollment samples curve
├── tests/
│   ├── conftest.py
│   ├── test_audio_pipeline.py       # VAD, normalization, resampler, flush
│   ├── test_speaker_pipeline.py     # Encoder, enrollment, tracker, policy
│   ├── test_window_buffer.py        # Sliding window conservation + resequencing
│   ├── test_pipeline_integration.py # Both loops end to end with stubbed models
│   └── test_config_override.py      # --set / --config propagation
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
[capture / file_capture] — 20ms float32 frames
       │
       ▼
[windowed_source] — SlidingWindowBuffer assembles overlapping 1s windows
                    (seq, window, new_samples) → window_queue
       │
       ▼
[extractor.separate] — parallel Conv-TasNet workers → [n_src, T]
[window_buffer]      — ResequencingBuffer restores order
[extractor.SourceSelector] — picks the target's source, in order, on one thread
       │
       ▼
new-sample tail only — the overlap is model context, never re-processed
       │
       ▼
[vad.py] — energy + ZCR + hangover → complete speech segments
           flush() on EOF routes the final segment through the full path
       │
       ▼
[stages.SegmentPipeline]
       │
       ├──▶ [features.py]  — L2 peak normalization (encoder input only)
       ├──▶ [encoder.py]   — ResNet34-LM → 256-dim embedding (worker thread)
       ├──▶ [tracker.py]   — assign / register speaker ID; EMA centroid updates
       ├──▶ [enrollment.py]— cosine match against profiles/*.npy
       │
       ▼
[policy.py] — ENROLLED: pass if matched user
              ALL:      pass everyone
              DYNAMIC:  pass if speaker ID == DYNAMIC_TARGET
       │
       ▼
[ultravox_client.py] — unnormalized segment → 48kHz int16 → WebSocket → Ultravox
       │
       ▼
Speakers (AI response playback, on its own thread)
       │
[logger.py] — logs/<session>.jsonl (always) + console (if debug=true)
```

1. **Enroll** — `enroll.py` records N voice samples, quality-gates each embedding (norm + entropy), averages them, and saves to `profiles/<username>.npy` + `profiles/<username>_meta.json`.
2. **Select users** — `main.py` prompts you to choose enrolled users (single or multi-user, up to 10) at startup.
3. **Capture** — the source streams 20ms frames into a bounded queue. In extractor mode they are assembled into overlapping 1 s windows; a window's sequence number is allocated only once it is safely enqueued, so a dropped window cannot stall the resequencer.
4. **Separate** — Conv-TasNet splits each window into its sources in parallel worker threads. A `ResequencingBuffer` restores order, then the target's source is chosen on the consumer thread using permutation continuity across the window overlap, re-anchored periodically by speaker embedding. Only the newest `hop_s` of each window continues downstream; the overlap exists as model context and is never processed twice.
5. **VAD** — cleaned frames accumulate into speech segments using energy and zero-crossing thresholds with a hangover window. The adaptive noise floor is clamped so digital silence cannot collapse the speech threshold. At EOF, `vad.flush()` recovers a segment that never saw trailing silence and sends it through the same gate-and-forward path as any other.
6. **Embed** — each segment is peak-normalized and passed through `pyannote/wespeaker-voxceleb-resnet34-LM` on a worker thread. Normalization is for the encoder only; Ultravox receives the original audio so relative loudness between utterances survives.
7. **Identify** — the embedding is compared against all enrolled profiles in a single cosine sweep whose score feeds both the log and the gate. Matched = username label; unmatched = tracker ID (`S1`, `S2`, …).
8. **Gate** — segments from non-enrolled (or non-target) speakers are dropped, and the decision is logged with the similarity that produced it.
9. **Stream** — approved segments are resampled to 48 kHz int16 PCM and sent over a WebSocket. Silence frames keep the stream alive between segments; playback of the AI response happens on a separate thread so it cannot stall the uplink.

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

> `asteroid` is needed only for extractor mode. Gatekeeper mode never imports it,
> so you can skip the install and run with `--set extractor.enabled=false`.
> Note that `asteroid` pins an older `torch` range than the one in
> `requirements.txt` and may downgrade it; use a separate environment if that
> matters to you.

### 4. Configure API keys

Create `.env` at the repo root:

```env
ULTRAVOX_API_KEY=your_ultravox_api_key_here
HF_TOKEN=your_huggingface_token_here
```

> Accept the model terms at [hf.co/pyannote/wespeaker-voxceleb-resnet34-LM](https://hf.co/pyannote/wespeaker-voxceleb-resnet34-LM) before generating your HF token.

`main.py` creates a new Ultravox call each run using your API key. To reuse a
specific call instead, set `ULTRAVOX_JOIN_URL` as an environment variable or set
`ultravox.join_url` in `config/base.yaml`; the environment variable wins. When a
join URL is present, no call is created and the startup banner says so.

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

On startup you will be prompted to choose a mode — **[1] Single-user** or **[2] Multi-user** (up to 10) — and enter enrolled usernames. The pipeline then initialises, loads the models, connects to Ultravox, and begins listening.

Press `Ctrl+C` to stop cleanly. Shutdown runs exactly once, in order: mic → WebSocket → VAD state → tracker state → capture state → log file.

### Useful flags

```bash
# Run against a WAV file instead of the mic (no hardware required)
python3 src/main.py --source path/to/file.wav --no-ultravox

# Skip Ultravox entirely (offline / evaluation mode)
python3 src/main.py --no-ultravox

# Disable extraction and use the lightweight gatekeeper mode
python3 src/main.py --set extractor.enabled=false

# Override any config value at runtime
python3 src/main.py --set vad.hangover_ms=300 --set debug=false

# Pass everyone through — the baseline arm for evaluation
python3 src/main.py --set policy.mode=ALL

# Point to a different config file
python3 src/main.py --config config/container.yaml
```

`--set` and `--config` are applied before any pipeline module is imported. If a
future change imports a pipeline module too early, `config.rebind()` raises
rather than letting the override silently do nothing.

---

## Testing

Tests run without any hardware (no microphone, GPU, API keys, or network):

```bash
pytest tests/ -v
```

**85 tests**, distributed as:

| File                           | Tests | Covers                                                                                                                                                                                                                                     |
| ------------------------------ | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `test_audio_pipeline.py`       | 12    | VAD segmentation and flush, peak normalization, 16→48 kHz resampling and clipping                                                                                                                                                          |
| `test_speaker_pipeline.py`     | 20    | Encoder token check, enrollment load/match/quality/metadata, tracker assignment and EMA, policy modes                                                                                                                                      |
| `test_window_buffer.py`        | 19    | Sliding-window sample conservation and overlap, drop-safe sequence allocation, resequencing order, gaps, `force_advance`, lock release under concurrency                                                                                   |
| `test_pipeline_integration.py` | 21    | Both consumer loops end to end with stubbed encoder and separator: segment delivery, final-segment flush, slow-encoder EOF, no audio duplication, gating decisions, logged drops, separator failure recovery, raw-vs-normalized forwarding |
| `test_config_override.py`      | 13    | `--set` parsing and coercion, rebinding to flat constants, the import-ordering guard                                                                                                                                                       |

Not covered: `llm/ultravox_client.py` (no fake-WebSocket harness yet),
`audio/extractor.py` model paths (the separator itself is stubbed everywhere;
only passthrough and error paths are exercised), and `enroll.py`.

Lint:

```bash
ruff check src/ eval/ tests/
```

---

## Offline Evaluation

The `eval/` directory contains scripts for measuring pipeline quality without a
live mic or Ultravox connection.

**No results are committed yet, because no corpus exists in the repository.**
[`eval/README.md`](eval/README.md) specifies the corpus layout, the minimum
viable size, and the order to run things in. Until that is built, the thresholds
in `config/base.yaml` remain assertions rather than measurements.

| Script                            | What it measures                                                                                                                                          |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `eval/run_speaker_eval.py`        | FAR, FRR, EER over labelled test + impostor WAV directories; outputs `scores.csv`, `metrics.json`, `det_curve.png`                                        |
| `eval/tracker_eval.py`            | Tracker purity from a sequenced multi-speaker session; outputs `tracker_results.json`, `timeline.png`                                                     |
| `eval/latency_report.py`          | Per-stage percentiles **and drop counts** from a session JSONL log; outputs a histogram PNG with `--plot`                                                 |
| `eval/vad_eval.py`                | Precision/recall/F1 sweep over `energy_multiplier`, `hangover_ms`, `zcr_threshold`, and `noise_floor_min_ratio`; outputs `param_sweep.csv`, `heatmap.png` |
| `eval/enrollment_quality_eval.py` | EER vs. number of enrollment samples; outputs `eer_vs_samples.png`                                                                                        |

Example — generate a latency report from a recorded session:

```bash
python3 eval/latency_report.py --log logs/<session>.jsonl --plot
```

The report separates `speech_onset_to_send` (what the user actually waits) from
`vad_close_to_send`, and lists dropped segments alongside the percentiles, since
a table covering only delivered work reads optimistic.

---

## Verifying the Setup

| Check                    | Command / Action                                                                                                   |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| Dependencies installed   | `pip show pyannote.audio torch sounddevice` (add `asteroid` for extractor mode)                                    |
| HF token valid           | `python3 -c "from huggingface_hub import whoami; print(whoami())"`                                                 |
| Profiles exist           | `ls profiles/`                                                                                                     |
| Encoder loads            | Run `src/enroll.py` — watch for `[encoder] loading …` without errors                                               |
| Extractor loads          | Run `src/main.py` — watch for `[extractor] ready — conv_tasnet on mps`                                             |
| Extractor keeps up       | Watch the `[extractor] realtime check` line at startup; a load factor above 1.0x means windows will be dropped     |
| Overrides work           | `python3 src/main.py --set extractor.enabled=false --no-ultravox` — the banner must read `Pipeline : gatekeeper`   |
| VAD fires                | Run with `debug: true` and speak — watch for `[vad] segment_closed`                                                |
| Policy passing           | Check `logs/<session>.jsonl` for `"event": "decision", "data": {"decision": "PASS"}` when the enrolled user speaks |
| Nothing is being dropped | Check the log for `segment_dropped` / `window_dropped` events                                                      |
| Ultravox connected       | Watch for `[ultravox] connected` in the console after startup                                                      |

---

## Configuration Reference

All settings live in [`config/base.yaml`](config/base.yaml). Override any value at runtime with `--set key.subkey=value` or by editing the YAML directly. The `src/config.py` shim re-exports every constant for backward compatibility with existing `import config` calls.

| Key                             | Default                                      | Description                                                                                                                     |
| ------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `audio.sample_rate`             | `16000`                                      | Internal pipeline sample rate (Hz)                                                                                              |
| `audio.frame_ms`                | `20`                                         | VAD frame size (ms)                                                                                                             |
| `vad.energy_multiplier`         | `3.0`                                        | Speech energy must exceed `noise_floor × this`                                                                                  |
| `vad.zcr_threshold`             | `0.3`                                        | Zero-crossing rate ceiling for speech frames. Not yet derived from data; likely too low for unvoiced fricatives — see `EVAL-04` |
| `vad.hangover_ms`               | `400`                                        | Keep speech flag active N ms after energy drops                                                                                 |
| `vad.noise_floor_init`          | `0.01`                                       | Initial noise floor estimate                                                                                                    |
| `vad.noise_floor_min_ratio`     | `0.1`                                        | Noise floor is clamped to `init × this`, so digital silence cannot collapse the speech threshold                                |
| `vad.max_segment_s`             | `30`                                         | Segment is force-closed and a new one started at this length, rather than silently truncated                                    |
| `extractor.enabled`             | `true`                                       | Set `false` to skip extraction and use gatekeeper mode                                                                          |
| `extractor.window_s`            | `1.0`                                        | Sliding window length fed to the separator (s)                                                                                  |
| `extractor.hop_s`               | `0.25`                                       | Window advance per step (s) — the overlap is model context only                                                                 |
| `extractor.model`               | `conv_tasnet`                                | Only supported value; `speakerbeam` was removed                                                                                 |
| `extractor.checkpoint`          | `JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k` | Must be natively 16 kHz; asserted at load                                                                                       |
| `extractor.device`              | `mps`                                        | Inference device (`mps` / `cpu` / `cuda`)                                                                                       |
| `extractor.max_workers`         | `2`                                          | Parallel separation threads                                                                                                     |
| `extractor.reanchor_every`      | `8`                                          | Re-identify the target source by embedding every N windows                                                                      |
| `extractor.log_every`           | `20`                                         | Sample queue depth and pending counts every N windows                                                                           |
| `extractor.startup_check`       | `true`                                       | Time the separator at startup and warn if it misses its realtime budget                                                         |
| `encoder.model`                 | `pyannote/wespeaker-voxceleb-resnet34-LM`    | HuggingFace model ID                                                                                                            |
| `encoder.device`                | `mps`                                        | Inference device; falls back to CPU automatically                                                                               |
| `encoder.min_segment_s`         | `0.5`                                        | Minimum segment length accepted by the encoder (s)                                                                              |
| `encoder.max_pending_segments`  | `3`                                          | Segments buffered ahead of the encoder before drops start (drops are logged)                                                    |
| `tracker.similarity_threshold`  | `0.65`                                       | Cosine similarity floor for tracker assignment                                                                                  |
| `tracker.ema_alpha`             | `0.25`                                       | Centroid update rate (higher = faster adaptation)                                                                               |
| `tracker.max_speakers`          | `4`                                          | Maximum simultaneous tracked speakers per session                                                                               |
| `enrollment.threshold`          | `0.65`                                       | Cosine similarity floor for enrolled user match. Not yet derived — see `EVAL-02`                                                |
| `enrollment.duration_s`         | `12`                                         | Recording length per sample during enrollment (s)                                                                               |
| `enrollment.num_samples`        | `3`                                          | Default number of samples to average (`--samples N` overrides)                                                                  |
| `policy.mode`                   | `ENROLLED`                                   | `ENROLLED` passes matched users; `ALL` passes everyone; `DYNAMIC` passes a fixed tracker label. Validated at startup            |
| `policy.dynamic_target`         | `S1`                                         | Tracker label passed in `DYNAMIC` mode                                                                                          |
| `ultravox.in_rate`              | `48000`                                      | WebSocket input sample rate (Hz)                                                                                                |
| `ultravox.out_rate`             | `48000`                                      | Playback sample rate (Hz)                                                                                                       |
| `ultravox.chunk_ms`             | `20`                                         | PCM chunk size sent per WebSocket frame (ms)                                                                                    |
| `ultravox.join_url`             | `""`                                         | Rejoin an existing call; `ULTRAVOX_JOIN_URL` env var takes precedence                                                           |
| `ultravox.system_prompt`        | `"You are a helpful assistant."`             | Sent when creating a new call                                                                                                   |
| `internal.norm_floor`           | `1e-6`                                       | Minimum L2 norm before treating a vector as zero                                                                                |
| `internal.noise_floor_ema_fast` | `0.05`                                       | Fast EMA weight for noise floor adaptation                                                                                      |
| `internal.noise_floor_ema_slow` | `0.95`                                       | Slow EMA weight for noise floor adaptation                                                                                      |
| `random_seed`                   | `42`                                         | Seed for NumPy and PyTorch (reproducibility)                                                                                    |
| `debug`                         | `true`                                       | Print per-event console logs alongside JSONL output                                                                             |

---

## License

This project is licensed under the [MIT License](LICENSE).
