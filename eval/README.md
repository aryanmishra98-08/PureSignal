# Evaluation

**Status: no numbers yet.** The scripts here work, but none has been run against
a real corpus, because no corpus exists in the repository. Nothing is committed
under `eval/results/`, and `enrollment.threshold: 0.65` and
`tracker.similarity_threshold: 0.65` are still assumed values rather than
measured ones.

Building the corpus is the blocker for every other measurement.
Audio is gitignored because it is generally not redistributable; this file is
how you rebuild it.

## Corpus layout

```
eval/test_audio/<username>/<utterance_N>.wav   # genuine trials, enrolled speakers
eval/imposters/<speaker_id>/<utterance_N>.wav  # impostor trials
eval/vad_audio/<name>.wav                      # VAD sweep audio
eval/vad_labels/<name>.lab                     # "start_s end_s speech" per line
                                               # (or <name>.TextGrid)
eval/tracker_audio/<session>.wav               # multi-speaker sequenced session
eval/tracker_audio/<session>.rttm              # ground-truth speaker turns
```

All audio: 16 kHz mono WAV. The loaders resample if needed, but recording at the
pipeline rate avoids a resampling confound.

## Minimum viable corpus

| Part              | Target                                      | Notes                                                                     |
| ----------------- | ------------------------------------------- | ------------------------------------------------------------------------- |
| Enrolled speakers | 3 speakers × 20 utterances                  | 3–8 s each, at least two acoustic conditions per speaker                  |
| Impostors         | 5 speakers × 20 utterances                  | VoxCeleb test split or LibriSpeech `test-clean` are acceptable            |
| Tracker session   | 1 session with turn labels                  | for `tracker_eval.py`                                                     |
| VAD session       | 1 session with hand-labelled speech/silence | for `vad_eval.py`; TextGrid supported via the optional `textgrid` package |

Caveat worth writing down when you publish numbers: LibriSpeech and VoxCeleb are
in the encoder's training distribution, so impostor trials drawn from them
flatter the system. Note it rather than hiding it.

## Running the evaluations

Order matters — later steps consume earlier results.

```bash
# 1. Derive the enrollment threshold (do this first)
python3 eval/run_speaker_eval.py \
  --enrolled-dir profiles \
  --test-dir eval/test_audio \
  --imposters-dir eval/imposters \
  --threshold-sweep 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 \
  --out eval/results/speaker_eval

# 2. VAD parameter sweep. Four parameters are sweepable; any you omit is held
#    at its config/base.yaml value.
python3 eval/vad_eval.py \
  --audio-dir eval/vad_audio --labels-dir eval/vad_labels \
  --param-sweep energy_multiplier=2.0,3.0,4.0 hangover_ms=200,400,600 \
                zcr_threshold=0.3,0.4,0.5 noise_floor_min_ratio=0.05,0.1,0.2 \
  --out eval/results/vad

# 3. Latency and enrollment quality
python3 eval/latency_report.py --log logs/<session>.jsonl --plot
python3 eval/enrollment_quality_eval.py --out eval/results/enrollment

# 4. The three-arm extractor comparison (see below)
```

## The comparison that matters

The premise of the extractor stage is that target-speaker separation beats
segment-level gating. That is unmeasured. All three arms are reachable from the
command line without editing config:

| Arm | Command                                                    | Isolates             |
| --- | ---------------------------------------------------------- | -------------------- |
| A   | `--set policy.mode=ALL --set extractor.enabled=false`      | passthrough baseline |
| B   | `--set policy.mode=ENROLLED --set extractor.enabled=false` | gatekeeper           |
| C   | `--set policy.mode=ENROLLED --set extractor.enabled=true`  | extractor            |

Input: overlapped two-speaker mixtures, one speaker enrolled, across a range of
target-to-interferer ratios.

Report per arm: interferer leakage (fraction of forwarded duration belonging to
the non-target), target retention, `speech_onset_to_send` P50/P95, and for arm C
the realtime factor from `extractor.realtime_check()`.

The claim under test is that C reduces leakage relative to B without materially
hurting retention or latency. If it does not, the extractor is not earning its
complexity — which is a legitimate finding and should be published as one.

## What to commit

- `eval/results/**/metrics.json`, `*.csv`, `*.png`
- The chosen operating point and its FAR/FRR, in the top-level README
- What `0.65` turned out to be worth. Expect the measured EER threshold to land
  lower — normalized wespeaker ResNet34-LM embeddings usually put it in the
  0.45–0.60 band, which would mean the shipped default is over-strict and has
  been rejecting the enrolled speaker more often than intended.
