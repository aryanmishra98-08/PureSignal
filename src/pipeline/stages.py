# =============================================================================
# pipeline/stages.py — Segment handling shared by both pipeline loops
#
# SegmentPipeline owns everything that happens once the VAD closes a segment:
# encoder submission, tracker assignment, enrollment match, the policy gate, and
# the handoff to Ultravox.  Both loops drive the same instance, which is what
# stops the extractor and gatekeeper paths from drifting apart again.
#
# Ordering guarantees:
#   - Segments are submitted the moment they close, never one segment late.
#   - A bounded FIFO absorbs bursts ahead of the single encoder worker; only
#     when it is full is a segment dropped, and the drop is always logged.
#   - drain_all() blocks, so no in-flight segment is lost at end-of-stream.
# =============================================================================
from __future__ import annotations

import time
from collections import deque

import config
from audio import features
from speaker import encoder, enrollment, policy, tracker
from utils.logger import get_logger

_log = get_logger()


class SegmentPipeline:
    """Drives closed speech segments through encode → gate → forward."""

    def __init__(self, enc_pool, no_ultravox: bool) -> None:
        """
        Args:
            enc_pool:    single-worker ThreadPoolExecutor for encoder.embed.
            no_ultravox: when True, passing segments are logged but not sent.
        """
        self._pool = enc_pool
        self._no_ultravox = no_ultravox
        self._pending: deque[dict] = deque()
        self._max_pending = max(1, int(config.ENCODER_MAX_PENDING))

    # -- submission ---------------------------------------------------------

    def submit(self, segment, close_ts: float | None = None) -> None:
        """
        Accept a freshly closed speech segment.

        Args:
            segment:  float32 ndarray at 16kHz.
            close_ts: monotonic timestamp at which the segment closed.  Taken
                      after the VAD returned it, so it is a true close time
                      rather than a frame-arrival time.
        """
        if segment is None or len(segment) == 0:
            return
        close_ts = time.monotonic() if close_ts is None else close_ts
        dur = len(segment) / config.SAMPLE_RATE
        _log.vad_segment_closed(duration_s=dur, sample_count=len(segment))

        self.drain_ready()

        if len(self._pending) >= self._max_pending:
            _log.segment_dropped(reason="encoder_busy", duration_s=dur,
                                 sample_count=len(segment))
            return

        # Normalization helps the encoder (scale invariance) but must not leak
        # into what Ultravox hears, or every utterance arrives at full scale and
        # relative loudness between segments is destroyed. Carry both.
        normalized = features.normalize(segment)
        enc_start = time.monotonic()
        self._pending.append({
            "future": self._pool.submit(encoder.embed, normalized),
            "raw": segment,
            "normalized": normalized,
            "timing": {
                "segment_start_ts": close_ts - dur,
                "vad_close_ts": close_ts,
                "encoder_start_ts": enc_start,
            },
        })

    # -- draining -----------------------------------------------------------

    def drain_ready(self) -> int:
        """Handle every completed encoder future without blocking."""
        handled = 0
        while self._pending and self._pending[0]["future"].done():
            self._handle(self._pending.popleft())
            handled += 1
        return handled

    def drain_all(self) -> int:
        """
        Handle every in-flight segment, blocking until each completes.

        Called at end-of-stream, where blocking is exactly what is wanted: the
        alternative is discarding the last segment because its encoder pass had
        not finished yet.
        """
        handled = 0
        while self._pending:
            self._handle(self._pending.popleft())
            handled += 1
        return handled

    @property
    def pending(self) -> int:
        return len(self._pending)

    # -- internals ----------------------------------------------------------

    def _handle(self, item: dict) -> None:
        """Score, gate, and forward one completed segment."""
        embedding = item["future"].result()
        enc_done_ts = time.monotonic()
        timing = item["timing"]
        normalized = item["normalized"]
        raw = item["raw"]
        seg_dur = len(normalized) / config.SAMPLE_RATE

        if embedding is None:
            _log.encoder_segment_too_short(segment_duration_s=seg_dur)
            return

        _log.encoder_embed_complete(
            latency_ms=(enc_done_ts - timing["encoder_start_ts"]) * 1000,
            segment_duration_s=seg_dur,
        )

        gallery_before = tracker.get_gallery()
        speaker_id, best_sim = tracker.assign_with_score(embedding)
        _log.tracker_speaker_assigned(
            speaker_id=speaker_id, best_sim=best_sim,
            gallery_size=len(tracker.get_gallery()),
            is_new=speaker_id not in gallery_before,
        )

        # One cosine sweep per segment: the score feeds the log and the name
        # feeds the gate, so policy must not recompute it.
        matched_name, match_sim = enrollment.match_with_score(embedding)
        _log.enrollment_match_result(
            matched_name=matched_name, best_sim=match_sim)

        policy_ts = time.monotonic()
        passes = policy.should_pass(
            speaker_id, embedding, matched_name=matched_name)
        _log.policy_decision(speaker_id=speaker_id, matched_name=matched_name,
                             decision="PASS" if passes else "DROP",
                             mode=config.POLICY_MODE)

        send_ts = time.monotonic()
        if passes and not self._no_ultravox:
            from llm import ultravox_client
            ultravox_client.send_segment(raw)
            _log.ultravox_segment_sent(
                pcm_bytes=len(raw) * 3 * 2,
                queue_depth=ultravox_client.audio_send_queue.qsize(),
            )
        elif passes and config.DEBUG:
            print(
                f"[policy] {matched_name or speaker_id} → PASS (no-ultravox)")

        _log.latency_record({
            "segment_start_ts": timing["segment_start_ts"],
            "vad_close_ts": timing["vad_close_ts"],
            "encoder_start_ts": timing["encoder_start_ts"],
            "encoder_done_ts": enc_done_ts,
            "policy_ts": policy_ts,
            "send_ts": send_ts,
            "segment_duration_s": seg_dur,
            "speech_onset_to_send_ms": (send_ts - timing["segment_start_ts"]) * 1000,
            "delivered": bool(passes and not self._no_ultravox),
        })
