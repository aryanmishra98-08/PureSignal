# =============================================================================
# pipeline/loops.py — The two consumer loops
#
# EXTRACTOR MODE (extractor.enabled = true):
#   window_queue → parallel separator workers → ResequencingBuffer
#     → SourceSelector → new-samples tail → VAD → SegmentPipeline
#
# GATEKEEPER MODE (extractor.enabled = false):
#   frame_queue → VAD → SegmentPipeline
#
# Both loops end on a None sentinel, flush the VAD, and block-drain the encoder,
# so the final utterance of a session is delivered rather than merely counted.
# =============================================================================
from __future__ import annotations

import concurrent.futures
import time

import numpy as np

import config
from audio import vad
from pipeline.stages import SegmentPipeline
from utils.logger import get_logger

_log = get_logger()


def _feed_vad(audio, segments: SegmentPipeline) -> None:
    """
    Slice `audio` into VAD frames and push closed segments downstream.

    `audio` must contain only new samples — never a whole overlapping window,
    or every sample is processed window_s / hop_s times over.
    """
    frame_size = config.FRAME_SAMPLES
    offset = 0
    while offset < len(audio):
        frame = audio[offset: offset + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))
        offset += frame_size
        segment = vad.process_frame(frame)
        if segment is not None:
            # Timestamp after the VAD returns, so this is a segment-close time
            # and not the arrival time of the frame that happened to close it.
            segments.submit(segment, close_ts=time.monotonic())


def _finish(segments: SegmentPipeline) -> None:
    """Flush the VAD and deliver everything still in flight at end-of-stream."""
    leftover = vad.flush()
    if leftover is not None:
        # Route the flushed tail through the full path. Logging it and dropping
        # it, as the previous code did, is worse than losing it silently: the
        # count says it was handled.
        segments.submit(leftover, close_ts=time.monotonic())
    segments.drain_all()


# ---------------------------------------------------------------------------
# EXTRACTOR MODE
# ---------------------------------------------------------------------------

def process_loop_extractor(target_emb, no_ultravox: bool) -> None:
    """
    Extractor-mode pipeline loop.

    Reads (seq_no, window, new_samples) tuples from capture.window_queue and
    submits each window to a separator thread pool.  Results come back through
    a ResequencingBuffer so ordering is restored, the target source is selected
    on this thread (in order, with the only encoder reference), and only the
    new-sample tail of each window is fed to the VAD.
    """
    if config.DEBUG:
        print("[main] extractor pipeline started")
    from audio import capture as _capture
    from audio.extractor import separate
    from audio.source_select import SourceSelector
    from audio.window_buffer import ResequencingBuffer

    if target_emb is None:
        print("[main] WARNING: no target embedding — using blind separation")

    reseq = ResequencingBuffer()
    selector = SourceSelector()
    stall_limit = 2 * config.EXTRACTOR_MAX_WORKERS + 4
    processed = 0

    def _run_extraction(seq_no, window, new_samples):
        """Worker body — must fill its slot even on failure."""
        try:
            sources = separate(window)
        except Exception as e:  # noqa: BLE001 — a hole here stalls the pipeline
            _log("extractor", "extract_failed", seq_no=seq_no, error=repr(e))
            sources = window[np.newaxis, :]  # passthrough, never leave a hole
        # new_samples travels with the result so the consumer needs no parallel
        # bookkeeping that a force_advance would then have to compensate for.
        reseq.put(seq_no, (sources, new_samples))

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=config.EXTRACTOR_MAX_WORKERS, thread_name_prefix="extractor"
    ) as ext_pool, concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="encoder"
    ) as enc_pool:

        segments = SegmentPipeline(enc_pool, no_ultravox)

        def _drain_into_vad() -> None:
            for sources, new_samples in reseq.drain():
                cleaned = selector.select(sources, target_emb, new_samples)
                tail = cleaned[-new_samples:] if 0 < new_samples < len(
                    cleaned) else cleaned
                _feed_vad(tail, segments)
            segments.drain_ready()

        while True:
            item = _capture.window_queue.get()

            if item is None:  # EOF sentinel
                # Wait out the workers so no separated window is abandoned.
                ext_pool.shutdown(wait=True)
                _drain_into_vad()
                _finish(segments)
                break

            seq_no, window, new_samples = item
            ext_pool.submit(_run_extraction, seq_no, window, new_samples)
            _drain_into_vad()

            processed += 1
            if config.EXTRACTOR_LOG_EVERY and processed % config.EXTRACTOR_LOG_EVERY == 0:
                _log("extractor", "backpressure",
                     window_queue_depth=_capture.window_queue.qsize(),
                     reseq_pending=reseq.pending_count(),
                     encoder_pending=segments.pending)

            # Staleness guard: an unforeseen path that leaves a sequence number
            # unfilled must degrade, not deadlock.
            if reseq.pending_count() > stall_limit:
                skipped = reseq.force_advance()
                if skipped:
                    _log("extractor", "sequence_skipped", count=skipped,
                         pending=reseq.pending_count())
                    _drain_into_vad.next_out += skipped
                    _drain_into_vad()


# ---------------------------------------------------------------------------
# GATEKEEPER MODE
# ---------------------------------------------------------------------------

def process_loop_gatekeeper(no_ultravox: bool) -> None:
    """
    Gatekeeper-mode pipeline loop (original v1 behaviour, minus the lag).

    Reads raw 20ms frames from capture.frame_queue and passes each through the
    VAD.  Closed segments go to SegmentPipeline immediately rather than waiting
    for the next segment to arrive.
    """
    if config.DEBUG:
        print("[main] gatekeeper pipeline started")
    from audio import capture as _capture

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="encoder"
    ) as pool:
        segments = SegmentPipeline(pool, no_ultravox)

        while True:
            frame = _capture.frame_queue.get()
            if frame is None:  # EOF sentinel
                _finish(segments)
                break

            segment = vad.process_frame(frame)
            if segment is not None:
                segments.submit(segment, close_ts=time.monotonic())
            segments.drain_ready()
