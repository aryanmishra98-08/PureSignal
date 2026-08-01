# =============================================================================
# audio/source_select.py — Target source selection after separation
#
# Conv-TasNet returns its sources in an arbitrary order that can flip from one
# window to the next, so choosing the target speaker's source needs cross-window
# state.  That state is why this is separate from extractor.separate(): the
# separator runs on worker threads, this runs on the consumer thread where
# windows arrive in order and the encoder is safe to touch.
# =============================================================================
from __future__ import annotations

import numpy as np

import config


class SourceSelector:
    """
    Picks the target speaker's source out of the separator's output.

    Conv-TasNet emits sources in an arbitrary order that can flip between
    windows, so the choice needs cross-window state.  Two mechanisms:

      permutation continuity — the overlap region shared with the previous
        window should match the previously chosen source.  Cheap: one dot
        product per candidate, no model involved.

      periodic re-anchoring — every `reanchor_every` windows, re-identify the
        target by speaker embedding so continuity drift cannot compound.

    Call from the consumer thread only — this is deliberately not thread-safe.
    Embedding from the extraction workers instead would mean one encoder pass
    per source per window (8/s of audio at the default hop) and concurrent use
    of a shared pyannote Inference object, which makes no thread-safety
    guarantee.  Re-anchoring on this thread keeps both problems away.
    """

    def __init__(self, embed_fn=None) -> None:
        """
        Args:
            embed_fn: callable(np.ndarray) -> np.ndarray | None.  Defaults to
                      speaker.encoder.embed, injected for testing.
        """
        self._embed_fn = embed_fn
        self._prev_tail: np.ndarray | None = None
        self._counter = 0

    def reset(self) -> None:
        self._prev_tail = None
        self._counter = 0

    def select(self, sources: np.ndarray, target_embedding: np.ndarray | None,
               new_samples: int) -> np.ndarray:
        """
        Return the source belonging to the target speaker.

        Args:
            sources:          [n_src, T] separator output.
            target_embedding: L2-normalized enrollment embedding, or None for
                              blind selection by energy.
            new_samples:      trailing samples of this window that are new;
                              T - new_samples is the overlap with the previous
                              window, which is what continuity compares.
        """
        if sources.shape[0] == 1:
            chosen = sources[0]
        else:
            overlap = max(0, sources.shape[1] - int(new_samples))
            idx = self._choose(sources, target_embedding, overlap)
            chosen = sources[idx]
        self._counter += 1
        overlap = max(0, sources.shape[1] - int(new_samples))
        self._prev_tail = chosen[-overlap:].copy() if overlap > 0 else None
        return chosen

    # -- internals ----------------------------------------------------------

    def _choose(self, sources: np.ndarray, target_embedding, overlap: int) -> int:
        due = self._counter % max(1, config.EXTRACTOR_REANCHOR_EVERY) == 0
        if target_embedding is not None and (due or self._prev_tail is None):
            idx = self._by_embedding(sources, target_embedding)
            if idx is not None:
                return idx
        if self._prev_tail is not None and overlap > 0:
            return self._by_continuity(sources, overlap)
        return int(np.argmax(np.abs(sources).mean(axis=1)))

    def _by_embedding(self, sources: np.ndarray, target_embedding) -> int | None:
        """Identify the target by cosine similarity of speaker embeddings."""
        embed = self._embed_fn
        if embed is None:
            from speaker.encoder import embed as _embed
            embed = _embed
        from audio.features import normalize
        best_idx, best_sim = None, -1.0
        for i in range(sources.shape[0]):
            emb = embed(normalize(sources[i]))
            if emb is None:
                continue
            sim = float(np.dot(emb, target_embedding))
            if sim > best_sim:
                best_sim, best_idx = sim, i
        return best_idx

    def _by_continuity(self, sources: np.ndarray, overlap: int) -> int:
        """Pick the source whose leading overlap best matches the previous pick."""
        ref = self._prev_tail
        n = min(len(ref), overlap, sources.shape[1])
        if n <= 0:
            return int(np.argmax(np.abs(sources).mean(axis=1)))
        ref_seg = ref[-n:]
        ref_norm = float(np.linalg.norm(ref_seg))
        if ref_norm < config.NORM_FLOOR:
            return int(np.argmax(np.abs(sources).mean(axis=1)))
        best_idx, best_score = 0, -np.inf
        for i in range(sources.shape[0]):
            cand = sources[i][:n]
            cand_norm = float(np.linalg.norm(cand))
            if cand_norm < config.NORM_FLOOR:
                continue
            score = float(np.dot(ref_seg, cand) / (ref_norm * cand_norm))
            if score > best_score:
                best_score, best_idx = score, i
        return best_idx
