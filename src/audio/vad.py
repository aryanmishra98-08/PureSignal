"""Frame-level Voice Activity Detection.

Energy + zero-crossing rate against an adaptive noise floor, with a hangover
window so brief pauses do not split an utterance.

Tunables are captured when the detector is constructed, so a caller that
overrides config (the evaluators, --set on the CLI) simply builds a detector
afterwards. Each instance owns its state, so several can run side by side.
"""

from __future__ import annotations

import numpy as np

import config

# Hard cap on one segment. The buffer is allocated once and reused, and 30s is
# far beyond any expected utterance.
_MAX_SEGMENT_S = 30.0


class VoiceActivityDetector:
    """Splits a stream of fixed-size frames into speech segments."""

    def __init__(
        self,
        *,
        sample_rate: int | None = None,
        frame_ms: int | None = None,
        energy_multiplier: float | None = None,
        zcr_threshold: float | None = None,
        hangover_ms: int | None = None,
        noise_floor_init: float | None = None,
        noise_floor_ema_fast: float | None = None,
        noise_floor_ema_slow: float | None = None,
        max_segment_s: float = _MAX_SEGMENT_S,
    ) -> None:
        """Build a detector. Any argument left as None is read from config."""
        self.sample_rate = config.SAMPLE_RATE if sample_rate is None else sample_rate
        self.frame_ms = config.FRAME_MS if frame_ms is None else frame_ms
        self.energy_multiplier = (
            config.ENERGY_MULTIPLIER if energy_multiplier is None else energy_multiplier
        )
        self.zcr_threshold = (
            config.ZCR_THRESHOLD if zcr_threshold is None else zcr_threshold
        )
        self.hangover_ms = config.HANGOVER_MS if hangover_ms is None else hangover_ms
        self.noise_floor_init = (
            config.NOISE_FLOOR_INIT if noise_floor_init is None else noise_floor_init
        )
        self._ema_fast = (
            config.NOISE_FLOOR_EMA_FAST
            if noise_floor_ema_fast is None
            else noise_floor_ema_fast
        )
        self._ema_slow = (
            config.NOISE_FLOOR_EMA_SLOW
            if noise_floor_ema_slow is None
            else noise_floor_ema_slow
        )

        self._hangover_frames = int(self.hangover_ms / self.frame_ms)
        self._max_samples = int(max_segment_s * self.sample_rate)
        self._buffer = np.empty(self._max_samples, dtype=np.float32)

        self._noise_floor = self.noise_floor_init
        self._hangover_left = 0
        self._speech_active = False
        self._fill = 0

    # -- internals -------------------------------------------------------

    @staticmethod
    def _rms(frame: np.ndarray) -> float:
        return float(np.sqrt(np.mean(frame**2)))

    @staticmethod
    def _zcr(frame: np.ndarray) -> float:
        crossings = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
        return float(crossings / len(frame))

    def _update_noise_floor(self, rms: float) -> None:
        """Adapt the noise floor toward the level observed during silence."""
        self._noise_floor = self._ema_slow * self._noise_floor + self._ema_fast * rms

    def _append(self, frame: np.ndarray) -> None:
        """Add a frame to the open segment, discarding overflow past the cap."""
        end = self._fill + len(frame)
        if end <= self._max_samples:
            self._buffer[self._fill : end] = frame
            self._fill = end

    def _close_segment(self) -> np.ndarray:
        """Detach the buffered segment and clear speech state."""
        segment = self._buffer[: self._fill].copy()
        self._speech_active = False
        self._fill = 0
        return segment

    # -- public API ------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> np.ndarray | None:
        """Feed one frame to the detector.

        Args:
            frame: One frame of float32 audio, ``frame_samples`` long.

        Returns:
            The completed speech segment when one closes, otherwise None. A
            segment closes once speech has been seen, energy drops below the
            threshold, and the hangover counter expires.
        """
        rms = self._rms(frame)
        is_speech = (
            rms > self._noise_floor * self.energy_multiplier
            and self._zcr(frame) < self.zcr_threshold
        )

        if is_speech:
            self._speech_active = True
            self._hangover_left = self._hangover_frames
            self._append(frame)
            return None

        if not self._speech_active:
            self._update_noise_floor(rms)  # pure silence
            return None

        if self._hangover_left > 0:
            # Still inside the hangover window — keep the frame.
            self._hangover_left -= 1
            self._append(frame)
            return None

        # Hangover expired — close the segment.
        if self._fill == 0:
            self._speech_active = False
            return None

        segment = self._close_segment()
        self._update_noise_floor(rms)
        return segment

    def flush(self) -> np.ndarray | None:
        """Return any segment still open, for example when the source hits EOF.

        Returns:
            The buffered segment, or None if no speech is currently open.
        """
        if self._speech_active and self._fill > 0:
            return self._close_segment()
        return None

    def reset(self) -> None:
        """Clear all state, keeping the tunables this detector was built with."""
        self._noise_floor = self.noise_floor_init
        self._hangover_left = 0
        self._speech_active = False
        self._fill = 0

    @property
    def frame_samples(self) -> int:
        """Number of samples in one frame at this detector's settings."""
        return int(self.sample_rate * self.frame_ms / 1000)

    @property
    def hangover_frames(self) -> int:
        """Number of silent frames tolerated before a segment closes."""
        return self._hangover_frames
