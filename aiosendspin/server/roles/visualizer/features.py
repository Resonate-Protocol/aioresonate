"""Feature extraction for draft visualizer role."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from aiosendspin.models.visualizer import StreamStartVisualizer


@dataclass(frozen=True)
class ExtractedFrame:
    """Computed visualizer features for one audio chunk."""

    timestamp_us: int
    loudness: int | None = None
    f_peak: int | None = None
    spectrum: np.ndarray | None = None


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


class VisualizerFeatureExtractor:
    """Compute visualizer features from PCM chunks."""

    def __init__(
        self,
        *,
        sample_rate: int,
        channels: int,
        config: StreamStartVisualizer,
    ) -> None:
        """Create a feature extractor for negotiated stream config."""
        self._sample_rate = sample_rate
        self._channels = channels
        self._config = config
        self._last_spectrum_ts_us: int | None = None
        self._last_spectrum: np.ndarray | None = None
        self._a_weight_cache: np.ndarray | None = None

    def reset(self) -> None:
        """Reset extractor state at stream boundaries."""
        self._last_spectrum_ts_us = None
        self._last_spectrum = None

    def process_chunk(self, pcm: bytes, timestamp_us: int) -> ExtractedFrame:
        """Compute a single frame from a PCM chunk."""
        mono = self._decode_pcm_to_mono_float32(pcm)

        needs_fft = any(t in self._config.types for t in ("loudness", "f_peak", "spectrum"))

        loudness: int | None = None
        f_peak: int | None = None
        spectrum: np.ndarray | None = None

        if needs_fft:
            freqs, magnitude = self._fft_magnitude(mono)
            compensated = self._apply_psychoacoustic_compensation(freqs, magnitude)

            if "loudness" in self._config.types:
                # A-weighted RMS via Parseval's theorem for one-sided rfft:
                # RMS ≈ sqrt(2 * sum(|X|^2)) / N
                if compensated.size == 0:
                    loudness = 0
                else:
                    n = mono.size
                    weighted_power = float(np.sum(np.square(compensated, dtype=np.float64)))
                    rms = np.sqrt(2.0 * weighted_power) / max(n, 1)
                    # A full-scale sine through Hanning has RMS ≈ sqrt(3/16) ≈ 0.43
                    # (amplitude 1.0, mean(sin^2)=0.5, mean(hann^2)=3/8).
                    # A-weight gain ~1.0 at 1-4 kHz.
                    ref = np.sqrt(3.0 / 16.0)
                    normalized = float(np.clip(rms / ref, 0.0, 1.0))
                    loudness = int(normalized * 65535.0)

            if "f_peak" in self._config.types:
                if compensated.size == 0:
                    f_peak = 0
                else:
                    idx = int(np.argmax(compensated))
                    peak_hz = int(freqs[idx]) if idx < freqs.size else 0
                    f_peak = int(np.clip(peak_hz, 0, 65535))

            if "spectrum" in self._config.types:
                spectrum = self._maybe_compute_spectrum(freqs, compensated, timestamp_us)

        return ExtractedFrame(
            timestamp_us=timestamp_us,
            loudness=loudness,
            f_peak=f_peak,
            spectrum=spectrum,
        )

    def _decode_pcm_to_mono_float32(self, pcm: bytes) -> np.ndarray:
        """Decode little-endian signed PCM16 to mono float32 in [-1, 1]."""
        raw = np.frombuffer(pcm, dtype="<i2")
        if raw.size == 0:
            return np.zeros(1, dtype=np.float32)

        if self._channels > 1:
            frame_count = raw.size // self._channels
            if frame_count <= 0:
                return np.zeros(1, dtype=np.float32)
            reshaped = raw[: frame_count * self._channels].reshape(frame_count, self._channels)
            mono = np.mean(reshaped.astype(np.float32), axis=1, dtype=np.float32)
        else:
            mono = raw.astype(np.float32)

        return np.clip(mono / 32768.0, -1.0, 1.0)  # type: ignore[return-value]

    def _fft_magnitude(self, mono: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return FFT frequency bins and normalized magnitudes."""
        if mono.size <= 1:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

        windowed = mono * np.hanning(mono.size).astype(np.float32)
        spectrum = np.fft.rfft(windowed)
        magnitude = np.abs(spectrum).astype(np.float32)
        freqs = np.fft.rfftfreq(mono.size, d=1.0 / float(self._sample_rate)).astype(np.float32)

        # Drop DC for peak/spectrum display.
        if magnitude.size > 0:
            magnitude[0] = 0.0
        return freqs, magnitude

    def _apply_psychoacoustic_compensation(
        self, freqs: np.ndarray, magnitude: np.ndarray
    ) -> np.ndarray:
        """Apply A-weighting gain so display tracks perceived loudness better."""
        if freqs.size == 0 or magnitude.size == 0:
            return magnitude

        # Cache A-weight gains for the current FFT size.
        if self._a_weight_cache is None or self._a_weight_cache.size != freqs.size:
            f = np.maximum(freqs.astype(np.float64), 1.0)
            f2 = f * f
            ra_num = (12194.0**2) * (f2**2)
            ra_den = (f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194.0**2)
            ra = np.clip(ra_num / np.maximum(ra_den, 1e-20), 1e-12, None)
            a_db = 2.0 + (20.0 * np.log10(ra))
            a_db = np.clip(a_db, -50.0, 6.0)
            self._a_weight_cache = np.power(10.0, a_db / 20.0).astype(np.float32)

        result: np.ndarray = magnitude * self._a_weight_cache
        return result

    def _maybe_compute_spectrum(
        self, freqs: np.ndarray, magnitude: np.ndarray, timestamp_us: int
    ) -> np.ndarray:
        spectrum_cfg = self._config.spectrum
        if spectrum_cfg is None:
            raise ValueError("spectrum in config.types but config.spectrum is None")

        if self._last_spectrum_ts_us is not None:
            min_delta_us = int(1_000_000 / spectrum_cfg.rate_max)
            if (
                timestamp_us - self._last_spectrum_ts_us < min_delta_us
                and self._last_spectrum is not None
            ):
                return self._last_spectrum

        computed = self._compute_binned_spectrum(
            freqs=freqs,
            magnitude=magnitude,
            n_bins=spectrum_cfg.n_disp_bins,
            f_min=spectrum_cfg.f_min,
            f_max=spectrum_cfg.f_max,
            scale=spectrum_cfg.scale,
        )
        self._last_spectrum_ts_us = timestamp_us
        self._last_spectrum = computed
        return computed

    def _compute_binned_spectrum(
        self,
        *,
        freqs: np.ndarray,
        magnitude: np.ndarray,
        n_bins: int,
        f_min: int,
        f_max: int,
        scale: Literal["lin", "log", "mel"],
    ) -> np.ndarray:
        if freqs.size == 0 or magnitude.size == 0:
            return np.zeros(n_bins, dtype=np.uint16)

        lo = max(0, f_min)
        hi = min(int(self._sample_rate / 2), f_max)
        if hi <= lo:
            return np.zeros(n_bins, dtype=np.uint16)

        edges = self._frequency_bin_edges(n_bins=n_bins, f_min=lo, f_max=hi, scale=scale)

        # Vectorized binning via np.digitize.
        bin_indices = np.digitize(freqs, edges) - 1
        valid = (bin_indices >= 0) & (bin_indices < n_bins)
        idx = bin_indices[valid]
        mag = magnitude[valid]

        sq = mag.astype(np.float64) ** 2
        sums = np.bincount(idx, weights=sq, minlength=n_bins).astype(np.float32)
        binned = np.sqrt(sums)

        # dB scale: map [-60 dB, 0 dB] relative to full-scale sine → [0, 65535].
        # Full-scale sine has rfft peak N/2, reduced to N/4 by Hanning window.
        n = freqs.size * 2 - 1  # original time-domain sample count
        ref = max(float(n) / 4.0, 1.0)
        ratio = np.maximum(binned / ref, 1e-10)
        db = 20.0 * np.log10(ratio)
        db_floor = -60.0
        normalized = np.clip((db - db_floor) / -db_floor, 0.0, 1.0)
        return (normalized * 65535.0).astype(np.uint16)

    def _frequency_bin_edges(
        self, *, n_bins: int, f_min: int, f_max: int, scale: Literal["lin", "log", "mel"]
    ) -> np.ndarray:
        if scale == "lin":
            return np.linspace(float(f_min), float(f_max), n_bins + 1, dtype=np.float32)
        if scale == "log":
            safe_min = max(f_min, 1)
            return np.logspace(
                np.log10(float(safe_min)),
                np.log10(float(f_max)),
                n_bins + 1,
                dtype=np.float32,
            )

        mel_edges = np.linspace(
            _hz_to_mel(np.array([float(f_min)], dtype=np.float32))[0],
            _hz_to_mel(np.array([float(f_max)], dtype=np.float32))[0],
            n_bins + 1,
            dtype=np.float32,
        )
        return _mel_to_hz(mel_edges).astype(np.float32)
