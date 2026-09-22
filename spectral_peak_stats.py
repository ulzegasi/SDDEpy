"""Frozen, observation-relative spectral peak loss for the 271-year Sun record.

One scalar is passed to SABC: combine the six losses BEFORE the usual prior-CDF
transform. This preserves their chosen relative weights without changing SABC.
No Julia or neural runtime is needed to evaluate or unpickle this adapter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks


@dataclass(frozen=True)
class SpectralPeakConfig:
    """Versioned scoring choices and observation identity, retained in pickles."""

    observed_sha256: str
    gleissberg_target_frequency: float
    version: str = "spectral_peaks_v1"
    n_samples: int = 271
    sample_interval_years: float = 1.0
    window: str = "numpy.hanning; no detrending"
    feature_names: tuple[str, ...] = ("main", "Gleissberg", "15y", "17y", "27y", "39y")
    native_target_bins: tuple[int, ...] = (25, 3, 18, 16, 10, 7)
    weights: tuple[float, ...] = (5.0, 5.0, 1.0, 1.0, 1.0, 1.0)
    tolerances_bins: tuple[float, ...] = (3.0, 1.0, 1.5, 1.5, 1.5, 1.5)
    native_candidate_bins: tuple[int, int] = (2, 27)
    dominant_bins: tuple[int, int] = (14, 33)
    visibility_regions: tuple[tuple[int, int], ...] = ((2, 5), (6, 13), (14, 27))
    relative_prominence: float = 0.1
    regional_prominence: float = 0.02
    padding_factor: int = 32
    gleissberg_period_band: tuple[float, float] = (60.0, 140.0)
    edge_taper_bins: float = 0.2
    missing_penalty: float = 1.0


def _native_candidates(
    spectrum: np.ndarray, config: SpectralPeakConfig,
) -> tuple[np.ndarray, int]:
    peaks, properties = find_peaks(spectrum, prominence=0)
    available = []
    lo_candidate, hi_candidate = config.native_candidate_bins
    for k, prominence in zip(peaks, properties["prominences"]):
        if not lo_candidate <= k <= hi_candidate:
            continue
        lo, hi = next(region for region in config.visibility_regions if region[0] <= k <= region[1])
        if (
            prominence >= config.relative_prominence * spectrum[k]
            and prominence >= config.regional_prominence * np.max(spectrum[lo:hi + 1])
        ):
            available.append(int(k))
    lo, hi = config.dominant_bins
    dominant = int(np.argmax(spectrum[lo:hi + 1]) + lo)
    return np.asarray(available, dtype=int), dominant


def _native_losses(spectrum: np.ndarray, config: SpectralPeakConfig) -> np.ndarray:
    """Match once per candidate, with the dominant main cycle assigned first."""
    available, dominant = _native_candidates(spectrum, config)
    targets = np.asarray(config.native_target_bins)
    tolerances = np.asarray(config.tolerances_bins)
    weights = np.asarray(config.weights)
    losses = np.full(6, config.missing_penalty)
    if dominant in available and abs(dominant - targets[0]) < tolerances[0]:
        losses[0] = abs(dominant - targets[0]) / tolerances[0]
        available = available[available != dominant]

    costs = np.minimum(abs(targets[1:, None] - available[None, :]) / tolerances[1:, None], 1)
    costs *= weights[1:, None]
    costs = np.concatenate((costs, np.tile(weights[1:, None], (1, 5))), axis=1)
    rows, columns = linear_sum_assignment(costs)
    for r, c in zip(rows, columns):
        if c < len(available) and abs(targets[r + 1] - available[c]) < tolerances[r + 1]:
            losses[r + 1] = abs(targets[r + 1] - available[c]) / tolerances[r + 1]
    # The native Gleissberg term is replaced by the continuous term below.
    return losses


def _gleissberg_candidates(
    spectrum: np.ndarray, config: SpectralPeakConfig,
) -> list[tuple[float, float]]:
    """Return refined frequency and visibility for actual 60--140-year maxima."""
    n = config.n_samples
    factor = config.padding_factor
    frequency = np.fft.rfftfreq(n * factor, d=config.sample_interval_years)
    low, high = 1 / config.gleissberg_period_band[1], 1 / config.gleissberg_period_band[0]
    bin_width = 1 / (n * config.sample_interval_years)
    region = (frequency >= low) & (frequency <= high)
    regional_maximum = np.max(spectrum[region])
    peaks, properties = find_peaks(spectrum, prominence=0)
    candidates = []
    for k, prominence in zip(peaks, properties["prominences"]):
        if not low - bin_width / factor <= frequency[k] <= high + bin_width / factor:
            continue
        values = np.log(np.maximum(spectrum[k - 1:k + 2], 1e-300))
        denominator = values[0] - 2 * values[1] + values[2]
        shift = 0.0 if denominator == 0 else float(
            np.clip(0.5 * (values[0] - values[2]) / denominator, -0.5, 0.5)
        )
        refined = (k + shift) * bin_width / factor
        if not low < refined < high:
            continue
        visibility = min(
            1.0,
            prominence / max(config.relative_prominence * spectrum[k], 1e-300),
            prominence / max(config.regional_prominence * regional_maximum, 1e-300),
        )
        edge = min(
            1.0,
            max(0.0, (refined - low) / (bin_width * config.edge_taper_bins)),
            max(0.0, (high - refined) / (bin_width * config.edge_taper_bins)),
        )
        candidates.append((float(refined), float(visibility * edge)))
    return candidates


def _gleissberg_loss(candidates: list[tuple[float, float]], config: SpectralPeakConfig) -> float:
    width = config.tolerances_bins[1] / (config.n_samples * config.sample_interval_years)
    return min(
        (1 - visibility * (1 - min(1.0, abs(frequency - config.gleissberg_target_frequency) / width))
         for frequency, visibility in candidates),
        default=config.missing_penalty,
    )


@dataclass(frozen=True)
class SpectralPeakStats:
    """Picklable batch adapter; stats are the loss to fixed observed targets."""

    config: SpectralPeakConfig

    @property
    def ss_obs(self) -> np.ndarray:
        # batch(y) already measures discrepancy to the stored observation.
        return np.zeros(1, dtype=np.float64)

    def feature_losses(self, y: np.ndarray) -> np.ndarray:
        """Compute the six component penalties, for auditing or scalar reduction."""
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 2 or y.shape[1] != self.config.n_samples:
            raise ValueError(f"Expected a batch of shape (n, {self.config.n_samples}); got {y.shape}.")
        losses = np.ones((len(y), 6), dtype=np.float64)
        window = np.hanning(self.config.n_samples)
        # Bounded chunks avoid allocating a full population's padded spectra.
        for start in range(0, len(y), 32):
            block = y[start:start + 32]
            valid = np.all(np.isfinite(block), axis=1)
            valid[valid] &= np.ptp(block[valid], axis=1) > 0
            indices = np.flatnonzero(valid)
            windowed = block[valid] * window
            native = abs(np.fft.rfft(windowed, axis=-1)) / self.config.n_samples
            dense = abs(np.fft.rfft(
                windowed, n=self.config.n_samples * self.config.padding_factor, axis=-1,
            )) / self.config.n_samples
            for index, native_spectrum, dense_spectrum in zip(indices, native, dense):
                row = _native_losses(native_spectrum, self.config)
                row[1] = _gleissberg_loss(_gleissberg_candidates(dense_spectrum, self.config), self.config)
                losses[start + index] = row
        return losses

    def batch(self, y: np.ndarray, ss_out: np.ndarray) -> None:
        """Fill one weighted scalar loss per simulation, in [0, 1]."""
        if ss_out.shape != (len(y), 1):
            raise ValueError(f"Expected output shape ({len(y)}, 1); got {ss_out.shape}.")
        weights = np.asarray(self.config.weights)
        ss_out[:, 0] = self.feature_losses(y) @ weights / weights.sum()

    def metadata(self) -> dict:
        metadata = asdict(self.config)
        frequencies = np.asarray(self.config.native_target_bins, dtype=float) / (
            self.config.n_samples * self.config.sample_interval_years
        )
        frequencies[1] = self.config.gleissberg_target_frequency
        metadata.update(
            target_frequencies_per_year=frequencies.tolist(),
            target_periods_years=(1 / frequencies).tolist(),
            loss="(5*main + 5*Gleissberg + 15y + 17y + 27y + 39y) / 14",
            native_loss="min(1, frequency-bin error / tolerance); absent or ineligible peak = 1",
            gleissberg_loss="min over candidates: 1 - visibility * (1 - min(1, frequency error / width)); absent = 1",
            gleissberg_refinement="three-point parabola on log magnitude; Hann window BEFORE padding",
            assignment="dominant main first; one-to-one weighted assignment for native peaks",
            amplitude_policy="height/prominence only determine visibility; absolute amplitudes are not matched",
            sabc_distance="one scalar loss against zero; usual SABC prior-CDF transform follows aggregation",
            invalid_record_policy="flat or nonfinite simulations receive loss 1",
            resolution_note="padding adds no observations or physical frequency resolution",
        )
        return metadata


def build_spectral_peak_stats(observed: np.ndarray) -> SpectralPeakStats:
    """Freeze the tested score for obsSN; reject different native peak targets."""
    observed = np.asarray(observed, dtype=np.float64)
    if observed.shape != (271,) or not np.all(np.isfinite(observed)):
        raise ValueError("spectral_peaks_v1 requires the finite, 271-sample annual obsSN record.")
    digest = hashlib.sha256(np.asarray(observed, dtype="<f8").tobytes()).hexdigest()
    provisional = SpectralPeakConfig(observed_sha256=digest, gleissberg_target_frequency=0.0)
    windowed = observed * np.hanning(provisional.n_samples)
    native = abs(np.fft.rfft(windowed)) / provisional.n_samples
    if np.any(_native_losses(native, provisional) != 0):
        raise ValueError("Observed native peaks differ from the frozen spectral_peaks_v1 targets.")
    dense = abs(np.fft.rfft(windowed, n=provisional.n_samples * provisional.padding_factor)) / provisional.n_samples
    candidates = _gleissberg_candidates(dense, provisional)
    if len(candidates) != 1:
        raise ValueError("Expected exactly one observed Gleissberg peak in 60--140 years.")
    adapter = SpectralPeakStats(SpectralPeakConfig(
        observed_sha256=digest, gleissberg_target_frequency=candidates[0][0],
    ))
    if np.any(adapter.feature_losses(observed[None, :]) > 1e-12):
        raise ValueError("Observed peaks do not satisfy the frozen visibility requirements.")
    return adapter


def validate_spectral_peak_resume(previous_stats_fn, current_stats_fn) -> None:
    """Prevent a saved distance/CDF from being silently reused for another score."""
    previous = getattr(previous_stats_fn, "__self__", None)
    current = getattr(current_stats_fn, "__self__", None)
    if not (isinstance(previous, SpectralPeakStats) or isinstance(current, SpectralPeakStats)):
        return
    if (
        not isinstance(previous, SpectralPeakStats)
        or not isinstance(current, SpectralPeakStats)
        or previous.config != current.config
    ):
        raise ValueError(
            "Cannot change to/from spectral_peaks, its scoring settings, or its observation "
            "when resuming. Start a fresh inference with a new run name."
        )


def save_spectral_peak_metadata(adapter: SpectralPeakStats, path: Path) -> None:
    """Write a readable score definition before starting a long inference."""
    metadata = adapter.metadata()
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    print("Spectral peak scoring: " + json.dumps(metadata, sort_keys=True), flush=True)
    print(f"Spectral peak settings saved to: {path}", flush=True)
