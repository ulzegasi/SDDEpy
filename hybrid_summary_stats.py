"""Original ENCAfftCNN z=5 outputs plus one Jupiter-period FFT magnitude."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
from pathlib import Path

import numpy as np
from sdde_model.solar_dynamo_jupiter import JUPITER_ORBITAL_PERIOD_YEARS

from enca_summary_stats import (
    EncaSummaryStats,
    EncaSummaryStatsConfig,
    _checkpoint_prefix,
    build_enca_fft_cnn_summary_stats,
)

HYBRID_MODE = "enca_fft_cnn_5+1"


@dataclass(frozen=True)
class HybridSummaryStatsConfig:
    encoder: EncaSummaryStatsConfig
    checkpoint_sha256: str
    fourier_bin: int  # NumPy index; DC is bin zero.
    jupiter_period_years: float = JUPITER_ORBITAL_PERIOD_YEARS
    sample_interval_years: float = 1.0
    version: str = "enca_fft_cnn_5+1_v1"


class HybridSummaryStats:
    """Picklable adapter; TensorFlow stays in the parent process, as for CNN."""

    def __init__(self, config: HybridSummaryStatsConfig):
        self.config = config

    def batch(self, y: np.ndarray, ss_out: np.ndarray) -> None:
        samples = np.asarray(y, dtype=np.float64)
        n = self.config.encoder.len_timeseries
        if samples.ndim != 2 or samples.shape[1] != n:
            raise ValueError(f"{HYBRID_MODE} expects a (batch, {n}) time-series array.")
        if ss_out.shape != (samples.shape[0], 6):
            raise ValueError(f"{HYBRID_MODE} requires exactly six output statistics.")
        # Preserve the original encoder's preprocessing and coordinate order.
        EncaSummaryStats(self.config.encoder).batch(samples, ss_out[:, :5])
        # abs(FFT)/N equals the magnitude of the inverse FFT used by FFT mode
        # for real input. No log, squared power, detrending, or peak searching.
        spectrum = np.fft.rfft(samples * np.hanning(n), axis=1)
        ss_out[:, 5] = np.abs(spectrum[:, self.config.fourier_bin]) / n

    def observed(self, sn_data: np.ndarray) -> np.ndarray:
        samples = np.asarray(sn_data, dtype=np.float64)
        if samples.ndim != 1:
            raise ValueError("Observed hybrid data must be a one-dimensional time series.")
        out = np.empty((1, 6), dtype=np.float64)
        self.batch(samples[None, :], out)
        return out[0]

    def metadata(self) -> dict:
        n = self.config.encoder.len_timeseries
        k = self.config.fourier_bin
        return {
            "version": self.config.version,
            "encoder_run": str(self.config.encoder.run_dir),
            "encoder_checkpoint": self.config.encoder.checkpoint_basename,
            "checkpoint_sha256": self.config.checkpoint_sha256,
            "encoder_model": "original",
            "inference_model": "jupiter",
            "statistics": ["encoder_tau", "encoder_T", "encoder_Nd",
                           "encoder_sigma", "encoder_Bmax", "jupiter_fft_magnitude"],
            "n_samples": n,
            "sample_interval_years": self.config.sample_interval_years,
            "target_period_years": self.config.jupiter_period_years,
            "fourier_bin_numpy": k,
            "fourier_index_julia": k + 1,
            "bin_period_years": n * self.config.sample_interval_years / k,
            "sixth_statistic": "abs(rfft(x * numpy.hanning(N)))[k] / N",
        }


def validate_hybrid_years(years: np.ndarray) -> None:
    years = np.asarray(years, dtype=float)
    if years.ndim != 1 or years.size < 2 or not np.allclose(
        np.diff(years), 1.0, rtol=0, atol=1e-8
    ):
        raise ValueError(f"{HYBRID_MODE} requires evenly spaced annual observations.")


def _checkpoint_digest(prefix: Path) -> str:
    shards = sorted(prefix.parent.glob(prefix.name + ".data-*-of-*"))
    if not shards:
        raise FileNotFoundError(f"No checkpoint data shards found for {prefix}")
    digest = hashlib.sha256()
    for path in [Path(str(prefix) + ".index"), *shards]:
        digest.update(path.name[len(prefix.name):].encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def build_hybrid_summary_stats(
    *,
    run_dir: str | Path,
    checkpoint_basename: str = "model_best_ckpt",
    expected_tobs: int | None = None,
    expected_model: str = "jupiter",
) -> HybridSummaryStats:
    if expected_model != "jupiter":
        raise ValueError(f"{HYBRID_MODE} requires --model jupiter.")
    # This explicit hybrid mode alone permits an original-trained encoder with
    # the Jupiter simulator. The ordinary CNN model-matching guard is unchanged.
    encoder = build_enca_fft_cnn_summary_stats(
        run_dir=run_dir,
        checkpoint_basename=checkpoint_basename,
        expected_tobs=expected_tobs,
        expected_model="original",
    )
    if encoder.config.ndims_latent != 5:
        raise ValueError(f"{HYBRID_MODE} requires an original-model z=5 encoder.")
    n = encoder.config.len_timeseries
    k = int(np.floor(n / JUPITER_ORBITAL_PERIOD_YEARS + 0.5))
    if not 1 <= k <= n // 2:
        raise ValueError("The record is too short for a Jupiter-period Fourier bin.")
    prefix = Path(_checkpoint_prefix(encoder.config.run_dir, checkpoint_basename))
    # Pin the exact step so later best checkpoints cannot silently replace it.
    config = replace(encoder.config, checkpoint_basename=str(prefix) + ".index")
    return HybridSummaryStats(HybridSummaryStatsConfig(
        encoder=config, checkpoint_sha256=_checkpoint_digest(prefix), fourier_bin=k,
    ))


def _identity(stats: HybridSummaryStats) -> dict:
    identity = asdict(stats.config)
    # Cluster/Mac paths may differ; checkpoint bytes and preprocessing may not.
    identity["encoder"].pop("run_dir")
    identity["encoder"].pop("checkpoint_basename")
    return identity


def validate_hybrid_resume(previous_stats_fn, current_stats_fn) -> None:
    previous = getattr(previous_stats_fn, "__self__", None)
    current = getattr(current_stats_fn, "__self__", None)
    if not isinstance(previous, HybridSummaryStats) and not isinstance(current, HybridSummaryStats):
        return
    if not isinstance(previous, HybridSummaryStats) or not isinstance(current, HybridSummaryStats):
        raise ValueError("Cannot switch into or out of enca_fft_cnn_5+1 when resuming a run.")
    if _identity(previous) != _identity(current):
        raise ValueError("Cannot change the hybrid checkpoint or FFT settings when resuming/filtering.")


def rebuild_hybrid_summary_stats(
    saved: HybridSummaryStats, *, run_dir: str | Path,
    checkpoint_basename: str = "model_best_ckpt", expected_tobs: int,
    expected_model: str,
) -> HybridSummaryStats:
    """Reload the saved checkpoint at a possibly different cluster/Mac path."""
    if checkpoint_basename == "model_best_ckpt":
        checkpoint_basename = Path(saved.config.encoder.checkpoint_basename).name
    current = build_hybrid_summary_stats(
        run_dir=run_dir, checkpoint_basename=checkpoint_basename,
        expected_tobs=expected_tobs, expected_model=expected_model,
    )
    validate_hybrid_resume(saved.batch, current.batch)
    return current
