"""ENCA encoder-backed summary statistics for solar-dynamo SABC runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import re

import numpy as np


@dataclass(frozen=True)
class EncaSummaryStatsConfig:
    run_dir: Path
    checkpoint_basename: str
    len_timeseries: int
    ndims_latent: int
    representation_mode: str = "time"
    num_fft_components: int | None = None
    fft_log_eps: float = 1e-8


_ENCODER_CACHE: dict[EncaSummaryStatsConfig, object] = {}


def _load_hyper_parameters(run_dir: Path) -> dict:
    path = run_dir / "hyper_parameters.json"
    if not path.exists():
        raise FileNotFoundError(f"ENCA hyper-parameter file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _checkpoint_prefix(run_dir: Path, basename: str) -> str:
    if not basename:
        raise ValueError("ENCA checkpoint basename cannot be empty")

    if basename.endswith(".index"):
        candidate = Path(basename)
        if not candidate.is_absolute():
            candidate = run_dir / candidate
        if not candidate.exists():
            raise FileNotFoundError(f"ENCA checkpoint index not found: {candidate}")
        return str(candidate.with_suffix(""))

    candidate = Path(basename)
    if candidate.is_absolute() or candidate.parent != Path("."):
        index_path = candidate.with_suffix(".index")
        if not index_path.exists():
            raise FileNotFoundError(f"ENCA checkpoint index not found: {index_path}")
        return str(candidate)

    pattern = re.compile(rf"^{re.escape(basename)}-(\d+)\.index$")
    matches: list[tuple[int, Path]] = []
    for path in run_dir.glob(f"{basename}-*.index"):
        match = pattern.match(path.name)
        if match:
            matches.append((int(match.group(1)), path))

    if not matches:
        raise FileNotFoundError(
            f"No ENCA checkpoints matching '{basename}-*.index' found in {run_dir}"
        )

    _, latest = max(matches, key=lambda item: item[0])
    return str(latest.with_suffix(""))


def _import_tensorflow():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorFlow is required for --summary-stats enca. Install TensorFlow "
            "in the environment used to run SABC, or use --summary-stats fft."
        ) from exc

    tf.get_logger().setLevel("ERROR")
    return tf


def _build_encoder(
    tf,
    *,
    len_timeseries: int,
    ndims_latent: int,
    representation_mode: str,
    num_fft_components: int | None,
):
    if representation_mode == "fourier_amplitude":
        if num_fft_components is None:
            raise ValueError("num_fft_components is required for MLP/Fourier ENCA")

        x_input = tf.keras.layers.Input(shape=[num_fft_components], name="fft_amplitudes")
        x = tf.keras.layers.Dense(256, activation="relu", name="enc_dense_1")(x_input)
        x = tf.keras.layers.Dense(256, activation="relu", name="enc_dense_2")(x)
        x = tf.keras.layers.Dense(128, activation="relu", name="enc_dense_3")(x)
        latent_space = tf.keras.layers.Dense(ndims_latent, activation=None, name="latent_space")(x)
        return tf.keras.Model(inputs=x_input, outputs=latent_space)

    if representation_mode != "time":
        raise ValueError(f"Unknown ENCA representation_mode: {representation_mode}")

    conv_fn = lambda filters, act=None, name=None: tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=3,
        activation=act,
        name=name,
    )
    x_input = tf.keras.layers.Input(shape=[len_timeseries, 1], name="x_observation")
    x = x_input
    for i, filters_for_stage in enumerate(([16, 16], [32, 32])):
        if i != 0:
            x = tf.keras.layers.MaxPool1D(pool_size=2, name=f"maxpool{i + 1}")(x)
        for j, n_filters in enumerate(filters_for_stage):
            x = conv_fn(
                filters=n_filters,
                act="relu",
                name=f"conv{i + 1}_{j + 1}",
            )(x)
    x = conv_fn(filters=ndims_latent, act=None, name="final_conv")(x)
    latent_space = tf.keras.layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    return tf.keras.Model(inputs=x_input, outputs=latent_space)


def _load_encoder(config: EncaSummaryStatsConfig):
    encoder = _ENCODER_CACHE.get(config)
    if encoder is not None:
        return encoder

    tf = _import_tensorflow()
    encoder = _build_encoder(
        tf,
        len_timeseries=config.len_timeseries,
        ndims_latent=config.ndims_latent,
        representation_mode=config.representation_mode,
        num_fft_components=config.num_fft_components,
    )
    ckpt = tf.train.Checkpoint(encoder=encoder)
    status = ckpt.restore(_checkpoint_prefix(config.run_dir, config.checkpoint_basename))
    status.assert_existing_objects_matched()
    status.expect_partial()
    _ENCODER_CACHE[config] = encoder
    return encoder


def _timeseries_to_fft_log_amplitudes(
    samples: np.ndarray,
    *,
    num_fft_components: int,
    fft_log_eps: float,
) -> np.ndarray:
    n_time = samples.shape[1]
    window = np.hanning(n_time).astype(np.float32)
    amplitudes = np.abs(np.fft.fft(samples * window[None, :], axis=1)) / float(n_time)
    return np.log(amplitudes[:, :num_fft_components] + fft_log_eps).astype(np.float32)


def _prepare_samples(config: EncaSummaryStatsConfig, data: np.ndarray) -> np.ndarray:
    samples = np.asarray(data, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    if samples.ndim != 2:
        raise ValueError(f"ENCA samples must be 1D or 2D, got shape {samples.shape}")
    if samples.shape[1] != config.len_timeseries:
        raise ValueError(
            f"ENCA encoder expects time-series length {config.len_timeseries}, "
            f"got {samples.shape[1]}"
        )

    if config.representation_mode == "fourier_amplitude":
        if config.num_fft_components is None:
            raise ValueError("num_fft_components is required for MLP/Fourier ENCA")
        return _timeseries_to_fft_log_amplitudes(
            samples,
            num_fft_components=config.num_fft_components,
            fft_log_eps=config.fft_log_eps,
        )

    if config.representation_mode == "time":
        return samples[:, :, np.newaxis]

    raise ValueError(f"Unknown ENCA representation_mode: {config.representation_mode}")


def _encode(config: EncaSummaryStatsConfig, data: np.ndarray) -> np.ndarray:
    encoder = _load_encoder(config)
    samples = _prepare_samples(config, data)
    z = encoder(samples, training=False).numpy()
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != config.ndims_latent:
        raise ValueError(
            f"ENCA encoder returned shape {z.shape}; expected "
            f"(batch, {config.ndims_latent})"
        )
    return z


def enca_stats_fn_batch(
    y: np.ndarray,
    ss_out: np.ndarray,
    *,
    config: EncaSummaryStatsConfig,
) -> None:
    """Compute ENCA latent summary statistics in-place for a simulated batch."""
    ss_out[:, :] = _encode(config, y)


class EncaSummaryStats:
    """Picklable ENCA summary-statistics adapter."""

    def __init__(self, config: EncaSummaryStatsConfig):
        self.config = config

    def observed(self, sn_data: np.ndarray) -> np.ndarray:
        return _encode(self.config, sn_data).reshape(-1)

    def batch(self, y: np.ndarray, ss_out: np.ndarray) -> None:
        enca_stats_fn_batch(y, ss_out, config=self.config)


def build_enca_summary_stats(
    *,
    run_dir: str | Path,
    checkpoint_basename: str = "model_best_ckpt",
    expected_tobs: int | None = None,
) -> EncaSummaryStats:
    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"ENCA run directory not found: {run_dir}")
    if not run_dir.is_dir():
        raise NotADirectoryError(f"ENCA run path is not a directory: {run_dir}")

    hyper_parameters = _load_hyper_parameters(run_dir)
    len_timeseries = int(hyper_parameters["len_timeseries"])
    ndims_latent = int(hyper_parameters["ndims_latent"])
    representation_mode = str(hyper_parameters.get("representation_mode", "time"))
    num_fft_components = hyper_parameters.get("num_fft_components")
    if num_fft_components is not None:
        num_fft_components = int(num_fft_components)
    fft_log_eps = float(hyper_parameters.get("fft_log_eps", 1e-8))

    if expected_tobs is not None and len_timeseries != int(expected_tobs):
        raise ValueError(
            f"ENCA encoder was trained for len_timeseries={len_timeseries}, "
            f"but the selected dataset has Tobs={expected_tobs}."
        )

    config = EncaSummaryStatsConfig(
        run_dir=run_dir,
        checkpoint_basename=checkpoint_basename,
        len_timeseries=len_timeseries,
        ndims_latent=ndims_latent,
        representation_mode=representation_mode,
        num_fft_components=num_fft_components,
        fft_log_eps=fft_log_eps,
    )
    _checkpoint_prefix(config.run_dir, config.checkpoint_basename)
    return EncaSummaryStats(config)


def build_mlp_summary_stats(
    *,
    run_dir: str | Path,
    checkpoint_basename: str = "model_best_ckpt",
    expected_tobs: int | None = None,
) -> EncaSummaryStats:
    stats = build_enca_summary_stats(
        run_dir=run_dir,
        checkpoint_basename=checkpoint_basename,
        expected_tobs=expected_tobs,
    )
    if stats.config.representation_mode != "fourier_amplitude":
        raise ValueError(
            "--summary-stats mlp requires an ENCA run with "
            'representation_mode="fourier_amplitude"; '
            f"got {stats.config.representation_mode!r}"
        )
    return stats
