"""Neural encoder-backed summary statistics for solar-dynamo SABC runs."""

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
    fft_window: str = "none"
    model: str = "original"
    num_model_parameters: int = 5
    simulation_backend: str | None = None


_ENCODER_CACHE: dict[EncaSummaryStatsConfig, object] = {}

VALID_FFT_WINDOWS = ("none", "hann")
VALID_SDDE_MODELS = ("original", "jupiter")
CANONICAL_NOISEGRID_BACKEND = "sdde_model_sddeproblem_em_noisegrid_v2"


@dataclass(frozen=True)
class FnoSummaryStatsConfig:
    run_dir: Path
    checkpoint_basename: str
    len_timeseries: int
    ndims_latent: int
    representation_mode: str = "time"
    reconstruction_loss_domain: str = "time"
    fno_modes: int = 32
    fno_width: int = 64
    fno_layers: int = 4
    use_time_coordinate: bool = True


_FNO_ENCODER_CACHE: dict[FnoSummaryStatsConfig, object] = {}
_FNO_ENCODE_FN_CACHE: dict[FnoSummaryStatsConfig, object] = {}


def _load_hyper_parameters(run_dir: Path) -> dict:
    path = run_dir / "hyper_parameters.json"
    if not path.exists():
        raise FileNotFoundError(f"ENCA hyper-parameter file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _neural_model_contract(
    hyper_parameters: dict,
    expected_model: str | None,
) -> tuple[str, int]:
    """Validate the physical-parameter contract saved with an encoder."""
    model = str(
        hyper_parameters.get("model", hyper_parameters.get("MODEL", "original"))
    ).strip().lower()
    if model not in VALID_SDDE_MODELS:
        raise ValueError(
            f"Unknown encoder SDDE model {model!r}; expected one of {VALID_SDDE_MODELS}."
        )
    required_parameters = 6 if model == "jupiter" else 5
    num_model_parameters = int(
        hyper_parameters.get("num_model_parameters", required_parameters)
    )
    if num_model_parameters != required_parameters:
        raise ValueError(
            f"Encoder metadata declares model={model!r}, which requires "
            f"{required_parameters} parameter regressors, but "
            f"num_model_parameters={num_model_parameters}."
        )
    if expected_model is not None:
        expected_model = str(expected_model).strip().lower()
        if expected_model not in VALID_SDDE_MODELS:
            raise ValueError(
                f"Unknown requested SDDE model {expected_model!r}; expected one "
                f"of {VALID_SDDE_MODELS}."
            )
        if model != expected_model:
            raise ValueError(
                f"SABC requested model={expected_model!r}, but the encoder was "
                f"trained with model={model!r}. Select a matching training run."
            )
    return model, num_model_parameters


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
            "TensorFlow is required for neural summary statistics. Install TensorFlow "
            "in the environment used to run SABC, or use --summary-stats fft."
        ) from exc

    tf.get_logger().setLevel("ERROR")
    return tf


def _hyper_parameter_int(
    hyper_parameters: dict,
    names: tuple[str, ...],
    default: int,
) -> int:
    for name in names:
        if name in hyper_parameters and hyper_parameters[name] is not None:
            return int(hyper_parameters[name])
    return default


def _hyper_parameter_bool(
    hyper_parameters: dict,
    names: tuple[str, ...],
    default: bool,
) -> bool:
    for name in names:
        if name in hyper_parameters and hyper_parameters[name] is not None:
            value = hyper_parameters[name]
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "y", "on"}
            return bool(value)
    return default


def _normalize_fft_window(value: object, *, allow_auto: bool = False) -> str:
    """Return the canonical Fourier preprocessing window name."""
    normalized = str(value).strip().lower()
    aliases = {
        "": "none",
        "no": "none",
        "null": "none",
        "rectangular": "none",
        "boxcar": "none",
        "hanning": "hann",
    }
    normalized = aliases.get(normalized, normalized)
    valid = (*VALID_FFT_WINDOWS, "auto") if allow_auto else VALID_FFT_WINDOWS
    if normalized not in valid:
        choices = ", ".join(valid)
        raise ValueError(f"Unknown FFT window {value!r}; expected one of: {choices}")
    return normalized


def _resolve_fft_window(hyper_parameters: dict, requested_window: str | None) -> str:
    """Resolve an explicit window or read it from a training run's metadata."""
    requested = _normalize_fft_window(
        "auto" if requested_window is None else requested_window,
        allow_auto=True,
    )
    if requested != "auto":
        return requested

    for name in ("fft_window", "window", "WINDOW"):
        if name in hyper_parameters and hyper_parameters[name] is not None:
            return _normalize_fft_window(hyper_parameters[name])

    # Fourier-CNN runs created before window metadata was introduced used the
    # raw signal. Preserve that behavior when their metadata has no window key.
    return "none"


def _build_encoder(
    tf,
    *,
    len_timeseries: int,
    ndims_latent: int,
    representation_mode: str,
    num_fft_components: int | None,
):
    if representation_mode == "enca_fft_cnn":
        if num_fft_components is None:
            raise ValueError("num_fft_components is required for Fourier-CNN ENCA")

        x_input = tf.keras.layers.Input(
            shape=[num_fft_components, 1],
            name="fourier_log_amplitude",
        )
        x = tf.keras.layers.Conv1D(
            16, 3, padding="same", activation="relu", name="enc_conv_1"
        )(x_input)
        x = tf.keras.layers.Conv1D(
            16, 3, padding="same", activation="relu", name="enc_conv_2"
        )(x)
        x = tf.keras.layers.MaxPool1D(pool_size=2, name="enc_maxpool")(x)
        x = tf.keras.layers.Conv1D(
            32, 3, padding="same", activation="relu", name="enc_conv_3"
        )(x)
        x = tf.keras.layers.Conv1D(
            32, 3, padding="same", activation="relu", name="enc_conv_4"
        )(x)
        x = tf.keras.layers.Conv1D(
            ndims_latent,
            3,
            padding="same",
            activation=None,
            name="latent_channels",
        )(x)
        latent_space = tf.keras.layers.GlobalAveragePooling1D(
            name="global_avg_pool"
        )(x)
        return tf.keras.Model(inputs=x_input, outputs=latent_space)

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


def _timeseries_to_rfft_log1p_amplitudes(
    samples: np.ndarray,
    *,
    num_fft_components: int,
    fft_window: str = "none",
) -> np.ndarray:
    """Match ENCAFourierCNN training preprocessing exactly."""
    max_components = samples.shape[1] // 2 + 1
    if num_fft_components > max_components:
        raise ValueError(
            f"num_fft_components={num_fft_components} exceeds the rFFT size "
            f"{max_components} for time-series length {samples.shape[1]}"
        )
    fft_window = _normalize_fft_window(fft_window)
    if fft_window == "hann":
        window = np.hanning(samples.shape[1]).astype(np.float32)
        samples = samples * window[np.newaxis, :]

    amplitudes = np.abs(np.fft.rfft(samples, axis=1))[:, :num_fft_components]
    return np.log1p(amplitudes).astype(np.float32)[:, :, np.newaxis]


def _make_spectral_conv1d_layer(tf):
    @tf.keras.utils.register_keras_serializable(package="sddepy")
    class SpectralConv1D(tf.keras.layers.Layer):
        def __init__(self, out_channels: int, modes: int, **kwargs):
            super().__init__(**kwargs)
            self.out_channels = int(out_channels)
            self.modes = int(modes)

        def build(self, input_shape):
            in_channels = int(input_shape[-1])
            scale = 1.0 / max(1, in_channels * self.out_channels)
            self.weight_real = self.add_weight(
                name="weight_real",
                shape=(self.modes, in_channels, self.out_channels),
                initializer=tf.keras.initializers.RandomNormal(stddev=scale),
                trainable=True,
            )
            self.weight_imag = self.add_weight(
                name="weight_imag",
                shape=(self.modes, in_channels, self.out_channels),
                initializer=tf.keras.initializers.RandomNormal(stddev=scale),
                trainable=True,
            )
            super().build(input_shape)

        def call(self, x):
            n_time = tf.shape(x)[1]
            x_ft = tf.signal.rfft(tf.transpose(x, perm=[0, 2, 1]))
            x_ft = tf.transpose(x_ft, perm=[0, 2, 1])

            weights = tf.complex(self.weight_real, self.weight_imag)
            x_ft_low = x_ft[:, : self.modes, :]
            out_ft_low = tf.einsum("bmi,mio->bmo", x_ft_low, weights)

            n_freq = tf.shape(x_ft)[1]
            pad_modes = n_freq - self.modes
            out_ft = tf.pad(out_ft_low, [[0, 0], [0, pad_modes], [0, 0]])
            out_ft = tf.transpose(out_ft, perm=[0, 2, 1])
            x_out = tf.signal.irfft(out_ft, fft_length=[n_time])
            return tf.transpose(x_out, perm=[0, 2, 1])

        def compute_output_shape(self, input_shape):
            input_shape = tf.TensorShape(input_shape).as_list()
            input_shape[-1] = self.out_channels
            return tf.TensorShape(input_shape)

        def get_config(self):
            config = super().get_config()
            config.update({"out_channels": self.out_channels, "modes": self.modes})
            return config

    return SpectralConv1D


def _make_fno_block1d_layer(tf):
    SpectralConv1D = _make_spectral_conv1d_layer(tf)

    @tf.keras.utils.register_keras_serializable(package="sddepy")
    class FNOBlock1D(tf.keras.layers.Layer):
        def __init__(self, width: int, modes: int, activation: str = "gelu", **kwargs):
            super().__init__(**kwargs)
            self.width = int(width)
            self.modes = int(modes)
            self.activation_name = activation
            self.spectral = SpectralConv1D(self.width, self.modes)
            self.pointwise = tf.keras.layers.Conv1D(self.width, kernel_size=1)
            self.activation = tf.keras.layers.Activation(activation)

        def call(self, x):
            x = self.spectral(x) + self.pointwise(x)
            return self.activation(x)

        def compute_output_shape(self, input_shape):
            input_shape = tf.TensorShape(input_shape).as_list()
            input_shape[-1] = self.width
            return tf.TensorShape(input_shape)

        def get_config(self):
            config = super().get_config()
            config.update(
                {
                    "width": self.width,
                    "modes": self.modes,
                    "activation": self.activation_name,
                }
            )
            return config

    return FNOBlock1D


def _time_coordinate_layer(tf, len_timeseries: int, name: str):
    time_grid = tf.linspace(0.0, 1.0, len_timeseries)
    time_grid = tf.reshape(time_grid, [1, len_timeseries, 1])
    return tf.keras.layers.Lambda(
        function=lambda x: tf.tile(tf.cast(time_grid, x.dtype), [tf.shape(x)[0], 1, 1]),
        name=name,
    )


def _build_fno_encoder(
    tf,
    *,
    len_timeseries: int,
    ndims_latent: int,
    representation_mode: str,
    fno_modes: int,
    fno_width: int,
    fno_layers: int,
    use_time_coordinate: bool,
):
    if representation_mode != "time":
        raise ValueError(
            "--summary-stats fno expects a time-domain FNO run "
            f"(representation_mode='time'); got {representation_mode!r}."
        )

    FNOBlock1D = _make_fno_block1d_layer(tf)
    input_length = int(len_timeseries)
    fno_modes = min(int(fno_modes), input_length // 2 + 1)
    if fno_modes < 1:
        raise ValueError("fno_modes must be at least 1.")

    x_input = tf.keras.layers.Input(shape=[input_length, 1], name="x_observation")
    x = x_input

    if use_time_coordinate:
        t = _time_coordinate_layer(tf, input_length, name="encoder_time_coordinate")(x_input)
        x = tf.keras.layers.Concatenate(axis=-1, name="encoder_concat_time")([x, t])

    x = tf.keras.layers.Dense(fno_width, activation=None, name="encoder_lift")(x)
    for index in range(int(fno_layers)):
        x = FNOBlock1D(fno_width, fno_modes, name=f"encoder_fno_block_{index + 1}")(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dense(fno_width, activation="gelu", name="encoder_dense")(x)
    z = tf.keras.layers.Dense(ndims_latent, activation=None, name="latent_space")(x)
    return tf.keras.Model(inputs=x_input, outputs=z)


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

    if config.representation_mode == "enca_fft_cnn":
        if config.num_fft_components is None:
            raise ValueError("num_fft_components is required for Fourier-CNN ENCA")
        return _timeseries_to_rfft_log1p_amplitudes(
            samples,
            num_fft_components=config.num_fft_components,
            fft_window=config.fft_window,
        )

    if config.representation_mode == "time":
        return samples[:, :, np.newaxis]

    raise ValueError(f"Unknown ENCA representation_mode: {config.representation_mode}")


def _prepare_fno_samples(config: FnoSummaryStatsConfig, data: np.ndarray) -> np.ndarray:
    samples = np.asarray(data, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    if samples.ndim != 2:
        raise ValueError(f"FNO samples must be 1D or 2D, got shape {samples.shape}")
    if samples.shape[1] != config.len_timeseries:
        raise ValueError(
            f"FNO encoder expects time-series length {config.len_timeseries}, "
            f"got {samples.shape[1]}"
        )

    if config.representation_mode != "time":
        raise ValueError(
            "FNO inference expects the encoder input representation to be the "
            f"raw time series; got representation_mode={config.representation_mode!r}."
        )
    return samples[:, :, np.newaxis]


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


def _load_fno_encoder(config: FnoSummaryStatsConfig):
    encoder = _FNO_ENCODER_CACHE.get(config)
    if encoder is not None:
        return encoder

    tf = _import_tensorflow()
    encoder = _build_fno_encoder(
        tf,
        len_timeseries=config.len_timeseries,
        ndims_latent=config.ndims_latent,
        representation_mode=config.representation_mode,
        fno_modes=config.fno_modes,
        fno_width=config.fno_width,
        fno_layers=config.fno_layers,
        use_time_coordinate=config.use_time_coordinate,
    )
    ckpt = tf.train.Checkpoint(encoder=encoder)
    status = ckpt.restore(_checkpoint_prefix(config.run_dir, config.checkpoint_basename))
    status.assert_existing_objects_matched()
    status.expect_partial()
    _FNO_ENCODER_CACHE[config] = encoder
    return encoder


def _encode_fno(config: FnoSummaryStatsConfig, data: np.ndarray) -> np.ndarray:
    encoder = _load_fno_encoder(config)
    samples = _prepare_fno_samples(config, data)
    encode_fn = _FNO_ENCODE_FN_CACHE.get(config)
    if encode_fn is None:
        tf = _import_tensorflow()

        @tf.function(reduce_retracing=True)
        def encode_fn(x):
            return encoder(x, training=False)

        _FNO_ENCODE_FN_CACHE[config] = encode_fn

    z = encode_fn(samples).numpy()
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != config.ndims_latent:
        raise ValueError(
            f"FNO encoder returned shape {z.shape}; expected "
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


def fno_stats_fn_batch(
    y: np.ndarray,
    ss_out: np.ndarray,
    *,
    config: FnoSummaryStatsConfig,
) -> None:
    """Compute FNO latent summary statistics in-place for a simulated batch."""
    ss_out[:, :] = _encode_fno(config, y)


class FnoSummaryStats:
    """Picklable FNO summary-statistics adapter."""

    def __init__(self, config: FnoSummaryStatsConfig):
        self.config = config

    def observed(self, sn_data: np.ndarray) -> np.ndarray:
        return _encode_fno(self.config, sn_data).reshape(-1)

    def batch(self, y: np.ndarray, ss_out: np.ndarray) -> None:
        fno_stats_fn_batch(y, ss_out, config=self.config)


def build_enca_summary_stats(
    *,
    run_dir: str | Path,
    checkpoint_basename: str = "model_best_ckpt",
    expected_tobs: int | None = None,
    expected_model: str | None = None,
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
    simulation_backend = hyper_parameters.get("simulation_backend")
    model, num_model_parameters = _neural_model_contract(
        hyper_parameters, expected_model
    )

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
        model=model,
        num_model_parameters=num_model_parameters,
        simulation_backend=simulation_backend,
    )
    _checkpoint_prefix(config.run_dir, config.checkpoint_basename)
    return EncaSummaryStats(config)


def build_mlp_summary_stats(
    *,
    run_dir: str | Path,
    checkpoint_basename: str = "model_best_ckpt",
    expected_tobs: int | None = None,
    expected_model: str | None = None,
) -> EncaSummaryStats:
    stats = build_enca_summary_stats(
        run_dir=run_dir,
        checkpoint_basename=checkpoint_basename,
        expected_tobs=expected_tobs,
        expected_model=expected_model,
    )
    if stats.config.representation_mode != "fourier_amplitude":
        raise ValueError(
            "--summary-stats mlp requires an ENCA run with "
            'representation_mode="fourier_amplitude"; '
            f"got {stats.config.representation_mode!r}"
        )
    if (
        expected_model is not None
        and stats.config.simulation_backend != CANONICAL_NOISEGRID_BACKEND
    ):
        raise ValueError(
            "SABC neural inference requires an MLP checkpoint trained with "
            f"simulation_backend={CANONICAL_NOISEGRID_BACKEND!r}; got "
            f"{stats.config.simulation_backend!r}. Retrain in a fresh directory."
        )
    return stats


def build_enca_fft_cnn_summary_stats(
    *,
    run_dir: str | Path,
    checkpoint_basename: str = "model_best_ckpt",
    expected_tobs: int | None = None,
    fft_window: str | None = "auto",
    expected_model: str | None = None,
) -> EncaSummaryStats:
    """Build the Fourier-CNN ENCA encoder used by solar_dynamo training."""
    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Fourier-CNN ENCA run directory not found: {run_dir}")
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Fourier-CNN ENCA run path is not a directory: {run_dir}")

    hyper_parameters = _load_hyper_parameters(run_dir)
    len_timeseries = int(hyper_parameters["len_timeseries"])
    ndims_latent = int(hyper_parameters["ndims_latent"])
    num_fft_components = int(hyper_parameters["num_fft_components"])
    resolved_fft_window = _resolve_fft_window(hyper_parameters, fft_window)
    simulation_backend = hyper_parameters.get("simulation_backend")
    model, num_model_parameters = _neural_model_contract(
        hyper_parameters, expected_model
    )
    if (
        expected_model is not None
        and simulation_backend != CANONICAL_NOISEGRID_BACKEND
    ):
        raise ValueError(
            "SABC neural inference requires an ENCAfftCNN checkpoint trained "
            f"with simulation_backend={CANONICAL_NOISEGRID_BACKEND!r}; got "
            f"{simulation_backend!r}. Retrain in a fresh directory."
        )
    if expected_model is not None and resolved_fft_window != "hann":
        raise ValueError(
            "SABC neural inference requires ENCAfftCNN checkpoint metadata "
            "with mandatory Hann preprocessing; got "
            f"fft_window={resolved_fft_window!r}. Retrain in a fresh directory."
        )

    if expected_tobs is not None and len_timeseries != int(expected_tobs):
        raise ValueError(
            f"Fourier-CNN ENCA encoder was trained for len_timeseries={len_timeseries}, "
            f"but the selected dataset has Tobs={expected_tobs}."
        )

    max_components = len_timeseries // 2 + 1
    if num_fft_components > max_components:
        raise ValueError(
            f"num_fft_components={num_fft_components} exceeds the rFFT size "
            f"{max_components} for len_timeseries={len_timeseries}."
        )

    config = EncaSummaryStatsConfig(
        run_dir=run_dir,
        checkpoint_basename=checkpoint_basename,
        len_timeseries=len_timeseries,
        ndims_latent=ndims_latent,
        representation_mode="enca_fft_cnn",
        num_fft_components=num_fft_components,
        fft_window=resolved_fft_window,
        model=model,
        num_model_parameters=num_model_parameters,
        simulation_backend=simulation_backend,
    )
    _checkpoint_prefix(config.run_dir, config.checkpoint_basename)
    return EncaSummaryStats(config)


def build_fno_summary_stats(
    *,
    run_dir: str | Path,
    checkpoint_basename: str = "model_best_ckpt",
    expected_tobs: int | None = None,
) -> FnoSummaryStats:
    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"FNO run directory not found: {run_dir}")
    if not run_dir.is_dir():
        raise NotADirectoryError(f"FNO run path is not a directory: {run_dir}")

    hyper_parameters = _load_hyper_parameters(run_dir)
    len_timeseries = int(hyper_parameters["len_timeseries"])
    ndims_latent = int(hyper_parameters["ndims_latent"])
    representation_mode = str(hyper_parameters.get("representation_mode", "time"))
    reconstruction_loss_domain = str(hyper_parameters.get("reconstruction_loss_domain", "time"))

    if representation_mode != "time":
        raise ValueError(
            "--summary-stats fno requires a time-domain FNO run "
            f"(representation_mode='time'); got {representation_mode!r}. "
            "The Fourier FNO runs still use representation_mode='time'; "
            "their reconstruction_loss_domain controls only the training loss."
        )

    if expected_tobs is not None and len_timeseries != int(expected_tobs):
        raise ValueError(
            f"FNO encoder was trained for len_timeseries={len_timeseries}, "
            f"but the selected dataset has Tobs={expected_tobs}."
        )

    config = FnoSummaryStatsConfig(
        run_dir=run_dir,
        checkpoint_basename=checkpoint_basename,
        len_timeseries=len_timeseries,
        ndims_latent=ndims_latent,
        representation_mode=representation_mode,
        reconstruction_loss_domain=reconstruction_loss_domain,
        fno_modes=_hyper_parameter_int(
            hyper_parameters,
            ("fno_modes", "num_fno_modes", "modes", "n_modes"),
            32,
        ),
        fno_width=_hyper_parameter_int(
            hyper_parameters,
            ("fno_width", "width", "hidden_channels", "channels"),
            64,
        ),
        fno_layers=_hyper_parameter_int(
            hyper_parameters,
            ("fno_layers", "fno_depth", "n_fno_layers", "num_fno_layers", "n_layers"),
            4,
        ),
        use_time_coordinate=_hyper_parameter_bool(
            hyper_parameters,
            ("use_time_coordinate", "fno_use_grid", "use_grid", "append_grid"),
            True,
        ),
    )
    _checkpoint_prefix(config.run_dir, config.checkpoint_basename)
    return FnoSummaryStats(config)
