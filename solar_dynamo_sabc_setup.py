"""Import-safe helpers for solar-dynamo SABC problems."""

from __future__ import annotations

from contextlib import redirect_stdout
from functools import partial
from pathlib import Path
import io

import numpy as np

from sdde_model import (
    hann_window,
    init_julia,
    sn_batch as sn_batch_original,
    summary_statistics,
    summary_statistics_batch,
)
from sdde_model.solar_dynamo_jupiter import sn_from_noise_batch as sn_from_noise_batch_jupiter

DatasetName = str
ModelName = str
VALID_MODELS = ("original", "jupiter")
SDDE_DT = 0.1


def load_observed_sn(datadir: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Load and crop the observed yearly sunspot-number record."""
    data_path = datadir / "silso_SN_y_202601.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data file: {data_path}")

    data = np.loadtxt(data_path, delimiter=",", dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("Expected a 2-column CSV with [year, sunspot_number].")

    years = data[:, 0][49:-6]
    values = data[:, 1][49:-6]
    return years, values, int(values.size)


def load_c14_sn(datadir: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Load the C14-reconstructed yearly sunspot-number record."""
    data_path = datadir / "SN_Usoskin.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data file: {data_path}")

    data = np.loadtxt(
        data_path,
        delimiter=",",
        dtype=float,
        usecols=(0, 1),
        skiprows=2,
        encoding="utf-8-sig",
    )
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("Expected a 2-column CSV with [year, sunspot_number].")

    years = data[:, 0]
    values = data[:, 1]
    return years, values, int(values.size)


def load_synthetic_sn(data_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Load a user-provided synthetic yearly sunspot-number record."""
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find synthetic data file: {data_path}")

    data = np.loadtxt(
        data_path,
        delimiter=",",
        dtype=float,
        usecols=(0, 1),
        skiprows=1,
        encoding="utf-8-sig",
    )
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("Expected a 2-column CSV with [time, sunspot_number].")

    years = data[:, 0]
    values = data[:, 1]
    return years, values, int(values.size)


def load_dataset(
    dataset: DatasetName,
    datadir: Path,
    synthetic_data_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Load one of the supported solar-dynamo datasets."""
    if dataset == "obsSN":
        return load_observed_sn(datadir)
    if dataset == "C14":
        return load_c14_sn(datadir)
    if dataset == "synthetic":
        if synthetic_data_path is None:
            raise ValueError("Missing data file, specify --synthetic-data-file")
        return load_synthetic_sn(synthetic_data_path)
    raise ValueError(f"Unknown dataset '{dataset}'. Expected 'obsSN', 'C14', or 'synthetic'.")


def simulator_batch(
    theta: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    *,
    Twarmup: int,
    Tobs: int,
    model: ModelName,
) -> None:
    """Simulate yearly sunspot traces for a batch of inference parameters.

    The Jupiter inference has six parameters. Its phase is a nuisance variable:
    draw one independent bare-noise path and phase per realization, then call
    the same explicit-noise EM integrator used during encoder training. Neither
    nuisance variable is part of the inferred parameter vector.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 2:
        raise ValueError(f"theta must be a 2D array, got shape {theta.shape}")

    expected_n_parameters = 5 if model == "original" else 6
    if model not in VALID_MODELS:
        raise ValueError(f"Unknown model {model!r}; expected one of {VALID_MODELS}")
    if theta.shape[1] != expected_n_parameters:
        raise ValueError(
            f"model={model!r} expects {expected_n_parameters} inference parameters, "
            f"got {theta.shape[1]}"
        )

    if model == "original":
        seeds = rng.integers(
            0,
            np.iinfo(np.int32).max,
            size=theta.shape[0],
            dtype=np.int64,
        )
        y_sim = sn_batch_original(
            theta,
            Twarmup=Twarmup,
            Tobs=Tobs,
            dt=SDDE_DT,
            seeds=seeds,
        )
    else:
        n_increments = int(round((Twarmup + Tobs) / SDDE_DT))
        eps_batch = rng.standard_normal(size=(theta.shape[0], n_increments))
        phases = rng.uniform(0.0, 2.0 * np.pi, size=(theta.shape[0], 1))
        theta_simulator = np.concatenate((theta, phases), axis=1)
        y_sim = sn_from_noise_batch_jupiter(
            theta_simulator,
            eps_batch,
            Twarmup=Twarmup,
            Tobs=Tobs,
            dt=SDDE_DT,
        )
    y[:, :] = np.asarray(y_sim, dtype=np.float64)


def build_simulator(*, Twarmup: int, Tobs: int, model: ModelName = "original"):
    """Create a picklable simulator with fixed warmup and observation length."""
    if model not in VALID_MODELS:
        raise ValueError(f"Unknown model {model!r}; expected one of {VALID_MODELS}")
    return partial(simulator_batch, Twarmup=Twarmup, Tobs=Tobs, model=model)


def stats_fn_batch(y: np.ndarray, ss_out: np.ndarray) -> None:
    """Compute batch summary statistics in-place."""
    ss_out[:, :] = np.asarray(summary_statistics_batch(y), dtype=np.float64)


def stats_fn_batch_fourier(
    y: np.ndarray,
    ss_out: np.ndarray,
    *,
    fourier_range: tuple[int, ...],
) -> None:
    """Compute batch Fourier summary statistics for a custom index range."""
    window = hann_window(y.shape[1])
    ss_out[:, :] = np.asarray(
        summary_statistics_batch(y, window=window, fourier_range=list(fourier_range)),
        dtype=np.float64,
    )


def build_stats_fn(*, fourier_range: tuple[int, ...] | None = None):
    """Create a picklable summary-statistics function."""
    if fourier_range is None:
        return stats_fn_batch
    return partial(stats_fn_batch_fourier, fourier_range=fourier_range)


def observed_summary_statistics(
    sn_data: np.ndarray,
    *,
    fourier_range: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Compute observed summary statistics."""
    window = None if fourier_range is None else hann_window(len(sn_data))
    indices = None if fourier_range is None else list(fourier_range)
    return np.asarray(
        summary_statistics(sn_data, window=window, fourier_range=indices),
        dtype=np.float64,
    ).reshape(-1)


def init_julia_quiet() -> None:
    """Initialize Julia in worker processes without printing the startup banner."""
    with redirect_stdout(io.StringIO()):
        init_julia()
