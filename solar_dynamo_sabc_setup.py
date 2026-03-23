"""Import-safe helpers for solar-dynamo SABC problems."""

from __future__ import annotations

from contextlib import redirect_stdout
from functools import partial
from pathlib import Path
import io

import numpy as np

from sdde_model import init_julia, sn_batch, summary_statistics, summary_statistics_batch

DatasetName = str


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


def load_dataset(dataset: DatasetName, datadir: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Load one of the supported solar-dynamo datasets."""
    if dataset == "obsSN":
        return load_observed_sn(datadir)
    if dataset == "C14":
        return load_c14_sn(datadir)
    raise ValueError(f"Unknown dataset '{dataset}'. Expected 'obsSN' or 'C14'.")


def simulator_batch(
    theta: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    *,
    Twarmup: int,
    Tobs: int,
) -> None:
    """Simulate yearly sunspot traces for a batch of parameters."""
    theta = np.asarray(theta, dtype=float)
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=theta.shape[0], dtype=np.int64)
    y_sim = sn_batch(theta, Twarmup=Twarmup, Tobs=Tobs, seeds=seeds)
    y[:, :] = np.asarray(y_sim, dtype=np.float64)


def build_simulator(*, Twarmup: int, Tobs: int):
    """Create a picklable simulator with fixed warmup and observation length."""
    return partial(simulator_batch, Twarmup=Twarmup, Tobs=Tobs)


def stats_fn_batch(y: np.ndarray, ss_out: np.ndarray) -> None:
    """Compute batch summary statistics in-place."""
    ss_out[:, :] = np.asarray(summary_statistics_batch(y), dtype=np.float64)


def observed_summary_statistics(sn_data: np.ndarray) -> np.ndarray:
    """Compute observed summary statistics."""
    return np.asarray(summary_statistics(sn_data), dtype=np.float64).reshape(-1)


def init_julia_quiet() -> None:
    """Initialize Julia in worker processes without printing the startup banner."""
    with redirect_stdout(io.StringIO()):
        init_julia()
