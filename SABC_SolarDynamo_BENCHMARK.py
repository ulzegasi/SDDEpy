"""
Benchmark driver for SABC solar dynamo runs.

Default benchmark settings:
- dataset: "obsSN"
- algorithm: "single_eps"
- n_particles: 1000
- n_simulation: 5_000_000
- no posterior outputs are written

Each completed run appends one row to a benchmark CSV so multiple Slurm jobs
with different ``n_workers`` values can write to the same file safely.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
import time

import fcntl
import numpy as np

from process_fdist import make_process_f_dist
from sdde_model import init_julia
from solar_dynamo_sabc_setup import (
    build_simulator,
    init_julia_quiet,
    load_dataset,
    observed_summary_statistics,
    stats_fn_batch,
)


N_WORKERS = 4
DATASET = "obsSN"
ALGORITHM = "single_eps"
PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_DATA_DIR = PROJECT_DIR / "data"
LOCAL_OUT_DIR = PROJECT_DIR / "output"
VALID_DATASETS = ("obsSN", "C14")
VALID_ALGORITHMS = ("single_eps", "multi_eps")
VALID_WORKER_COUNTS = (1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32)
N_PARTICLES = 1_000
N_SIMULATION = 5_000_000
SHOW_CHECKPOINT = 500


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=VALID_DATASETS, default=DATASET)
    parser.add_argument("--algorithm", choices=VALID_ALGORITHMS, default=ALGORITHM)
    parser.add_argument("--n-workers", type=int, default=N_WORKERS)
    parser.add_argument("--benchmark-file", default=None)
    return parser.parse_args()


def _algorithm_label(algorithm: str) -> str:
    return "single" if algorithm == "single_eps" else "multi"


def _default_benchmark_path(dataset: str, algorithm: str) -> Path:
    return LOCAL_OUT_DIR / f"benchmark_{dataset}_{_algorithm_label(algorithm)}.csv"


def _import_sabc_package():
    """Import simulated_annealing_abc, with local sibling-repo fallback."""
    try:
        from simulated_annealing_abc import DifferentialEvolution, SABCConfig, make_f_dist, sabc
    except ModuleNotFoundError:
        candidate = PROJECT_DIR.parent / "SimulatedAnnealingABC" / "src"
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            from simulated_annealing_abc import (
                DifferentialEvolution,
                SABCConfig,
                make_f_dist,
                sabc,
            )
        else:
            raise ModuleNotFoundError(
                "Could not import 'simulated_annealing_abc'. "
                "Install it with: pip install -e ../SimulatedAnnealingABC"
            )

    return DifferentialEvolution, SABCConfig, make_f_dist, sabc


class Prior:
    """Independent uniform prior on (tau, T, Nd, sigma, Bmax)."""

    def __init__(self, lower: np.ndarray, upper: np.ndarray):
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        if self.lower.shape != self.upper.shape:
            raise ValueError("lower and upper must have same shape")
        if np.any(self.upper <= self.lower):
            raise ValueError("Each upper bound must be greater than lower bound")
        self._log_volume = float(np.sum(np.log(self.upper - self.lower)))

    def rvs(self, rng: np.random.Generator, size: int = 1) -> np.ndarray:
        return rng.uniform(self.lower, self.upper, size=(size, self.lower.size))

    def logpdf(self, theta: np.ndarray) -> np.ndarray:
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        in_bounds = np.all((theta >= self.lower) & (theta <= self.upper), axis=1)
        lp = np.full(theta.shape[0], -np.inf, dtype=float)
        lp[in_bounds] = -self._log_volume
        return lp


def _resolve_paths() -> tuple[Path, Path]:
    datadir = LOCAL_DATA_DIR
    outdir = LOCAL_OUT_DIR
    if not datadir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {datadir}")
    outdir.mkdir(parents=True, exist_ok=True)
    return datadir, outdir


def _append_benchmark_row(csv_path: Path, row: dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    fieldnames = [
        "run_id",
        "dataset",
        "algorithm",
        "n_workers",
        "n_particles",
        "n_simulation",
        "elapsed_time_seconds",
    ]

    with csv_path.open("a+", newline="") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0, os.SEEK_END)
        empty_file = handle.tell() == 0
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if empty_file or not file_exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def main() -> None:
    args = _parse_args()
    if args.n_workers < 1:
        raise ValueError("Set --n-workers to an integer >= 1")

    init_julia()  # Julia must be initialized early in the parent process.

    DifferentialEvolution, SABCConfig, make_f_dist, sabc = _import_sabc_package()

    datadir, outdir = _resolve_paths()
    benchmark_path = Path(args.benchmark_file) if args.benchmark_file else _default_benchmark_path(
        args.dataset, args.algorithm
    )

    SNyrs, SNdata, Tobs_without_warmup = load_dataset(args.dataset, datadir)

    simulator = build_simulator(Twarmup=200, Tobs=Tobs_without_warmup)
    stats_fn = stats_fn_batch
    ss_obs = observed_summary_statistics(SNdata)
    n_stats = int(ss_obs.size)

    worker_backend = "process" if args.n_workers > 1 else "thread"

    if worker_backend == "process":
        f_dist = make_process_f_dist(
            n_samples=Tobs_without_warmup,
            ss_obs=ss_obs,
            simulator=simulator,
            stats_fn=stats_fn,
            seed=123,
            distance="abs",
            n_workers=args.n_workers,
            worker_setup=init_julia_quiet,
            mp_start_method="spawn",
        )
    else:
        f_dist = make_f_dist(
            n_samples=Tobs_without_warmup,
            ss_obs=ss_obs,
            simulator=simulator,
            stats_fn=stats_fn,
            seed=123,
            distance="abs",
            n_workers=args.n_workers,
            use_numba=False,
        )

    lower = np.array([0.1, 0.1, 1.0, 0.01, 1.0], dtype=float)
    upper = np.array([10.0, 10.0, 15.0, 0.3, 15.0], dtype=float)
    prior = Prior(lower=lower, upper=upper)

    rng_alg = np.random.default_rng(18)
    rng_prop = np.random.default_rng(22)

    proposal = DifferentialEvolution(n_para=lower.size, rng=rng_prop)
    config = SABCConfig(
        f_dist=f_dist,
        prior=prior,
        n_particles=N_PARTICLES,
        algorithm=args.algorithm,
        proposal=proposal,
        rng=rng_alg,
        show_checkpoint=SHOW_CHECKPOINT,
        show_progressbar=True,
        parallel_batches=False,
    )

    run_id = os.environ.get("RUN_ID", "")

    print("---------------------------------------------------")
    print(f"Benchmark dataset: {args.dataset}")
    print(f"Benchmark algorithm: {args.algorithm}")
    print(f"Benchmark n_workers: {args.n_workers}")
    print(f"Benchmark n_particles: {N_PARTICLES}")
    print(f"Benchmark n_simulation: {N_SIMULATION}")
    print(f"Observed years used: {SNyrs[0]} - {SNyrs[-1]} (n={Tobs_without_warmup})")
    print(f"Number of observed summary stats: {n_stats}")
    print(f"Benchmark CSV: {benchmark_path}")
    print("---------------------------------------------------")

    sabc_wallclock_start = time.perf_counter()
    _ = sabc(config, n_simulation=N_SIMULATION)
    sabc_wallclock = time.perf_counter() - sabc_wallclock_start

    _append_benchmark_row(
        benchmark_path,
        {
            "run_id": run_id,
            "dataset": args.dataset,
            "algorithm": args.algorithm,
            "n_workers": args.n_workers,
            "n_particles": N_PARTICLES,
            "n_simulation": N_SIMULATION,
            "elapsed_time_seconds": f"{sabc_wallclock:.6f}",
        },
    )

    print("---------------------------------------------------")
    print("Benchmark finished successfully")
    print(f"n_workers used: {args.n_workers} ({worker_backend})")
    print(f"SABC wall-clock time: {sabc_wallclock:.2f} s")
    print(f"Appended benchmark row to: {benchmark_path}")
    print("---------------------------------------------------")


if __name__ == "__main__":
    main()
