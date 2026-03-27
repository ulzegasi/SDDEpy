"""
Unified SABC inference driver for the solar dynamo model.

Supports:
- datasets: "obsSN" or "C14"
- algorithms: "single_eps" or "multi_eps"
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

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


# ##### First or update run? #####
FROM_PREVIOUS = 0  # set 0 for first run, 1 for updating previous run

# ##### Parallel workers #####
# Use 1 for the original serial path, or >1 to use the process-backed Julia-safe path.
N_WORKERS = 4

# ##### Problem selection #####
DATASET = "obsSN"  # "obsSN" or "C14"
ALGORITHM = "single_eps"  # "single_eps" or "multi_eps"

# ##### Output naming #####
RUN_NAME = None
PREVIOUS_RUN_NAME = None

PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_DATA_DIR = PROJECT_DIR / "data"
LOCAL_OUT_DIR = PROJECT_DIR / "output"
VALID_DATASETS = ("obsSN", "C14")
VALID_ALGORITHMS = ("single_eps", "multi_eps")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=VALID_DATASETS, default=DATASET)
    parser.add_argument("--algorithm", choices=VALID_ALGORITHMS, default=ALGORITHM)
    parser.add_argument("--from-previous", type=int, choices=(0, 1), default=FROM_PREVIOUS)
    parser.add_argument("--n-workers", type=int, default=N_WORKERS)
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--previous-run-name", default=PREVIOUS_RUN_NAME)
    return parser.parse_args()


def _default_run_name(dataset: str, algorithm: str) -> str:
    return f"{dataset}_{algorithm}"


def _resolve_run_names(args: argparse.Namespace) -> tuple[str, str | None]:
    run_name = args.run_name or _default_run_name(args.dataset, args.algorithm)
    previous_run_name = args.previous_run_name
    if args.from_previous == 1 and not previous_run_name:
        raise ValueError(
            "Set --previous-run-name when --from-previous=1 so the script knows which "
            "saved population to continue."
        )
    return run_name, previous_run_name


def _import_sabc_package():
    """Import simulated_annealing_abc, with local sibling-repo fallback."""
    try:
        from simulated_annealing_abc import (
            DifferentialEvolution,
            SABCConfig,
            load_sabc_result,
            make_f_dist,
            sabc,
            save_sabc_result,
            update_population,
        )
    except ModuleNotFoundError:
        candidate = PROJECT_DIR.parent / "SimulatedAnnealingABC" / "src"
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            from simulated_annealing_abc import (
                DifferentialEvolution,
                SABCConfig,
                load_sabc_result,
                make_f_dist,
                sabc,
                save_sabc_result,
                update_population,
            )
        else:
            raise ModuleNotFoundError(
                "Could not import 'simulated_annealing_abc'. "
                "Install it with: pip install -e ../SimulatedAnnealingABC"
            )

    return (
        DifferentialEvolution,
        SABCConfig,
        load_sabc_result,
        make_f_dist,
        sabc,
        save_sabc_result,
        update_population,
    )


class Prior:
    """
    Independent uniform prior on (tau, T, Nd, sigma, Bmax).

    Batch API:
      - ``rvs(rng, size=n_particles)`` -> ``(n_particles, 5)``
      - ``logpdf(theta_batch)`` -> ``(n_particles,)`` where ``theta_batch`` is ``(n_particles, 5)``
    """

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


def _format_elapsed_dd_hh_mm(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    days, remainder = divmod(total_seconds, 24 * 60 * 60)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, _ = divmod(remainder, 60)
    return f"{days:02d}:{hours:02d}:{minutes:02d}"


def main() -> None:
    args = _parse_args()
    if args.n_workers < 1:
        raise ValueError("Set --n-workers to an integer >= 1")

    run_name, previous_run_name = _resolve_run_names(args)

    init_julia()  # Julia must be initialized early in the parent process.

    (
        DifferentialEvolution,
        SABCConfig,
        load_sabc_result,
        make_f_dist,
        sabc,
        save_sabc_result,
        update_population,
    ) = _import_sabc_package()

    datadir, outdir = _resolve_paths()
    SNyrs, SNdata, Tobs_without_warmup = load_dataset(args.dataset, datadir)

    # Problem-specific simulator/statistics functions live in an import-safe helper
    # module so they can be used safely from multiprocessing "spawn" workers.
    simulator = build_simulator(Twarmup=200, Tobs=Tobs_without_warmup)
    stats_fn = stats_fn_batch

    # Observed summary statistics.
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

    # Prior ranges (yearly resolution).
    lower = np.array([0.1, 0.1, 1.0, 0.01, 1.0], dtype=float)
    upper = np.array([10.0, 10.0, 15.0, 0.3, 15.0], dtype=float)
    prior = Prior(lower=lower, upper=upper)

    # SABC parameters (parity with Julia script).
    n_particles = 1_000
    n_simulation = 1_000_000_000

    rng_alg = np.random.default_rng(18)
    rng_prop = np.random.default_rng(22)

    proposal = DifferentialEvolution(n_para=lower.size, rng=rng_prop)
    config = SABCConfig(
        f_dist=f_dist,
        prior=prior,
        n_particles=n_particles,
        algorithm=args.algorithm,
        proposal=proposal,
        rng=rng_alg,
        show_checkpoint=500,
        show_progressbar=True,
        parallel_batches=False,
    )

    if args.from_previous == 0:
        sabc_wallclock_start = time.perf_counter()
        out = sabc(config, n_simulation=n_simulation)
    else:
        prev_path = outdir / f"SABCresult_{previous_run_name}.pkl"
        if not prev_path.exists():
            raise FileNotFoundError(f"Previous result not found: {prev_path}")

        out_prev = load_sabc_result(prev_path)
        sabc_wallclock_start = time.perf_counter()
        out = update_population(out_prev, n_simulation=n_simulation)

    sabc_wallclock = time.perf_counter() - sabc_wallclock_start

    np.savetxt(outdir / f"post_population_{run_name}.csv", out.population, delimiter=",")

    eps_hist = np.asarray(out.state.epsilon_history, dtype=float)
    rho_hist = np.asarray(out.state.rho_history, dtype=float)
    u_hist = np.asarray(out.state.u_history, dtype=float)

    np.savetxt(outdir / f"epsilon_history_{run_name}.csv", eps_hist, delimiter=",")
    np.savetxt(outdir / f"rho_history_{run_name}.csv", rho_hist, delimiter=",")
    np.savetxt(outdir / f"u_history_{run_name}.csv", u_hist, delimiter=",")

    save_sabc_result(out, outdir / f"SABCresult_{run_name}.pkl")

    print(f"Saved outputs to: {outdir}")
    print(f"Dataset used: {args.dataset}")
    print(f"Algorithm used: {args.algorithm}")
    print(f"Observed years used: {SNyrs[0]} - {SNyrs[-1]} (n={Tobs_without_warmup})")
    print(f"Number of observed summary stats: {n_stats}")
    print(f"n_workers used: {args.n_workers} ({worker_backend})")
    print(f"Run name: {run_name}")
    if previous_run_name is not None:
        print(f"Previous run name: {previous_run_name}")
    print(f"SABC wall-clock time (DD:HH:MM): {_format_elapsed_dd_hh_mm(sabc_wallclock)}")


if __name__ == "__main__":
    main()
