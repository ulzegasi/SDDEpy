"""
SABC inference with solar dynamo model
Data: observed SN record (yearly resolution)
Inference algorithm: single-epsilon SABC with DE proposal
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np

from process_fdist import make_process_f_dist
from sdde_model import init_julia
from solar_dynamo_obs_sn_problem import (
    build_simulator,
    init_julia_quiet,
    load_observed_sn,
    observed_summary_statistics,
    stats_fn_batch,
)


# ##### Cluster or Local? #####
run_on_cluster = 0  # 0 is local, 1 is cluster
if run_on_cluster not in (0, 1):
    raise ValueError("Set 'run_on_cluster' to 0 or 1")

# ##### First or update run? #####
from_previous = 0  # set 0 for first run, 1 for updating previous run
if from_previous not in (0, 1):
    raise ValueError("Set 'from_previous' to 0 or 1")

# ##### Parallel workers #####
# Use 1 for the original serial path, or >1 to use the process-backed Julia-safe path.
n_workers = 1
if n_workers < 1:
    raise ValueError("Set 'n_workers' to an integer >= 1")

PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_DATA_DIR = PROJECT_DIR / "data"
LOCAL_OUT_DIR = PROJECT_DIR / "output"


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
    if run_on_cluster == 0:
        datadir = LOCAL_DATA_DIR
        outdir = LOCAL_OUT_DIR
    else:
        datadir = Path("/cfs/earth/scratch/ulzg/julia/SABC/SDDEpy/data")
        outdir = Path("/cfs/earth/scratch/ulzg/julia/SABC/SDDEpy/output")

    if not datadir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {datadir}")
    outdir.mkdir(parents=True, exist_ok=True)
    return datadir, outdir


def main() -> None:
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
    SNyrs, SNdata, Tobs_without_warmup = load_observed_sn(datadir)

    # Problem-specific simulator/statistics functions live in an import-safe helper
    # module so they can be used safely from multiprocessing "spawn" workers.
    simulator = build_simulator(Twarmup=200, Tobs=Tobs_without_warmup)
    stats_fn = stats_fn_batch

    # Observed summary statistics.
    ss_obs = observed_summary_statistics(SNdata)
    n_stats = int(ss_obs.size)

    worker_backend = "process" if n_workers > 1 else "thread"

    if worker_backend == "process":
        f_dist = make_process_f_dist(
            n_samples=Tobs_without_warmup,
            ss_obs=ss_obs,
            simulator=simulator,
            stats_fn=stats_fn,
            seed=123,
            distance="abs",
            n_workers=n_workers,
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
            n_workers=n_workers,
            use_numba=False,
        )

    # Prior ranges (yearly resolution).
    lower = np.array([0.1, 0.1, 1.0, 0.01, 1.0], dtype=float)
    upper = np.array([10.0, 10.0, 15.0, 0.3, 15.0], dtype=float)
    prior = Prior(lower=lower, upper=upper)

    # SABC parameters (parity with Julia script).
    n_particles = 1_000
    n_simulation = 1_000_000

    rng_alg = np.random.default_rng(18)
    rng_prop = np.random.default_rng(22)

    proposal = DifferentialEvolution(n_para=lower.size, rng=rng_prop)
    config = SABCConfig(
        f_dist=f_dist,
        prior=prior,
        n_particles=n_particles,
        algorithm="single_eps",
        proposal=proposal,
        rng=rng_alg,
        show_checkpoint=50,
        show_progressbar=True,
        parallel_batches=False,
    )

    if from_previous == 0:
        fname = "obsSN_single_77_py"
        sabc_wallclock_start = time.perf_counter()
        out = sabc(config, n_simulation=n_simulation)
    else:
        fname = "obsSN_single_77_10kd"
        fname_previous = "obsSN_single_77_10kc"

        prev_path = outdir / f"SABCresult_{fname_previous}.pkl"
        if not prev_path.exists():
            raise FileNotFoundError(f"Previous result not found: {prev_path}")

        out_prev = load_sabc_result(prev_path)
        sabc_wallclock_start = time.perf_counter()
        out = update_population(out_prev, n_simulation=n_simulation)

    sabc_wallclock = time.perf_counter() - sabc_wallclock_start

    np.savetxt(outdir / f"post_population_{fname}.csv", out.population, delimiter=",")

    eps_hist = np.asarray(out.state.epsilon_history, dtype=float)
    rho_hist = np.asarray(out.state.rho_history, dtype=float)
    u_hist = np.asarray(out.state.u_history, dtype=float)

    np.savetxt(outdir / f"epsilon_history_{fname}.csv", eps_hist, delimiter=",")
    np.savetxt(outdir / f"rho_history_{fname}.csv", rho_hist, delimiter=",")
    np.savetxt(outdir / f"u_history_{fname}.csv", u_hist, delimiter=",")

    save_sabc_result(out, outdir / f"SABCresult_{fname}.pkl")

    print(f"Saved outputs to: {outdir}")
    print(f"Observed years used: {SNyrs[0]} - {SNyrs[-1]} (n={Tobs_without_warmup})")
    print(f"Number of observed summary stats: {n_stats}")
    print(f"n_workers used: {n_workers} ({worker_backend})")
    print(f"SABC wall-clock time: {sabc_wallclock:.2f} s")


if __name__ == "__main__":
    main()
