"""Importance-sampling-style filtering for saved SABC posterior populations.

Two filtering modes are supported for each saved run:
1. Reconstruct distances by resimulating the posterior sample and averaging
   the summary-statistic distances over repeated realizations.
2. Recover the final SABC distances saved by the inference run.

The cutoff is chosen automatically from a 1D KDE so that the retained mass is
approximately equal to a requested target (for example 70% kept / 30% dropped).
To keep the realized retained count close to the requested mass, the script
tries a small grid of KDE bandwidths and keeps the KDE-based cutoff whose
empirical retained fraction is closest to the target.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import gaussian_kde

from process_fdist import make_process_f_dist
from sdde_model import init_julia
from solar_dynamo_sabc_setup import (
    build_simulator,
    init_julia_quiet,
    load_dataset,
    observed_summary_statistics,
    stats_fn_batch,
)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"

VALID_DATASETS = ("obsSN", "C14")
VALID_ALGORITHMS = ("single", "multi")


class Prior:
    """Compatibility copy of the original run-script prior for unpickling.

    The filtering workflow only needs ``result.rho`` from the saved SABC result,
    not the prior itself. However, old pickles reference ``__main__.Prior``
    because they were created from the run script, so we provide the same class
    here to let pickle reconstruct the full object graph consistently.
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=(*VALID_DATASETS, "all"),
        default="all",
        help="Dataset to process.",
    )
    parser.add_argument(
        "--algorithm",
        choices=(*VALID_ALGORITHMS, "all"),
        default="all",
        help="Algorithm family encoded in the run name.",
    )
    parser.add_argument(
        "--tag",
        default="77py",
        help="Run suffix used in filenames like post_population_obsSN_single_77py.csv.",
    )
    parser.add_argument(
        "--run-name",
        action="append",
        default=[],
        help="Optional explicit run name(s). If given, dataset/algorithm expansion is skipped.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=50,
        help="Number of posterior resimulations per particle for reconstructed distances.",
    )
    parser.add_argument(
        "--keep-mass",
        type=float,
        default=0.70,
        help="Target mass to retain below the cutoff.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Number of worker processes for reconstructed distances.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Base random seed for reconstructed distances.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show histogram overlays interactively for each processed run.",
    )
    return parser.parse_args()


def _import_sabc_io():
    try:
        from simulated_annealing_abc import load_sabc_result
    except ModuleNotFoundError:
        candidate = PROJECT_DIR.parent / "SimulatedAnnealingABC" / "src"
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            from simulated_annealing_abc import load_sabc_result
        else:
            raise ModuleNotFoundError(
                "Could not import 'simulated_annealing_abc'. "
                "Install it with: pip install -e ../SimulatedAnnealingABC"
            )
    return load_sabc_result


def _expand_run_names(args: argparse.Namespace) -> list[str]:
    if args.run_name:
        return list(dict.fromkeys(args.run_name))

    datasets = VALID_DATASETS if args.dataset == "all" else (args.dataset,)
    algorithms = VALID_ALGORITHMS if args.algorithm == "all" else (args.algorithm,)
    return [f"{dataset}_{algorithm}_{args.tag}" for dataset in datasets for algorithm in algorithms]


def _dataset_from_run_name(run_name: str) -> str:
    for dataset in VALID_DATASETS:
        if run_name.startswith(f"{dataset}_"):
            return dataset
    raise ValueError(f"Could not infer dataset from run name '{run_name}'.")


def _load_population(run_name: str) -> np.ndarray:
    path = OUTPUT_DIR / f"post_population_{run_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Posterior population not found: {path}")

    population = np.loadtxt(path, delimiter=",", dtype=float)
    population = np.atleast_2d(population)
    if population.ndim != 2:
        raise ValueError(f"Expected a 2D posterior population in {path}.")
    return population


def _recover_sabc_rho(run_name: str, n_particles: int) -> np.ndarray:
    load_sabc_result = _import_sabc_io()
    pkl_path = OUTPUT_DIR / f"SABCresult_{run_name}.pkl"
    if pkl_path.exists():
        try:
            main_module = sys.modules.get("__main__")
            if main_module is not None and not hasattr(main_module, "Prior"):
                setattr(main_module, "Prior", Prior)
            result = load_sabc_result(pkl_path)
            rho = np.asarray(result.rho, dtype=float)
            rho = np.atleast_2d(rho)
            if rho.shape[0] != n_particles and rho.shape[1] == n_particles:
                rho = rho.T
            if rho.shape[0] != n_particles:
                raise ValueError(
                    f"Saved final rho in {pkl_path} has shape {rho.shape}, "
                    f"expected {n_particles} particles."
                )
            return rho
        except Exception as exc:
            raise RuntimeError(
                f"Could not recover final SABC rho from {pkl_path}. "
                f"rho_history is not a valid fallback here because it stores "
                f"population means, not the final particle-wise rho matrix. "
                f"Original error: {exc}"
            )
    raise FileNotFoundError(f"Final SABC result not found: {pkl_path}")


def _build_reconstruction_f_dist(dataset: str, n_workers: int, seed: int):
    _, obs_data, t_obs = load_dataset(dataset, DATA_DIR)
    ss_obs = observed_summary_statistics(obs_data)
    simulator = build_simulator(Twarmup=200, Tobs=t_obs)

    return make_process_f_dist(
        n_samples=t_obs,
        ss_obs=ss_obs,
        simulator=simulator,
        stats_fn=stats_fn_batch,
        seed=seed,
        distance="abs",
        n_workers=n_workers,
        worker_setup=init_julia_quiet,
        mp_start_method="spawn",
    )


def _reconstruct_rho(
    population: np.ndarray,
    *,
    run_name: str,
    dataset: str,
    n_repeats: int,
    n_workers: int,
    seed: int,
) -> np.ndarray:
    if n_repeats < 1:
        raise ValueError("--n-repeats must be >= 1.")

    f_dist = _build_reconstruction_f_dist(dataset, n_workers=n_workers, seed=seed)
    n_stats = int(f_dist.ss_obs.size)
    rho_sum = np.zeros((population.shape[0], n_stats), dtype=float)

    try:
        for rep in range(n_repeats):
            print(f"[{run_name}] reconstructing distances: repeat {rep + 1}/{n_repeats}", flush=True)
            rho_sum += f_dist(population)
    finally:
        del f_dist

    return rho_sum / float(n_repeats)


def _norms(rho: np.ndarray) -> np.ndarray:
    return np.linalg.norm(rho, axis=1)


@dataclass
class FilterResult:
    method: str
    n_particles: int
    keep_mass_target: float
    cutoff: float
    kept_count: int
    dropped_count: int
    kept_fraction_empirical: float
    dropped_fraction_empirical: float
    cutoff_method: str


def _kde_cutoff(values: np.ndarray, keep_mass: float) -> tuple[float, str]:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("Need at least one value to estimate a cutoff.")
    if not (0.0 < keep_mass < 1.0):
        raise ValueError("--keep-mass must lie strictly between 0 and 1.")
    if values.size == 1 or np.allclose(values, values[0]):
        return float(values[0]), "degenerate"

    std = float(np.std(values, ddof=1))
    span = float(np.max(values) - np.min(values))
    pad = max(5.0 * std, span, 1.0)
    hi = float(np.max(values) + pad)
    bandwidth_scales = (
        1.0,
        0.9,
        0.8,
        0.7,
        0.65,
        0.6,
        0.55,
        0.5,
        0.45,
        0.4,
        0.35,
        0.3,
        0.25,
        0.2,
        0.175,
        0.15,
        0.125,
        0.1,
    )
    best_cutoff: float | None = None
    best_method: str | None = None
    best_error = float("inf")

    for scale in bandwidth_scales:
        kde = gaussian_kde(values, bw_method=lambda kde, scale=scale: kde.scotts_factor() * scale)
        positive_mass = float(kde.integrate_box_1d(0.0, hi))
        if positive_mass <= 0.0:
            continue

        def objective(x: float) -> float:
            return float(kde.integrate_box_1d(0.0, x) / positive_mass - keep_mass)

        f_lo = objective(0.0)
        f_hi = objective(hi)
        if not (f_lo <= 0.0 <= f_hi):
            continue

        cutoff = float(brentq(objective, 0.0, hi, xtol=1e-10, rtol=1e-10))
        empirical_keep = float(np.mean(values <= cutoff))
        error = abs(empirical_keep - keep_mass)

        if error < best_error:
            best_cutoff = cutoff
            best_method = f"kde_cdf_bwscale_{scale:g}"
            best_error = error

        if error <= 0.01:
            return cutoff, f"kde_cdf_bwscale_{scale:g}"

    if best_cutoff is not None and best_method is not None:
        return best_cutoff, f"{best_method}_best_match"

    return float(np.quantile(values, keep_mass)), "empirical_quantile_fallback"


def _filter_from_norms(method: str, values: np.ndarray, keep_mass: float) -> tuple[np.ndarray, FilterResult]:
    cutoff, cutoff_method = _kde_cutoff(values, keep_mass)
    keep_mask = values <= cutoff
    kept_count = int(np.count_nonzero(keep_mask))
    n_particles = int(values.size)
    dropped_count = n_particles - kept_count

    result = FilterResult(
        method=method,
        n_particles=n_particles,
        keep_mass_target=float(keep_mass),
        cutoff=float(cutoff),
        kept_count=kept_count,
        dropped_count=dropped_count,
        kept_fraction_empirical=kept_count / n_particles,
        dropped_fraction_empirical=dropped_count / n_particles,
        cutoff_method=cutoff_method,
    )
    return keep_mask, result


def _save_arrays(run_name: str, method: str, population: np.ndarray, values: np.ndarray, keep_mask: np.ndarray) -> None:
    del population
    del values
    kept_indices = np.flatnonzero(keep_mask)
    safe_method = method.lower().replace(" ", "_")
    if safe_method == "reconstructed":
        safe_method = "reconst"
    np.savetxt(OUTPUT_DIR / f"kept_ind_{safe_method}_{run_name}.csv", kept_indices, delimiter=",", fmt="%d")


def _show_overlay_plot(run_name: str, reconstructed: np.ndarray, sabc: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        reconstructed,
        bins=30,
        alpha=0.4,
        label="Reconstructed distances",
        color="tab:blue",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.hist(
        sabc,
        bins=30,
        alpha=0.4,
        label="SABC distances",
        color="tab:red",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xlabel("Distance norm")
    ax.set_ylabel("Count")
    ax.set_title(f"Distance comparison: {run_name}")
    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def _process_one_run(args: argparse.Namespace, run_name: str) -> None:
    dataset = _dataset_from_run_name(run_name)
    population = _load_population(run_name)
    n_particles = population.shape[0]

    reconstructed_rho = _reconstruct_rho(
        population,
        run_name=run_name,
        dataset=dataset,
        n_repeats=args.n_repeats,
        n_workers=args.n_workers,
        seed=args.seed,
    )
    sabc_rho = _recover_sabc_rho(run_name, n_particles=n_particles)

    reconstructed_norms = _norms(reconstructed_rho)
    sabc_norms = _norms(sabc_rho)

    reconstructed_keep, reconstructed_result = _filter_from_norms(
        "reconstructed",
        reconstructed_norms,
        args.keep_mass,
    )
    sabc_keep, sabc_result = _filter_from_norms(
        "sabc",
        sabc_norms,
        args.keep_mass,
    )

    _save_arrays(run_name, "reconstructed", population, reconstructed_norms, reconstructed_keep)
    _save_arrays(run_name, "sabc", population, sabc_norms, sabc_keep)

    if args.show_plots:
        _show_overlay_plot(run_name, reconstructed_norms, sabc_norms)

    print(
        f"[{run_name}] retained particles: "
        f"reconst = {reconstructed_result.kept_count}/{reconstructed_result.n_particles}, "
        f"sabc = {sabc_result.kept_count}/{sabc_result.n_particles}"
    )


def main() -> None:
    args = _parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    init_julia()

    for run_name in _expand_run_names(args):
        _process_one_run(args, run_name)


if __name__ == "__main__":
    main()
