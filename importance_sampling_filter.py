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
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import gaussian_kde

from enca_summary_stats import (
    build_enca_fft_cnn_summary_stats,
    build_enca_summary_stats,
    build_fno_summary_stats,
    build_mlp_summary_stats,
)
from process_fdist import make_process_f_dist, make_process_sim_then_stats_f_dist
from sdde_model import init_julia
from solar_dynamo_sabc_setup import (
    build_stats_fn,
    build_simulator,
    init_julia_quiet,
    load_dataset,
    observed_summary_statistics,
)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic_data"
OUTPUT_DIR = PROJECT_DIR / "output"

DEFAULT_SYNTHETIC_DATA_FILE = "sn_t6_T7_N12_s002_B8_tobs271_seed1822.csv"
VALID_DATASETS = ("obsSN", "C14", "synthetic")
DEFAULT_DATASETS = ("obsSN", "C14")
VALID_ALGORITHMS = ("single", "multi")
VALID_SUMMARY_STATS = ("fft", "enca", "mlp", "enca_fft_cnn", "fno")


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
        help=(
            "Run suffix used in filenames like post_population_obsSN_single_77py.csv. "
            "Ignored when --run-name is supplied."
        ),
    )
    parser.add_argument(
        "--synthetic-data-file",
        default=DEFAULT_SYNTHETIC_DATA_FILE,
        help=(
            "Synthetic CSV path, or filename under data/synthetic_data, used for "
            "synthetic runs."
        ),
    )
    parser.add_argument(
        "--summary-stats",
        choices=VALID_SUMMARY_STATS,
        default="fft",
        help=(
            "Summary-statistics backend used to reconstruct distances. Must match "
            "the backend used by the original SABC inference run."
        ),
    )
    parser.add_argument(
        "--fourier-range",
        default=None,
        help=(
            "Optional 1-based Fourier indices for --summary-stats fft. Accepts "
            "Julia-style start:step:stop, e.g. 1:6:120, or a Python-style list."
        ),
    )
    parser.add_argument(
        "--train-run-dir",
        "--enca-run-dir",
        dest="train_run_dir",
        default=None,
        help="Training run directory required for a neural summary-statistics backend.",
    )
    parser.add_argument(
        "--enca-checkpoint-basename",
        default="model_best_ckpt",
        help="Checkpoint basename to load for a neural summary-statistics backend.",
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
    args = parser.parse_args()
    if args.summary_stats in ("enca", "mlp", "enca_fft_cnn", "fno") and args.train_run_dir is None:
        parser.error(f"--summary-stats {args.summary_stats} requires --train-run-dir")
    if args.summary_stats in ("enca", "mlp", "enca_fft_cnn", "fno") and args.fourier_range is not None:
        parser.error("--fourier-range can only be used with --summary-stats fft")
    return args


def _parse_fourier_range(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None

    value = value.strip()
    if not value:
        raise ValueError("--fourier-range cannot be empty")

    if ":" in value:
        parts = value.split(":")
        if len(parts) != 3:
            raise ValueError("Use --fourier-range start:step:stop, for example 1:6:120")
        start, step, stop = (int(part.strip()) for part in parts)
        if step == 0:
            raise ValueError("--fourier-range step cannot be 0")
        if (stop - start) * step < 0:
            raise ValueError("--fourier-range step points away from stop")

        indices = []
        current = start
        if step > 0:
            while current <= stop:
                indices.append(current)
                current += step
        else:
            while current >= stop:
                indices.append(current)
                current += step
    else:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, int):
            indices = [parsed]
        else:
            indices = list(parsed)

    fourier_range = tuple(int(index) for index in indices)
    if not fourier_range:
        raise ValueError("--fourier-range must contain at least one index")
    if any(index < 1 for index in fourier_range):
        raise ValueError("--fourier-range uses Julia/1-based indices, so all values must be >= 1")
    return fourier_range


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

    datasets = DEFAULT_DATASETS if args.dataset == "all" else (args.dataset,)
    algorithms = VALID_ALGORITHMS if args.algorithm == "all" else (args.algorithm,)
    return [f"{dataset}_{algorithm}_{args.tag}" for dataset in datasets for algorithm in algorithms]


def _dataset_from_run_name(run_name: str) -> str:
    for dataset in VALID_DATASETS:
        if run_name.startswith(f"{dataset}_"):
            return dataset
    if run_name.startswith("synthetic"):
        return "synthetic"
    raise ValueError(f"Could not infer dataset from run name '{run_name}'.")


def _resolve_synthetic_data_path(filename_or_path: str | None) -> Path | None:
    if filename_or_path is None:
        return None

    path = Path(filename_or_path).expanduser()
    if path.is_absolute() or path.parent != Path("."):
        return path
    return SYNTHETIC_DATA_DIR / path


def _load_population(run_name: str) -> np.ndarray:
    path = OUTPUT_DIR / f"post_population_{run_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Posterior population not found: {path}")

    population = np.loadtxt(path, delimiter=",", dtype=float)
    population = np.atleast_2d(population)
    if population.ndim != 2:
        raise ValueError(f"Expected a 2D posterior population in {path}.")
    return population


def _model_from_population(population: np.ndarray) -> str:
    """Infer the forward model from the saved parameter dimension."""
    n_parameters = population.shape[1]
    if n_parameters == 5:
        return "original"
    if n_parameters == 6:
        return "jupiter"
    raise ValueError(
        f"Cannot infer a forward model from a population with {n_parameters} parameters; "
        "expected 5 (original) or 6 (jupiter)."
    )


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


def _build_reconstruction_f_dist(
    dataset: str,
    model: str,
    n_workers: int,
    seed: int,
    synthetic_data_path: Path | None,
    summary_stats: str,
    fourier_range: tuple[int, ...] | None,
    train_run_dir: str | None,
    enca_checkpoint_basename: str,
):
    _, obs_data, t_obs = load_dataset(dataset, DATA_DIR, synthetic_data_path=synthetic_data_path)
    simulator = build_simulator(Twarmup=200, Tobs=t_obs, model=model)

    if summary_stats == "fft":
        stats_fn = build_stats_fn(fourier_range=fourier_range)
        ss_obs = observed_summary_statistics(obs_data, fourier_range=fourier_range)
    elif summary_stats == "enca":
        enca_stats = build_enca_summary_stats(
            run_dir=train_run_dir,
            checkpoint_basename=enca_checkpoint_basename,
            expected_tobs=t_obs,
        )
        if enca_stats.config.representation_mode != "time":
            raise ValueError(
                "--summary-stats enca requires an original time-series ENCA run "
                f"(representation_mode='time'); got {enca_stats.config.representation_mode!r}. "
                "Use --summary-stats mlp for Fourier/MLP ENCA runs."
            )
        stats_fn = enca_stats.batch
        ss_obs = enca_stats.observed(obs_data)
    elif summary_stats == "mlp":
        mlp_stats = build_mlp_summary_stats(
            run_dir=train_run_dir,
            checkpoint_basename=enca_checkpoint_basename,
            expected_tobs=t_obs,
        )
        stats_fn = mlp_stats.batch
        ss_obs = mlp_stats.observed(obs_data)
    elif summary_stats == "enca_fft_cnn":
        enca_fft_cnn_stats = build_enca_fft_cnn_summary_stats(
            run_dir=train_run_dir,
            checkpoint_basename=enca_checkpoint_basename,
            expected_tobs=t_obs,
        )
        stats_fn = enca_fft_cnn_stats.batch
        ss_obs = enca_fft_cnn_stats.observed(obs_data)
    elif summary_stats == "fno":
        fno_stats = build_fno_summary_stats(
            run_dir=train_run_dir,
            checkpoint_basename=enca_checkpoint_basename,
            expected_tobs=t_obs,
        )
        stats_fn = fno_stats.batch
        ss_obs = fno_stats.observed(obs_data)
    else:
        raise ValueError(f"Unknown summary-statistics backend: {summary_stats}")

    make_process_distance = (
        make_process_sim_then_stats_f_dist
        if summary_stats == "fno"
        else make_process_f_dist
    )
    return make_process_distance(
        n_samples=t_obs,
        ss_obs=ss_obs,
        simulator=simulator,
        stats_fn=stats_fn,
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
    model: str,
    synthetic_data_path: Path | None,
    summary_stats: str,
    fourier_range: tuple[int, ...] | None,
    train_run_dir: str | None,
    enca_checkpoint_basename: str,
    n_repeats: int,
    n_workers: int,
    seed: int,
) -> np.ndarray:
    if n_repeats < 1:
        raise ValueError("--n-repeats must be >= 1.")

    f_dist = _build_reconstruction_f_dist(
        dataset,
        model=model,
        n_workers=n_workers,
        seed=seed,
        synthetic_data_path=synthetic_data_path,
        summary_stats=summary_stats,
        fourier_range=fourier_range,
        train_run_dir=train_run_dir,
        enca_checkpoint_basename=enca_checkpoint_basename,
    )
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
    fourier_range = _parse_fourier_range(args.fourier_range)
    synthetic_data_path = None
    if dataset == "synthetic":
        synthetic_data_path = _resolve_synthetic_data_path(args.synthetic_data_file)
        if synthetic_data_path is None:
            raise ValueError("Synthetic runs require --synthetic-data-file.")

    population = _load_population(run_name)
    n_particles = population.shape[0]
    model = _model_from_population(population)

    print(
        f"[{run_name}] reconstructing with model={model}, summary_stats={args.summary_stats}",
        flush=True,
    )
    reconstructed_rho = _reconstruct_rho(
        population,
        run_name=run_name,
        dataset=dataset,
        model=model,
        synthetic_data_path=synthetic_data_path,
        summary_stats=args.summary_stats,
        fourier_range=fourier_range,
        train_run_dir=args.train_run_dir,
        enca_checkpoint_basename=args.enca_checkpoint_basename,
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
