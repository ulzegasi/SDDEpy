"""Process-based batch distance function for non-thread-safe simulators."""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

DistanceMode = Literal["abs", "sq", "weighted_sq"]
SimulatorFn = Callable[[np.ndarray, np.ndarray, np.random.Generator], None]
StatsFn = Callable[[np.ndarray, np.ndarray], None]
InitFn = Callable[[], None]

_SIMULATOR: SimulatorFn | None = None
_STATS_FN: StatsFn | None = None
_N_SAMPLES: int | None = None
_N_STATS: int | None = None


def _init_process_worker(
    simulator: SimulatorFn,
    stats_fn: StatsFn,
    n_samples: int,
    n_stats: int,
    worker_setup: InitFn | None,
) -> None:
    """Bind worker-local globals once per child process."""
    global _SIMULATOR, _STATS_FN, _N_SAMPLES, _N_STATS
    _SIMULATOR = simulator
    _STATS_FN = stats_fn
    _N_SAMPLES = n_samples
    _N_STATS = n_stats

    if worker_setup is not None:
        worker_setup()


def _run_process_chunk(theta_chunk: np.ndarray, seed: int) -> np.ndarray:
    """Run simulator + summary stats for one chunk inside a worker process."""
    if _SIMULATOR is None or _STATS_FN is None or _N_SAMPLES is None or _N_STATS is None:
        raise RuntimeError("Process worker was not initialized.")

    theta_chunk = np.asarray(theta_chunk, dtype=np.float64)
    n_chunk = theta_chunk.shape[0]
    y = np.empty((n_chunk, _N_SAMPLES), dtype=np.float64)
    ss = np.empty((n_chunk, _N_STATS), dtype=np.float64)
    rng = np.random.default_rng(seed)

    _SIMULATOR(theta_chunk, y, rng)
    _STATS_FN(y, ss)
    return ss


def _init_process_sim_worker(
    simulator: SimulatorFn,
    n_samples: int,
    worker_setup: InitFn | None,
) -> None:
    """Bind only the simulator in workers; summary stats run in the parent."""
    global _SIMULATOR, _STATS_FN, _N_SAMPLES, _N_STATS
    _SIMULATOR = simulator
    _STATS_FN = None
    _N_SAMPLES = n_samples
    _N_STATS = None

    if worker_setup is not None:
        worker_setup()


def _run_process_sim_chunk(theta_chunk: np.ndarray, seed: int) -> np.ndarray:
    """Run only the simulator for one chunk inside a worker process."""
    if _SIMULATOR is None or _N_SAMPLES is None:
        raise RuntimeError("Process simulator worker was not initialized.")

    theta_chunk = np.asarray(theta_chunk, dtype=np.float64)
    y = np.empty((theta_chunk.shape[0], _N_SAMPLES), dtype=np.float64)
    rng = np.random.default_rng(seed)

    _SIMULATOR(theta_chunk, y, rng)
    return y


@dataclass
class ProcessFDist:
    """Picklable process-backed batch distance function."""

    n_samples: int
    ss_obs: np.ndarray
    simulator: SimulatorFn
    stats_fn: StatsFn
    seed: int | None = None
    distance: DistanceMode = "abs"
    weights: np.ndarray | None = None
    n_workers: int = 1
    worker_setup: InitFn | None = None
    mp_start_method: str = "spawn"

    _rng: np.random.Generator | None = field(init=False, repr=False, compare=False, default=None)
    _ctx: object | None = field(init=False, repr=False, compare=False, default=None)
    _pool: object | None = field(init=False, repr=False, compare=False, default=None)
    _transform: Callable[[np.ndarray], None] | None = field(
        init=False,
        repr=False,
        compare=False,
        default=None,
    )

    def __post_init__(self) -> None:
        self.ss_obs = np.asarray(self.ss_obs, dtype=np.float64).reshape(-1)
        n_stats = self.ss_obs.size

        if self.distance == "weighted_sq":
            if self.weights is None:
                raise ValueError("weights must be provided when distance='weighted_sq'.")
            self.weights = np.asarray(self.weights, dtype=np.float64).reshape(-1)
            if self.weights.shape != (n_stats,):
                raise ValueError(f"weights must have shape ({n_stats},), got {self.weights.shape}.")
        elif self.distance not in ("abs", "sq"):
            raise ValueError(f"Unknown distance='{self.distance}'.")

        if self.n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {self.n_workers}.")

        self._rng = np.random.default_rng(self.seed)
        self._ctx = None
        self._pool = None
        self._bind_transform()

    def _bind_transform(self) -> None:
        if self.distance == "abs":
            self._transform = self._transform_abs
        elif self.distance == "sq":
            self._transform = self._transform_sq
        else:
            self._transform = self._transform_weighted_sq

    def _ensure_pool(self):
        if self._pool is None:
            self._ctx = mp.get_context(self.mp_start_method)
            self._pool = self._ctx.Pool(
                processes=self.n_workers,
                initializer=_init_process_worker,
                initargs=(
                    self.simulator,
                    self.stats_fn,
                    self.n_samples,
                    self.ss_obs.size,
                    self.worker_setup,
                ),
            )
        return self._pool

    def __call__(self, theta: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        n_batch_particles = theta.shape[0]
        n_stats = self.ss_obs.size

        if out is None:
            out = np.empty((n_batch_particles, n_stats), dtype=np.float64)
        else:
            if out.shape != (n_batch_particles, n_stats):
                raise ValueError(
                    f"out must have shape ({n_batch_particles}, {n_stats}), got {out.shape}."
                )
            if out.dtype != np.float64:
                raise ValueError(f"out must have dtype float64, got {out.dtype}.")

        if self.n_workers <= 1:
            y = np.empty((n_batch_particles, self.n_samples), dtype=np.float64)
            ss = np.empty((n_batch_particles, n_stats), dtype=np.float64)
            self.simulator(theta, y, self._rng)
            self.stats_fn(y, ss)
            np.subtract(ss, self.ss_obs, out=out)
            self._transform(out)  # type: ignore[misc]
            return out

        pool = self._ensure_pool()
        boundaries = np.linspace(0, n_batch_particles, self.n_workers + 1, dtype=int)
        tasks: list[tuple[np.ndarray, int]] = []
        slices: list[tuple[int, int]] = []

        for k in range(self.n_workers):
            lo = int(boundaries[k])
            hi = int(boundaries[k + 1])
            if hi <= lo:
                continue
            seed = int(self._rng.integers(0, np.iinfo(np.int32).max, dtype=np.int64))
            tasks.append((theta[lo:hi].copy(), seed))
            slices.append((lo, hi))

        ss_combined = np.empty((n_batch_particles, n_stats), dtype=np.float64)
        ss_chunks = pool.starmap(_run_process_chunk, tasks)
        for (lo, hi), ss_chunk in zip(slices, ss_chunks, strict=True):
            ss_combined[lo:hi] = ss_chunk

        np.subtract(ss_combined, self.ss_obs, out=out)
        self._transform(out)  # type: ignore[misc]
        return out

    def clone(self, seed: int | None = None) -> "ProcessFDist":
        return ProcessFDist(
            n_samples=self.n_samples,
            ss_obs=self.ss_obs.copy(),
            simulator=self.simulator,
            stats_fn=self.stats_fn,
            seed=seed,
            distance=self.distance,
            weights=self.weights.copy() if self.weights is not None else None,
            n_workers=self.n_workers,
            worker_setup=self.worker_setup,
            mp_start_method=self.mp_start_method,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_ctx"] = None
        state["_pool"] = None
        state["_rng"] = None
        state["_transform"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._rng = np.random.default_rng(self.seed)
        self._ctx = None
        self._pool = None
        self._bind_transform()

    def __del__(self) -> None:
        pool = getattr(self, "_pool", None)
        if pool is not None:
            pool.close()
            pool.terminate()
            pool.join()

    @staticmethod
    def _transform_abs(out: np.ndarray) -> None:
        np.abs(out, out=out)

    @staticmethod
    def _transform_sq(out: np.ndarray) -> None:
        np.square(out, out=out)

    def _transform_weighted_sq(self, out: np.ndarray) -> None:
        np.square(out, out=out)
        out *= self.weights


def make_process_f_dist(
    *,
    n_samples: int,
    ss_obs: np.ndarray,
    simulator: SimulatorFn,
    stats_fn: StatsFn,
    seed: int | None = None,
    distance: DistanceMode = "abs",
    weights: np.ndarray | None = None,
    n_workers: int = 1,
    worker_setup: InitFn | None = None,
    mp_start_method: str = "spawn",
) -> ProcessFDist:
    """Build a process-backed distance function for non-thread-safe simulators."""
    return ProcessFDist(
        n_samples=n_samples,
        ss_obs=ss_obs,
        simulator=simulator,
        stats_fn=stats_fn,
        seed=seed,
        distance=distance,
        weights=weights,
        n_workers=n_workers,
        worker_setup=worker_setup,
        mp_start_method=mp_start_method,
    )


@dataclass
class ProcessSimThenStatsFDist(ProcessFDist):
    """Process-backed simulator with parent-process summary statistics.

    This is useful for expensive TensorFlow-backed summary statistics: workers
    keep Julia simulation parallelism, while the parent evaluates the neural
    encoder once on the full batch.
    """

    def _ensure_pool(self):
        if self._pool is None:
            self._ctx = mp.get_context(self.mp_start_method)
            self._pool = self._ctx.Pool(
                processes=self.n_workers,
                initializer=_init_process_sim_worker,
                initargs=(
                    self.simulator,
                    self.n_samples,
                    self.worker_setup,
                ),
            )
        return self._pool

    def __call__(self, theta: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        n_batch_particles = theta.shape[0]
        n_stats = self.ss_obs.size

        if out is None:
            out = np.empty((n_batch_particles, n_stats), dtype=np.float64)
        else:
            if out.shape != (n_batch_particles, n_stats):
                raise ValueError(
                    f"out must have shape ({n_batch_particles}, {n_stats}), got {out.shape}."
                )
            if out.dtype != np.float64:
                raise ValueError(f"out must have dtype float64, got {out.dtype}.")

        y_combined = np.empty((n_batch_particles, self.n_samples), dtype=np.float64)

        if self.n_workers <= 1:
            self.simulator(theta, y_combined, self._rng)
        else:
            pool = self._ensure_pool()
            boundaries = np.linspace(0, n_batch_particles, self.n_workers + 1, dtype=int)
            tasks: list[tuple[np.ndarray, int]] = []
            slices: list[tuple[int, int]] = []

            for k in range(self.n_workers):
                lo = int(boundaries[k])
                hi = int(boundaries[k + 1])
                if hi <= lo:
                    continue
                seed = int(self._rng.integers(0, np.iinfo(np.int32).max, dtype=np.int64))
                tasks.append((theta[lo:hi].copy(), seed))
                slices.append((lo, hi))

            y_chunks = pool.starmap(_run_process_sim_chunk, tasks)
            for (lo, hi), y_chunk in zip(slices, y_chunks, strict=True):
                y_combined[lo:hi] = y_chunk

        ss = np.empty((n_batch_particles, n_stats), dtype=np.float64)
        self.stats_fn(y_combined, ss)
        np.subtract(ss, self.ss_obs, out=out)
        self._transform(out)  # type: ignore[misc]
        return out

    def clone(self, seed: int | None = None) -> "ProcessSimThenStatsFDist":
        return ProcessSimThenStatsFDist(
            n_samples=self.n_samples,
            ss_obs=self.ss_obs.copy(),
            simulator=self.simulator,
            stats_fn=self.stats_fn,
            seed=seed,
            distance=self.distance,
            weights=self.weights.copy() if self.weights is not None else None,
            n_workers=self.n_workers,
            worker_setup=self.worker_setup,
            mp_start_method=self.mp_start_method,
        )


def make_process_sim_then_stats_f_dist(
    *,
    n_samples: int,
    ss_obs: np.ndarray,
    simulator: SimulatorFn,
    stats_fn: StatsFn,
    seed: int | None = None,
    distance: DistanceMode = "abs",
    weights: np.ndarray | None = None,
    n_workers: int = 1,
    worker_setup: InitFn | None = None,
    mp_start_method: str = "spawn",
) -> ProcessSimThenStatsFDist:
    """Build a process simulator + parent stats distance function."""
    return ProcessSimThenStatsFDist(
        n_samples=n_samples,
        ss_obs=ss_obs,
        simulator=simulator,
        stats_fn=stats_fn,
        seed=seed,
        distance=distance,
        weights=weights,
        n_workers=n_workers,
        worker_setup=worker_setup,
        mp_start_method=mp_start_method,
    )
