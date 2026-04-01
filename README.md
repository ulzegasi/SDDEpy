# SDDEpy

This repository contains the scripts and data needed to run SABC inference using the Python SABC algorithm and the Python-wrapped SDDE Solar Dynamo model from Julia.

## Environment setup

Activate the project environment:

```bash
conda activate sddepy_env
```

Install the shared SDDE model package (-e -> editable means changes in that repo are immediately visible without reinstalling.):

```bash
pip install -e /path/to/SDDE-model
```

The shared `sdde_model` package now owns the Julia bootstrap and pinned Julia
environment. In scripts in this repo, initialize Julia with:

```python
from sdde_model import init_julia
init_julia()
```

Call `init_julia()` before importing `tensorflow` or other native-library-heavy
modules.

## Parallel SABC with the Julia model

`SimulatedAnnealingABC` uses threads when `n_workers > 1`. That works for the
NumPy-only examples, but it is not reliable for this repository's Julia-backed
solar dynamo model: multiple Python threads would share the same `juliacall`
bridge and could hang during initialization or simulation.

To avoid that, [`SABC_SolarDynamo.py`](/Users/ulzg/SABC/SDDEpy/SABC_SolarDynamo.py)
now uses process-based parallelism for the Julia case. Separate processes do not
share one Julia session; each worker starts its own Julia runtime, which is
heavier but much safer than threads here.

Two helper modules support this:

- [`process_fdist.py`](/Users/ulzg/SABC/SDDEpy/process_fdist.py) provides a
  process-backed replacement for the usual threaded `f_dist` path, so batches
  of particles can still be evaluated in parallel.
- [`solar_dynamo_sabc_setup.py`](/Users/ulzg/SABC/SDDEpy/solar_dynamo_sabc_setup.py)
  holds the solar-dynamo-specific simulator, summary-statistics, data-loading,
  and quiet Julia worker-initialization helpers in an import-safe module for
  multiprocessing.

On the local `obsSN_single` run, this reduced the measured SABC wall-clock time
from `246.95 s` with `n_workers = 1` to `85.09 s` with `n_workers = 4`
(about `2.9x` faster).

## Unified Solar Dynamo Driver

[`SABC_SolarDynamo.py`](/Users/ulzg/SABC/SDDEpy/SABC_SolarDynamo.py) is the
unified entry point for the solar-dynamo SABC runs. It supports both datasets
(`obsSN` and `C14`) and both epsilon-update strategies (`single_eps` and
`multi_eps`).

Example runs:

```bash
python3 SABC_SolarDynamo.py --dataset obsSN --algorithm single_eps
python3 SABC_SolarDynamo.py --dataset obsSN --algorithm multi_eps
python3 SABC_SolarDynamo.py --dataset C14 --algorithm single_eps
python3 SABC_SolarDynamo.py --dataset C14 --algorithm multi_eps
```

Supported command-line arguments:

- `--dataset`
  Selects the dataset. Choices: `obsSN`, `C14`.
- `--algorithm`
  Selects the epsilon-update strategy. Choices: `single_eps`, `multi_eps`.
- `--from-previous`
  Selects whether to start a fresh run or continue a saved one. Choices: `0` for
  a fresh run, `1` to continue a previous run.
- `--n-workers`
  Number of workers for the distance-function evaluation. Use `1` for the
  serial/thread path or `>1` for the Julia-safe process path.
- `--run-name`
  Optional custom output name. If omitted, the default is
  `<dataset>_<algorithm>`.
- `--previous-run-name`
  Name of the saved run to continue from when `--from-previous 1` is used.

Example of continuing a previous run:

```bash
python3 SABC_SolarDynamo.py \
  --dataset obsSN \
  --algorithm multi_eps \
  --from-previous 1 \
  --previous-run-name obsSN_multi_eps \
  --run-name obsSN_multi_eps_continue
```

## Importance-Sampling Filtering

[`importance_sampling_filter.py`](/Users/ulzg/SABC/SDDEpy/importance_sampling_filter.py)
post-processes a saved final SABC population and keeps particles using two
distance definitions:

- reconstructed distances: for each posterior particle, resimulate the model
  repeatedly and average the summary-statistic distances
- SABC distances: use the final particle-wise `rho` matrix stored in the saved
  SABC result

For each run, the script computes the norm of the distance vector, builds a KDE
for those norms, and chooses a cutoff targeting a retained mass of about `70%`
by default. To keep the realized retained count close to the requested target,
it tries several KDE bandwidths and uses the KDE-based cutoff that best matches
the requested retained fraction.

The script writes only two files per processed run to
[`output/`](/Users/ulzg/SABC/SDDEpy/output):

- `kept_ind_reconst_<run_name>.csv`
- `kept_ind_sabc_<run_name>.csv`

Examples:

```bash
python3 importance_sampling_filter.py --dataset obsSN --algorithm single --tag 77py
python3 importance_sampling_filter.py --dataset obsSN --algorithm multi --tag 77py
python3 importance_sampling_filter.py --dataset C14 --algorithm single --tag 77py
python3 importance_sampling_filter.py --dataset C14 --algorithm multi --tag 77py
```

To display the overlaid histogram comparison interactively, add:

```bash
--show-plots
```

Supported command-line arguments:

- `--dataset`
  Dataset to process. Choices: `obsSN`, `C14`, `all`.
- `--algorithm`
  Algorithm family encoded in the saved run name. Choices: `single`, `multi`,
  `all`.
- `--tag`
  Run suffix used in filenames such as
  `post_population_obsSN_single_77py.csv`. Default: `77py`.
- `--run-name`
  Optional explicit run name. This can be passed multiple times; if used, the
  script skips the automatic dataset/algorithm expansion.
- `--n-repeats`
  Number of repeated simulations per posterior particle for the reconstructed
  distances. Default: `50`.
- `--keep-mass`
  Target retained mass for the KDE-based cutoff. Default: `0.70`.
- `--n-workers`
  Number of worker processes used for the reconstructed-distance step.
  Default: `4`.
- `--seed`
  Base random seed used for the reconstructed-distance simulations.
  Default: `123`.
- `--show-plots`
  Show one interactive figure per processed run with the reconstructed and SABC
  distance histograms overlaid.

## Benchmarking

[`SABC_SolarDynamo_BENCHMARK.py`](/Users/ulzg/SABC/SDDEpy/SABC_SolarDynamo_BENCHMARK.py)
is a benchmark-only driver for timing runs with fixed benchmark settings. By
default it uses:

- dataset `obsSN`
- algorithm `single_eps`
- `n_particles = 1000`
- `n_simulation = 4_000_000`

Unlike the main driver, it does not save posterior populations or histories. It
only measures the SABC wall-clock time and appends one row to a benchmark CSV.

Run one benchmark directly with:

```bash
python3 SABC_SolarDynamo_BENCHMARK.py --n-workers 8
```

This writes one row to:

```bash
output/benchmark_obsSN_single.csv
```

with the columns:

- `run_id`
- `dataset`
- `algorithm`
- `n_workers`
- `n_particles`
- `n_simulation`
- `elapsed_time_seconds`

For cluster benchmarking, two helper scripts are provided:

- [`runjob_benchmark.sh`](/Users/ulzg/SABC/SDDEpy/runjob_benchmark.sh)
  runs one benchmark job for a given worker count.
- [`submit_benchmarks.sh`](/Users/ulzg/SABC/SDDEpy/submit_benchmarks.sh)
  submits a sweep over the worker counts `1,2,4,6,8,10,12,16,20,24,28,32`.

On the cluster, submit the default `obsSN` + `single_eps` benchmark sweep with:

```bash
bash submit_benchmarks.sh
```

The worker sweep is controlled by
[`submit_benchmarks.sh`](/Users/ulzg/SABC/SDDEpy/submit_benchmarks.sh). The
dataset and algorithm are configured directly in
[`runjob_benchmark.sh`](/Users/ulzg/SABC/SDDEpy/runjob_benchmark.sh), where the
batch job also derives `N_WORKERS` from `SLURM_CPUS_PER_TASK` and writes the
benchmark CSV for the selected dataset/algorithm pair.

If you want to benchmark a different setup, edit the values near the top of
[`runjob_benchmark.sh`](/Users/ulzg/SABC/SDDEpy/runjob_benchmark.sh), for
example:

```bash
DATASET="C14"
ALGORITHM="multi_eps"
```

You can still override the Slurm submission settings from
[`runjob_benchmark.sh`](/Users/ulzg/SABC/SDDEpy/runjob_benchmark.sh), such as
time, partition, and memory, when launching the sweep. These values are passed
to `sbatch` by [`submit_benchmarks.sh`](/Users/ulzg/SABC/SDDEpy/submit_benchmarks.sh)
and therefore override the corresponding `#SBATCH` defaults:

```bash
TIME_LIMIT=02-00:00:00 PARTITION=earth-3 MEMORY=256G bash submit_benchmarks.sh
```

All benchmark jobs append to the same benchmark CSV using a file lock, so the
rows are written safely even when multiple worker-count jobs run
simultaneously.
