# SDDEpy

This repository contains the scripts and data needed to run SABC inference using the Python SABC algorithm and the Python-wrapped SDDE Solar Dynamo model from Julia.

## Environment setup

For standard FFT summary-statistics runs, activate the lightweight project
environment:

```bash
conda activate sddepy_env
```

For ENCA encoder summary-statistics runs, use the TensorFlow-capable environment
defined in [`environment-enca.yml`](/Users/ulzg/SABC/SDDEpy/environment-enca.yml):

```bash
conda env create -f environment-enca.yml
conda activate sddepy_enca_env
```

Install the shared SDDE model package (-e -> editable means changes in that repo are immediately visible without reinstalling.):

```bash
pip install -e /path/to/SDDE-model
```

Also install the SABC package in the same environment if it is not already
available:

```bash
pip install -e /path/to/SimulatedAnnealingABC
```

The ENCA environment pins Python to `3.11` and `juliacall` to `0.9.31`, because
TensorFlow and the Julia bridge are sensitive to very new Python/Juliacall
releases. Standard FFT runs do not need TensorFlow.

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
(`obsSN`, `C14`, and user-provided synthetic CSV files) and both epsilon-update
strategies (`single_eps` and `multi_eps`).

Example runs:

```bash
python3 SABC_SolarDynamo.py --dataset obsSN --algorithm single_eps
python3 SABC_SolarDynamo.py --dataset obsSN --algorithm multi_eps
python3 SABC_SolarDynamo.py --dataset C14 --algorithm single_eps
python3 SABC_SolarDynamo.py --dataset C14 --algorithm multi_eps
python3 SABC_SolarDynamo.py \
  --dataset synthetic \
  --synthetic-data-file sn_t6_T7_N12_s002_B8_tobs271_seed1822.csv \
  --algorithm single_eps
```

Supported command-line arguments:

- `--dataset`
  Selects the dataset. Choices: `obsSN`, `C14`, `synthetic`.
- `--synthetic-data-file`
  CSV filename required when `--dataset synthetic`. The file is loaded from
  `data/synthetic_data/` and should have a header row and two columns:
  time/year and sunspot number.
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
- `--summary-stats`
  Selects the summary-statistics backend. Choices: `fft`, `enca`. Default:
  `fft`.
- `--fourier-range`
  Optional custom Fourier indices for `--summary-stats fft`. If omitted, the run
  uses the default definition from the shared `sdde_model` package: `1:6:120`,
  which gives 20 summary statistics.
- `--enca-run-dir`
  ENCA training-run directory used when `--summary-stats enca`. It must contain
  `hyper_parameters.json` and TensorFlow checkpoint files.
- `--enca-checkpoint-basename`
  Checkpoint family to load when `--summary-stats enca`. Default:
  `model_best_ckpt`.

### Custom Fourier Summary Statistics

By default, SABC uses the FFT-based summary statistics defined in the shared
`sdde_model` package. For each time series, the model applies a Hann window,
computes the inverse FFT, and keeps the absolute values at Fourier indices
`1:6:120`:

```text
1, 7, 13, ..., 115
```

These are Julia/1-based indices and produce 20 summary statistics.

To try a different regular range, pass `--fourier-range start:step:stop`.
For example, this uses 10 Fourier components:

```bash
python3 SABC_SolarDynamo.py \
  --dataset synthetic \
  --synthetic-data-file sn_t6_T7_N12_s002_B8_tobs271_seed1822.csv \
  --algorithm single_eps \
  --fourier-range 1:6:60
```

which selects:

```text
1, 7, 13, ..., 55
```

Irregular selections are also supported by passing a Python-style list:

```bash
python3 SABC_SolarDynamo.py \
  --dataset synthetic \
  --synthetic-data-file sn_t6_T7_N12_s002_B8_tobs271_seed1822.csv \
  --algorithm single_eps \
  --fourier-range '[1,2,5,9,12,28]'
```

The current default FFT summary-statistics function remains unchanged and is
used whenever `--fourier-range` is not provided.

### ENCA Encoder Summary Statistics

The driver can also use a trained ENCA encoder as the summary-statistics
generator. In this mode, each simulated or observed time series is reshaped from
`(Tobs,)` to `(Tobs, 1)` and passed through the encoder. The encoder output
becomes the SABC summary-statistics vector.

Use ENCA summaries by setting:

```bash
--summary-stats enca
--enca-run-dir /path/to/enca/run
```

The ENCA run directory must contain:

```text
hyper_parameters.json
model_best_ckpt-*.index
model_best_ckpt-*.data-00000-of-00001
```

or another checkpoint family selected with `--enca-checkpoint-basename`.

For example, an ENCA run trained with:

```json
{
  "len_timeseries": 271,
  "ndims_latent": 10
}
```

will produce 10 summary statistics for each time series. These can be interpreted
as the ENCA latent variables, for example 5 parameter-regression coordinates plus
5 additional latent coordinates, depending on how the ENCA model was trained.

Example, using a placeholder ENCA run path:

```bash
python3 SABC_SolarDynamo.py \
  --dataset synthetic \
  --synthetic-data-file sn_t6_T7_N12_s002_B8_tobs271_seed1822.csv \
  --algorithm multi_eps \
  --summary-stats enca \
  --enca-run-dir /path/to/sdde_ENCA_runs/20260504_enca_z10_3
```

To use a specific checkpoint family instead of the default `model_best_ckpt`,
pass:

```bash
--enca-checkpoint-basename model_ckpt
```

The loader selects the highest-numbered matching checkpoint file, for example
`model_best_ckpt-675000` among all `model_best_ckpt-*.index` files.

The ENCA run path is machine-specific. Use the absolute path that exists on the
machine where the inference is running. For example, on one local workstation it
may look like:

```bash
--enca-run-dir /Users/ulzg/switchdrive/ZHAW_BISTOM/RENKU/enca-inca/sdde_ENCA_runs/20260504_enca_z10_3
```

The ENCA backend checks that the selected dataset length matches the encoder's
`len_timeseries`. It will fail early if, for example, an encoder trained for
`len_timeseries = 271` is used with a dataset of a different length.

TensorFlow is required only for `--summary-stats enca`. Standard FFT runs do not
import TensorFlow.

For Slurm or other batch-system runs, [`runjob.sh`](/Users/ulzg/SABC/SDDEpy/runjob.sh)
exposes the same choice in the editable variables section. Set
`ENCA_RUN_DIR` to the absolute ENCA run directory on the compute system you are
using:

```bash
SUMMARY_STATS="enca"     # "fft" or "enca"
FOURIER_RANGE=""         # used only when SUMMARY_STATS="fft"
ENCA_RUN_DIR="/path/on/your/cluster/sdde_ENCA_runs/20260504_enca_z10_3"
ENCA_CHECKPOINT_BASENAME="model_best_ckpt"
```

On a different cluster or filesystem, only the path values and environment
activation commands should need to change; the Python options are the same.

When `SUMMARY_STATS="enca"`, `FOURIER_RANGE` is ignored and not passed to the
Python driver.

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
