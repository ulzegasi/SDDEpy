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
