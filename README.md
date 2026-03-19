# SDDEpy

This repository contains the scripts and data needed to run SABC inference using the Python SABC algorithm and the Python-wrapped SDDE Solar Dynamo model from Julia.

## Environment setup

Activate the project environment:

```bash
conda activate sddepy_env
```

Install the shared SDDE model package (-e -> editable means changes in that repo are immediately visible without reinstalling.):

```bash
pip install -e /Users/ulzg/SABC/SDDE-model
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

To avoid that, [`SABC_SolarDynamo_obsSN_single.py`](/Users/ulzg/SABC/SDDEpy/SABC_SolarDynamo_obsSN_single.py)
now uses process-based parallelism for the Julia case. Separate processes do not
share one Julia session; each worker starts its own Julia runtime, which is
heavier but much safer than threads here.

Two helper modules support this:

- [`process_fdist.py`](/Users/ulzg/SABC/SDDEpy/process_fdist.py) provides a
  process-backed replacement for the usual threaded `f_dist` path, so batches
  of particles can still be evaluated in parallel.
- [`solar_dynamo_obs_sn_problem.py`](/Users/ulzg/SABC/SDDEpy/solar_dynamo_obs_sn_problem.py)
  holds the solar-dynamo-specific simulator, summary-statistics, data-loading,
  and quiet Julia worker-initialization helpers in an import-safe module for
  multiprocessing.

On the local `obsSN_single` run, this reduced the measured SABC wall-clock time
from `246.95 s` with `n_workers = 1` to `85.09 s` with `n_workers = 4`
(about `2.9x` faster).
