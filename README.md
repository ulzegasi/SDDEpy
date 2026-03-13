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

## Julia setup

This repo now carries its own pinned Julia project in [`julia_env/Project.toml`](/Users/ulzg/SABC/SDDEpy/julia_env/Project.toml).
`julia_bootstrap.init_julia()` points `juliacall` at that project automatically, so you should not need ad-hoc `export JULIA_PROJECT=...` commands for normal use.

First-time setup in `sddepy_env`:

```bash
/opt/miniconda3/envs/sddepy_env/bin/python -c "from julia_bootstrap import init_julia; init_julia()"
```

After that, start scripts with `from julia_bootstrap import init_julia; init_julia()` before importing `sdde_model`.

If you want to reuse the same approach from another repository, either:

```bash
export PYTHON_JULIACALL_PROJECT=/absolute/path/to/SDDEpy/julia_env
```

or call:

```python
from julia_bootstrap import init_julia
init_julia("/absolute/path/to/SDDEpy/julia_env")
```
