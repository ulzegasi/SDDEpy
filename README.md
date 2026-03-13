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
