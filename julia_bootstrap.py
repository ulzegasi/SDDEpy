"""
Initialize Julia through a pinned project-local environment.

This avoids `juliacall`/`juliapkg` resolving against whatever SciML versions
currently happen to be newest in the active Conda environment.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

_INITIALIZED = False
_JL = None


def _default_project_dir() -> Path:
    return Path(__file__).resolve().parent / "julia_env"


def _configure_juliacall_env(project_dir: Path) -> None:
    import juliapkg

    project_dir = project_dir.resolve()
    if not project_dir.exists():
        raise FileNotFoundError(f"Julia project directory does not exist: {project_dir}")

    project = str(project_dir)
    julia_exe = juliapkg.executable()

    os.environ["PYTHON_JULIACALL_PROJECT"] = project
    os.environ.setdefault("PYTHON_JULIACALL_EXE", julia_exe)
    os.environ["JULIA_PROJECT"] = project

    _ensure_julia_project(project_dir, julia_exe)


def _ensure_julia_project(project_dir: Path, julia_exe: str) -> None:
    subprocess.run(
        [
            julia_exe,
            f"--project={project_dir}",
            "--startup-file=no",
            "-e",
            "import Pkg; Pkg.instantiate()",
        ],
        check=True,
    )


def init_julia(project_dir: str | os.PathLike[str] | None = None):
    """
    Initialize Julia once and return `juliacall.Main`.

    Parameters
    ----------
    project_dir:
        Optional Julia project directory. If omitted, use the pinned project in
        `./julia_env`. You can also override this globally with the environment
        variable `PYTHON_JULIACALL_PROJECT`.
    """
    global _INITIALIZED, _JL
    if _INITIALIZED:
        return _JL

    chosen_project = (
        Path(project_dir)
        if project_dir is not None
        else Path(os.environ.get("PYTHON_JULIACALL_PROJECT", _default_project_dir()))
    )
    _configure_juliacall_env(chosen_project)

    from juliacall import Main as jl

    version = jl.seval("VERSION")
    active_project = jl.seval("Base.active_project()")
    print(f"--- Julia engine: ON (Julia {version}, project={active_project}) ---")

    _JL = jl
    _INITIALIZED = True
    return jl


if __name__ == "__main__":
    init_julia()
