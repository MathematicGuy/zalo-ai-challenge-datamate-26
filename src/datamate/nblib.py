"""Use a notebook as a library: run only its cells tagged "lib" in the caller's namespace.

01_data.ipynb and 02_agent.ipynb hold the code; their tagged cells define functions and classes,
the untagged cells explore or run. Another notebook loads the definitions without re-running the
exploration:  run_lib_cells("01_data.ipynb", globals())
"""
from __future__ import annotations

import sys
import types
from pathlib import Path


def notebooks_dir() -> Path:
    here = Path.cwd()
    for d in (here, *here.parents):
        if (d / "notebooks" / "01_data.ipynb").exists():
            return d / "notebooks"
    raise FileNotFoundError("notebooks/01_data.ipynb not found above the working directory")


def run_lib_cells(notebook: str, namespace: dict) -> None:
    import nbformat
    nb = nbformat.read(notebooks_dir() / notebook, as_version=4)
    # dataclasses (and pickle) need the defining module in sys.modules. A Jupyter kernel's globals()
    # is the __main__ module's dict; anything else (scripts, tests) gets a registered module.
    mod = sys.modules.get(namespace.get("__name__", ""))
    if mod is not None and mod.__dict__ is namespace:
        target = namespace
    else:
        mod = types.ModuleType(f"nblib_{Path(notebook).stem}")
        sys.modules[mod.__name__] = mod
        mod.__dict__.update({k: v for k, v in namespace.items() if not k.startswith("__")})
        target = mod.__dict__
    for cell in nb.cells:
        if cell.cell_type == "code" and "lib" in cell.metadata.get("tags", []):
            exec(compile(cell.source, f"{notebook}:lib", "exec"), target)
    if target is not namespace:
        namespace.update({k: v for k, v in target.items() if not k.startswith("__")})
