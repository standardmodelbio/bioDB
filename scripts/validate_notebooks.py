"""Validate saved tutorials and optionally execute them with network access blocked.

Run ``python scripts/validate_notebooks.py --execute`` after installing
``biodb[docs]``. Execution uses temporary copies and never rewrites saved outputs.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import nbformat
from IPython.core.inputtransformer2 import TransformerManager
from jupyter_client import KernelManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]

# This runs inside the kernel after its control sockets have been established.
# Unexpected HTTP requests fail immediately instead of depending on CI networking.
NETWORK_GUARD = """
import socket

def _biodb_no_network(*args, **kwargs):
    raise RuntimeError("Tutorial execution is offline; mock the network boundary")

socket.socket.connect = _biodb_no_network
socket.socket.connect_ex = _biodb_no_network
socket.create_connection = _biodb_no_network
socket.getaddrinfo = _biodb_no_network
"""


def validate(path: Path):
    """Check the schema, executable syntax, and absence of saved errors."""
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    transformer = TransformerManager()
    for index, cell in enumerate(notebook.cells, 1):
        if cell.cell_type != "code":
            continue
        if "skip-execution" in cell.metadata.get("tags", []):
            raise ValueError(f"{path}:cell {index} uses a forbidden skip-execution tag")
        if cell.source.strip() and cell.get("execution_count") is None:
            raise ValueError(f"{path}:cell {index} has not been executed in the saved notebook")
        compile(transformer.transform_cell(cell.source), f"{path}:cell {index}", "exec")
        for output in cell.get("outputs", []):
            if output.output_type == "error":
                raise ValueError(f"{path}:cell {index} contains a saved {output.ename} error")
    return notebook


def execute(path: Path, notebook, timeout: int):
    """Run every code cell in an isolated directory, using this Python runtime."""
    if notebook.metadata.get("biodb", {}).get("offline") is not True:
        raise ValueError(f"{path}: execution requires metadata.biodb.offline = true")
    execution_copy = copy.deepcopy(notebook)
    for cell in execution_copy.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    execution_copy.cells.insert(0, nbformat.v4.new_code_cell(NETWORK_GUARD))
    with TemporaryDirectory(prefix="biodb-tutorial-") as workdir:
        kernel = KernelManager(
            kernel_name="python3", connection_file=str(Path(workdir) / "kernel.json")
        )
        kernel.kernel_spec.argv = [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ]
        client = NotebookClient(
            execution_copy,
            km=kernel,
            timeout=timeout,
            startup_timeout=60,
            force_raise_errors=True,
            resources={"metadata": {"path": workdir}},
        )
        client.execute(
            cleanup_kc=True,
            env={**os.environ, "IPYTHONDIR": str(Path(workdir) / "ipython")},
        )
        for index, cell in enumerate(execution_copy.cells, 1):
            if cell.cell_type == "code" and cell.source.strip() and cell.execution_count is None:
                raise ValueError(f"{path}:cell {index} was skipped during execution")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Defaults to tutorials/*.ipynb")
    parser.add_argument("--execute", action="store_true", help="Execute all offline notebooks")
    parser.add_argument("--timeout", type=int, default=120, help="Maximum seconds per cell")
    args = parser.parse_args()
    paths = args.paths or sorted((ROOT / "tutorials").glob("*.ipynb"))
    if not paths:
        parser.error("No tutorial notebooks found")
    failures = 0
    for path in paths:
        try:
            notebook = validate(path)
            if args.execute:
                execute(path, notebook, args.timeout)
        except Exception as exc:
            failures += 1
            print(f"FAIL {path.name}: {exc}", file=sys.stderr, flush=True)
        else:
            action = "executed offline" if args.execute else "validated"
            print(f"OK {path.name}: {action}", flush=True)
    print(f"{len(paths) - failures}/{len(paths)} notebooks passed", flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
