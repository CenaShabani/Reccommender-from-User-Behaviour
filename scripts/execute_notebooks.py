"""Execute every notebook headlessly (smoke test), or strip their outputs.

    python scripts/execute_notebooks.py               # run all, fail on the first error
    python scripts/execute_notebooks.py 03 05         # run only notebooks whose name starts with these
    python scripts/execute_notebooks.py --clear-only  # strip outputs in place (do this before committing)

Executed notebooks are not saved, so the committed copies stay output-free and
learners see results from *their* data.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def clear(nb) -> None:
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs, cell.execution_count = [], None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("prefixes", nargs="*")
    p.add_argument("--clear-only", action="store_true")
    p.add_argument("--timeout", type=int, default=3600, help="per-cell timeout in seconds")
    args = p.parse_args()

    paths = sorted(NB_DIR.glob("*.ipynb"))
    if args.prefixes:
        paths = [q for q in paths if q.name.startswith(tuple(args.prefixes))]

    for path in paths:
        nb = nbformat.read(path, as_version=4)
        if args.clear_only:
            clear(nb)
            nbformat.write(nb, path)
            print(f"cleared {path.name}")
            continue
        t0 = time.time()
        print(f"running {path.name} ...", end=" ", flush=True)
        NotebookClient(nb, timeout=args.timeout, kernel_name="python3", resources={"metadata": {"path": str(NB_DIR)}}).execute()
        print(f"ok ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
