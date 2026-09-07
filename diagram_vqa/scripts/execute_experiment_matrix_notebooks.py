from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from vqa_retrieval.experiment_matrix import ARCHITECTURES, DATASETS, write_matrix_summary


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks" / "experiments"
PROGRESS_PATH = ROOT / "reports" / "experiment_matrix_execution.json"


def write_progress(completed: list[str], current: str | None, status: str) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(
        json.dumps(
            {"status": status, "completed": completed, "completed_count": len(completed), "current": current},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def iter_notebooks(
    dataset: str | None = None,
    architecture: str | None = None,
    skip_architectures: set[str] | None = None,
) -> list[Path]:
    datasets = [dataset] if dataset else list(DATASETS)
    architectures = [architecture] if architecture else list(ARCHITECTURES)
    skip_architectures = skip_architectures or set()
    return [
        NOTEBOOK_ROOT / dataset_name / f"{dataset_name}_{architecture_name}.ipynb"
        for dataset_name in datasets
        for architecture_name in architectures
        if architecture_name not in skip_architectures
    ]


def execute_notebook(path: Path, timeout: int, kernel_name: str) -> Path:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=None if timeout <= 0 else timeout,
        kernel_name=kernel_name,
        resources={"metadata": {"path": str(ROOT)}},
    )
    try:
        client.execute()
    finally:
        nbformat.write(nb, path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute clean per-approach experiment notebooks.")
    parser.add_argument("--dataset", choices=DATASETS, default=None)
    parser.add_argument("--architecture", choices=ARCHITECTURES, default=None)
    parser.add_argument("--skip-architecture", action="append", choices=ARCHITECTURES, default=[])
    parser.add_argument(
        "--start-after",
        default=None,
        help="Skip notebooks through this notebook stem, then continue the matrix.",
    )
    parser.add_argument("--timeout", type=int, default=0, help="Per-cell timeout in seconds; 0 disables it.")
    parser.add_argument("--kernel-name", default="data-cu124")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    executed = []
    paths = iter_notebooks(args.dataset, args.architecture, set(args.skip_architecture))
    if args.start_after:
        stems = [path.stem for path in paths]
        if args.start_after not in stems:
            raise ValueError(f"Unknown --start-after notebook: {args.start_after}")
        paths = paths[stems.index(args.start_after) + 1 :]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        write_progress(executed, path.as_posix(), "running")
        executed.append(execute_notebook(path, timeout=args.timeout, kernel_name=args.kernel_name).as_posix())
        write_progress(executed, None, "running")
    write_matrix_summary()
    write_progress(executed, None, "completed")
    print(json.dumps({"mode": "full", "updated": executed, "count": len(executed)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
