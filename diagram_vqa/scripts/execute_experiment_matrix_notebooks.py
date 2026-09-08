from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vqa_retrieval.experiment_matrix import ARCHITECTURES, DATASETS
from vqa_retrieval.notebook_runs import notebook_paths, prepare_notebook, reserve_run


def iter_notebooks(dataset=None, architecture=None, skip_architectures=None):
    return [path for path in notebook_paths(ROOT, dataset, architecture)
            if path.stem.removeprefix(path.parent.name + "_") not in (skip_architectures or set())]


def execute_notebook(notebook: dict, config: dict, timeout: int) -> Path:
    # Planning and preparing notebooks require no Jupyter installation.
    import nbformat
    from nbclient import NotebookClient

    run_dir = reserve_run(config)
    output = run_dir / "notebook.executed.ipynb"
    nb = nbformat.from_dict(notebook)
    state = {"status": "running", "config": config}
    state_path = run_dir / "execution.json"
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    client = NotebookClient(nb, timeout=None if timeout <= 0 else timeout,
                            kernel_name=config["kernel_name"],
                            resources={"metadata": {"path": config["project_root"]}})
    try:
        client.execute()
        state["status"] = "completed"
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        nbformat.write(nb, output)
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Execute isolated copies of the 18 actual research notebooks.")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--architecture", choices=ARCHITECTURES)
    parser.add_argument("--skip-architecture", action="append", choices=ARCHITECTURES, default=[])
    parser.add_argument("--start-after", help="Skip through this notebook stem before continuing.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, help="Override each notebook's full-data epoch default.")
    parser.add_argument("--output-root", type=Path, required=True, help="Separate root for all new run artifacts.")
    parser.add_argument("--external-root", type=Path, help="Directory containing ai2d/, docvqa/, models/, etc.")
    parser.add_argument("--timeout", type=int, default=0, help="Per-cell seconds; 0 disables timeout.")
    parser.add_argument("--kernel-name", default="python3")
    parser.add_argument("--resume", action="store_true", help="Resume only matching recorded notebook configurations.")
    parser.add_argument("--dry-run", action="store_true", help="Print the exact plans without writing or running cells.")
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be unique")
    return args


def main():
    args = parse_args()
    paths = iter_notebooks(args.dataset, args.architecture, set(args.skip_architecture))
    if args.start_after:
        stems = [path.stem for path in paths]
        if args.start_after not in stems:
            raise ValueError(f"Unknown --start-after notebook: {args.start_after}")
        paths = paths[stems.index(args.start_after) + 1:]
    planned = [prepare_notebook(path, args.output_root, seed=seed, epochs=args.epochs,
                               external_root=args.external_root, kernel_name=args.kernel_name,
                               resume=args.resume) for path in paths for seed in args.seeds]
    if args.dry_run:
        print(json.dumps({"mode": "dry_run", "count": len(planned),
                          "runs": [config for _, config in planned]}, indent=2))
        return
    for notebook, config in planned:
        print(json.dumps({"status": "starting", **config}), flush=True)
        output = execute_notebook(notebook, config, args.timeout)
        print(json.dumps({"status": "completed", "notebook": str(output)}), flush=True)


if __name__ == "__main__":
    main()
