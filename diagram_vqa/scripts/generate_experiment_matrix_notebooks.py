"""Make unexecuted, parameterized copies; never replace the canonical model code."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vqa_retrieval.experiment_matrix import ARCHITECTURES, DATASETS
from vqa_retrieval.notebook_runs import notebook_paths, prepare_notebook, reserve_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--architecture", choices=ARCHITECTURES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--kernel-name", default="python3")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be unique")
    planned = [prepare_notebook(path, args.output_root, seed=seed, epochs=args.epochs,
                               external_root=args.external_root, kernel_name=args.kernel_name)
               for path in notebook_paths(ROOT, args.dataset, args.architecture) for seed in args.seeds]
    outputs = []
    for notebook, config in planned:
        path = Path(config["run_dir"]) / "notebook.ipynb"
        if not args.dry_run:
            reserve_run(config, require_manifest=False)
            path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        outputs.append({"notebook": str(path), "config": config})
    print(json.dumps({"mode": "dry_run" if args.dry_run else "prepared", "count": len(outputs), "runs": outputs}, indent=2))


if __name__ == "__main__":
    main()
