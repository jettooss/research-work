from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit notebooks, runs, checkpoints, metrics, and submissions for the 11x3 matrix.")
    parser.add_argument("--notebook-root", type=Path, default=ROOT / "notebooks/model_matrix")
    parser.add_argument("--aggregate", type=Path, default=ROOT / "reports/model_matrix_aggregate.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/model_matrix_audit.json")
    args = parser.parse_args()
    aggregate = json.loads(args.aggregate.read_text(encoding="utf-8")) if args.aggregate.exists() else {"cells": []}
    sources = [path for path in args.notebook_root.rglob("*.ipynb") if not path.name.endswith(".executed.ipynb")]
    executed = list(args.notebook_root.rglob("*.executed.ipynb"))
    cells = aggregate.get("cells", [])
    missing_checkpoints = []
    missing_submissions = []
    for cell in cells:
        if cell.get("status") not in {"available_full", "available_smoke"}:
            continue
        for run in cell.get("runs", []):
            metrics = Path(run["metrics_file"])
            run_dir = metrics.parent
            if cell["model"] not in {"random", "ocr_text"}:
                if cell["model"] == "qwen25_vl_qlora":
                    exists = Path(str(run.get("adapter_path", ""))).exists()
                else:
                    exists = (run_dir / "checkpoint_best.pt").exists()
                if not exists:
                    missing_checkpoints.append(f"{cell['dataset']}/{cell['model']}/seed{run.get('seed')}")
            if cell["dataset"] != "ai2d" and not (run_dir.parent / "test/submission.json").exists():
                missing_submissions.append(f"{cell['dataset']}/{cell['model']}/seed{run.get('seed')}")
    ai2d_audit = json.loads((ROOT.parent / "ai2d/model_matrix_v1/audit.json").read_text(encoding="utf-8"))
    payload = {
        "source_notebooks": len(sources), "executed_notebooks": len(executed),
        "available_full_cells": sum(cell.get("status") == "available_full" for cell in cells),
        "available_smoke_cells": sum(cell.get("status") in {"available_full", "available_smoke"} for cell in cells),
        "expected_cells": 33, "missing_checkpoints": missing_checkpoints, "missing_submissions": missing_submissions,
        "ai2d_ready": bool(ai2d_audit.get("ready")),
    }
    payload["defense_ready"] = payload["source_notebooks"] == 33 and payload["executed_notebooks"] == 33 and payload["available_full_cells"] == 33 and not missing_checkpoints and not missing_submissions and payload["ai2d_ready"]
    payload["smoke_ready"] = payload["source_notebooks"] == 33 and payload["executed_notebooks"] == 33 and payload["available_smoke_cells"] == 33 and not missing_checkpoints and not missing_submissions and payload["ai2d_ready"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
