from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vqa_retrieval.model_matrix import DATASETS, MODELS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified 11x3 model-matrix experiment runner.")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val",
                        help="For trainable models: val trains/selects on validation and evaluates test; test only evaluates an existing validation-selected checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/model_matrix")
    parser.add_argument("--data-root", type=Path, default=ROOT.parent,
                        help="Root containing dataset manifests, models and model_matrix_cache")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()

    common = [
        "--dataset", args.dataset,
        "--model", args.model,
        "--split", args.split,
        "--seed", str(args.seed),
        "--output-dir", str(args.output_dir),
        "--data-root", str(args.data_root.resolve()),
        "--resume" if args.resume else "--no-resume",
    ]
    if args.max_samples is not None:
        common.extend(["--max-samples", str(args.max_samples)])

    if args.model in {"random", "ocr_text"}:
        script = "scripts/run_model_matrix_baseline.py"
    elif args.model == "qwen25_vl_qlora":
        script = "scripts/run_model_matrix_qlora.py"
    elif args.model in {"clip", "siglip"}:
        script = "scripts/train_model_matrix_vision.py"
    else:
        script = "scripts/train_model_matrix_local.py"
    if args.model not in {"random", "ocr_text"}:
        mode = "evaluation only; requires a completed validation-selected checkpoint" if args.split == "test" else "train/select on validation, then final test evaluation"
        print(f"[MODEL MATRIX] {mode}", flush=True)
    command = [str(Path(sys.executable)), script, *common]
    print("[MODEL MATRIX]", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
