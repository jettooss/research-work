from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)


def prediction_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def is_full(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "available_full"
    except (OSError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restart the resumable DocVQA VLM evaluator after recoverable CUDA process failures."
    )
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "docvqa_qwen25_vl")
    parser.add_argument("--flush-every", type=int, default=1)
    parser.add_argument(
        "--fallback-max-pixels",
        type=int,
        default=1048576,
        help="Retry a stalled sample at this image budget before returning to the configured budget.",
    )
    parser.add_argument("--max-consecutive-stalls", type=int, default=5)
    parser.add_argument("--restart-delay-seconds", type=float, default=5.0)
    args, evaluator_args = parser.parse_known_args()

    split_dir = args.output_dir / args.split
    state_path = split_dir / "all_predictions.jsonl"
    metrics_path = split_dir / "metrics.json"
    consecutive_stalls = 0
    attempt = 0

    while not is_full(metrics_path):
        attempt += 1
        before = prediction_count(state_path)
        active_evaluator_args = list(evaluator_args)
        if consecutive_stalls > 0 and args.fallback_max_pixels:
            if "--max-pixels" in active_evaluator_args:
                index = active_evaluator_args.index("--max-pixels")
                active_evaluator_args[index + 1] = str(args.fallback_max_pixels)
            else:
                active_evaluator_args.extend(["--max-pixels", str(args.fallback_max_pixels)])
        command = [
            str(PYTHON),
            "scripts/evaluate_docvqa_vlm.py",
            "--split", args.split,
            "--output-dir", str(args.output_dir),
            "--flush-every", str(args.flush_every),
            "--resume",
            *active_evaluator_args,
        ]
        print(
            f"[VLM WATCHDOG] attempt={attempt} saved={before} "
            f"fallback_resolution={consecutive_stalls > 0}",
            flush=True,
        )
        result = subprocess.run(command, cwd=ROOT, check=False)
        after = prediction_count(state_path)
        if result.returncode == 0 and is_full(metrics_path):
            print(f"[VLM WATCHDOG] complete split={args.split} predictions={after}", flush=True)
            return

        if after > before:
            consecutive_stalls = 0
        else:
            consecutive_stalls += 1
        print(
            f"[VLM WATCHDOG] evaluator_exit={result.returncode} saved={after} "
            f"consecutive_stalls={consecutive_stalls}/{args.max_consecutive_stalls}",
            flush=True,
        )
        if consecutive_stalls >= args.max_consecutive_stalls:
            raise RuntimeError(
                f"DocVQA VLM made no saved progress in {consecutive_stalls} consecutive process restarts"
            )
        time.sleep(args.restart_delay_seconds)


if __name__ == "__main__":
    main()
