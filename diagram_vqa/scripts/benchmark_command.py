from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import threading
import time
from pathlib import Path

try:
    import psutil
except ImportError:  # Optional: wall time and VRAM still work without it.
    psutil = None


def gpu_memory_mb() -> float | None:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    result = subprocess.run(
        [executable, "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False
    )
    values = [float(line.strip()) for line in result.stdout.splitlines() if line.strip().replace(".", "", 1).isdigit()]
    return sum(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an experiment and record wall time and peak GPU memory.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--questions", type=int, default=None)
    parser.add_argument("--accuracy", type=float, default=None)
    parser.add_argument("--parameter-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("command is required after --")

    gpu_samples: list[float] = []
    ram_samples: list[float] = []
    stop = threading.Event()
    process_holder: list[subprocess.Popen] = []

    def poll() -> None:
        while not stop.wait(0.25):
            value = gpu_memory_mb()
            if value is not None:
                gpu_samples.append(value)
            if psutil is not None and process_holder:
                try:
                    root = psutil.Process(process_holder[0].pid)
                    family = [root, *root.children(recursive=True)]
                    ram_samples.append(sum(proc.memory_info().rss for proc in family) / (1024 * 1024))
                except (psutil.Error, ProcessLookupError):
                    pass

    worker = threading.Thread(target=poll, daemon=True)
    started = time.perf_counter()
    process = subprocess.Popen(command)
    process_holder.append(process)
    worker.start()
    exit_code = process.wait()
    elapsed = time.perf_counter() - started
    stop.set()
    worker.join(timeout=1)

    payload = {
        "name": args.name,
        "command": command,
        "exit_code": exit_code,
        "wall_time_seconds": elapsed,
        "peak_ram_mb": max(ram_samples) if ram_samples else None,
        "peak_vram_mb": max(gpu_samples) if gpu_samples else None,
        "platform": platform.platform(),
        "questions": args.questions,
        "accuracy": args.accuracy,
        "parameter_count": args.parameter_count,
        "seed": args.seed,
        "predictions": str(args.predictions.resolve()) if args.predictions else None,
        "latency_ms_per_question": (elapsed * 1000 / args.questions) if args.questions else None,
        "throughput_questions_per_second": (args.questions / elapsed) if args.questions else None
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
