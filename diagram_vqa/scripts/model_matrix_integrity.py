"""Immutable run identities and validation before reusing experiment artifacts."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path


SCHEMA_VERSION = 2


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def directory_digest(path: Path) -> str:
    files = sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"Artifact directory is empty: {path}")
    return digest({p.relative_to(path).as_posix(): file_digest(p) for p in files})


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def make_config(args, rows, *, kind: str, input_root: Path | None = None, **parameters) -> dict:
    if args.split not in {"val", "test"}:
        raise ValueError("split must be val or test")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive, or omitted for the full split")
    if not rows:
        raise ValueError("The input manifest is empty")
    script_root = Path(__file__).resolve().parent
    source_paths = [script_root / name for name in (
        "model_matrix_integrity.py", "run_model_matrix_baseline.py", "train_model_matrix_vision.py",
        "train_model_matrix_local.py", "run_model_matrix_qlora.py", "evaluate_model_matrix_qwen_retrieval.py",
        "train_vlm_qlora.py", "evaluate_ai2d_vlm.py", "evaluate_docvqa_vlm.py", "evaluate_infographicvqa_vlm.py")]
    source_paths += [script_root.parent / "src/vqa_retrieval" / name for name in
                     ("model_matrix.py", "metrics.py", "public_vqa_metrics.py")]
    versions = {}
    for name in ("numpy", "torch", "scikit-learn", "torch-geometric", "transformers", "sentence-transformers",
                 "clip", "torchvision", "Pillow", "ultralytics", "peft", "accelerate", "bitsandbytes",
                 "pytesseract", "tokenizers", "safetensors", "ftfy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    identity_rows = []
    for row in rows:
        canonical = dict(row)
        if input_root is not None:
            for key, value in canonical.items():
                if key.endswith("_path") and value:
                    path = Path(str(value))
                    if not path.is_absolute():
                        canonical[key] = "{data_root}/" + path.as_posix()
                    elif path.is_relative_to(input_root):
                        canonical[key] = "{data_root}/" + path.relative_to(input_root).as_posix()
        identity_rows.append(canonical)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "dataset": args.dataset, "model": args.model, "seed": args.seed,
        "split": args.split, "max_samples": args.max_samples,
        "input_rows_sha256": digest(identity_rows),
        # The CPU example image intentionally omits the optional training backends.
        "implementation_sha256": digest({p.relative_to(script_root.parent).as_posix():
                                          hashlib.sha256(p.read_text(encoding="utf-8").encode("utf-8")).hexdigest() if p.is_file() else None
                                          for p in source_paths}),
        "package_versions": versions,
        **parameters,
    }


def validate_directory(path: Path, config: dict, *, resume: bool) -> None:
    """Read only: never relabel or overwrite an incompatible/legacy run."""
    if not path.exists() or not any(path.iterdir()):
        return
    if not resume:
        raise ValueError(f"Run directory is not empty: {path}. Use a fresh --output-dir.")
    saved_path = path / "config.json"
    if not saved_path.is_file():
        raise ValueError(f"Cannot verify legacy run without config.json: {path}. Use a fresh --output-dir.")
    saved = read_json(saved_path)
    different = [key for key in sorted(saved.keys() | config.keys())
                 if key not in saved or key not in config or saved[key] != config[key]]
    if different:
        raise ValueError(f"Resume configuration mismatch in {path}: {', '.join(different)}. "
                         "Use a fresh --output-dir; existing artifacts were not changed.")


def write_config(path: Path, config: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    destination = path / "config.json"
    if not destination.exists():
        destination.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def completed_metrics(path: Path, config: dict, rows, *, trained: bool = False):
    metrics_path = path / "metrics.json"
    if not metrics_path.exists():
        return None
    result = read_json(metrics_path)
    expected_status = ("completed_" if trained else "available_") + ("full" if config["max_samples"] is None else "smoke")
    expected = {key: config[key] for key in ("dataset", "model", "seed", "split")}
    expected.update(status=expected_status, num_samples=len(rows), config_sha256=digest(config))
    wrong = [key for key, value in expected.items() if result.get(key) != value]
    if wrong:
        raise ValueError(f"Saved metrics do not match this run ({', '.join(wrong)}): {path}")
    predictions_path = path / "predictions.jsonl"
    if not predictions_path.is_file():
        raise ValueError(f"Completed run is missing predictions.jsonl: {path}")
    if result.get("predictions_sha256") != file_digest(predictions_path):
        raise ValueError(f"Completed run predictions are missing an integrity hash or have changed: {path}")
    predictions = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    identity = lambda x: (str(x["sample_id"]), str(x["image_id"]), str(x["question"]))
    if [identity(x) for x in predictions] != [identity(x) for x in rows]:
        raise ValueError(f"Saved predictions do not match the evaluation rows/order: {path}")
    if trained:
        checkpoint = path / "checkpoint_best.pt"
        if not checkpoint.is_file() or result.get("checkpoint_sha256") != file_digest(checkpoint):
            raise ValueError(f"Completed run checkpoint is missing or has changed: {path}")
    return result


def training_paths(args):
    override = getattr(args, "run_dir_override", None)
    train_dir = Path(override) if override else args.output_dir / args.dataset / args.model / f"seed{args.seed}" / "val"
    test_dir = train_dir / "test" if override else train_dir.parent / "test"
    return train_dir, test_dir


def training_config(args, rows, *, backend: str, **parameters):
    for key in ("epochs", "batch_size", "eval_batch_size"):
        if key in parameters and parameters[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if parameters.get("early_stopping_patience", 0) < 0:
        raise ValueError("early_stopping_patience must be nonnegative")
    config = make_config(args, rows, kind="training", backend=backend, **parameters)
    config["split"] = "val"
    config["checkpoint_selection_split"] = "val"
    config["checkpoint_selection"] = "0.5*vqa+0.5*mean_r1"
    return config


def checkpoint_config(args, rows, train_dir: Path, *, input_root: Path | None = None) -> dict:
    path = train_dir / "config.json"
    if not path.is_file():
        raise ValueError("--split test is evaluation-only and needs a completed validation-selected "
                         f"run at {train_dir}. Train with --split val first.")
    saved = read_json(path)
    request = make_config(args, rows, kind="training", input_root=input_root)
    keys = ("schema_version", "kind", "dataset", "model", "seed", "max_samples", "input_rows_sha256",
            "implementation_sha256", "package_versions")
    wrong = [key for key in keys if saved.get(key) != request[key]]
    if saved.get("split") != "val" or saved.get("checkpoint_selection_split") != "val":
        wrong.append("checkpoint_selection_split")
    if wrong:
        raise ValueError(f"Checkpoint is incompatible or has no verified validation selection ({', '.join(wrong)}): {train_dir}")
    return saved


def test_config(args, rows, saved_config: dict, checkpoint: Path, *, input_root: Path | None = None) -> dict:
    config = make_config(args, rows, kind="checkpoint_evaluation", input_root=input_root,
                         training_config_sha256=digest(saved_config),
                         checkpoint_sha256=file_digest(checkpoint), checkpoint_selection_split="val")
    config["split"] = "test"
    return config


def validate_partial_checkpoint(path: Path, config: dict, load_checkpoint):
    history_path, last_path, best_path = (path / name for name in ("history.json", "checkpoint_last.pt", "checkpoint_best.pt"))
    if not any(p.exists() for p in (history_path, last_path, best_path)):
        return None, []
    if not all(p.is_file() for p in (history_path, last_path, best_path)):
        raise ValueError(f"Incomplete resume state (history/last/best checkpoint): {path}. Use a fresh --output-dir.")
    history = read_json(history_path)
    last, best = load_checkpoint(last_path), load_checkpoint(best_path)
    expected_digest = digest(config)
    if last.get("config_sha256") != expected_digest or best.get("config_sha256") != expected_digest:
        raise ValueError(f"Checkpoint configuration identity is missing or different: {path}")
    if not history or [row["epoch"] for row in history] != list(range(1, len(history) + 1)):
        raise ValueError(f"Invalid epoch history: {path}")
    if last.get("epoch") != len(history) or last.get("history_sha256") != digest(history):
        raise ValueError(f"Last checkpoint and history disagree: {path}")
    if not all(math.isfinite(float(row["composite"])) for row in history):
        raise ValueError(f"Invalid validation composite: {path}")
    best_epoch = max(range(len(history)), key=lambda i: history[i]["composite"]) + 1
    if best.get("epoch") != best_epoch or "rng_state" not in last:
        raise ValueError(f"Best checkpoint or random state is inconsistent: {path}")
    return last, history


def capture_rng_state():
    import random
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng_state(state):
    import random
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
