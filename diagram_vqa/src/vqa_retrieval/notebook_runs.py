"""Prepare isolated copies of the actual research notebooks without changing models."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from .experiment_matrix import ARCHITECTURES, DATASETS
from .model_matrix import manifest_for


def notebook_paths(root: Path, dataset: str | None = None, architecture: str | None = None) -> list[Path]:
    if dataset is not None and dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    if architecture is not None and architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {architecture}")
    return [root / "notebooks/experiments" / d / f"{d}_{a}.ipynb"
            for d in ([dataset] if dataset else DATASETS)
            for a in ([architecture] if architecture else ARCHITECTURES)]


def _source(cell: dict[str, Any]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def notebook_defaults(notebook: dict[str, Any]) -> dict[str, int]:
    result = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse(_source(cell)).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in {"SEED", "EPOCHS"}:
                    # All current conditional epoch defaults use SMOKE; reproduction is full data.
                    value = node.value
                    if isinstance(value, ast.IfExp) and isinstance(value.test, ast.Name) and value.test.id == "SMOKE":
                        value = value.orelse
                    number = ast.literal_eval(value)
                    if type(number) is not int:
                        raise ValueError(f"Non-integer notebook default: {target.id}")
                    result[target.id] = number
    if set(result) != {"SEED", "EPOCHS"} or result["EPOCHS"] < 1:
        raise ValueError("Notebook must declare integer SEED and positive EPOCHS")
    return result


def _rewrite_cell(source: str, values: dict[str, str], found: dict[str, int], resume: bool) -> str:
    tree = ast.parse(source)
    edits = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {"load_manifest", "load_rows"}:
            base = "DATA_ROOT" if node.name == "load_manifest" else "external_root"
            for statement in node.body:
                if isinstance(statement, ast.Return) and isinstance(statement.value, ast.Name):
                    rows = statement.value.id
                    lines = [f"for _row in {rows}:",
                             "    for _key in ('ocr_path', 'ocr_v2_path'):",
                             "        _value = _row.get(_key)",
                             "        if _value:",
                             "            _path = Path(str(_value).replace('\\\\', '/'))",
                             f"            _row[_key] = str((_path if _path.is_absolute() else {base} / _path).resolve())",
                             f"return {rows}"]
                    replacement = ("\n" + " " * statement.col_offset).join(lines)
                    edits.append((statement, replacement))
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in values:
                found[target.id] = found.get(target.id, 0) + 1
                edits.append((node.value, values[target.id]))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "MatrixFeatureStore":
            if len(node.args) >= 3:
                edits.append((node.args[2], values["MATRIX_CACHE_ROOT"]))
            else:
                for keyword in node.keywords:
                    if keyword.arg == "cache_root":
                        edits.append((keyword.value, values["MATRIX_CACHE_ROOT"]))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "train_local_experiment":
            for keyword in node.keywords:
                if keyword.arg == "resume":
                    edits.append((keyword.value, repr(resume)))
    # AST column offsets count UTF-8 bytes, including on lines containing non-ASCII text.
    lines = source.encode("utf-8").splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    data = source.encode("utf-8")
    spans = [(offsets[n.lineno - 1] + n.col_offset,
              offsets[n.end_lineno - 1] + n.end_col_offset, text.encode("utf-8")) for n, text in edits]
    for start, end, replacement in sorted(spans, reverse=True):
        data = data[:start] + replacement + data[end:]
    return data.decode("utf-8")


def manifest_fingerprint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "sha256": None, "bytes": None, "missing": True}
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
            size += len(chunk)
    return {"path": str(path), "sha256": h.hexdigest(), "bytes": size, "missing": False}


def prepare_notebook(path: Path, output_root: Path, *, seed: int = 42, epochs: int | None = None,
                     external_root: Path | None = None, kernel_name: str = "python3",
                     resume: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    original = path.read_bytes()
    notebook = copy.deepcopy(json.loads(original))
    dataset = path.parent.name
    architecture = path.stem.removeprefix(dataset + "_")
    if dataset not in DATASETS or architecture not in ARCHITECTURES:
        raise ValueError(f"Not an active experiment notebook: {path}")
    defaults = notebook_defaults(notebook)
    epochs = defaults["EPOCHS"] if epochs is None else epochs
    if epochs < 1 or seed < 0:
        raise ValueError("epochs must be positive and seed nonnegative")
    root = path.resolve().parents[3]
    external_root = (external_root or root.parent).resolve()
    run_dir = output_root.resolve() / dataset / architecture / f"seed{seed}_full"
    source_hash = hashlib.sha256(original).hexdigest()
    manifest = manifest_fingerprint(manifest_for(dataset, external_root))
    cache_key = hashlib.sha256(json.dumps({"source": source_hash, "manifest": manifest,
                                         "external_root": str(external_root)}, sort_keys=True).encode()).hexdigest()
    feature_cache = output_root.resolve() / "feature_cache" / cache_key / dataset / architecture
    config = {"dataset": dataset, "architecture": architecture, "seed": seed, "epochs": epochs,
              "full_data": True, "resume": resume, "kernel_name": kernel_name,
              "source_notebook": str(path.resolve()), "source_sha256": source_hash,
              "project_root": str(root), "external_root": str(external_root), "run_dir": str(run_dir),
              "input_manifest": manifest, "feature_cache_key": cache_key, "feature_cache_dir": str(feature_cache),
              "fingerprint_scope": "Manifest bytes and external-root location; image, OCR, model and existing cache bytes are not hashed."}
    values = {"SEED": repr(seed), "EPOCHS": repr(epochs), "SMOKE": "False",
              "FORCE_RETRAIN": repr(not resume), "RUN_DIR": f"Path({str(run_dir)!r})",
              "ROOT": f"Path({str(root)!r})", "PROJECT_ROOT": f"Path({str(root)!r})",
              "EXTERNAL_ROOT": f"Path({str(external_root)!r})", "DATA_ROOT": f"Path({str(external_root)!r})",
              "FEATURE_CACHE": f"Path({str(feature_cache)!r})", "MATRIX_CACHE_ROOT": f"Path({str(feature_cache / 'matrix')!r})"}
    found: dict[str, int] = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["source"] = _rewrite_cell(_source(cell), values, found, resume).splitlines(keepends=True)
            cell["execution_count"] = None
            cell["outputs"] = []
    for name in ("SEED", "EPOCHS", "RUN_DIR"):
        if found.get(name) != 1:
            raise ValueError(f"Expected exactly one top-level {name} assignment; found {found.get(name, 0)}")
    if not (found.get("DATA_ROOT") or found.get("EXTERNAL_ROOT")):
        raise ValueError("Notebook does not expose a data-root assignment")
    notebook.setdefault("metadata", {})["kernelspec"] = {
        "display_name": kernel_name, "language": "python", "name": kernel_name}
    notebook["metadata"]["reproduction"] = config
    return notebook, config


def reserve_run(config: dict[str, Any], *, require_manifest: bool = True) -> Path:
    """Refuse unrelated existing outputs, including directories without a recorded config."""
    run_dir = Path(config["run_dir"])
    recorded_input = config["input_manifest"]
    current_input = manifest_fingerprint(Path(recorded_input["path"]))
    if require_manifest and current_input["missing"]:
        raise FileNotFoundError(f"Prepared dataset manifest is required before execution: {current_input['path']}")
    if current_input != recorded_input:
        raise ValueError(f"Input manifest changed after planning: {current_input['path']}; prepare a new run")
    record = run_dir / "notebook_run.json"
    if run_dir.exists() and any(run_dir.iterdir()):
        if not config["resume"] or not record.is_file():
            raise FileExistsError(f"Run directory is not empty: {run_dir}; choose a new output root")
        previous = json.loads(record.read_text(encoding="utf-8"))
        identity = {key: value for key, value in config.items() if key != "resume"}
        if identity != {key: value for key, value in previous.items() if key != "resume"}:
            raise ValueError(f"Cannot resume with a different notebook/configuration: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return run_dir
