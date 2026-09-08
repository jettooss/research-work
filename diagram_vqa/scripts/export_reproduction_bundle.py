"""Create or verify a portable, checksummed archive for the published runs.

Default: code, saved evaluation evidence and available per-run metadata/predictions.
--include-data adds portable prepared manifests and selected OCR files; source data is unchanged.
--include-weights adds best model checkpoints, never optimizer/last checkpoints.
Images, feature caches and pretrained model downloads are not included.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_name(value):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("Unsafe archive path")
    path = PurePosixPath(value)
    if (path.is_absolute() or ".." in path.parts or ":" in value or "\\" in value
            or str(path) != value or value == "." or value.endswith("/")):
        raise ValueError(f"Unsafe archive path: {value!r}")
    return value


def child(root, name):
    path = root / safe_name(name)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Source path escapes allowed root: {name}")
    return path


class BundleFile:
    def __init__(self, path, payload=None, original_sha256=None):
        self.path = path
        self.payload = payload
        self.original_sha256 = original_sha256

    @property
    def size(self):
        return len(self.payload) if self.payload is not None else self.path.stat().st_size

    def inventory(self, name):
        archived_digest = hashlib.sha256(self.payload).hexdigest() if self.payload is not None else digest_file(self.path)
        item = {"path": name, "bytes": self.size, "sha256": archived_digest}
        if self.payload is not None:
            item.update(original_sha256=self.original_sha256 or digest_file(self.path), transformation="manifest paths relative to data root")
        return item


def portable_data_path(value, dataset, data_root, *, ocr=False):
    """Relocate only dataset paths; never follow an arbitrary original absolute path."""
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"Invalid data path: {value!r}")
    normalized = value.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    if ".." in parts:
        raise ValueError(f"Unsafe data path: {value}")
    absolute = Path(value).is_absolute() or PureWindowsPath(value).is_absolute()
    if absolute:
        local = Path(value)
        if local.is_absolute() and local.is_relative_to(data_root):
            relative = local.relative_to(data_root).as_posix()
        else:
            # A moved Windows/Linux manifest can retain the old author's prefix.
            # Accept exactly one recognizable dataset/cache suffix, resolved only
            # against the explicitly selected new data root.
            starts = [i for i, part in enumerate(parts) if part == dataset and
                      (not ocr or i == 0 or parts[i - 1] != "ocr")]
            cache_starts = [i for i in range(len(parts) - 2)
                            if parts[i:i + 3] == ("model_matrix_cache", "ocr", dataset)]
            candidates = ["/".join(parts[i:]) for i in cache_starts] or ["/".join(parts[i:]) for i in starts]
            if len(candidates) != 1:
                raise ValueError(f"Cannot safely relocate data path: {value}")
            relative = candidates[0]
    else:
        relative = normalized
    safe_name(relative)
    if ocr:
        allowed = (f"{dataset}/prepared_v2/ocr_v2/", f"model_matrix_cache/ocr/{dataset}/",
                   *(f"{dataset}/{split}/ocr_results/" for split in ("train", "val", "test")))
        if not relative.startswith(allowed) or not relative.lower().endswith(".json"):
            raise ValueError(f"OCR path is outside recognized OCR directories: {relative}")
    elif not relative.startswith(dataset + "/"):
        raise ValueError(f"Image path is outside its dataset: {relative}")
    # This normalizes a reference only. collect() validates every file that will
    # actually be read against the resolved source root. Excluded image files
    # need no filesystem walk (many questions share each image).
    return relative


def collect(root, data_root, include_data, include_weights):
    root = root.resolve()
    data_root = data_root.resolve()
    evidence = json.loads(child(root, "reports/reproducibility/evidence.json").read_text(encoding="utf-8"))
    files = {}
    missing = []
    # Questions share images/OCR; resolve each referenced file once per bundle.
    resolved = lru_cache(maxsize=None)(lambda path: path.resolve())
    data_child = lru_cache(maxsize=None)(lambda name: data_root / safe_name(name))
    relocate = lru_cache(maxsize=None)(lambda value, dataset, ocr: portable_data_path(value, dataset, data_root, ocr=ocr))

    def add(path, name, required=False, payload=None, scope=None, original_sha256=None):
        safe_name(name)
        if name == "bundle_manifest.json":
            raise ValueError("Reserved archive path")
        previous = files.get(name)
        if previous is not None:
            if previous.path != path or previous.payload != payload:
                raise ValueError(f"Conflicting bundle inputs: {name}")
            return
        if not resolved(path).is_relative_to(resolved(scope or root)):
            raise ValueError(f"Source path escapes allowed root: {path}")
        if path.is_file():
            files[name] = BundleFile(path, payload, original_sha256)
        elif required:
            missing.append(name)

    for folder, suffixes in (("src", {".py"}), ("scripts", {".py"}), ("examples", {".py"}),
                             ("tests", {".py"}), ("notebooks/experiments", {".ipynb"}),
                             ("docs", {".md", ".json"}), ("experiments", {".json"}),
                             ("reports/reproducibility", {".json", ".md"}),
                             ("reports/presentation_metrics", {".py"})):
        for path in (root / folder).rglob("*"):
            if path.is_file() and path.suffix in suffixes:
                add(path, "diagram_vqa/" + path.relative_to(root).as_posix())
    for path in root.glob("requirements*.txt"):
        add(path, "diagram_vqa/" + path.name)
    add(root / "pyproject.toml", "diagram_vqa/pyproject.toml", True)
    for name in ("README.md", "Dockerfile", ".dockerignore"):
        add(root.parent / name, name, name == "README.md", scope=root.parent)
    add(root / "examples/retrieval_metrics.py", "diagram_vqa/examples/retrieval_metrics.py", True)
    for name in ("retrieval_matrix_audit.json", "evaluate_saved_retrieval.py"):
        add(root / "reports/presentation_metrics" / name, "diagram_vqa/reports/presentation_metrics/" + name, True)
    for group in evidence["accuracy"] + evidence["retrieval"]:
        for run in group["runs"]:
            metric = child(root, run["metrics_path"])
            add(metric, "diagram_vqa/" + run["metrics_path"], True)
            training_dir = child(root, run["training_run"]) if run.get("training_run") else None
            run_dirs = {metric.parent}
            if training_dir is not None:
                run_dirs.add(training_dir)
            for directory in run_dirs:
                for name in ("config.json", "history.json", "notebook_run.json", "metrics.json",
                             "predictions.jsonl", "predictions.json", "retrieval_unique_test.json"):
                    path = directory / name
                    add(path, "diagram_vqa/" + path.relative_to(root).as_posix())
                if include_weights:
                    path = directory / "checkpoint_best.pt"
                    required = directory == training_dir or (directory.name.endswith("_full") and group.get("architecture") != "random")
                    add(path, "diagram_vqa/" + path.relative_to(root).as_posix(), required)
    if include_data:
        for dataset, info in evidence["datasets"].items():
            if dataset not in {"ai2d", "docvqa", "infographicvqa"}:
                raise ValueError(f"Unknown dataset: {dataset}")
            manifest_name = safe_name(info["manifest_relative_to_data_root"])
            if not manifest_name.startswith(dataset + "/") or not manifest_name.endswith(".jsonl"):
                raise ValueError(f"Invalid dataset manifest path: {manifest_name}")
            manifest = child(data_root, manifest_name)
            if not manifest.is_file():
                missing.append(manifest_name)
                continue
            original_bytes = manifest.read_bytes()
            rows = []
            for line in original_bytes.decode("utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("image_path"):
                    row["image_path"] = relocate(row["image_path"], dataset, False)
                for key in ("ocr_path", "ocr_v2_path"):
                    if not row.get(key):
                        continue
                    name = relocate(row[key], dataset, True)
                    row[key] = name
                    add(data_child(name), name, True, scope=data_root)
                # Infographic notebooks can use a per-image OCR cache rather than
                # an OCR-path manifest field. Include only selected image IDs.
                image_id = str(row.get("image_id", ""))
                if image_id and image_id not in {".", ".."} and PurePosixPath(image_id).name == image_id and not any(c in image_id for c in ("\\", ":", "\x00")):
                    name = f"model_matrix_cache/ocr/{dataset}/{image_id}.json"
                    add(data_child(name), name, scope=data_root)
                rows.append(row)
            payload = ("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)).encode("utf-8")
            add(manifest, manifest_name, True, payload if payload != original_bytes else None, scope=data_root,
                original_sha256=hashlib.sha256(original_bytes).hexdigest())
            for name in ("split.json", "audit.json", "split_hybrid.json"):
                path = manifest.parent / name
                add(path, path.relative_to(data_root).as_posix(), scope=data_root)
    return files, sorted(set(missing))


def verify(path):
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [safe_name(info.filename) for info in infos]
        if len(names) != len(set(name.casefold() for name in names)):
            raise ValueError("Duplicate archive names")
        for info in infos:
            if stat.S_ISLNK(info.external_attr >> 16) or info.flag_bits & 1:
                raise ValueError("Symlink/encrypted archive entries are not supported")
        if "bundle_manifest.json" not in names:
            raise ValueError("Missing bundle_manifest.json")
        inventory = json.loads(archive.read("bundle_manifest.json"))
        if inventory.get("schema_version") != 1 or not isinstance(inventory.get("files"), list):
            raise ValueError("Invalid archive inventory schema")
        items = inventory["files"]
        if any(not isinstance(item, dict) or "path" not in item for item in items):
            raise ValueError("Invalid archive inventory entry")
        listed = [safe_name(item["path"]) for item in items]
        if "bundle_manifest.json" in listed or len(listed) != len(set(name.casefold() for name in listed)):
            raise ValueError("Duplicate/reserved inventory paths")
        if set(names) != set(listed) | {"bundle_manifest.json"}:
            raise ValueError("Archive inventory mismatch")
        for item in items:
            name = item["path"]
            if type(item.get("bytes")) is not int or item["bytes"] < 0 or archive.getinfo(name).file_size != item["bytes"]:
                raise ValueError(f"Archive size mismatch: {name}")
            h = hashlib.sha256()
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            if h.hexdigest() != item.get("sha256"):
                raise ValueError(f"Archive content mismatch: {name}")
    return len(items)


def write_bundle(output, files, include_data=False, include_weights=False):
    inventory = {"schema_version": 1, "includes_data": include_data, "includes_weights": include_weights,
                 "excluded": ["images", "feature caches", "pretrained model downloads", "optimizer checkpoints"],
                 "path_note": "Prepared manifest image/OCR paths are relative to the extracted data root. Original hashes identify transformed source manifests. Historical run configs retain provenance paths.",
                 "files": []}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, entry in sorted(files.items()):
            safe_name(name)
            h = hashlib.sha256()
            size = 0
            if entry.payload is None:
                with entry.path.open("rb") as source, archive.open(name, "w", force_zip64=True) as target:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(block)
                        h.update(block)
                        size += len(block)
            else:
                archive.writestr(name, entry.payload)
                h.update(entry.payload)
                size = len(entry.payload)
            item = {"path": name, "bytes": size, "sha256": h.hexdigest()}
            if entry.payload is not None:
                item.update(original_sha256=entry.original_sha256 or digest_file(entry.path),
                            transformation="manifest paths relative to data root")
            inventory["files"].append(item)
        archive.writestr("bundle_manifest.json", json.dumps(inventory, indent=2) + "\n")
    return verify(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path)
    action.add_argument("--verify", type=Path)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--include-data", action="store_true")
    parser.add_argument("--include-weights", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps({"verified_files": verify(args.verify)}))
        return
    project_root = args.project_root.resolve()
    files, missing = collect(project_root, (args.data_root or project_root.parent).resolve(), args.include_data, args.include_weights)
    print(json.dumps({"files": len(files), "uncompressed_bytes": sum(entry.size for entry in files.values()),
                      "transformed_files": sum(entry.payload is not None for entry in files.values()),
                      "includes_data": args.include_data, "includes_weights": args.include_weights,
                      "missing_required_inputs": missing}))
    if missing:
        parser.exit(1, "Required bundle inputs missing; see dry-run statistics above.\n")
    if args.dry_run:
        return
    if args.output.exists():
        parser.error("Output already exists; use a new archive path.")
    count = write_bundle(args.output, files, args.include_data, args.include_weights)
    print(json.dumps({"verified_files": count, "archive": str(args.output)}))


if __name__ == "__main__":
    main()
