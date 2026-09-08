from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = ROOT.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vqa_retrieval.ai2d_hybrid import (  # noqa: E402
    Ai2dHybridSample,
    assign_splits_to_samples,
    create_image_level_splits,
    write_manifest_hybrid,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path, path_root: Path) -> str:
    """Manifest paths are POSIX paths relative to the repository/data root."""
    try:
        return path.resolve().relative_to(path_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{path} is outside --path-root {path_root}; use a common data root") from exc


def load_official_test_ids(path: Path) -> list[str]:
    ids = []
    for row in csv.reader(path.read_text(encoding="utf-8-sig").splitlines()):
        if not row or not row[0].strip():
            continue
        token = row[0].strip()
        if not ids and token.lower() in {"id", "image_id", "imageid"}:
            continue
        if "/" in token or chr(92) in token:
            raise ValueError(f"Expected an image ID, not a path, in {path}: {token!r}")
        ids.append(Path(token).stem if token.lower().endswith(".png") else token)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"Official test IDs must be nonempty and unique: {path}")
    return ids


def validated_samples(source_manifest: Path, path_root: Path) -> list[Ai2dHybridSample]:
    samples = []
    for number, line in enumerate(source_manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        options = row.get("options")
        index = row.get("correct_option_idx")
        if (not isinstance(options, list) or not options or
                any(not isinstance(x, str) for x in options) or
                type(index) is not int or not 0 <= index < len(options) or not options[index].strip()):
            raise ValueError(f"Invalid answer options/index at {source_manifest}:{number}")
        for field in ("sample_id", "image_id", "image_path", "ocr_v2_path", "question"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Missing {field} at {source_manifest}:{number}; run prepare_ai2d_raw.py first")
        if row.get("correct_option_text") != options[index]:
            raise ValueError(f"Answer text/index disagree at {source_manifest}:{number}")
        for field in ("image_path", "ocr_v2_path"):
            path = Path(row[field].replace(chr(92), "/"))
            if not path.is_absolute():
                path = path_root / path
            if not path.is_file():
                raise ValueError(f"Missing {field} at {source_manifest}:{number}: {path}")
            row[field] = portable_path(path, path_root)
        samples.append(Ai2dHybridSample.from_dict(row))
    ids = [sample.sample_id for sample in samples]
    if not samples or len(set(ids)) != len(ids):
        raise ValueError("Question manifest must be nonempty with unique sample IDs")
    image_paths = {}
    for sample in samples:
        if sample.image_id in image_paths and image_paths[sample.image_id] != sample.image_path:
            raise ValueError(f"Different image paths share image_id={sample.image_id}")
        image_paths[sample.image_id] = sample.image_path
    return samples


def prepare_matrix(source_manifest: Path, official_test_ids: Path, output_dir: Path,
                   images_dir: Path, path_root: Path, seed: int = 42,
                   val_ratio: float = 0.1, allow_missing_document_only_test_images: bool = False) -> dict:
    """Create a fresh audited matrix; never replace previously generated results."""
    path_root = path_root.resolve()
    for path in (source_manifest, official_test_ids, output_dir, images_dir):
        portable_path(path, path_root)
    if not images_dir.is_dir():
        raise ValueError(f"Missing --images-dir: {images_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_dir}; choose a new directory")
    samples = validated_samples(source_manifest, path_root)
    image_files = sorted(images_dir.glob("*.png"))
    inventory = {path.stem for path in image_files}
    split = create_image_level_splits(
        (sample.image_id for sample in samples), load_official_test_ids(official_test_ids),
        val_ratio=val_ratio, seed=seed, available_image_ids=inventory,
    )
    if split["missing_test_ids"] and not allow_missing_document_only_test_images:
        raise ValueError(f"Official test images are missing from {images_dir}: {split['missing_test_ids'][:10]}")
    assigned = assign_splits_to_samples(samples, split)
    by_split = {name: [sample for sample in assigned if sample.split == name]
                for name in ("train", "val", "test")}
    image_sets = {name: {sample.image_id for sample in rows} for name, rows in by_split.items()}
    leakage = {f"{left}-{right}": sorted(image_sets[left] & image_sets[right])
               for left, right in (("train", "val"), ("train", "test"), ("val", "test"))}
    if any(leakage.values()):
        raise ValueError(f"Image leakage: {leakage}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = write_manifest_hybrid(assigned, output_dir / "manifest.jsonl")
    split_path = output_dir / "split.json"
    split_path.write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    audit = {
        "dataset": "ai2d", "schema": "ai2d_hybrid_matrix_v1",
        "path_convention": "POSIX paths relative to --path-root (repository/data root)",
        "manifest": portable_path(manifest_path, path_root),
        "manifest_sha256": sha256_file(manifest_path),
        "split_file": portable_path(split_path, path_root),
        "split_sha256": sha256_file(split_path),
        "source_manifest": portable_path(source_manifest, path_root),
        "source_manifest_sha256": sha256_file(source_manifest),
        "official_test_ids_source": portable_path(official_test_ids, path_root),
        "official_test_ids_sha256": sha256_file(official_test_ids),
        "images_dir": portable_path(images_dir, path_root),
        "image_inventory_verified": True,
        "seed": seed, "val_ratio": val_ratio,
        "questions": {name: len(rows) for name, rows in by_split.items()},
        "question_images": {name: len(image_sets[name]) for name in by_split},
        "official_test_images": len(split["test_image_ids"]),
        "document_only_test_images": len(split["document_only_test_ids"]),
        "missing_official_test_ids": len(split["missing_test_ids"]),
        "missing_official_test_image_ids": split["missing_test_ids"],
        "allow_missing_document_only_test_images": allow_missing_document_only_test_images,
        "image_leakage": leakage, "ready": not split["missing_test_ids"],
        "full_archive_ready": not split["missing_test_ids"], "question_dataset_ready": True,
        "warnings": (["Explicitly excluded missing official test images with no question rows: "
                      + ", ".join(split["missing_test_ids"])
                      + ". All question rows are retained; full archive validation is false."]
                     if split["missing_test_ids"] else []),
    }
    (output_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare AI2D matrix from an OCR hybrid manifest; refuses nonempty output directories.")
    parser.add_argument("--source-manifest", type=Path, default=EXTERNAL_ROOT / "ai2d/prepared_v2/manifest_hybrid.jsonl")
    parser.add_argument("--official-test-ids", type=Path, default=EXTERNAL_ROOT / "ai2d/ai2d_test_ids (1).csv")
    parser.add_argument("--output-dir", type=Path, default=EXTERNAL_ROOT / "ai2d/model_matrix_v1")
    parser.add_argument("--images-dir", type=Path, default=EXTERNAL_ROOT / "ai2d/images")
    parser.add_argument("--path-root", type=Path, default=EXTERNAL_ROOT,
                        help="Root against which image/OCR paths resolve in notebooks and matrix loaders")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--allow-missing-document-only-test-images", action="store_true",
                        help="Explicitly exclude absent official test images with no question rows; full_archive_ready remains false")
    args = parser.parse_args()
    try:
        audit = prepare_matrix(args.source_manifest, args.official_test_ids, args.output_dir,
                               args.images_dir, args.path_root, args.seed, args.val_ratio,
                               args.allow_missing_document_only_test_images)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"AI2D preparation failed: {exc}\nFor raw archives, run prepare_ai2d_raw.py --help.\n")
    for warning in audit["warnings"]:
        print(f"WARNING: {warning}", file=sys.stderr)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
