"""Prepare an unpacked official AI2D archive without modifying existing data.

Images and question JSON are the only annotation inputs. No captions, manually
annotated edges, or previously computed features are required. OCR v2 uses the
project's existing three-variant Tesseract recognizer. Historical caches may
have been produced with different OCR/model versions; this creates a new,
auditable preparation rather than promising identical historical features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

from prepare_ai2d_model_matrix import (
    EXTERNAL_ROOT,
    ROOT,
    load_official_test_ids,
    portable_path,
    prepare_matrix,
    sha256_file,
)
from vqa_retrieval.ai2d_hybrid import (
    assign_splits_to_samples,
    build_hybrid_samples_from_ai2d,
    create_image_level_splits,
    write_manifest_hybrid,
)


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def write_json_new(path: Path, value: dict) -> None:
    """Exclusive creation protects both existing caches and experiment outputs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def configure_ocr(tesseract_cmd: str | None, language: str) -> tuple[dict, object]:
    # Preparation --help and --check-inputs do not require OCR dependencies.
    try:
        import cv2
        import numpy
        import pytesseract
        from vqa_retrieval.ocr_v2 import DEFAULT_TESSERACT_VARIANTS, run_tesseract_ocr_v2
    except ImportError as exc:
        raise ValueError("OCR preparation needs opencv-python(-headless), numpy and pytesseract; "
                         "install the documented research environment") from exc
    command = tesseract_cmd or shutil.which("tesseract")
    if command is None:
        windows_default = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        command = str(windows_default) if windows_default.is_file() else None
    if not command:
        raise ValueError("Tesseract executable not found. Install Tesseract with the requested language "
                         "data and add it to PATH or pass --tesseract-cmd PATH")
    pytesseract.pytesseract.tesseract_cmd = command
    try:
        version = str(pytesseract.get_tesseract_version())
        available_languages = pytesseract.get_languages(config="")
        languages_report = subprocess.run([command, "--list-langs"], check=True, capture_output=True,
                                          text=True, encoding="utf-8", errors="replace")
    except Exception as exc:
        raise ValueError(f"Cannot start Tesseract at {command!r}: {exc}") from exc
    missing = set(language.split("+")) - set(available_languages)
    if missing:
        raise ValueError(f"Tesseract language data unavailable: {sorted(missing)}; available: {available_languages}")
    location = re.search(r'List of available languages in "([^"]+)"', languages_report.stdout + languages_report.stderr)
    language_hashes = {}
    if location:
        for name in language.split("+"):
            traineddata = Path(location.group(1)) / f"{name}.traineddata"
            if traineddata.is_file():
                language_hashes[name] = sha256_file(traineddata)
    # Bound OCR worker parallelism consistently across machines.
    os.environ["OMP_THREAD_LIMIT"] = "1"
    engine = {
        "name": "tesseract_v2", "tesseract_version": version,
        "pytesseract_version": pytesseract.__version__, "opencv_version": cv2.__version__,
        "numpy_version": numpy.__version__, "language": language, "omp_thread_limit": 1,
        "python_version": sys.version.split()[0], "language_data_sha256": language_hashes,
        "variants": [asdict(item) for item in DEFAULT_TESSERACT_VARIANTS],
    }
    return engine, run_tesseract_ocr_v2


def inspect_inputs(data_root: Path, prepared_dir: Path, official_test_ids: Path,
                   path_root: Path, seed: int, val_ratio: float,
                   allow_missing_document_only_test_images: bool = False) -> tuple[list, dict]:
    for path in (data_root, prepared_dir, official_test_ids):
        portable_path(path, path_root)
    samples = build_hybrid_samples_from_ai2d(data_root, prepared_dir, strict=True, include_captions=False)
    image_ids = [path.stem for path in (data_root / "images").glob("*.png")]
    split = create_image_level_splits(
        (sample.image_id for sample in samples), load_official_test_ids(official_test_ids),
        seed=seed, val_ratio=val_ratio, available_image_ids=image_ids,
    )
    if split["missing_test_ids"] and not allow_missing_document_only_test_images:
        raise ValueError(f"Official test image files are missing: {split['missing_test_ids'][:10]}")
    return assign_splits_to_samples(samples, split), split


def prepare_raw(args: argparse.Namespace) -> dict:
    data_root = args.data_root.resolve()
    path_root = args.path_root.resolve()
    prepared_dir = (args.prepared_dir or data_root / "prepared_v2").resolve()
    output_dir = (args.output_dir or data_root / "model_matrix_v1").resolve()
    test_ids_path = args.official_test_ids.resolve()
    portable_path(output_dir, path_root)
    if prepared_dir == output_dir or prepared_dir in output_dir.parents or output_dir in prepared_dir.parents:
        raise ValueError("--prepared-dir and --output-dir must be separate, non-nested directories")
    if not 0 <= args.min_conf <= 100:
        raise ValueError("--min-conf must be in [0, 100]")
    samples, split = inspect_inputs(data_root, prepared_dir, test_ids_path, path_root, args.seed, args.val_ratio,
                                   args.allow_missing_document_only_test_images)
    if split["missing_test_ids"]:
        print("WARNING: Explicitly excluded missing official test images with no question rows: "
              + ", ".join(split["missing_test_ids"])
              + ". All question rows are retained; full archive validation is false.", file=sys.stderr)
    if args.check_inputs:
        return {"dataset": "ai2d", "mode": "input-validation-only", "questions": len(samples),
                "questions_by_split": {name: sum(s.split == name for s in samples) for name in ("train", "val", "test")},
                "question_images": len({s.image_id for s in samples}),
                "document_only_test_images": len(split["document_only_test_ids"]),
                "missing_official_test_image_ids": split["missing_test_ids"],
                "full_archive_ready": not split["missing_test_ids"], "question_dataset_ready": True, "writes": 0}
    plan_path = prepared_dir / "preparation_plan.json"
    provenance_path = prepared_dir / "provenance.json"
    if prepared_dir.exists() and any(prepared_dir.iterdir()) and not (args.resume and plan_path.is_file()):
        raise ValueError(f"Prepared directory is not empty: {prepared_dir}; choose a new directory. "
                         "--resume is only for this script's matching preparation_plan.json")
    if output_dir.exists() and any(output_dir.iterdir()) and not (args.resume and provenance_path.is_file()):
        raise ValueError(f"Matrix output directory is not empty: {output_dir}; choose a new directory")
    engine, run_ocr = configure_ocr(args.tesseract_cmd, args.language)
    engine["min_conf"] = args.min_conf
    inputs = sorted(set((data_root / "images").glob("*.png")) |
                    set((data_root / "questions").glob("*.json")) | {test_ids_path})
    source_hashes = {portable_path(path, path_root): sha256_file(path) for path in inputs}
    plan = {
        "schema": "ai2d_raw_preparation_v1", "data_root": portable_path(data_root, path_root),
        "prepared_dir": portable_path(prepared_dir, path_root), "matrix_dir": portable_path(output_dir, path_root),
        "path_convention": "POSIX paths relative to --path-root (repository/data root)",
        "official_test_ids": portable_path(test_ids_path, path_root),
        "seed": args.seed, "val_ratio": args.val_ratio, "ocr": engine,
        "captions": "not generated or consumed",
        "answer_options_policy": "Preserve every official option slot, including blank distractors, and its original index",
        "questions_with_blank_distractors": [s.sample_id for s in samples if any(not x for x in s.options)],
        "allow_missing_document_only_test_images": args.allow_missing_document_only_test_images,
        "excluded_missing_document_only_test_image_ids": split["missing_test_ids"],
        "source_sha256": source_hashes,
        "code_sha256": {str(path.relative_to(ROOT).as_posix()): sha256_file(path) for path in (
            Path(__file__).resolve(), ROOT / "scripts/prepare_ai2d_model_matrix.py",
            ROOT / "src/vqa_retrieval/ai2d_hybrid.py", ROOT / "src/vqa_retrieval/ocr_v2.py")},
    }
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Cannot resume: source files, parameters or software changed; use new output directories")
        if provenance_path.exists():
            completed = json.loads(provenance_path.read_text(encoding="utf-8"))
            for relative, expected_hash in completed["output_sha256"].items():
                if sha256_file(path_root / relative) != expected_hash:
                    raise ValueError(f"Completed output hash changed: {relative}")
            return {"status": "already_complete_verified", "provenance": portable_path(provenance_path, path_root)}
    else:
        write_json_new(plan_path, plan)
    manifest_path = prepared_dir / "manifest_hybrid.jsonl"
    split_path = prepared_dir / "split_hybrid.json"
    if manifest_path.exists() or split_path.exists():
        raise ValueError("Incomplete finalization already wrote a manifest/split; inspect it and use new output directories")
    fingerprint = canonical_hash(plan)
    images = {sample.image_id: Path(sample.image_path) for sample in samples}
    ocr_paths = {}
    for index, (image_id, image_path) in enumerate(sorted(images.items()), 1):
        ocr_path = prepared_dir / "ocr_v2" / f"{image_id}.ocr.json"
        if ocr_path.exists():
            payload = json.loads(ocr_path.read_text(encoding="utf-8"))
            metadata = payload.pop("_preparation", {})
            if (metadata.get("fingerprint") != fingerprint or
                    metadata.get("payload_sha256") != canonical_hash(payload)):
                raise ValueError(f"OCR cache does not match this preparation: {ocr_path}")
        else:
            try:
                payload = run_ocr(image_path, lang=args.language, min_conf=args.min_conf).to_dict()
            except Exception as exc:
                raise ValueError(f"OCR failed for {image_path}: {exc}; fix the cause and rerun with --resume") from exc
            payload["image_path"] = portable_path(image_path, path_root)
            payload["_preparation"] = {"fingerprint": fingerprint, "payload_sha256": canonical_hash(payload)}
            write_json_new(ocr_path, payload)
        ocr_paths[image_id] = portable_path(ocr_path, path_root)
        if index == 1 or index % 25 == 0 or index == len(images):
            print(f"OCR {index}/{len(images)} images", file=sys.stderr, flush=True)
    portable_samples = [replace(sample, image_path=portable_path(Path(sample.image_path), path_root),
                                ocr_v2_path=ocr_paths[sample.image_id], short_description=None) for sample in samples]
    write_manifest_hybrid(portable_samples, manifest_path)
    write_json_new(split_path, split)
    audit = prepare_matrix(manifest_path, test_ids_path, output_dir, data_root / "images", path_root,
                           args.seed, args.val_ratio, args.allow_missing_document_only_test_images)
    outputs = [plan_path, manifest_path, split_path, output_dir / "manifest.jsonl",
               output_dir / "split.json", output_dir / "audit.json"] + [path_root / p for p in ocr_paths.values()]
    provenance = {
        "schema": "ai2d_raw_preparation_v1", "status": "complete", "preparation_fingerprint": fingerprint,
        "plan": portable_path(plan_path, path_root),
        "output_sha256": {portable_path(path, path_root): sha256_file(path) for path in outputs},
        "questions": audit["questions"], "question_images": audit["question_images"],
        "full_archive_ready": audit["full_archive_ready"], "question_dataset_ready": True,
        "warnings": audit["warnings"],
        "limitation": "New OCR depends on recorded Tesseract/language/software versions. Historical captions, "
                      "features and checkpoints are not regenerated or claimed equivalent. Official blank distractor "
                      "slots are preserved: this corrects historical preprocessing that shifted the training gold "
                      "answer for 2618.png:2618.png-3; use new dataset hashes for new runs.",
    }
    write_json_new(provenance_path, provenance)
    return {**audit, "provenance": portable_path(provenance_path, path_root)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True, help="Unpacked official AI2D root containing images/ and questions/")
    parser.add_argument("--official-test-ids", type=Path, required=True, help="Official AI2D test-image ID CSV, one ID per row")
    parser.add_argument("--prepared-dir", type=Path, help="New OCR/hybrid output directory; default: DATA_ROOT/prepared_v2")
    parser.add_argument("--output-dir", type=Path, help="New matrix output directory; default: DATA_ROOT/model_matrix_v1")
    parser.add_argument("--path-root", type=Path, default=EXTERNAL_ROOT, help="Repository/data root used to resolve portable paths")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--allow-missing-document-only-test-images", action="store_true",
                        help="Explicitly exclude absent official test images with no question rows; full_archive_ready remains false")
    parser.add_argument("--language", default="eng")
    parser.add_argument("--min-conf", type=float, default=35.0)
    parser.add_argument("--tesseract-cmd", help="Path to the Tesseract executable if it is not on PATH")
    parser.add_argument("--check-inputs", action="store_true", help="Validate complete raw metadata/splits without OCR or writes")
    parser.add_argument("--resume", action="store_true", help="Resume only this script's matching, unmodified preparation")
    args = parser.parse_args()
    try:
        result = prepare_raw(args)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"AI2D preparation failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
