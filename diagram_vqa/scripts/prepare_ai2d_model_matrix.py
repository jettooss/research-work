from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = ROOT.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vqa_retrieval.ai2d_hybrid import (  # noqa: E402
    assign_splits_to_samples,
    create_image_level_splits,
    load_manifest_hybrid,
    write_manifest_hybrid,
)


def load_official_test_ids(path: Path) -> list[str]:
    return [line.strip().split(",")[0] for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the audited AI2D model-matrix split.")
    parser.add_argument("--source-manifest", type=Path, default=EXTERNAL_ROOT / "ai2d/prepared_v2/manifest_hybrid.jsonl")
    parser.add_argument("--official-test-ids", type=Path, default=EXTERNAL_ROOT / "ai2d/ai2d_test_ids (1).csv")
    parser.add_argument("--output-dir", type=Path, default=EXTERNAL_ROOT / "ai2d/model_matrix_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    samples = load_manifest_hybrid(args.source_manifest)
    split = create_image_level_splits(
        (sample.image_id for sample in samples),
        load_official_test_ids(args.official_test_ids),
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    assigned = assign_splits_to_samples(samples, split)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = write_manifest_hybrid(assigned, args.output_dir / "manifest.jsonl")
    split_path = args.output_dir / "split.json"
    split_path.write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")

    by_split = {name: [sample for sample in assigned if sample.split == name] for name in ("train", "val", "test")}
    image_sets = {name: {sample.image_id for sample in rows} for name, rows in by_split.items()}
    leakage = {
        "train-val": sorted(image_sets["train"] & image_sets["val"]),
        "train-test": sorted(image_sets["train"] & image_sets["test"]),
        "val-test": sorted(image_sets["val"] & image_sets["test"]),
    }
    audit = {
        "dataset": "ai2d",
        "manifest": str(manifest_path.resolve()),
        "split_file": str(split_path.resolve()),
        "questions": {name: len(rows) for name, rows in by_split.items()},
        "question_images": {name: len(image_sets[name]) for name in by_split},
        "official_test_images": len(split["test_image_ids"]),
        "document_only_test_images": len(split["document_only_test_ids"]),
        "missing_official_test_ids": len(split["missing_test_ids"]),
        "image_leakage": leakage,
        "ready": not any(leakage.values()) and not split["missing_test_ids"],
    }
    (args.output_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
