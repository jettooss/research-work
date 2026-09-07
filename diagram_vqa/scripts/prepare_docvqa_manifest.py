from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = ROOT.parent


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_split_rows(data_root: Path, split: str) -> list[dict[str, Any]]:
    split_root = data_root / split
    payload = read_json(split_root / f"{split}_v1.0.json")
    rows: list[dict[str, Any]] = []
    for item in payload.get("data", []):
        relative_image = Path(str(item.get("image", "")).strip())
        image_path = split_root / relative_image
        image_id = relative_image.stem
        ocr_path = split_root / "ocr_results" / f"{image_id}.json"
        question_id = str(item.get("questionId", "")).strip()
        rows.append(
            {
                "dataset": "docvqa",
                "split": split,
                "sample_id": question_id or f"{split}:{len(rows)}",
                "question_id": question_id,
                "doc_id": str(item.get("docId", "")).strip(),
                "source_document_id": str(item.get("ucsf_document_id", "")).strip(),
                "page_number": str(item.get("ucsf_document_page_no", "")).strip(),
                "image_id": image_id,
                "image_path": str(image_path.resolve()),
                "ocr_path": str(ocr_path.resolve()),
                "question": str(item.get("question", "")).strip(),
                "answers": [str(answer) for answer in item.get("answers", []) if str(answer).strip()],
                "image_exists": image_path.exists(),
                "ocr_exists": ocr_path.exists(),
            }
        )
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the official DocVQA splits for reproducible experiments.")
    parser.add_argument("--data-root", type=Path, default=EXTERNAL_ROOT / "docvqa")
    parser.add_argument("--output-dir", type=Path, default=EXTERNAL_ROOT / "docvqa" / "prepared_v1")
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val", "test"])
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []
    split_payload: dict[str, Any] = {"dataset": "docvqa", "source_root": str(args.data_root.resolve()), "splits": {}}
    for split in args.splits:
        rows = iter_split_rows(args.data_root, split)
        all_rows.extend(rows)
        split_payload["splits"][split] = {
            "questions": len(rows),
            "images": len({row["image_id"] for row in rows}),
            "documents": len({row["doc_id"] for row in rows}),
            "source_documents": len({row["source_document_id"] for row in rows}),
            "missing_images": sum(not row["image_exists"] for row in rows),
            "missing_ocr": sum(not row["ocr_exists"] for row in rows),
            "questions_without_answers": sum(not row["answers"] for row in rows),
            "sample_ids": [row["sample_id"] for row in rows],
            "image_ids": sorted({row["image_id"] for row in rows}),
            "doc_ids": sorted({row["doc_id"] for row in rows}),
            "source_document_ids": sorted({row["source_document_id"] for row in rows}),
        }

    split_payload["image_leakage"] = {}
    split_payload["document_page_leakage"] = {}
    split_payload["source_document_overlap"] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if left not in split_payload["splits"] or right not in split_payload["splits"]:
            continue
        left_payload = split_payload["splits"][left]
        right_payload = split_payload["splits"][right]
        split_payload["image_leakage"][f"{left}-{right}"] = sorted(
            set(left_payload["image_ids"]) & set(right_payload["image_ids"])
        )
        split_payload["document_page_leakage"][f"{left}-{right}"] = sorted(
            set(left_payload["doc_ids"]) & set(right_payload["doc_ids"])
        )
        # Official DocVQA splits may contain different pages from the same UCSF
        # source document. Record this for transparency, but do not call it page leakage.
        split_payload["source_document_overlap"][f"{left}-{right}"] = sorted(
            set(left_payload["source_document_ids"]) & set(right_payload["source_document_ids"])
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    split_path = args.output_dir / "split.json"
    write_jsonl(all_rows, manifest_path)
    split_path.write_text(json.dumps(split_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "split": str(split_path), "rows": len(all_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
