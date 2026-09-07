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
    ann_dir = data_root / "InfographicVQA" / "infographicsvqa_qas"
    images_dir = data_root / "InfographicVQA" / "infographicsvqa_images"
    ann_path = ann_dir / f"infographicsVQA_{split}_v1.0.json"
    if split == "val":
        with_qt = ann_dir / "infographicsVQA_val_v1.0_withQT.json"
        ann_path = with_qt if with_qt.exists() else ann_path

    payload = read_json(ann_path)
    rows: list[dict[str, Any]] = []
    for item in payload.get("data", []):
        image_name = str(item.get("image_local_name", "")).strip()
        if not image_name:
            continue
        image_path = images_dir / image_name
        qid = str(item.get("questionId", "")).strip()
        rows.append(
            {
                "dataset": "infographicvqa",
                "split": split,
                "sample_id": qid or f"{split}:{len(rows)}",
                "question_id": qid,
                "image_id": Path(image_name).stem,
                "image_name": image_name,
                "image_path": str(image_path),
                "question": str(item.get("question", "")).strip(),
                "answers": [str(answer) for answer in item.get("answers", []) if str(answer).strip()],
                "answer_type": item.get("answer_type", []),
                "evidence": item.get("evidence", []),
                "operation_reasoning": item.get("operation/reasoning", []),
                "ocr_output_file": item.get("ocr_output_file"),
                "image_exists": image_path.exists(),
            }
        )
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare InfographicVQA manifest for defense experiments.")
    parser.add_argument("--data-root", type=Path, default=EXTERNAL_ROOT / "infographicvqa")
    parser.add_argument("--output-dir", type=Path, default=EXTERNAL_ROOT / "infographicvqa" / "prepared_v1")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []
    split_payload: dict[str, Any] = {"dataset": "infographicvqa", "source_root": str(args.data_root), "splits": {}}
    for split in args.splits:
        rows = iter_split_rows(args.data_root, split)
        all_rows.extend(rows)
        split_payload["splits"][split] = {
            "questions": len(rows),
            "images": len({row["image_id"] for row in rows}),
            "missing_images": sum(1 for row in rows if not row["image_exists"]),
            "sample_ids": [row["sample_id"] for row in rows],
            "image_ids": sorted({row["image_id"] for row in rows}),
        }

    leakage = {}
    split_images = {name: set(payload["image_ids"]) for name, payload in split_payload["splits"].items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if left in split_images and right in split_images:
            leakage[f"{left}-{right}"] = sorted(split_images[left] & split_images[right])
    split_payload["image_leakage"] = leakage

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    split_path = args.output_dir / "split.json"
    write_jsonl(all_rows, manifest_path)
    split_path.write_text(json.dumps(split_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "split": str(split_path), "rows": len(all_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
