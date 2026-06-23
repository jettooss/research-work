from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from vqa_retrieval.ai2d_hybrid import (
    Ai2dHybridSample,
    assign_splits_to_samples,
    create_image_level_splits,
    load_manifest_hybrid,
    read_test_ids_csv,
)
from vqa_retrieval.vlm_service import LineItem, build_analysis_prompt


@dataclass(frozen=True)
class SftRecord:
    split: str
    sample_id: str
    image_id: str
    image_path: str
    ocr_v2_path: str
    question: str
    prompt: str
    completion: str
    correct_option_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "sample_id": self.sample_id,
            "image_id": self.image_id,
            "image_path": self.image_path,
            "ocr_v2_path": self.ocr_v2_path,
            "question": self.question,
            "prompt": self.prompt,
            "completion": self.completion,
            "correct_option_text": self.correct_option_text,
        }


_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokens(text: str) -> set[str]:
    return {tok.lower() for tok in _TOKEN_RE.findall(str(text))}


def load_ocr_lines_from_json(ocr_path: str | Path, max_lines: int = 60) -> list[LineItem]:
    payload = json.loads(Path(ocr_path).read_text(encoding="utf-8"))
    lines_raw = payload.get("lines", [])
    out: list[LineItem] = []
    for idx, item in enumerate(lines_raw):
        text = str(item.get("text", "")).strip()
        bbox = item.get("bbox", [])
        if not text or len(bbox) != 4:
            continue
        conf = float(item.get("conf", 0.0))
        out.append(
            LineItem(
                line_index=idx,
                text=text,
                bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                confidence=max(0.0, min(1.0, conf / 100.0)),
            )
        )
        if len(out) >= max_lines:
            break
    return out


def _line_match_score(answer_text: str, line_text: str) -> float:
    answer_tokens = _tokens(answer_text)
    line_tokens = _tokens(line_text)
    if not answer_tokens or not line_tokens:
        return 0.0
    overlap = len(answer_tokens & line_tokens)
    if overlap == 0:
        return 0.0
    return overlap / max(1, len(answer_tokens))


def infer_gold_evidence_lines(
    correct_answer_text: str,
    ocr_lines: Sequence[LineItem],
    top_k: int = 3,
    min_score: float = 0.2,
) -> list[LineItem]:
    scored: list[tuple[float, LineItem]] = []
    for line in ocr_lines:
        score = _line_match_score(correct_answer_text, line.text)
        if score >= min_score:
            scored.append((score, line))
    scored.sort(key=lambda item: (item[0], item[1].confidence), reverse=True)
    return [line for _, line in scored[:top_k]]


def build_target_payload(
    answer: str,
    evidence_lines: Sequence[LineItem],
    extracted_text: Sequence[LineItem],
) -> dict[str, Any]:
    final_conf = 0.0
    if evidence_lines:
        final_conf = sum(item.confidence for item in evidence_lines) / len(evidence_lines)
    elif answer:
        final_conf = 0.25

    if not answer:
        status = "failed"
    elif final_conf < 0.35:
        status = "low_confidence"
    else:
        status = "ok"

    return {
        "answer": answer,
        "evidence_lines": [item.to_dict() for item in evidence_lines],
        "extracted_text": [item.to_dict() for item in extracted_text],
        "final_confidence": round(float(final_conf), 4),
        "status": status,
    }


def build_sft_record(
    sample: Ai2dHybridSample,
    ocr_lines: Sequence[LineItem],
    task_mode: str = "all",
) -> SftRecord:
    evidence = infer_gold_evidence_lines(sample.correct_option_text, ocr_lines)
    extracted = list(ocr_lines)
    target_payload = build_target_payload(
        answer=sample.correct_option_text,
        evidence_lines=evidence,
        extracted_text=extracted,
    )
    prompt = build_analysis_prompt(question=sample.question, ocr_lines=ocr_lines, task_mode=task_mode)
    completion = json.dumps(target_payload, ensure_ascii=False)
    return SftRecord(
        split=sample.split,
        sample_id=sample.sample_id,
        image_id=sample.image_id,
        image_path=sample.image_path,
        ocr_v2_path=str(sample.ocr_v2_path or ""),
        question=sample.question,
        prompt=prompt,
        completion=completion,
        correct_option_text=sample.correct_option_text,
    )


def load_or_build_manifest_with_splits(
    manifest_path: str | Path,
    split_json_path: Optional[str | Path],
    test_ids_csv_path: Optional[str | Path] = None,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> list[Ai2dHybridSample]:
    samples = load_manifest_hybrid(manifest_path)
    if split_json_path and Path(split_json_path).exists():
        split_payload = json.loads(Path(split_json_path).read_text(encoding="utf-8"))
        return assign_splits_to_samples(samples, split_payload)

    if test_ids_csv_path is None:
        return samples

    image_ids = sorted({sample.image_id for sample in samples})
    test_ids = read_test_ids_csv(test_ids_csv_path)
    split_payload = create_image_level_splits(
        image_ids=image_ids,
        test_ids=test_ids,
        val_ratio=val_ratio,
        seed=seed,
    )
    return assign_splits_to_samples(samples, split_payload)


def write_sft_jsonl(records: Iterable[SftRecord], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return output_path
