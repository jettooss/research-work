from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from vqa_retrieval.ai2d_hybrid import Ai2dHybridSample


OPTION_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class OcrEvidenceLine:
    line_index: int
    text: str
    bbox: tuple[int, int, int, int]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bbox"] = list(self.bbox)
        return payload


@dataclass(frozen=True)
class Ai2dVlmPrediction:
    sample_id: str
    image_id: str
    question: str
    options: tuple[str, ...]
    gold_option_idx: int
    gold_option_text: str
    pred_option_idx: int
    pred_option_text: str
    is_correct: bool
    confidence: float
    question_type: str
    mode: str
    raw_response: str
    evidence: tuple[OcrEvidenceLine, ...] = tuple()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["options"] = list(self.options)
        payload["evidence"] = [item.to_dict() for item in self.evidence]
        return payload


def load_ocr_evidence_lines(ocr_path: str | Path | None, max_lines: int = 60) -> list[OcrEvidenceLine]:
    if not ocr_path:
        return []
    path = Path(ocr_path)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: list[OcrEvidenceLine] = []
    for idx, item in enumerate(payload.get("lines", [])):
        text = str(item.get("text", "")).strip()
        bbox_raw = item.get("bbox", [])
        if not text or len(bbox_raw) != 4:
            continue
        conf = float(item.get("conf", 0.0))
        out.append(
            OcrEvidenceLine(
                line_index=int(item.get("line_index", idx)),
                text=text,
                bbox=tuple(int(v) for v in bbox_raw),
                confidence=max(0.0, min(1.0, conf / 100.0)),
            )
        )
        if max_lines > 0 and len(out) >= max_lines:
            break
    return out


def classify_ai2d_question(question: str) -> str:
    q = str(question or "").strip().lower()
    if re.search(r"\b(how many|count|number of|total)\b", q) or re.search(r"\d", q):
        return "numeric_counting"
    if re.search(r"\b(left|right|above|below|under|over|between|next to|closest|near|far|top|bottom)\b", q):
        return "spatial_relation"
    if re.search(r"\b(what|name|which)\b.*\b([a-z]|letter)\b", q) or re.search(r"\b(label|labeled|depicted as)\b", q):
        return "label_lookup"
    if re.search(r"\b(what object|what is shown|which picture|represents|identify|type of)\b", q):
        return "object_identity"
    return "science_knowledge"


def build_ai2d_multiple_choice_prompt(
    sample: Ai2dHybridSample,
    ocr_lines: Sequence[OcrEvidenceLine] = (),
    mode: str = "direct",
    hybrid_prediction: Optional[Mapping[str, Any]] = None,
) -> str:
    options_block = "\n".join(
        f"{OPTION_LABELS[idx]}. {option}"
        for idx, option in enumerate(sample.options)
    )
    ocr_block = "\n".join(
        f"{line.line_index}: text='{line.text}' bbox={list(line.bbox)} conf={line.confidence:.3f}"
        for line in ocr_lines
    ) or "NO_OCR_LINES"

    caption = str(sample.short_description or "").strip() or "NO_CAPTION"
    helper = ""
    if hybrid_prediction:
        pred_text = str(hybrid_prediction.get("pred_option_text", "")).strip()
        pred_idx = hybrid_prediction.get("pred_option_idx", None)
        helper = (
            "\nExisting graph model signal:\n"
            f"- predicted_index: {pred_idx}\n"
            f"- predicted_answer: {pred_text or 'UNKNOWN'}\n"
            "Use this only as weak evidence; override it when the image contradicts it.\n"
        )

    task_note = (
        "Choose the answer directly from the image."
        if mode == "direct"
        else "Rerank the answer choices using image evidence, OCR/caption context, and the weak graph-model signal."
    )

    return (
        "You are solving an AI2D scientific diagram multiple-choice question.\n"
        "Return STRICT JSON only.\n"
        "Schema: {\"answer_index\": 0, \"answer_label\": \"A\", \"answer_text\": \"...\", "
        "\"confidence\": 0.0, \"evidence\": [\"short reason\"]}\n\n"
        "Rules:\n"
        "- answer_index must be the 0-based index of one option.\n"
        "- answer_label must match the selected option label.\n"
        "- answer_text must exactly copy the selected option text.\n"
        "- Use the image first; OCR and caption are supporting evidence.\n"
        "- If labels such as A/B/C/D appear in the question, locate that label in the diagram.\n"
        "- Do not explain outside JSON.\n\n"
        f"Task: {task_note}\n"
        f"Question type hint: {classify_ai2d_question(sample.question)}\n"
        f"Question: {sample.question.strip()}\n\n"
        f"Options:\n{options_block}\n\n"
        f"Caption/context: {caption}\n\n"
        f"OCR lines:\n{ocr_block}\n"
        f"{helper}"
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    text = str(text or "").strip()
    if not text:
        raise ValueError("empty VLM response")
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    matches = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    for block in matches:
        try:
            payload = json.loads(block.strip())
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            continue
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("VLM response has no JSON object")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("VLM JSON response is not an object")
    return payload


def parse_ai2d_multiple_choice_response(raw_response: str, options: Sequence[str]) -> tuple[int, str, float]:
    if not options:
        raise ValueError("options must be non-empty")
    try:
        payload = _extract_json_object(raw_response)
    except Exception:
        payload = {}

    idx: Optional[int] = None
    raw_idx = payload.get("answer_index")
    if isinstance(raw_idx, bool):
        raw_idx = None
    if isinstance(raw_idx, (int, float)):
        idx = int(raw_idx)

    label = str(payload.get("answer_label", "")).strip().upper()
    if idx is None and label in OPTION_LABELS:
        label_idx = OPTION_LABELS.index(label)
        if label_idx < len(options):
            idx = label_idx

    answer_text = str(payload.get("answer_text", "")).strip()
    if idx is None and answer_text:
        norm = _norm(answer_text)
        for option_idx, option in enumerate(options):
            if _norm(option) == norm:
                idx = option_idx
                break

    if idx is None:
        match = re.search(r"\b([A-Z])\b", str(raw_response or "").upper())
        if match and match.group(1) in OPTION_LABELS:
            candidate = OPTION_LABELS.index(match.group(1))
            if candidate < len(options):
                idx = candidate

    if idx is None or idx < 0 or idx >= len(options):
        idx = 0

    conf = payload.get("confidence", 0.0)
    confidence = float(conf) if isinstance(conf, (int, float)) else 0.0
    return idx, str(options[idx]), max(0.0, min(1.0, confidence))


def predict_ai2d_sample_with_vlm(
    sample: Ai2dHybridSample,
    generator: Callable[[str | Path, str, int], str],
    mode: str = "direct",
    hybrid_prediction: Optional[Mapping[str, Any]] = None,
    max_ocr_lines: int = 60,
    max_new_tokens: int = 128,
) -> Ai2dVlmPrediction:
    if mode not in {"direct", "rerank"}:
        raise ValueError("mode must be 'direct' or 'rerank'")
    evidence = tuple(load_ocr_evidence_lines(sample.ocr_v2_path, max_lines=max_ocr_lines))
    prompt = build_ai2d_multiple_choice_prompt(
        sample=sample,
        ocr_lines=evidence,
        mode=mode,
        hybrid_prediction=hybrid_prediction,
    )
    raw = generator(sample.image_path, prompt, max_new_tokens)
    pred_idx, pred_text, confidence = parse_ai2d_multiple_choice_response(raw, sample.options)
    return Ai2dVlmPrediction(
        sample_id=sample.sample_id,
        image_id=sample.image_id,
        question=sample.question,
        options=sample.options,
        gold_option_idx=sample.correct_option_idx,
        gold_option_text=sample.correct_option_text,
        pred_option_idx=pred_idx,
        pred_option_text=pred_text,
        is_correct=pred_idx == sample.correct_option_idx,
        confidence=confidence,
        question_type=classify_ai2d_question(sample.question),
        mode=mode,
        raw_response=raw,
        evidence=evidence,
    )


def summarize_ai2d_vlm_predictions(predictions: Sequence[Ai2dVlmPrediction | Mapping[str, Any]]) -> dict[str, Any]:
    rows = [pred.to_dict() if isinstance(pred, Ai2dVlmPrediction) else dict(pred) for pred in predictions]
    total = len(rows)
    correct = sum(1 for row in rows if bool(row.get("is_correct")))
    by_type: dict[str, dict[str, float]] = {}
    confusion: dict[str, dict[str, int]] = {}
    for row in rows:
        qtype = str(row.get("question_type", "unknown"))
        bucket = by_type.setdefault(qtype, {"num_samples": 0, "correct": 0, "accuracy": 0.0})
        bucket["num_samples"] += 1
        bucket["correct"] += int(bool(row.get("is_correct")))

        gold = _label_for_index(int(row.get("gold_option_idx", -1)))
        pred = _label_for_index(int(row.get("pred_option_idx", -1)))
        confusion.setdefault(gold, {})
        confusion[gold][pred] = confusion[gold].get(pred, 0) + 1

    for bucket in by_type.values():
        bucket["accuracy"] = float(bucket["correct"] / max(1, bucket["num_samples"]))
    return {
        "num_samples": total,
        "accuracy": float(correct / max(1, total)),
        "correct": correct,
        "by_question_type": by_type,
        "confusion": confusion,
    }


def _label_for_index(idx: int) -> str:
    return OPTION_LABELS[idx] if 0 <= idx < len(OPTION_LABELS) else "?"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())
