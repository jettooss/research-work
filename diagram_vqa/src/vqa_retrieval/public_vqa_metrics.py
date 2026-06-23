from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


PUBLIC_VQA_TARGETS: dict[str, dict[str, Any]] = {
    "ai2d": {
        "metric": "accuracy",
        "primary_target": 0.947,
        "stretch_target": 0.9602,
        "qwen25_vl_3b_baseline": 0.815,
        "sources": [
            "https://evalscope.readthedocs.io/en/latest/benchmarks/ai2d.html",
            "https://llm-stats.com/benchmarks/ai2d",
            "https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/README.md",
        ],
    },
    "docvqa": {
        "metric": "anls",
        "primary_target": 0.964,
        "stretch_target": None,
        "qwen25_vl_3b_baseline": 0.939,
        "sources": [
            "https://www.docvqa.org/challenges/challenge-2020",
            "https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct",
            "https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/README.md",
        ],
    },
    "infographicvqa": {
        "metric": "anls",
        "primary_target": 0.834,
        "stretch_target": 0.873,
        "qwen25_vl_3b_baseline": 0.771,
        "aliases": ["infovqa", "infographicsvqa"],
        "sources": [
            "https://www.docvqa.org/datasets/infographicvqa",
            "https://evalscope.readthedocs.io/en/latest/benchmarks/infovqa.html",
            "https://llm-stats.com/benchmarks/infovqa",
            "https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/README.md",
        ],
    },
}

_ALIASES = {
    alias: dataset
    for dataset, payload in PUBLIC_VQA_TARGETS.items()
    for alias in payload.get("aliases", [])
}

_PUNCT_RE = re.compile(r"[^\w\s\.\-/%]", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class PublicVqaSampleResult:
    sample_id: str
    dataset: str
    metric: str
    raw_prediction: str
    prediction: str
    normalized_prediction: str
    gold_answers: tuple[str, ...]
    normalized_gold_answers: tuple[str, ...]
    score: float
    error_slice: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gold_answers"] = list(self.gold_answers)
        payload["normalized_gold_answers"] = list(self.normalized_gold_answers)
        return payload


def canonical_dataset_name(dataset_name: str) -> str:
    key = str(dataset_name).strip().lower().replace("-", "").replace("_", "")
    if key in PUBLIC_VQA_TARGETS:
        return key
    if key in _ALIASES:
        return _ALIASES[key]
    raise ValueError(f"Unsupported public VQA dataset: {dataset_name}")


def target_for_dataset(dataset_name: str) -> dict[str, Any]:
    key = canonical_dataset_name(dataset_name)
    return PUBLIC_VQA_TARGETS[key]


def normalize_answer(text: Any) -> str:
    value = str(text or "").strip().lower()
    value = value.replace("\u2013", "-").replace("\u2014", "-")
    value = value.replace(",", "")
    value = _PUNCT_RE.sub(" ", value)
    return _SPACE_RE.sub(" ", value).strip()


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    if len(left) < len(right):
        left, right = right, left

    prev = list(range(len(right) + 1))
    for i, left_ch in enumerate(left, start=1):
        curr = [i]
        for j, right_ch in enumerate(right, start=1):
            cost = 0 if left_ch == right_ch else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def normalized_levenshtein_similarity(prediction: str, gold: str, threshold: float = 0.5) -> float:
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)
    if not pred_norm and not gold_norm:
        return 1.0
    if not pred_norm or not gold_norm:
        return 0.0
    norm_dist = levenshtein_distance(pred_norm, gold_norm) / max(len(pred_norm), len(gold_norm))
    if norm_dist >= threshold:
        return 0.0
    return 1.0 - norm_dist


def anls_score(prediction: str, gold_answers: Sequence[str], threshold: float = 0.5) -> float:
    if not gold_answers:
        return 0.0
    return max(normalized_levenshtein_similarity(prediction, gold, threshold=threshold) for gold in gold_answers)


def accuracy_score(prediction: str, gold_answers: Sequence[str]) -> float:
    pred_norm = normalize_answer(prediction)
    return float(bool(pred_norm) and any(pred_norm == normalize_answer(gold) for gold in gold_answers))


def public_vqa_score(dataset_name: str, prediction: str, gold_answers: Sequence[str]) -> tuple[str, float]:
    target = target_for_dataset(dataset_name)
    metric = str(target["metric"])
    if metric == "accuracy":
        return metric, accuracy_score(prediction, gold_answers)
    if metric == "anls":
        return metric, anls_score(prediction, gold_answers)
    raise ValueError(f"Unsupported metric for {dataset_name}: {metric}")


def classify_error_slice(
    prediction: str,
    gold_answers: Sequence[str],
    score: float,
    raw_prediction: str = "",
    json_valid: Optional[bool] = None,
    status: str = "",
    ocr_text: str = "",
) -> str:
    if score >= 1.0:
        return "correct"
    if json_valid is False:
        return "invalid_json"
    if str(status).strip().lower() == "failed":
        return "failed_status"

    pred_norm = normalize_answer(prediction)
    if not pred_norm:
        return "empty_answer"

    ocr_norm = normalize_answer(ocr_text)
    gold_norms = [normalize_answer(gold) for gold in gold_answers if normalize_answer(gold)]
    if ocr_norm and gold_norms and not any(gold in ocr_norm for gold in gold_norms):
        return "ocr_miss"

    raw_norm = normalize_answer(raw_prediction)
    if gold_norms and any(gold in raw_norm for gold in gold_norms):
        return "answer_extraction_miss"
    if re.search(r"\d", pred_norm) or any(re.search(r"\d", gold) for gold in gold_norms):
        return "arithmetic_or_numeric_miss"
    return "wrong_answer"


def _row_gold_answers(row: Mapping[str, Any]) -> tuple[str, ...]:
    value = row.get("gold_answers", None)
    if value is None:
        value = row.get("answers", None)
    if isinstance(value, str):
        out = [value]
    elif isinstance(value, Iterable):
        out = [str(item) for item in value if str(item).strip()]
    else:
        out = []

    for key in ("gold_answer", "correct_option_text", "correct_text"):
        if row.get(key) is not None:
            out.append(str(row[key]))
    if row.get("gold_option_text") is not None:
        out.append(str(row["gold_option_text"]))
    dedup: list[str] = []
    for answer in out:
        if answer not in dedup:
            dedup.append(answer)
    return tuple(dedup)


def evaluate_public_vqa_rows(
    rows: Sequence[Mapping[str, Any]],
    dataset_name: str,
) -> tuple[dict[str, Any], list[PublicVqaSampleResult]]:
    dataset = canonical_dataset_name(dataset_name)
    target = target_for_dataset(dataset)
    metric = str(target["metric"])

    sample_results: list[PublicVqaSampleResult] = []
    for idx, row in enumerate(rows):
        prediction = str(
            row.get(
                "pred_answer",
                row.get("pred_option_text", row.get("prediction", row.get("answer", ""))),
            )
            or ""
        )
        raw_prediction = str(row.get("raw_response", row.get("raw_prediction", prediction)) or "")
        gold_answers = _row_gold_answers(row)
        _, score = public_vqa_score(dataset, prediction, gold_answers)
        normalized_gold = tuple(normalize_answer(answer) for answer in gold_answers)
        sample_results.append(
            PublicVqaSampleResult(
                sample_id=str(row.get("sample_id", row.get("questionId", idx))),
                dataset=dataset,
                metric=metric,
                raw_prediction=raw_prediction,
                prediction=prediction,
                normalized_prediction=normalize_answer(prediction),
                gold_answers=gold_answers,
                normalized_gold_answers=normalized_gold,
                score=float(score),
                error_slice=classify_error_slice(
                    prediction=prediction,
                    gold_answers=gold_answers,
                    score=float(score),
                    raw_prediction=raw_prediction,
                    json_valid=row.get("json_valid") if isinstance(row.get("json_valid"), bool) else None,
                    status=str(row.get("status", "")),
                    ocr_text=str(row.get("ocr_text", "")),
                ),
            )
        )

    scores = [result.score for result in sample_results]
    aggregate = sum(scores) / max(1, len(scores))
    primary_target = float(target["primary_target"])
    stretch_raw = target.get("stretch_target")
    stretch_target = float(stretch_raw) if stretch_raw is not None else None
    error_counts: dict[str, int] = {}
    for result in sample_results:
        error_counts[result.error_slice] = error_counts.get(result.error_slice, 0) + 1

    metrics = {
        "dataset": dataset,
        "metric": metric,
        "num_samples": len(sample_results),
        metric: aggregate,
        "score": aggregate,
        "primary_target": primary_target,
        "primary_target_gap": aggregate - primary_target,
        "beats_primary_target": aggregate > primary_target,
        "stretch_target": stretch_target,
        "stretch_target_gap": None if stretch_target is None else aggregate - stretch_target,
        "beats_stretch_target": False if stretch_target is None else aggregate > stretch_target,
        "qwen25_vl_3b_baseline": target.get("qwen25_vl_3b_baseline"),
        "error_slices": error_counts,
        "sources": target.get("sources", []),
    }
    return metrics, sample_results


def write_public_vqa_report(
    metrics: Mapping[str, Any],
    sample_results: Sequence[PublicVqaSampleResult],
    output_dir: str | Path,
    prefix: str = "public_vqa",
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{prefix}_metrics.json"
    predictions_path = output_dir / f"{prefix}_predictions.jsonl"
    errors_path = output_dir / f"{prefix}_error_slices.json"

    metrics_path.write_text(json.dumps(dict(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    with predictions_path.open("w", encoding="utf-8") as handle:
        for result in sample_results:
            handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
    errors_path.write_text(
        json.dumps(metrics.get("error_slices", {}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "metrics": str(metrics_path),
        "predictions": str(predictions_path),
        "error_slices": str(errors_path),
    }
