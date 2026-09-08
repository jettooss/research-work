from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

from .metrics import mean_reciprocal_rank_multi_positive, recall_at_k_multi_positive
from .public_vqa_metrics import accuracy_score, anls_score, classify_error_slice


DATASETS = ("ai2d", "infographicvqa", "docvqa")
MODELS = (
    "random",
    "clip",
    "siglip",
    "ocr_text",
    "gatv2_knn",
    "graph_transformer",
    "sam2_graph_transformer",
    "hybrid_gatv2_knn",
    "graphcolbert",
    "graphcolbert_film",
    "qwen25_vl_qlora",
)
TRAINABLE_LOCAL_MODELS = MODELS[1:3] + MODELS[4:10]
# The controlled matrix uses one reproducible run per model/dataset.  Older
# seed43/seed44 artifacts are preserved but are no longer scheduled.
SEEDS = (42,)


def manifest_for(dataset: str, external_root: Path) -> Path:
    paths = {
        "ai2d": external_root / "ai2d/model_matrix_v1/manifest.jsonl",
        "infographicvqa": external_root / "infographicvqa/prepared_v1/manifest.jsonl",
        "docvqa": external_root / "docvqa/prepared_v1/manifest.jsonl",
    }
    return paths[dataset]


def _resolve_data_path(value: str | Path, external_root: Path) -> Path:
    """Manifest paths are relative to the selected data root, never the cwd."""
    path = Path(value)
    return (path if path.is_absolute() else external_root / path).resolve()


def load_rows(dataset: str, external_root: Path) -> list[dict[str, Any]]:
    external_root = Path(external_root).resolve()
    path = manifest_for(dataset, external_root)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    normalized = []
    for row in rows:
        item = dict(row)
        image_path = _resolve_data_path(item["image_path"], external_root)
        item["image_path"] = str(image_path)
        for field in ("ocr_path", "ocr_v2_path"):
            if item.get(field):
                item[field] = str(_resolve_data_path(item[field], external_root))
        item["sample_id"] = str(item.get("sample_id", item.get("question_id", len(normalized))))
        item["image_id"] = str(item.get("image_id", item.get("doc_id", image_path.stem)))
        if dataset == "ai2d":
            item["answers"] = [str(item["correct_option_text"])]
        else:
            item["answers"] = [str(value) for value in item.get("answers", [])]
        normalized.append(item)
    return normalized


def ensure_free_space(path: Path, minimum_free_gb: float = 20.0) -> float:
    path.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(path.resolve().anchor).free / (1024**3)
    if free_gb < minimum_free_gb:
        raise RuntimeError(f"Only {free_gb:.2f} GiB free; {minimum_free_gb:.2f} GiB required")
    return free_gb


def split_rows(rows: Sequence[dict[str, Any]], split: str, max_samples: int | None = None) -> list[dict[str, Any]]:
    selected = [row for row in rows if row.get("split") == split]
    return selected[:max_samples] if max_samples is not None else selected


def _bbox_from_polygon(values: Sequence[float]) -> list[float]:
    xs = [float(values[idx]) for idx in range(0, len(values), 2)]
    ys = [float(values[idx]) for idx in range(1, len(values), 2)]
    return [min(xs), min(ys), max(xs), max(ys)] if xs and ys else [0.0, 0.0, 0.0, 0.0]


def _read_ai2d_ocr(row: dict[str, Any], external_root: Path) -> list[dict[str, Any]]:
    if not row.get("ocr_v2_path"):
        return []
    path = _resolve_data_path(row["ocr_v2_path"], external_root)
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [{"text": str(word.get("text", "")), "bbox": word.get("bbox", [0, 0, 0, 0])} for word in payload.get("words", [])]


def _read_azure_ocr(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("recognitionResults") or payload.get("analyzeResult", {}).get("readResults") or []
    spans = []
    for page in results:
        for line in page.get("lines", []):
            polygon = line.get("boundingBox") or line.get("polygon") or []
            spans.append({"text": str(line.get("text", "")), "bbox": _bbox_from_polygon(polygon)})
    return spans


def _tesseract_ocr(row: dict[str, Any], cache_dir: Path) -> list[dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{row['image_id']}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    import pytesseract

    if pytesseract.pytesseract.tesseract_cmd == "tesseract":
        windows_default = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if windows_default.exists():
            pytesseract.pytesseract.tesseract_cmd = str(windows_default)

    image = Image.open(row["image_path"]).convert("RGB")
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config="--psm 11")
    spans = []
    for idx, text in enumerate(data.get("text", [])):
        value = str(text).strip()
        if not value:
            continue
        x, y = int(data["left"][idx]), int(data["top"][idx])
        w, h = int(data["width"][idx]), int(data["height"][idx])
        spans.append({"text": value, "bbox": [x, y, x + w, y + h]})
    path.write_text(json.dumps(spans, ensure_ascii=False), encoding="utf-8")
    return spans


def ocr_spans(dataset: str, row: dict[str, Any], external_root: Path, cache_root: Path) -> list[dict[str, Any]]:
    if dataset == "ai2d":
        spans = _read_ai2d_ocr(row, external_root)
    elif dataset == "docvqa":
        path = row.get("ocr_path")
        spans = _read_azure_ocr(_resolve_data_path(path, external_root)) if path else []
    else:
        image_row = dict(row, image_path=str(_resolve_data_path(row["image_path"], external_root)))
        spans = _tesseract_ocr(image_row, cache_root / dataset)
    return [span for span in spans if str(span.get("text", "")).strip()]


def candidate_texts(spans: Sequence[dict[str, Any]], max_candidates: int = 256) -> list[str]:
    values = [str(span["text"]).strip() for span in spans if str(span.get("text", "")).strip()]
    candidates: list[str] = []
    for value in values:
        if value not in candidates:
            candidates.append(value)
    # Contiguous line/word windows make extractive evaluation useful for
    # answers that span several OCR tokens without consulting gold labels.
    for size in (2, 3, 4, 5, 6):
        for start in range(max(0, len(values) - size + 1)):
            value = " ".join(values[start : start + size])
            if value not in candidates:
                candidates.append(value)
            if len(candidates) >= max_candidates:
                return candidates
    return candidates[:max_candidates]


def retrieval_metrics(similarity: np.ndarray, question_doc_ids: Sequence[str], document_ids: Sequence[str]) -> dict[str, Any]:
    doc_index = {doc_id: idx for idx, doc_id in enumerate(document_ids)}
    q2d_positive = [{doc_index[doc_id]} for doc_id in question_doc_ids]
    questions_by_doc: dict[str, set[int]] = defaultdict(set)
    for index, doc_id in enumerate(question_doc_ids):
        questions_by_doc[doc_id].add(index)
    d2q_positive = [questions_by_doc[doc_id] for doc_id in document_ids]
    q2d = similarity.tolist()
    d2q = similarity.T.tolist()
    q2d_recall = recall_at_k_multi_positive(q2d, q2d_positive)
    d2q_recall = recall_at_k_multi_positive(d2q, d2q_positive)
    return {
        "question_to_document": {"recall_at_k": {str(k): v for k, v in q2d_recall.items()}, "mrr": mean_reciprocal_rank_multi_positive(q2d, q2d_positive)},
        "document_to_question": {"recall_at_k": {str(k): v for k, v in d2q_recall.items()}, "mrr": mean_reciprocal_rank_multi_positive(d2q, d2q_positive)},
        "mean_recall_at_k": {str(k): (q2d_recall[k] + d2q_recall[k]) / 2 for k in q2d_recall},
    }


def vqa_row(dataset: str, row: dict[str, Any], prediction: str) -> dict[str, Any]:
    answers = list(row.get("answers", []))
    score = accuracy_score(prediction, answers) if dataset == "ai2d" else anls_score(prediction, answers)
    exact = accuracy_score(prediction, answers)
    return {
        "sample_id": row["sample_id"],
        "question_id": row.get("question_id", row["sample_id"]),
        "image_id": row["image_id"],
        "question": row["question"],
        "pred_answer": prediction,
        "gold_answers": answers,
        "score": score,
        "exact": exact,
        "error_slice": classify_error_slice(prediction, answers, score),
    }


def random_vqa_prediction(dataset: str, row: dict[str, Any], train_answers: Sequence[str], rng: random.Random) -> str:
    if dataset == "ai2d":
        options = list(row.get("options", []))
        return str(rng.choice(options)) if options else ""
    return str(rng.choice(train_answers)) if train_answers else ""
