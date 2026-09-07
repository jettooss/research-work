from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import psutil


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vqa_retrieval.metrics import mean_reciprocal_rank_multi_positive, recall_at_k_multi_positive  # noqa: E402
from vqa_retrieval.public_vqa_metrics import accuracy_score, evaluate_public_vqa_rows, write_public_vqa_report  # noqa: E402


OPEN_ANSWER_MODES = ("most_frequent", "random_train_answer", "question_retrieval")
RETRIEVAL_MODES = ("random", "ocr_text")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def first_answer(row: dict[str, Any]) -> str:
    answers = row.get("answers") or []
    return str(answers[0]) if answers else ""


def azure_ocr_text(path: str | Path) -> str:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return "\n".join(
        str(line.get("text", "")).strip()
        for page in payload.get("recognitionResults", [])
        for line in page.get("lines", [])
        if str(line.get("text", "")).strip()
    )


def resource_metrics(started: float, num_samples: int) -> dict[str, Any]:
    elapsed = time.perf_counter() - started
    return {
        "wall_time_seconds": elapsed,
        "latency_ms_per_question": elapsed * 1000 / max(1, num_samples),
        "throughput_questions_per_second": num_samples / elapsed if elapsed > 0 else None,
        "peak_ram_mb": psutil.Process().memory_info().rss / (1024 * 1024),
        "peak_vram_mb": 0.0,
        "parameter_count": 0,
    }


def evaluate_open_answer(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if args.split == "test":
        raise SystemExit("Official DocVQA test has no answers; open-answer metrics are available on val only.")
    train_rows = [row for row in rows if row.get("split") == "train" and first_answer(row)]
    eval_rows = [row for row in rows if row.get("split") == args.split and row.get("answers")]
    if args.max_samples:
        eval_rows = eval_rows[: args.max_samples]
    rng = random.Random(args.seed)
    most_common = Counter(first_answer(row) for row in train_rows).most_common(1)[0][0]

    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()
    if args.mode == "question_retrieval":
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, max_features=200_000)
        train_matrix = vectorizer.fit_transform(str(row["question"]) for row in train_rows)
        eval_matrix = vectorizer.transform(str(row["question"]) for row in eval_rows)
        batch_size = 128
        predicted_answers: list[str] = []
        for start in range(0, len(eval_rows), batch_size):
            similarities = eval_matrix[start : start + batch_size] @ train_matrix.T
            best = similarities.argmax(axis=1).A1
            predicted_answers.extend(first_answer(train_rows[int(idx)]) for idx in best)
    elif args.mode == "random_train_answer":
        predicted_answers = [first_answer(rng.choice(train_rows)) for _ in eval_rows]
    else:
        predicted_answers = [most_common for _ in eval_rows]

    for row, prediction in zip(eval_rows, predicted_answers):
        predictions.append(
            {
                "sample_id": row["sample_id"],
                "question_id": row["question_id"],
                "image_id": row["image_id"],
                "question": row["question"],
                "pred_answer": prediction,
                "gold_answers": row["answers"],
            }
        )
    metrics, public_results = evaluate_public_vqa_rows(predictions, dataset_name="docvqa")
    exact = [accuracy_score(row["pred_answer"], row["gold_answers"]) for row in predictions]
    metrics.update(
        {
            "mode": args.mode,
            "metric_family": "open_answer",
            "split": args.split,
            "accuracy": sum(exact) / max(1, len(exact)),
            "status": "available_full" if args.max_samples is None else "available_partial",
            "num_samples": len(predictions),
            **resource_metrics(started, len(predictions)),
        }
    )
    out_dir = args.output_dir / "open_answer" / args.mode / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(predictions, out_dir / "predictions.jsonl")
    write_public_vqa_report(metrics, public_results, out_dir, prefix="public_vqa")
    return metrics


def evaluate_retrieval(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    eval_rows = [row for row in rows if row.get("split") == args.split]
    if args.max_samples:
        eval_rows = eval_rows[: args.max_samples]
    documents: dict[str, dict[str, Any]] = {}
    for row in eval_rows:
        documents.setdefault(str(row["image_id"]), row)
    document_rows = list(documents.values())
    document_index = {str(row["image_id"]): idx for idx, row in enumerate(document_rows)}
    started = time.perf_counter()

    if args.mode == "ocr_text":
        from sklearn.feature_extraction.text import TfidfVectorizer

        document_texts = [azure_ocr_text(row["ocr_path"]) for row in document_rows]
        question_texts = [str(row["question"]) for row in eval_rows]
        vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, max_features=250_000)
        all_matrix = vectorizer.fit_transform(document_texts + question_texts)
        doc_matrix = all_matrix[: len(document_texts)]
        question_matrix = all_matrix[len(document_texts) :]
        similarities = (question_matrix @ doc_matrix.T).toarray()
    else:
        rng = random.Random(args.seed)
        similarities = [
            [rng.random() for _ in document_rows]
            for _ in eval_rows
        ]

    q2d_positives = [{document_index[str(row["image_id"])]} for row in eval_rows]
    q2d = recall_at_k_multi_positive(similarities, q2d_positives)
    q2d_mrr = mean_reciprocal_rank_multi_positive(similarities, q2d_positives)

    image_questions: dict[str, set[int]] = defaultdict(set)
    for question_idx, row in enumerate(eval_rows):
        image_questions[str(row["image_id"])].add(question_idx)
    transposed = [
        [float(similarities[q_idx][d_idx]) for q_idx in range(len(eval_rows))]
        for d_idx in range(len(document_rows))
    ]
    d2q_positives = [image_questions[str(row["image_id"])] for row in document_rows]
    d2q = recall_at_k_multi_positive(transposed, d2q_positives)
    d2q_mrr = mean_reciprocal_rank_multi_positive(transposed, d2q_positives)

    metrics = {
        "dataset": "docvqa",
        "mode": args.mode,
        "metric_family": "retrieval",
        "split": args.split,
        "status": "available_full" if args.max_samples is None else "available_partial",
        "num_samples": len(eval_rows),
        "num_documents": len(document_rows),
        "question_to_document": {"recall_at_k": {str(k): v for k, v in q2d.items()}, "mrr": q2d_mrr},
        "document_to_question": {"recall_at_k": {str(k): v for k, v in d2q.items()}, "mrr": d2q_mrr},
        "mean_recall_at_k": {str(k): (q2d[k] + d2q[k]) / 2 for k in q2d},
        **resource_metrics(started, len(eval_rows)),
    }
    out_dir = args.output_dir / "retrieval" / args.mode / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DocVQA open-answer and retrieval baselines.")
    parser.add_argument("--manifest", type=Path, default=ROOT.parent / "docvqa" / "prepared_v1" / "manifest.jsonl")
    parser.add_argument("--family", choices=["open_answer", "retrieval"], required=True)
    parser.add_argument("--mode", choices=OPEN_ANSWER_MODES + RETRIEVAL_MODES, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "docvqa_baselines")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Accepted for CLI parity; baselines are deterministic and overwrite compact outputs.")
    args = parser.parse_args()
    if args.family == "open_answer" and args.mode not in OPEN_ANSWER_MODES:
        parser.error(f"mode {args.mode!r} does not belong to open_answer")
    if args.family == "retrieval" and args.mode not in RETRIEVAL_MODES:
        parser.error(f"mode {args.mode!r} does not belong to retrieval")
    rows = load_jsonl(args.manifest)
    metrics = evaluate_open_answer(args, rows) if args.family == "open_answer" else evaluate_retrieval(args, rows)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
