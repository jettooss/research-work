from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer
if __package__:
    from .model_matrix_integrity import completed_metrics, digest, file_digest, make_config, validate_directory, write_config
else:
    from model_matrix_integrity import completed_metrics, digest, file_digest, make_config, validate_directory, write_config


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = ROOT.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vqa_retrieval.model_matrix import (  # noqa: E402
    DATASETS,
    candidate_texts,
    ensure_free_space,
    load_rows,
    ocr_spans,
    random_vqa_prediction,
    retrieval_metrics,
    split_rows,
    vqa_row,
)


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    global EXTERNAL_ROOT
    parser = argparse.ArgumentParser(description="Run fresh random or OCR-text model-matrix baselines.")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--model", choices=["random", "ocr_text"], required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/model_matrix")
    parser.add_argument("--data-root", type=Path, default=ROOT.parent,
                        help="Root containing dataset manifests, models and model_matrix_cache")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    EXTERNAL_ROOT = args.data_root.resolve()
    run_baseline_config(args)


def run_baseline_config(args: argparse.Namespace) -> dict:
    run_dir = args.output_dir / args.dataset / args.model / f"seed{args.seed}" / args.split
    rows = load_rows(args.dataset, EXTERNAL_ROOT)
    config = make_config(args, rows, input_root=EXTERNAL_ROOT, kind="baseline", trainable=False)
    validate_directory(run_dir, config, resume=args.resume)
    eval_rows = split_rows(rows, args.split, args.max_samples)
    if not eval_rows:
        raise ValueError(f"Evaluation split {args.split!r} is empty")
    if args.split == "val":
        test_dir = run_dir.parent / "test"
        test_config = dict(config, split="test")
        validate_directory(test_dir, test_config, resume=args.resume)
        completed_metrics(test_dir, test_config, split_rows(rows, "test", args.max_samples))
    metrics_path = run_dir / "metrics.json"
    existing = completed_metrics(run_dir, config, eval_rows)
    if existing is not None:
        print(json.dumps(existing, ensure_ascii=False, indent=2))
        if args.split == "val":
            run_baseline_config(argparse.Namespace(**(vars(args) | {"split": "test"})))
        return existing
    ensure_free_space(run_dir)
    write_config(run_dir, config)

    started = time.perf_counter()
    train_rows = split_rows(rows, "train", args.max_samples)
    unique_docs: dict[str, dict] = {}
    for row in eval_rows:
        unique_docs.setdefault(row["image_id"], row)
    docs = list(unique_docs.values())
    doc_ids = [row["image_id"] for row in docs]
    question_doc_ids = [row["image_id"] for row in eval_rows]
    rng = random.Random(args.seed)

    if args.model == "random":
        similarity = np.asarray([[rng.random() for _ in docs] for _ in eval_rows], dtype=np.float32)
        train_answers = [answer for row in train_rows for answer in row.get("answers", []) if answer]
        predictions = [vqa_row(args.dataset, row, random_vqa_prediction(args.dataset, row, train_answers, rng)) for row in eval_rows]
    else:
        cache_root = EXTERNAL_ROOT / "model_matrix_cache/ocr"
        document_spans = {row["image_id"]: ocr_spans(args.dataset, row, EXTERNAL_ROOT, cache_root) for row in docs}
        doc_texts = [" ".join(span["text"] for span in document_spans[row["image_id"]]) for row in docs]
        vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, max_features=100_000)
        doc_matrix = vectorizer.fit_transform(doc_texts)
        question_matrix = vectorizer.transform([row["question"] for row in eval_rows])
        similarity = (question_matrix @ doc_matrix.T).toarray().astype(np.float32)
        predictions = []
        for row in eval_rows:
            candidates = candidate_texts(document_spans[row["image_id"]])
            if candidates:
                candidate_matrix = vectorizer.transform(candidates)
                scores = (vectorizer.transform([row["question"]]) @ candidate_matrix.T).toarray()[0]
                prediction = candidates[int(scores.argmax())]
            else:
                prediction = ""
            predictions.append(vqa_row(args.dataset, row, prediction))

    retrieval = retrieval_metrics(similarity, question_doc_ids, doc_ids)
    elapsed = time.perf_counter() - started
    vqa_score = float(np.mean([row["score"] for row in predictions])) if predictions else 0.0
    exact = float(np.mean([row["exact"] for row in predictions])) if predictions else 0.0
    metrics = {
        "dataset": args.dataset,
        "model": args.model,
        "split": args.split,
        "seed": args.seed,
        "status": "available_full" if args.max_samples is None else "available_smoke",
        "num_samples": len(eval_rows),
        "num_documents": len(docs),
        "config_sha256": digest(config),
        "retrieval": retrieval,
        "vqa": {"metric": "accuracy" if args.dataset == "ai2d" else "anls", "score": vqa_score, "exact_accuracy": exact},
        "resources": {
            "wall_time_seconds": elapsed,
            "latency_ms_per_question": elapsed * 1000 / max(1, len(eval_rows)),
            "throughput_questions_per_second": len(eval_rows) / elapsed if elapsed else None,
            "peak_ram_mb": psutil.Process().memory_info().rss / (1024 * 1024),
            "peak_vram_mb": 0.0,
            "parameter_count": 0,
        },
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(predictions, run_dir / "predictions.jsonl")
    metrics["predictions_sha256"] = file_digest(run_dir / "predictions.jsonl")
    if args.split == "test" and args.dataset != "ai2d":
        submission = [{"questionId": row["question_id"], "answer": row["pred_answer"]} for row in predictions]
        (run_dir / "submission.json").write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.split == "val":
        run_baseline_config(argparse.Namespace(**(vars(args) | {"split": "test"})))
    return metrics


if __name__ == "__main__":
    main()
