from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vqa_retrieval.public_vqa_metrics import accuracy_score, evaluate_public_vqa_rows, write_public_vqa_report  # noqa: E402


TOKEN_RE = re.compile(r"[a-z0-9]+", flags=re.IGNORECASE)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "based",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "here",
    "how",
    "in",
    "is",
    "it",
    "many",
    "much",
    "of",
    "on",
    "or",
    "shown",
    "the",
    "there",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tokenize(text: str) -> Counter[str]:
    return Counter(token.lower() for token in TOKEN_RE.findall(text) if token.lower() not in STOPWORDS)


def cosine(left: Counter[str], right: Counter[str]) -> float:
    common = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def first_answer(row: dict[str, Any]) -> str:
    answers = row.get("answers") or []
    return str(answers[0]) if answers else ""


def choose_prediction(mode: str, row: dict[str, Any], train_rows: list[dict[str, Any]], train_tokens: list[Counter[str]], rng: random.Random, most_common: str) -> str:
    if mode == "most_frequent":
        return most_common
    if mode == "random_train_answer":
        return first_answer(rng.choice(train_rows))
    if mode == "question_retrieval":
        q_tokens = tokenize(str(row.get("question", "")))
        best_idx = max(range(len(train_rows)), key=lambda idx: cosine(q_tokens, train_tokens[idx]))
        return first_answer(train_rows[best_idx])
    raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate lightweight InfographicVQA open-answer baselines.")
    parser.add_argument("--manifest", type=Path, default=ROOT.parent / "infographicvqa" / "prepared_v1" / "manifest.jsonl")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--mode", choices=["most_frequent", "random_train_answer", "question_retrieval"], default="question_retrieval")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "infographicvqa_baselines")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    train_rows = [row for row in rows if row.get("split") == "train" and first_answer(row)]
    eval_rows = [row for row in rows if row.get("split") == args.split and row.get("answers")]
    if args.max_samples:
        eval_rows = eval_rows[: args.max_samples]
    if not train_rows or not eval_rows:
        raise SystemExit("Prepared manifest must contain train rows and evaluated rows with answers.")

    answer_counts = Counter(first_answer(row) for row in train_rows)
    most_common = answer_counts.most_common(1)[0][0]
    train_tokens = [tokenize(str(row.get("question", ""))) for row in train_rows]
    inverted: dict[str, set[int]] = defaultdict(set)
    for idx, tokens in enumerate(train_tokens):
        for token in tokens:
            inverted[token].add(idx)
    rng = random.Random(args.seed)

    started = time.perf_counter()
    predictions = []
    for row in eval_rows:
        if args.mode == "question_retrieval":
            q_tokens = tokenize(str(row.get("question", "")))
            candidate_ids: set[int] = set()
            query_tokens = [token for token, _ in q_tokens.most_common()]
            for token in query_tokens[:8]:
                candidate_ids.update(inverted.get(token, set()))
            if not candidate_ids:
                candidate_ids = set(range(len(train_rows)))
            if len(candidate_ids) > 2000:
                ranked = sorted(
                    candidate_ids,
                    key=lambda idx: sum(train_tokens[idx].get(token, 0) for token in query_tokens),
                    reverse=True,
                )
                candidate_ids = set(ranked[:2000])
            best_idx = max(candidate_ids, key=lambda idx: cosine(q_tokens, train_tokens[idx]))
            pred = first_answer(train_rows[best_idx])
        else:
            pred = choose_prediction(args.mode, row, train_rows, train_tokens, rng, most_common)
        predictions.append(
            {
                "sample_id": row["sample_id"],
                "image_id": row["image_id"],
                "question": row["question"],
                "pred_answer": pred,
                "gold_answers": row["answers"],
                "answer_type": row.get("answer_type", []),
                "evidence": row.get("evidence", []),
                "operation_reasoning": row.get("operation_reasoning", []),
            }
        )
    elapsed = time.perf_counter() - started

    metrics, public_results = evaluate_public_vqa_rows(predictions, dataset_name="infographicvqa")
    exact_scores = [accuracy_score(row["pred_answer"], row["gold_answers"]) for row in predictions]
    metrics.update(
        {
            "mode": args.mode,
            "split": args.split,
            "accuracy": sum(exact_scores) / max(1, len(exact_scores)),
            "wall_time_seconds": elapsed,
            "latency_ms_per_question": elapsed * 1000 / max(1, len(predictions)),
            "throughput_questions_per_second": len(predictions) / elapsed if elapsed > 0 else None,
            "parameter_count": 0,
        }
    )

    out_dir = args.output_dir / args.mode / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_public_vqa_report(metrics, public_results, out_dir, prefix="public_vqa")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
