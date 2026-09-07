"""Runnable retrieval example: no datasets, model weights, or GPU required."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vqa_retrieval.metrics import (  # noqa: E402
    mean_reciprocal_rank_multi_positive,
    recall_at_k_multi_positive,
)

# Three questions, two unique documents. Questions 0 and 1 refer to document 0.
similarity = [[0.9, 0.1], [0.4, 0.6], [0.2, 0.8]]
question_positives = [{0}, {0}, {1}]
document_positives = [{0, 1}, {2}]
transposed = list(map(list, zip(*similarity)))

q2d = recall_at_k_multi_positive(similarity, question_positives, ks=(1, 5, 10))
d2q = recall_at_k_multi_positive(transposed, document_positives, ks=(1, 5, 10))
for k in q2d:
    print(f"R@{k}: q->doc={q2d[k]:.2%}, doc->q={d2q[k]:.2%}, mean={(q2d[k] + d2q[k]) / 2:.2%}")
print(f"MRR q->doc: {mean_reciprocal_rank_multi_positive(similarity, question_positives):.4f}")
