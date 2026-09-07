from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vqa_retrieval.metrics import (  # noqa: E402
    mean_reciprocal_rank_multi_positive,
    recall_at_k_from_sim,
    recall_at_k_multi_positive,
)
from vqa_retrieval.public_vqa_metrics import anls_score  # noqa: E402


class MultiPositiveMetricsTest(unittest.TestCase):
    def test_existing_one_to_one_api_is_unchanged(self) -> None:
        self.assertEqual(recall_at_k_from_sim([[2.0, 1.0], [1.0, 2.0]], ks=(1,)), {1: 1.0})

    def test_multiple_questions_for_one_document_are_positive(self) -> None:
        similarities = [[0.9, 0.1, 0.0], [0.8, 0.7, 0.1]]
        positives = [{0, 1}, {1}]
        self.assertEqual(recall_at_k_multi_positive(similarities, positives, ks=(1, 2)), {1: 0.5, 2: 1.0})
        self.assertAlmostEqual(mean_reciprocal_rank_multi_positive(similarities, positives), 0.75)

    def test_invalid_positive_index_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            recall_at_k_multi_positive([[1.0]], [{1}])

    def test_docvqa_anls_uses_best_gold_answer(self) -> None:
        self.assertEqual(anls_score("January 10 1999", ["irrelevant", "January 10, 1999"]), 1.0)


if __name__ == "__main__":
    unittest.main()
