from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vqa_retrieval.model_matrix import DATASETS, MODELS, SEEDS, candidate_texts, retrieval_metrics


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = ROOT.parent


def test_matrix_has_exactly_33_cells() -> None:
    assert len(DATASETS) == 3
    assert len(MODELS) == 11
    assert len(DATASETS) * len(MODELS) == 33
    assert SEEDS == (42,)


def test_ai2d_official_document_only_split_is_audited() -> None:
    audit = json.loads((EXTERNAL_ROOT / "ai2d/model_matrix_v1/audit.json").read_text(encoding="utf-8"))
    assert audit["official_test_images"] == 982
    assert audit["document_only_test_images"] == 168
    assert audit["missing_official_test_ids"] == 0
    assert audit["ready"] is True
    assert not any(audit["image_leakage"].values())


def test_candidate_spans_do_not_require_gold_answers() -> None:
    values = candidate_texts([{"text": "New"}, {"text": "York"}, {"text": "City"}])
    assert "New York" in values
    assert "New York City" in values


def test_multi_positive_retrieval_supports_repeated_documents() -> None:
    similarity = np.asarray([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]], dtype=np.float32)
    metrics = retrieval_metrics(similarity, ["a", "a", "b"], ["a", "b"])
    assert metrics["question_to_document"]["recall_at_k"]["1"] == 1.0
    assert metrics["document_to_question"]["recall_at_k"]["1"] == 1.0


def test_working_notebook_tree_keeps_only_experiments() -> None:
    notebook_dirs = sorted(path.name for path in (ROOT / "notebooks").iterdir() if path.is_dir())
    assert notebook_dirs == ["experiments"]
