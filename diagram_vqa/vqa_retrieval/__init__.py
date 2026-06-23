from __future__ import annotations

from pathlib import Path


_SRC_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "vqa_retrieval"
if _SRC_PACKAGE.exists():
    __path__.append(str(_SRC_PACKAGE))

from .datasets import (  # noqa: E402,F401
    Ai2dRetrievalDataset,
    DocVQARetrievalDataset,
    InfographicVQARetrievalDataset,
    RetrievalSample,
    dataset_from_name,
)
from .metrics import recall_at_k_from_sim  # noqa: E402,F401
from .public_vqa_metrics import (  # noqa: E402,F401
    PUBLIC_VQA_TARGETS,
    anls_score,
    evaluate_public_vqa_rows,
    normalize_answer,
    public_vqa_score,
)
from .ai2d_vlm import (  # noqa: E402,F401
    Ai2dVlmPrediction,
    classify_ai2d_question,
    parse_ai2d_multiple_choice_response,
    summarize_ai2d_vlm_predictions,
)
from .ai2d_hybrid import resolve_sample_file_paths  # noqa: E402,F401

__all__ = [
    "RetrievalSample",
    "Ai2dRetrievalDataset",
    "DocVQARetrievalDataset",
    "InfographicVQARetrievalDataset",
    "dataset_from_name",
    "recall_at_k_from_sim",
    "PUBLIC_VQA_TARGETS",
    "anls_score",
    "evaluate_public_vqa_rows",
    "normalize_answer",
    "public_vqa_score",
    "Ai2dVlmPrediction",
    "classify_ai2d_question",
    "parse_ai2d_multiple_choice_response",
    "summarize_ai2d_vlm_predictions",
    "resolve_sample_file_paths",
]
