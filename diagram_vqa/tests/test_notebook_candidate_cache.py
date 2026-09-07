from __future__ import annotations

import ast
import json
import random
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
ANLS = runpy.run_path(str(ROOT / "src/vqa_retrieval/public_vqa_metrics.py"))["anls_score"]


def feature_store(dataset):
    path = ROOT / "notebooks/experiments" / dataset / f"{dataset}_hybrid_v4_multipos_training.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    classes = [node for cell in notebook["cells"] if cell["cell_type"] == "code"
               for node in ast.parse("".join(cell["source"])).body
               if isinstance(node, ast.ClassDef) and node.name == "MatrixFeatureStore"]
    assert len(classes) == 1
    future = ast.parse("from __future__ import annotations").body[0]
    module = ast.fix_missing_locations(ast.Module(body=[future, classes[0]], type_ignores=[]))
    namespace = {"np": np, "torch": torch, "anls_score": ANLS,
                 "SentenceTransformer": lambda *_args, **_kwargs: None}
    exec(compile(module, str(path), "exec"), namespace)
    store = namespace["MatrixFeatureStore"](dataset, "hybrid_gatv2_knn", ROOT / "runs", torch.device("cpu"))
    return store, namespace


def uncached_target(values, answers):
    scores = [max((ANLS(value, [answer]) for answer in answers), default=0.0) for value in values]
    return int(np.argmax(scores)) if scores else 0


@pytest.mark.parametrize("dataset", ["docvqa", "infographicvqa"])
@pytest.mark.parametrize("values,answers", [
    (["", "January 10, 1999", "Alpha"], ["January 10 1999"]),
    (["Alpha", "alpha"], ["ALPHA"]),
    (["x", "y"], []),
    ([], ["a"]),
    (["abcd", "abce", "xy"], ["abce", "irrelevant"]),
])
def test_targets_and_embeddings_are_unchanged(dataset, values, answers):
    store, namespace = feature_store(dataset)
    graph = SimpleNamespace(candidate_texts=values, candidate_embeddings=torch.ones(len(values), 384, dtype=torch.float64))
    expected = uncached_target(values, answers)
    first = store.candidates({"answers": answers}, graph)
    namespace["anls_score"] = lambda *_args: pytest.fail("Cached target was recomputed")
    second = store.candidates({"answers": answers}, graph)
    for actual in (first, second):
        assert actual[0] == values and actual[2] == expected
        torch.testing.assert_close(actual[1], graph.candidate_embeddings.float(), rtol=0, atol=0)


@pytest.mark.parametrize("dataset", ["docvqa", "infographicvqa"])
def test_changed_answers_and_candidates_use_new_cache_entries(dataset):
    store, _ = feature_store(dataset)
    graph = SimpleNamespace(candidate_texts=["first", "second"], candidate_embeddings=torch.ones(2, 384))
    row = {"answers": ["second"]}
    assert store.candidates(row, graph)[2] == 1
    row["answers"] = ["first"]
    assert store.candidates(row, graph)[2] == 0
    graph.candidate_texts.reverse()
    assert store.candidates(row, graph)[2] == 1
    assert len(store._candidate_targets) == 3


@pytest.mark.parametrize("dataset", ["docvqa", "infographicvqa"])
def test_cached_targets_match_original_on_local_dataset(dataset):
    manifest = ROOT.parent / dataset / "prepared_v1/manifest.jsonl"
    if not manifest.exists():
        pytest.skip("Local dataset is not installed")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row["split"] in {"train", "val"}]
    random.Random(42).shuffle(rows)
    store, _ = feature_store(dataset)
    graph_dir = ROOT.parent / "model_matrix_cache/graphs" / dataset / "hybrid_gatv2_knn"
    checked = 0
    for row in rows:
        path = graph_dir / f"{row['image_id']}.pt"
        if not path.exists():
            continue
        graph = torch.load(path, map_location="cpu", weights_only=False)
        expected = uncached_target(graph.candidate_texts, row.get("answers", []))
        first = store.candidates(row, graph)
        repeated = store.candidates(row, graph)
        assert first[2] == repeated[2] == expected
        assert first[0] == repeated[0] == graph.candidate_texts
        torch.testing.assert_close(first[1], graph.candidate_embeddings.float(), rtol=0, atol=0)
        checked += 1
        if checked == 200:
            break
    assert checked == 200
