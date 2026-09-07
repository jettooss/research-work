from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from vqa_retrieval.graph_builder import parse_azure_ocr  # noqa: E402
from train_docvqa_gnn import VARIANTS, apply_variant, multi_positive_loss  # noqa: E402


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class DocVqaPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prepared = WORKSPACE / "docvqa" / "prepared_v1"
        cls.split = json.loads((cls.prepared / "split.json").read_text(encoding="utf-8"))

    def test_official_counts_and_files(self) -> None:
        self.assertEqual(self.split["splits"]["train"]["questions"], 39463)
        self.assertEqual(self.split["splits"]["val"]["questions"], 5349)
        self.assertEqual(self.split["splits"]["test"]["questions"], 5188)
        for payload in self.split["splits"].values():
            self.assertEqual(payload["missing_images"], 0)
            self.assertEqual(payload["missing_ocr"], 0)
        self.assertFalse(any(self.split["image_leakage"].values()))
        self.assertFalse(any(self.split["document_page_leakage"].values()))

    def test_azure_ocr_parser(self) -> None:
        path = next((WORKSPACE / "docvqa" / "val" / "ocr_results").glob("*.json"))
        nodes = parse_azure_ocr(str(path))
        self.assertTrue(nodes)
        self.assertTrue(all(node.text and len(node.bbox) == 4 for node in nodes))

    def test_every_graph_ablation_has_valid_edges(self) -> None:
        x = torch.zeros((4, 394), dtype=torch.float32)
        x[:, 384:386] = torch.tensor([[0.1, 0.1], [0.4, 0.1], [0.1, 0.5], [0.8, 0.8]])
        x[:, 389:393] = torch.tensor([[0.0, 0.0, 0.2, 0.2], [0.3, 0.0, 0.5, 0.2], [0.0, 0.4, 0.2, 0.6], [0.7, 0.7, 0.9, 0.9]])
        x[:, 393] = torch.tensor([1, 0, 1, 0])
        graph = Data(x=x, edge_index=torch.empty((2, 0), dtype=torch.long))
        for variant in VARIANTS:
            transformed = apply_variant(graph, variant)
            self.assertEqual(transformed.x.shape, x.shape)
            self.assertEqual(transformed.edge_index.shape[0], 2)
            self.assertEqual(transformed.edge_index.shape[1], transformed.edge_type.shape[0])

    def test_multi_positive_loss_is_finite(self) -> None:
        image = torch.nn.functional.normalize(torch.randn(3, 8), dim=1)
        text = torch.nn.functional.normalize(torch.randn(3, 8), dim=1)
        loss = multi_positive_loss(image, text, ["page-a", "page-a", "page-b"], 0.07)
        self.assertTrue(torch.isfinite(loss))

    def test_test_submission_has_official_shape(self) -> None:
        module = load_script("evaluate_docvqa_vlm")
        rows = [
            {"sample_id": "1", "question_id": "1", "image_id": "a"},
            {"sample_id": "2", "question_id": "2", "image_id": "b"},
        ]
        predictions = [
            {**rows[0], "question": "q1", "pred_answer": "a1", "raw_response": "{}", "gold_answers": []},
            {**rows[1], "question": "q2", "pred_answer": "a2", "raw_response": "{}", "gold_answers": []},
        ]
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            module.save_outputs(predictions, rows, output, "test", 0.0, 0.0, output, 1, "cpu", False, 128, 2)
            submission = json.loads((output / "submission.json").read_text(encoding="utf-8"))
            self.assertEqual(submission, [{"questionId": 1, "answer": "a1"}, {"questionId": 2, "answer": "a2"}])


if __name__ == "__main__":
    unittest.main()
