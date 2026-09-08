from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reproduce_results", ROOT / "scripts/reproduce_results.py")
results = importlib.util.module_from_spec(spec)
spec.loader.exec_module(results)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


class ResultsIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.evidence_path = self.root / "evidence.json"
        self.evidence = {"accuracy": [{"dataset": "ai2d", "architecture": "clip", "name": "CLIP", "num_samples": 3088, "runs": []}],
                         "retrieval": [{"dataset": "ai2d", "architecture": "clip", "name": "CLIP", "num_samples": 3088, "num_documents": 814, "runs": []}]}
        for seed in (42, 43, 44):
            metric = {"dataset": "ai2d", "model": "clip", "seed": seed, "split": "test", "status": "available_full",
                      "num_samples": 3088, "num_documents": 814, "vqa": {"score": (seed - 40) / 10},
                      "retrieval": {"question_to_document": {"recall_at_k": {"1": 0.1, "5": 0.2, "10": 0.3}},
                                    "document_to_question": {"recall_at_k": {"1": 0.2, "5": 0.3, "10": 0.4}},
                                    "mean_recall_at_k": {"1": 0.15, "5": 0.25, "10": 0.35}}}
            name = f"seed{seed}.json"
            (self.root / name).write_text(json.dumps(metric), encoding="utf-8")
            item = {"seed": seed, "metrics_path": name, "content_sha256": digest(metric)}
            self.evidence["accuracy"][0]["runs"].append({**item, "score_field": "vqa.score"})
            self.evidence["retrieval"][0]["runs"].append(item.copy())

    def compute(self):
        self.evidence_path.write_text(json.dumps(self.evidence), encoding="utf-8")
        return results.compute(self.evidence_path, self.root)

    def mutate_metric(self, seed, *, update_digest=True, **updates):
        name = f"seed{seed}.json"
        path = self.root / name
        value = json.loads(path.read_text())
        value.update(updates)
        path.write_text(json.dumps(value), encoding="utf-8")
        if update_digest:
            for kind in ("accuracy", "retrieval"):
                for group in self.evidence[kind]:
                    for item in group["runs"]:
                        if item["metrics_path"] == name:
                            item["content_sha256"] = digest(value)

    def test_valid_full_measurements_aggregate_population_sd(self):
        value = self.compute()
        self.assertAlmostEqual(value["accuracy"][0]["mean"], 30)
        self.assertAlmostEqual(value["accuracy"][0]["std_ddof0"], (2 / 3) ** 0.5 * 10)
        self.assertEqual(value["retrieval"][0]["n"], 3)

    def test_accuracy_rejects_duplicate_seeds(self):
        self.evidence["accuracy"][0]["runs"][1] = self.evidence["accuracy"][0]["runs"][0].copy()
        with self.assertRaises(ValueError):
            self.compute()

    def test_accuracy_rejects_source_identity_mismatch(self):
        self.evidence["retrieval"] = []
        self.mutate_metric(42, dataset="docvqa")
        with self.assertRaises(ValueError):
            self.compute()

    def test_accuracy_rejects_smoke_measurements(self):
        self.mutate_metric(42, status="available_smoke")
        self.evidence["retrieval"] = []
        with self.assertRaises(ValueError):
            self.compute()

    def test_accuracy_rejects_changed_score_with_stale_content_digest(self):
        self.mutate_metric(42, update_digest=False, vqa={"score": 0.99})
        self.evidence["retrieval"] = []
        with self.assertRaises(ValueError):
            self.compute()

    def test_retrieval_rejects_another_model_even_with_matching_digest(self):
        self.mutate_metric(42, model="siglip")
        self.evidence["accuracy"] = []
        with self.assertRaises(ValueError):
            self.compute()

    def test_retrieval_rejects_duplicate_seed_and_candidate_count_change(self):
        self.evidence["accuracy"] = []
        self.mutate_metric(42, num_documents=10)
        with self.assertRaises(ValueError):
            self.compute()
        self.mutate_metric(42, num_documents=814)
        self.evidence["retrieval"][0]["runs"][1] = self.evidence["retrieval"][0]["runs"][0].copy()
        with self.assertRaises(ValueError):
            self.compute()

    def test_canonical_digest_survives_json_reformatting_and_crlf(self):
        expected = self.compute()
        for path in self.root.glob("seed*.json"):
            value = json.loads(path.read_text())
            path.write_bytes((json.dumps(value, indent=2) + "\n").replace("\n", "\r\n").encode("utf-8"))
        self.assertEqual(self.compute(), expected)

    def test_check_detects_drift_in_readme_marked_table(self):
        value = self.compute()
        project = self.root / "repository/diagram_vqa"
        (project / "docs").mkdir(parents=True)
        snapshot = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        (project / "docs/readme_metrics_verification.json").write_text(snapshot, encoding="utf-8")
        markdown = "Intro\n" + results.START + "\n" + results.table(value) + "\n" + results.END + "\n"
        readme = project.parent / "README.md"
        readme.write_text(markdown, encoding="utf-8")
        (project / "docs/results.md").write_text(markdown, encoding="utf-8")
        with mock.patch.object(results, "ROOT", project), mock.patch.object(results, "compute", return_value=value), mock.patch("sys.argv", ["reproduce_results.py", "--check"]):
            with contextlib.redirect_stdout(io.StringIO()):
                results.main()
            readme.write_text(markdown.replace("15.00", "99.00"), encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                results.main()
            self.assertEqual(error.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
