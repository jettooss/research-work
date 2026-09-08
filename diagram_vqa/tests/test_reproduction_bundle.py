from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import warnings
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reproduction_bundle", ROOT / "scripts/export_reproduction_bundle.py")
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


class ReproductionBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "repository/diagram_vqa"
        self.data = self.base / "data"
        self.data.mkdir()
        self.evidence = {"accuracy": [], "retrieval": [], "datasets": {}}
        for name, payload in (("pyproject.toml", "[project]\nname='fixture'\n"),
                              ("examples/retrieval_metrics.py", "print('example')\n"),
                              ("reports/presentation_metrics/evaluate_saved_retrieval.py", "# evaluator\n"),
                              ("reports/presentation_metrics/retrieval_matrix_audit.json", "{}")):
            self.put(self.root / name, payload)
        self.put(self.root.parent / "README.md", "fixture")
        self.put(self.root.parent / "Dockerfile", "FROM python:3.11-slim")
        self.put(self.root.parent / ".dockerignore", "models")

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")

    def collect(self, data=False, weights=False):
        self.put(self.root / "reports/reproducibility/evidence.json", json.dumps(self.evidence))
        return bundle.collect(self.root, self.data, data, weights)

    def portable_fixture(self):
        name = "docvqa/prepared_v1/manifest.jsonl"
        row = {"image_id": "q", "image_path": "C:\\former\\data\\docvqa\\train\\documents\\q.png",
               "ocr_path": "C:\\former\\data\\docvqa\\train\\ocr_results\\q.json"}
        self.put(self.data / name, json.dumps(row) + "\n")
        self.put(self.data / "docvqa/train/ocr_results/q.json", '{"recognitionResults": []}')
        self.put(self.data / "docvqa/prepared_v1/split.json", "{}")
        self.put(self.data / "docvqa/prepared_v1/credentials.json", '{"private": "must-not-export"}')
        self.evidence["datasets"] = {"docvqa": {"manifest_relative_to_data_root": name}}
        return name

    def test_metadata_contains_example_evaluator_and_docker_inputs(self):
        files, missing = self.collect()
        self.assertEqual(missing, [])
        self.assertTrue({"diagram_vqa/examples/retrieval_metrics.py", "diagram_vqa/reports/presentation_metrics/evaluate_saved_retrieval.py", "Dockerfile", ".dockerignore"} <= set(files))
        self.assertEqual(bundle.write_bundle(self.base / "metadata.zip", files), len(files))

    def test_include_data_relocates_archive_only_and_hashes_transformed_bytes(self):
        name = self.portable_fixture()
        original = (self.data / name).read_bytes()
        files, missing = self.collect(data=True)
        self.assertEqual(missing, [])
        self.assertNotIn("docvqa/prepared_v1/credentials.json", files)
        self.assertNotIn("docvqa/train/documents/q.png", files)
        target = self.base / "portable.zip"
        bundle.write_bundle(target, files, include_data=True)
        self.assertEqual((self.data / name).read_bytes(), original)
        with zipfile.ZipFile(target) as archive:
            archived = archive.read(name)
            row = json.loads(archived)
            self.assertEqual(row["image_path"], "docvqa/train/documents/q.png")
            self.assertEqual(row["ocr_path"], "docvqa/train/ocr_results/q.json")
            inventory = json.loads(archive.read("bundle_manifest.json"))
            item = next(item for item in inventory["files"] if item["path"] == name)
            self.assertEqual(item["sha256"], hashlib.sha256(archived).hexdigest())
            self.assertEqual(item["original_sha256"], hashlib.sha256(original).hexdigest())
            self.assertEqual(item["bytes"], len(archived))

    def test_missing_referenced_ocr_is_reported(self):
        self.portable_fixture()
        (self.data / "docvqa/train/ocr_results/q.json").unlink()
        _, missing = self.collect(data=True)
        self.assertIn("docvqa/train/ocr_results/q.json", missing)

    def test_ocr_reference_cannot_sweep_other_json_or_escape_data_root(self):
        name = self.portable_fixture()
        for bad in ("docvqa/prepared_v1/credentials.json", "../outside.json", "C:\\private\\keys.json"):
            row = json.loads((self.data / name).read_text())
            row["ocr_path"] = bad
            self.put(self.data / name, json.dumps(row))
            with self.assertRaises(ValueError):
                self.collect(data=True)

    def test_unsafe_evidence_path_is_rejected_before_collecting_it(self):
        self.evidence["accuracy"] = [{"runs": [{"metrics_path": "../README.md"}]}]
        with self.assertRaises(ValueError):
            self.collect()

    def test_only_best_weights_are_opted_in(self):
        path = "runs/ai2d/learned/seed42_full/metrics.json"
        self.put(self.root / path, "{}")
        self.put((self.root / path).parent / "checkpoint_best.pt", "best")
        self.put((self.root / path).parent / "checkpoint_last.pt", "optimizer-and-last")
        self.evidence["accuracy"] = [{"architecture": "learned", "runs": [{"metrics_path": path}]}]
        normal, _ = self.collect()
        weighted, missing = self.collect(weights=True)
        self.assertEqual(missing, [])
        self.assertFalse(any(name.endswith(".pt") for name in normal))
        self.assertEqual([name for name in weighted if name.endswith(".pt")], ["diagram_vqa/" + path.replace("metrics.json", "checkpoint_best.pt")])

    def write_zip(self, name="a.txt", content=b"original", *, duplicated_inventory=False, duplicate_zip=False, omit_file=False):
        target = self.base / "modified.zip"
        item = {"path": name, "bytes": 8, "sha256": hashlib.sha256(b"original").hexdigest()}
        inventory = {"schema_version": 1, "files": [item, item] if duplicated_inventory else [item]}
        with warnings.catch_warnings(), zipfile.ZipFile(target, "w") as archive:
            warnings.simplefilter("ignore", UserWarning)
            if not omit_file:
                archive.writestr(name, content)
                if duplicate_zip:
                    archive.writestr(name, content)
            archive.writestr("bundle_manifest.json", json.dumps(inventory))
        return target

    def test_tampered_archive_content_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "content mismatch"):
            bundle.verify(self.write_zip(content=b"modified"))

    def test_missing_archive_member_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            bundle.verify(self.write_zip(omit_file=True))

    def test_duplicate_inventory_and_zip_entries_are_rejected(self):
        for kwargs in ({"duplicated_inventory": True}, {"duplicate_zip": True}):
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                bundle.verify(self.write_zip(**kwargs))

    def test_unsafe_archive_paths_are_rejected(self):
        for name in ("../escape", "C:/escape", "folder\\escape", "/escape", "a/./escape"):
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                bundle.verify(self.write_zip(name=name))

    def test_inventory_hash_describes_bytes_actually_written_after_source_change(self):
        files, _ = self.collect()
        self.put(self.root.parent / "README.md", "edited after collection")
        target = self.base / "late-edit.zip"
        bundle.write_bundle(target, files)
        with zipfile.ZipFile(target) as archive:
            content = archive.read("README.md")
            row = next(row for row in json.loads(archive.read("bundle_manifest.json"))["files"] if row["path"] == "README.md")
            self.assertEqual(row["sha256"], hashlib.sha256(content).hexdigest())


if __name__ == "__main__":
    unittest.main()
