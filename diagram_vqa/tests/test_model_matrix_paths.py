"""Manifest/OCR path regressions with real files and no model dependencies."""
from __future__ import annotations

import ast
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch


MODULE = Path(__file__).resolve().parents[1] / "src/vqa_retrieval/model_matrix.py"


def path_functions():
    # Execute the actual I/O functions without importing unrelated ML/plot packages.
    names = {"manifest_for", "_resolve_data_path", "load_rows", "_bbox_from_polygon",
             "_read_ai2d_ocr", "_read_azure_ocr", "ocr_spans"}
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    body.extend(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names)
    namespace = {"Path": Path, "json": json}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(MODULE), "exec"), namespace)
    return namespace


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class MatrixPathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "separate-data"
        self.outside = self.root / "unrelated-cwd"
        self.data.mkdir()
        self.outside.mkdir()
        self.functions = path_functions()

    def write_json(self, relative, value):
        path = self.data / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def manifest(self, dataset, row):
        path = self.functions["manifest_for"](dataset, self.data)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return path

    def snapshot(self):
        return {str(path.relative_to(self.data)): (path.read_bytes(), path.stat().st_mtime_ns)
                for path in self.data.rglob("*") if path.is_file()}

    def test_docvqa_relative_manifest_and_ocr_work_outside_data_root_without_mutation(self):
        ocr = self.write_json("docvqa/train/ocr_results/page.json", {
            "recognitionResults": [{"lines": [{"text": "invoice total", "boundingBox": [1, 2, 5, 2, 5, 6, 1, 6]}]}]})
        row = {"image_path": "docvqa/train/documents/page.png", "ocr_path": str(ocr.relative_to(self.data)),
               "question_id": 7, "question": "Total?", "answers": [42]}
        self.manifest("docvqa", row)
        before = self.snapshot()
        with working_directory(self.outside):
            loaded = self.functions["load_rows"]("docvqa", self.data)[0]
            for source in (row, loaded):
                spans = self.functions["ocr_spans"]("docvqa", source, self.data, self.root / "cache")
                self.assertEqual(spans, [{"text": "invoice total", "bbox": [1.0, 2.0, 5.0, 6.0]}])
        self.assertEqual(loaded["ocr_path"], str(ocr.resolve()))
        self.assertEqual(loaded["image_path"], str((self.data / row["image_path"]).resolve()))
        self.assertEqual(loaded["sample_id"], "7")
        self.assertEqual(loaded["answers"], ["42"])
        self.assertEqual(self.snapshot(), before)

    def test_ai2d_relative_ocr_is_normalized_for_downstream_consumers(self):
        ocr = self.write_json("ai2d/ocr/page.json", {"words": [{"text": "root", "bbox": [2, 3, 8, 9]}]})
        row = {"image_path": "ai2d/images/page.png", "ocr_v2_path": str(ocr.relative_to(self.data)),
               "correct_option_text": "root", "question": "Which part?"}
        self.manifest("ai2d", row)
        before = self.snapshot()
        with working_directory(self.outside):
            loaded = self.functions["load_rows"]("ai2d", self.data)[0]
            self.assertEqual(loaded["ocr_v2_path"], str(ocr.resolve()))
            self.assertEqual(loaded["answers"], ["root"])
            for source in (row, loaded):
                self.assertEqual(self.functions["ocr_spans"]("ai2d", source, self.data, self.root / "cache"),
                                 [{"text": "root", "bbox": [2, 3, 8, 9]}])
        self.assertEqual(self.snapshot(), before)

    def test_absolute_ocr_paths_remain_usable(self):
        ocr = self.write_json("outside-manifest/page.json", {"analyzeResult": {"readResults": [
            {"lines": [{"text": "absolute", "polygon": [0, 0, 4, 0, 4, 2, 0, 2]}]}]}})
        row = {"image_path": str(self.data / "page.png"), "ocr_path": str(ocr), "question": "Text?"}
        self.manifest("docvqa", row)
        with working_directory(self.outside):
            loaded = self.functions["load_rows"]("docvqa", self.data)[0]
            self.assertEqual(loaded["ocr_path"], str(ocr.resolve()))
            self.assertEqual(self.functions["ocr_spans"]("docvqa", loaded, self.data, self.root / "cache")[0]["text"],
                             "absolute")

    def test_optional_missing_or_directory_ocr_is_empty(self):
        with working_directory(self.outside):
            for dataset, field in (("ai2d", "ocr_v2_path"), ("docvqa", "ocr_path")):
                for row in ({}, {field: None}, {field: ""}, {field: "missing.json"}, {field: "."}):
                    with self.subTest(dataset=dataset, row=row):
                        self.assertEqual(self.functions["ocr_spans"](dataset, row, self.data, self.root / "cache"), [])

    def test_infographic_ocr_receives_resolved_image_without_changing_input_row(self):
        tesseract = Mock(return_value=[{"text": "graphic", "bbox": [0, 0, 1, 1]}])
        row = {"image_id": "page", "image_path": "infographicvqa/images/page.png"}
        original = dict(row)
        with working_directory(self.outside), patch.dict(self.functions["ocr_spans"].__globals__, {"_tesseract_ocr": tesseract}):
            self.functions["ocr_spans"]("infographicvqa", row, self.data, self.root / "cache")
        self.assertEqual(tesseract.call_args.args[0]["image_path"], str((self.data / row["image_path"]).resolve()))
        self.assertEqual(tesseract.call_args.args[1], self.root / "cache/infographicvqa")
        self.assertEqual(row, original)


if __name__ == "__main__":
    unittest.main()
