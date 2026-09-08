"""Regression tests for raw AI2D labels, audited splits and safe preparation."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
for directory in (PROJECT / "src", PROJECT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import prepare_ai2d_raw as raw
from prepare_ai2d_model_matrix import prepare_matrix, sha256_file
from vqa_retrieval.ai2d_hybrid import (
    Ai2dHybridSample,
    build_hybrid_samples_from_ai2d,
    create_image_level_splits,
    resolve_sample_file_paths,
)


class Ai2dPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "ai2d"
        (self.data / "images").mkdir(parents=True)
        (self.data / "questions").mkdir()
        for index in range(5):
            (self.data / "images" / f"{index}.png").write_bytes(f"fixture image {index}".encode())
            if index == 4:  # Official image without any questions is valid.
                continue
            self.write_question(index, ["C", "B", "", "A"] if index == 0 else ["A", "B"], 3 if index == 0 else 1)
        self.test_ids = self.data / "test.csv"
        self.test_ids.write_text("3\n4\n", encoding="utf-8")
        self.args = argparse.Namespace(
            data_root=self.data, official_test_ids=self.test_ids, prepared_dir=self.data / "prepared_v2",
            output_dir=self.data / "model_matrix_v1", path_root=self.root,
            seed=42, val_ratio=0.25, language="eng", min_conf=35.0,
            tesseract_cmd=None, check_inputs=False, resume=False, allow_missing_document_only_test_images=False,
        )

    def write_question(self, index, options, answer):
        payload = {"imageName": f"{index}.png", "questions": {
            "Which label?": {"answerTexts": options, "correctAnswer": answer, "questionId": f"{index}.png-0"}}}
        (self.data / "questions" / f"{index}.png.json").write_text(json.dumps(payload), encoding="utf-8")

    @staticmethod
    def fake_ocr(path, **kwargs):
        payload = {"image_path": str(path), "image_size": [40, 40], "engine": "tesseract_v2",
                   "language": "eng", "variant": "test", "words": [], "lines": []}
        return SimpleNamespace(to_dict=lambda: dict(payload))

    def prepare(self):
        with patch.object(raw, "configure_ocr", return_value=({"version": "test"}, self.fake_ocr)):
            return raw.prepare_raw(self.args)

    def test_official_blank_distractor_preserves_gold_index_and_roundtrip(self):
        samples = build_hybrid_samples_from_ai2d(self.data, self.args.prepared_dir, strict=True)
        sample = samples[0]
        self.assertEqual(sample.options, ("C", "B", "", "A"))
        self.assertEqual((sample.correct_option_idx, sample.correct_option_text), (3, "A"))
        self.assertEqual(Ai2dHybridSample.from_dict(sample.to_dict()), sample)

    def test_explicit_data_roots_take_precedence_over_cwd_files(self):
        sample = build_hybrid_samples_from_ai2d(self.data, self.args.prepared_dir, strict=True)[0]
        relative_image = Path("ai2d/images/0.png")
        relative_ocr = Path("ai2d/prepared_v2/ocr_v2/0.ocr.json")
        chosen_ocr = self.root / relative_ocr
        chosen_ocr.parent.mkdir(parents=True)
        chosen_ocr.write_text('{"words": []}', encoding="utf-8")
        unrelated_cwd = self.root / "unrelated_cwd"
        for relative in (relative_image, relative_ocr):
            stale = unrelated_cwd / relative
            stale.parent.mkdir(parents=True, exist_ok=True)
            stale.write_text("unrelated file", encoding="utf-8")
        relative_sample = replace(sample, image_path=relative_image.as_posix(),
                                  ocr_v2_path=relative_ocr.as_posix())
        original_cwd = Path.cwd()
        try:
            os.chdir(unrelated_cwd)
            resolved = resolve_sample_file_paths([relative_sample], roots=[self.root])[0]
            self.assertEqual(Path(resolved.image_path), self.root / relative_image)
            self.assertEqual(Path(resolved.ocr_v2_path), chosen_ocr)
            absolute = resolve_sample_file_paths([sample], roots=[unrelated_cwd])[0]
            self.assertEqual(absolute.image_path, sample.image_path)
            self.assertIsNone(absolute.ocr_v2_path)
        finally:
            os.chdir(original_cwd)

    def test_invalid_gold_is_rejected_instead_of_silently_clamped(self):
        self.write_question(0, ["A", "B"], 3)
        with self.assertRaisesRegex(ValueError, "Out-of-range correctAnswer"):
            build_hybrid_samples_from_ai2d(self.data, self.args.prepared_dir, strict=True)

    def test_document_only_and_actually_missing_test_images_are_distinct(self):
        split = create_image_level_splits(["0", "1", "2", "3"], ["3", "4", "missing"],
                                          available_image_ids=["0", "1", "2", "3", "4"])
        self.assertEqual(split["document_only_test_ids"], ["4"])
        self.assertEqual(split["missing_test_ids"], ["missing"])
        self.assertEqual(split["test_image_ids"], ["3", "4"])
        self.test_ids.write_text("3\n4\nmissing\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Official test image files are missing"):
            self.prepare()
        self.assertFalse(self.args.prepared_dir.exists())

    def test_explicit_missing_document_only_policy_keeps_questions_but_not_archive_ready(self):
        self.test_ids.write_text("3\n4\nmissing\n", encoding="utf-8")
        self.args.allow_missing_document_only_test_images = True
        audit = self.prepare()
        self.assertEqual(sum(audit["questions"].values()), 4)
        self.assertTrue(audit["question_dataset_ready"])
        self.assertFalse(audit["ready"])
        self.assertFalse(audit["full_archive_ready"])
        self.assertEqual(audit["missing_official_test_image_ids"], ["missing"])
        self.assertTrue(audit["warnings"])
        # The escape hatch must never hide missing images which DO have questions.
        self.args.check_inputs = True
        (self.data / "images" / "0.png").unlink()
        with self.assertRaisesRegex(ValueError, "Missing image referenced"):
            self.prepare()

    def test_raw_pipeline_paths_splits_gold_hashes_and_matrix_replay(self):
        audit = self.prepare()
        self.assertTrue(audit["ready"])
        self.assertEqual(audit["questions"], {"train": 2, "val": 1, "test": 1})
        self.assertEqual(audit["document_only_test_images"], 1)
        self.assertFalse(any(audit["image_leakage"].values()))
        matrix = self.args.output_dir / "manifest.jsonl"
        rows = [json.loads(line) for line in matrix.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows[0]["correct_option_text"], "A")
        for row in rows:
            self.assertEqual(len(row["options"]), 4 if row["image_id"] == "0" else 2)
            for field in ("image_path", "ocr_v2_path"):
                self.assertFalse(Path(row[field]).is_absolute())
                self.assertNotIn(chr(92), row[field])
                self.assertTrue((self.root / row[field]).is_file())
        provenance = json.loads((self.args.prepared_dir / "provenance.json").read_text(encoding="utf-8"))
        for relative, digest in provenance["output_sha256"].items():
            self.assertEqual(sha256_file(self.root / relative), digest)
        replay = prepare_matrix(self.args.prepared_dir / "manifest_hybrid.jsonl", self.test_ids,
                                self.data / "replayed_matrix", self.data / "images", self.root,
                                seed=42, val_ratio=0.25)
        self.assertEqual(replay["manifest_sha256"], audit["manifest_sha256"])
        self.assertEqual(replay["split_sha256"], audit["split_sha256"])

    def test_unrelated_existing_outputs_are_never_overwritten(self):
        self.args.prepared_dir.mkdir()
        original = self.args.prepared_dir / "existing_manifest.jsonl"
        original.write_text("keep me", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Prepared directory is not empty"):
            self.prepare()
        self.assertEqual(original.read_text(encoding="utf-8"), "keep me")
        self.assertFalse(self.args.output_dir.exists())

    def test_completed_resume_checks_hashes_and_refuses_changed_inputs(self):
        self.prepare()
        self.args.resume = True
        self.assertEqual(self.prepare()["status"], "already_complete_verified")
        (self.data / "images" / "0.png").write_bytes(b"different source")
        with self.assertRaisesRegex(ValueError, "source files, parameters or software changed"):
            self.prepare()

    def test_interrupted_ocr_resumes_only_valid_own_cache(self):
        count = 0

        def interrupted(path, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError("interruption")
            return self.fake_ocr(path, **kwargs)

        with patch.object(raw, "configure_ocr", return_value=({"version": "test"}, interrupted)):
            with self.assertRaisesRegex(ValueError, "OCR failed"):
                raw.prepare_raw(self.args)
        first_cache = self.args.prepared_dir / "ocr_v2/0.ocr.json"
        original_hash = sha256_file(first_cache)
        self.args.resume = True
        self.assertTrue(self.prepare()["ready"])
        self.assertEqual(sha256_file(first_cache), original_hash)

    def test_input_validation_needs_no_ocr_and_makes_no_outputs(self):
        self.args.check_inputs = True
        with patch.object(raw, "configure_ocr", side_effect=AssertionError("must not initialize OCR")):
            result = raw.prepare_raw(self.args)
        self.assertEqual(result["writes"], 0)
        self.assertEqual(result["questions"], 4)
        self.assertFalse(self.args.prepared_dir.exists())
        self.assertFalse(self.args.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
