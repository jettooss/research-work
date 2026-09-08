"""CPU/stdlib regression tests: no pretrained models, downloads or GPU required."""
from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import math
from pathlib import Path
import random
import statistics
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import Mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import model_matrix_integrity as integrity


def functions_from_file(name, names, namespace):
    """Execute the production control flow with tiny CPU test doubles for ML I/O."""
    path = SCRIPTS / name
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    body.extend(node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names)
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec"), namespace)
    return namespace


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_predictions(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class RunIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = [dict(sample_id=f"{split}-{i}", image_id=f"{split}-doc", question=f"Question {i}",
                          split=split, options=["yes", "no"], answers=["yes"], correct_option_text="yes")
                     for split in ("train", "val", "test") for i in range(3)]
        self.args = argparse.Namespace(dataset="ai2d", model="random", split="test", seed=42,
                                       max_samples=None, output_dir=self.root, resume=True,
                                       epochs=2, early_stopping_patience=0, batch_size=2, eval_batch_size=2)

    def namespace(self):
        values = {name: getattr(integrity, name) for name in dir(integrity) if not name.startswith("_")}
        values.update(argparse=argparse, Path=Path, json=json, Any=object,
                      EXTERNAL_ROOT=self.root, load_rows=lambda *a: self.rows,
                      split_rows=lambda rows, split, limit=None: [r for r in rows if r["split"] == split][:limit],
                      ensure_free_space=Mock(side_effect=AssertionError("must not mutate before validation")),
                      write_jsonl=write_predictions)
        return values

    def baseline(self):
        return functions_from_file("run_model_matrix_baseline.py", ["run_baseline_config"], self.namespace())["run_baseline_config"]

    def save_baseline(self, args):
        path = args.output_dir / args.dataset / args.model / f"seed{args.seed}" / args.split
        config = integrity.make_config(args, self.rows, kind="baseline", trainable=False)
        integrity.write_config(path, config)
        selected = [x for x in self.rows if x["split"] == args.split][:args.max_samples]
        result = {"dataset": args.dataset, "model": args.model, "seed": args.seed, "split": args.split,
                  "status": "available_full" if args.max_samples is None else "available_smoke",
                  "num_samples": len(selected), "config_sha256": integrity.digest(config)}
        write_predictions(selected, path / "predictions.jsonl")
        result["predictions_sha256"] = integrity.file_digest(path / "predictions.jsonl")
        write_json(path / "metrics.json", result)
        return path, result

    def test_smoke_cannot_be_resumed_as_full_or_relabel_config(self):
        smoke = argparse.Namespace(**(vars(self.args) | {"max_samples": 1}))
        path, _ = self.save_baseline(smoke)
        before = {p.name: p.read_bytes() for p in path.iterdir()}
        with self.assertRaisesRegex(ValueError, "max_samples"):
            self.baseline()(self.args)
        self.assertEqual(before, {p.name: p.read_bytes() for p in path.iterdir()})

    def test_matching_completed_baseline_returns_before_compute_or_write(self):
        path, expected = self.save_baseline(self.args)
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in path.iterdir()}
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(expected, self.baseline()(self.args))
        self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in path.iterdir()})

    def test_limit_larger_than_split_uses_actual_count_for_safe_resume(self):
        self.args.max_samples = 100
        _, expected = self.save_baseline(self.args)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(expected, self.baseline()(self.args))

    def test_changed_labels_in_same_manifest_location_reject_resume(self):
        path, _ = self.save_baseline(self.args)
        before = (path / "config.json").read_bytes()
        self.rows[0]["answers"] = ["different"]
        with self.assertRaisesRegex(ValueError, "input_rows_sha256"):
            self.baseline()(self.args)
        self.assertEqual(before, (path / "config.json").read_bytes())

    def test_changed_clip_package_version_rejects_resume_without_writes(self):
        installed_version = integrity.importlib.metadata.version
        with unittest.mock.patch.object(integrity.importlib.metadata, "version",
                                        side_effect=lambda name: "1.0" if name == "clip" else installed_version(name)):
            path, _ = self.save_baseline(self.args)
        self.assertEqual("1.0", integrity.read_json(path / "config.json")["package_versions"]["clip"])
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in path.iterdir()}
        with unittest.mock.patch.object(integrity.importlib.metadata, "version",
                                        side_effect=lambda name: "1.1" if name == "clip" else installed_version(name)):
            with self.assertRaisesRegex(ValueError, "package_versions"):
                self.baseline()(self.args)
        self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in path.iterdir()})

    def test_data_root_relocation_preserves_semantic_identity(self):
        first_root, second_root = self.root / "machine-one", self.root / "machine-two"
        first = [dict(row, image_path=str(first_root / "ai2d/images/1.png"), ocr_v2_path="ai2d/ocr/1.json") for row in self.rows]
        second = [dict(row, image_path=str(second_root / "ai2d/images/1.png"), ocr_v2_path="ai2d/ocr/1.json") for row in self.rows]
        a = integrity.make_config(self.args, first, kind="baseline", input_root=first_root)
        b = integrity.make_config(self.args, second, kind="baseline", input_root=second_root)
        self.assertEqual(a, b)

    def test_dispatcher_resolves_output_from_caller_before_changing_child_cwd(self):
        other = self.root / "separate-data"
        command = Mock()
        ns = {"__file__": str(SCRIPTS / "run_model_matrix.py"), "ROOT": SCRIPTS.parent,
              "Path": Path, "argparse": argparse, "DATASETS": ["ai2d"], "MODELS": ["random"],
              "sys": types.SimpleNamespace(executable="python", argv=["runner", "--dataset", "ai2d", "--model", "random", "--data-root", str(other), "--output-dir", "relative-results"]),
              "subprocess": types.SimpleNamespace(run=command, list2cmdline=lambda args: str(args))}
        # argparse reads process argv, while the command builder uses the injected sys executable.
        functions_from_file("run_model_matrix.py", ["main"], ns)
        with contextlib.chdir(self.root), unittest.mock.patch.object(sys, "argv", ns["sys"].argv), contextlib.redirect_stdout(io.StringIO()):
            ns["main"]()
        argv = command.call_args.args[0]
        self.assertEqual(str(other.resolve()), argv[argv.index("--data-root") + 1])
        self.assertEqual(str(self.root / "relative-results"), argv[argv.index("--output-dir") + 1])
        self.assertEqual(SCRIPTS.parent, command.call_args.kwargs["cwd"])

    def test_direct_backend_clis_resolve_output_from_caller(self):
        for script, model, function in (
            ("run_model_matrix_baseline.py", "random", "run_baseline_config"),
            ("train_model_matrix_vision.py", "clip", "run_vision_experiment_config"),
            ("train_model_matrix_local.py", "gatv2_knn", "run_local_experiment_config"),
            ("run_model_matrix_qlora.py", "qwen25_vl_qlora", "run_qlora_config"),
        ):
            with self.subTest(script=script):
                run = Mock(return_value={})
                ns = {"ROOT": SCRIPTS.parent, "Path": Path, "argparse": argparse,
                      "DATASETS": ["ai2d"], "SEEDS": [42], "TRAINABLE_LOCAL_MODELS": [model],
                      "json": json, function: run}
                functions_from_file(script, ["main"], ns)
                argv = ["runner", "--dataset", "ai2d", "--model", model,
                        "--output-dir", "relative-results", "--data-root", "relative-data"]
                with contextlib.chdir(self.root), unittest.mock.patch.object(sys, "argv", argv):
                    ns["main"]()
                self.assertEqual(self.root / "relative-results", run.call_args.args[0].output_dir)
                self.assertEqual(self.root / "relative-data", ns["EXTERNAL_ROOT"])

    def test_no_resume_preserves_nonempty_directory(self):
        path, _ = self.save_baseline(self.args)
        self.args.resume = False
        with self.assertRaisesRegex(ValueError, "fresh --output-dir"):
            self.baseline()(self.args)
        self.assertTrue((path / "metrics.json").is_file())

    def test_missing_or_reordered_predictions_are_not_completed(self):
        path, _ = self.save_baseline(self.args)
        rows = [x for x in self.rows if x["split"] == "test"]
        write_predictions(list(reversed(rows)), path / "predictions.jsonl")
        with self.assertRaisesRegex(ValueError, "have changed"):
            self.baseline()(self.args)

    def test_legacy_aggregates_without_identity_cannot_be_reused(self):
        path = self.root / "ai2d/random/seed42/test"
        write_json(path / "metrics.json", {"status": "available_full", "num_samples": 3})
        with self.assertRaisesRegex(ValueError, "legacy run"):
            self.baseline()(self.args)

    def training_namespace(self, backend):
        filename = f"train_model_matrix_{backend}.py"
        function = f"run_{'vision' if backend == 'vision' else 'local'}_experiment_config"
        evaluator = f"evaluate_{backend}_test"
        namespace = self.namespace()
        namespace["torch"] = Mock(side_effect=AssertionError("test selection must not start training"))
        functions_from_file(filename, [function, evaluator], namespace)
        return namespace, function

    def test_trainable_test_request_without_checkpoint_never_starts_training(self):
        for backend, model in (("vision", "clip"), ("local", "gatv2_knn")):
            with self.subTest(backend=backend):
                self.args.model = model
                ns, name = self.training_namespace(backend)
                with self.assertRaisesRegex(ValueError, "evaluation-only"):
                    ns[name](self.args)
                ns["ensure_free_space"].assert_not_called()
                self.assertEqual([], list(self.root.iterdir()))

    def test_training_hyperparameter_change_rejected_before_any_write(self):
        self.args.model, self.args.split = "clip", "val"
        path, _ = integrity.training_paths(self.args)
        config = integrity.training_config(self.args, self.rows, backend="frozen_vision", epochs=2,
                                           early_stopping_patience=0, batch_size=2, frozen_backbone=True)
        integrity.write_config(path, config)
        self.args.epochs = 3
        ns, name = self.training_namespace("vision")
        before = (path / "config.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "epochs"):
            ns[name](self.args)
        self.assertEqual(before, (path / "config.json").read_bytes())
        ns["ensure_free_space"].assert_not_called()

    def test_checkpoint_cannot_claim_a_different_history_or_config(self):
        self.args.model, self.args.split = "clip", "val"
        config = integrity.training_config(self.args, self.rows, backend="frozen_vision", epochs=2)
        path, _ = integrity.training_paths(self.args)
        integrity.write_config(path, config)
        history = [{"epoch": 1, "composite": 0.5}]
        write_json(path / "history.json", history)
        write_json(path / "checkpoint_best.pt", {"epoch": 1, "config_sha256": integrity.digest(config)})
        last = {"epoch": 1, "config_sha256": integrity.digest(config),
                "history_sha256": "wrong", "rng_state": {}}
        write_json(path / "checkpoint_last.pt", last)
        with self.assertRaisesRegex(ValueError, "history disagree"):
            integrity.validate_partial_checkpoint(path, config, integrity.read_json)

    def test_qlora_test_request_does_not_train_an_adapter(self):
        self.args.model = "qwen25_vl_qlora"
        ns = self.namespace()
        functions_from_file("run_model_matrix_qlora.py", ["run_qlora_config"], ns)
        with self.assertRaisesRegex(ValueError, "evaluation-only"):
            ns["run_qlora_config"](self.args)
        ns["ensure_free_space"].assert_not_called()
        self.assertEqual([], list(self.root.iterdir()))

    def test_qlora_shared_smoke_adapter_cannot_be_reused_for_full(self):
        self.args.model = "qwen25_vl_qlora"
        training = self.root / "ai2d/qwen25_vl_qlora/seed42/training"
        integrity.write_config(training, {"schema_version": 2, "max_samples": 1})
        write_json(training / "completion.json", {})
        ns = self.namespace()
        functions_from_file("run_model_matrix_qlora.py", ["run_qlora_config"], ns)
        before = (training / "config.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "max_samples"):
            ns["run_qlora_config"](self.args)
        self.assertEqual(before, (training / "config.json").read_bytes())
        ns["ensure_free_space"].assert_not_called()

    def test_validation_selects_checkpoint_and_test_mode_only_evaluates(self):
        self.args.model, self.args.split = "clip", "val"
        self.args.max_samples = 2
        ns = self.namespace()
        calls = []

        class Head:
            def to(self, device): return self
            def train(self): pass
            def state_dict(self): return {}
            def load_state_dict(self, state): pass
            def parameters(self): return []

        class Optimizer:
            def state_dict(self): return {}
            def load_state_dict(self, state): pass

        def evaluate(model, rows, store, device):
            split = rows[0]["split"]
            calls.append(split)
            # Test would win any checkpoint-selection contest; it must never enter one.
            score = 0.99 if split == "test" else (0.6 if calls.count("val") != 2 else 0.4)
            return {"retrieval": {"mean_recall_at_k": {"1": score}}, "vqa": {"score": score}}, rows

        cuda = types.SimpleNamespace(is_available=lambda: False)
        optimizer_factory = Mock(return_value=Optimizer())
        ns.update(torch=types.SimpleNamespace(cuda=cuda, device=lambda x: x, manual_seed=lambda x: None,
                                             optim=types.SimpleNamespace(AdamW=optimizer_factory),
                                             save=lambda value, path: write_json(path, value)),
                  random=random, math=math, time=time,
                  np=types.SimpleNamespace(random=types.SimpleNamespace(seed=lambda x: None),
                                           mean=lambda values: statistics.mean(values) if values else 0.0),
                  FrozenVisionStore=lambda *a: types.SimpleNamespace(feature_dim=2), VisionHeads=lambda *a: Head(),
                  RowDataset=lambda x: x, DataLoader=lambda *a, **k: [], tqdm=lambda x, **k: x,
                  ensure_free_space=lambda p: None, load_torch=integrity.read_json, evaluate=evaluate,
                  validation_loss=lambda *a: 0.0, capture_rng_state=lambda: {"test": True},
                  restore_rng_state=lambda x: None,
                  psutil=types.SimpleNamespace(Process=lambda: types.SimpleNamespace(memory_info=lambda: types.SimpleNamespace(rss=0))))
        functions_from_file("train_model_matrix_vision.py", ["run_vision_experiment_config", "evaluate_vision_test"], ns)
        with contextlib.redirect_stdout(io.StringIO()):
            result = ns["run_vision_experiment_config"](self.args)
        self.assertEqual(["val", "val", "val", "test"], calls)
        self.assertEqual(1, result["best_epoch"])
        self.assertEqual("completed_smoke", result["status"])
        self.assertFalse(result["training_execution"]["full_training_was_run"])
        self.assertEqual("val", result["test"]["checkpoint_selection_split"])
        train_dir, test_dir = integrity.training_paths(self.args)
        checkpoint_before = (train_dir / "checkpoint_best.pt").read_bytes()
        history_before = (train_dir / "history.json").read_bytes()
        (test_dir / "metrics.json").unlink()  # simulate interrupted final evaluation in this temporary fixture
        self.args.split = "test"
        optimizer_factory.reset_mock()
        with contextlib.redirect_stdout(io.StringIO()):
            evaluated = ns["run_vision_experiment_config"](self.args)
        optimizer_factory.assert_not_called()
        self.assertEqual("test", calls[-1])
        self.assertEqual("test", evaluated["split"])
        self.assertEqual(checkpoint_before, (train_dir / "checkpoint_best.pt").read_bytes())
        self.assertEqual(history_before, (train_dir / "history.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
