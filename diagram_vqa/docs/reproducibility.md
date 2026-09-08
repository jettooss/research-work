# Reproduce and verify the experiments

The complete clean-clone walkthrough is now in the **[main README](../../README.md#reproducibility)**, including official dataset downloads, archive extraction, exact directory layouts, model downloads and troubleshooting. This document retains technical details and the validation record.

Run commands from the repository root. This guide distinguishes recalculating a table, evaluating a saved checkpoint, and training a new model. The saved results were audited on 2026-09-08; they have not all been retrained in the new environment. See [results and scientific limitations](results.md) and the [change record](reproduction_changes_2026-09-08.md).

## Check the published tables without datasets

```sh
python diagram_vqa/scripts/reproduce_results.py --check
```

This standard-library command validates the versioned evidence and recomputes five AI2D Accuracy aggregates and 12 retrieval rows from 36 principal runs. It checks source identity, canonical JSON SHA256, seeds, status, sample counts, retrieval directions and Markdown table consistency. Canonical JSON hashing makes the evidence independent of checkout line endings. It does not run inference or recover missing training configurations.

After intentionally updating the underlying measurements and their provenance, regenerate the two marked retrieval tables and verification JSON with:

```sh
python diagram_vqa/scripts/reproduce_results.py --write
python diagram_vqa/scripts/reproduce_results.py --check
```

Values are percentages; variation is population standard deviation across seeds 42, 43 and 44.

## Install a reference environment

Use Python 3.11. The tested reference is Windows, Python 3.11.9, PyTorch 2.6.0 CPU. Create a separate environment:

```sh
python -m venv .venv-repro
```

Activate with `.venv-repro/Scripts/Activate.ps1` in PowerShell or `source .venv-repro/bin/activate` in a POSIX shell. On Windows, `py -3.11 -m venv .venv-repro` explicitly selects Python 3.11. If activation is unavailable, call the environment's Python executable directly.

For the minimal Random/metric example:

```sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
python -m pip install -r diagram_vqa/requirements-cpu.txt
python -m pip install -e ./diagram_vqa
python diagram_vqa/examples/retrieval_metrics.py
```

For graph notebooks and saved-model evaluation, install the matched CPU pair, then research dependencies:

```sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 torchvision==0.21.0
python -m pip install -r diagram_vqa/requirements-research.txt
python -m pip install -e ./diagram_vqa
python -m pip check
```

Alternatively, on Windows/Python 3.11 CPU, install `requirements-lock-win-py311-cpu.txt` in a fresh environment and then install the local package. This complete reference freeze includes optional VLM packages. It records the environment tested here, not the undocumented original environment of every historical run.

For CUDA 12.4, use a separate environment and install the same version pair from `https://download.pytorch.org/whl/cu124` instead of the CPU index. The pairing is documented in [PyTorch's version-specific installation instructions](https://pytorch.org/get-started/previous-versions/). Then install `requirements-research.txt`; optional QLoRA dependencies are in `requirements-vlm.txt`. CPU and CUDA wheels must come from the same selected index. CUDA training, QLoRA generation and a full model retraining matrix were not executed during this repair.

OCR additionally requires the [Tesseract executable and English language data](https://tesseract-ocr.github.io/tessdoc/Installation.html). `pytesseract` alone does not install them. Check `tesseract --version` and `tesseract --list-langs`, or pass `--tesseract-cmd` to AI2D preparation. Model weights, text encoders and feature caches are separate inputs; initial extraction may download pretrained models. Keep their revisions and hashes with any released experiment archive.

The CLIP baseline uses the OpenAI implementation pinned to commit `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`; installing the research requirements therefore also requires Git. The installed package versions and performed checks are recorded in [environment_reference.json](environment_reference.json).

## Prepare AI2D from the official archive

Place official raw `images/`, `questions/` and the official test-ID CSV under `ai2d/`. First check metadata without OCR or output writes:

```sh
python diagram_vqa/scripts/prepare_ai2d_raw.py --data-root ai2d --official-test-ids "ai2d/ai2d_test_ids (1).csv" --check-inputs
```

On a complete archive, prepare new OCR and both manifest stages:

```sh
python diagram_vqa/scripts/prepare_ai2d_raw.py --data-root ai2d --official-test-ids "ai2d/ai2d_test_ids (1).csv" --seed 42 --val-ratio 0.1
```

The default outputs are `ai2d/prepared_v2/` and `ai2d/model_matrix_v1/`. Existing output directories are protected. To inspect a rebuild alongside historical data, add `--prepared-dir ai2d/prepared_v2_rebuilt --output-dir ai2d/model_matrix_v1_rebuilt`. To train on a new dataset version without replacing historical files, unpack a separate data root with the standard `ai2d/` layout, pass that root as `--path-root` during preparation, and use it as `--data-root`/`--external-root` during experiments.

The official AI2D ZIP checked on 2026-09-08 also lacks test image 4325 and its question JSON, although the official test-ID CSV includes it. This is not specific to the author's local copy, and downloading the same ZIP again will not supply it. Strict preflight therefore rejects that archive. To reproduce the available question dataset, add `--allow-missing-document-only-test-images` to preflight and preparation. This records `full_archive_ready=false`; it never excuses a missing image referenced by a question. The available question dataset contains 15,501 questions: 11,145 train, 1,268 validation and 3,088 test. The [main README](../../README.md#ai2d) documents the source links and explicit preparation commands.

Preparation preserves empty answer-option slots and their original indices. Historical preprocessing removed empty slots and changed the training label for `2618.png:2618.png-3` from A to C. The new version fixes that label. Therefore new manifests, OCR, features and runs receive new provenance; historical metrics/checkpoints are retained as measurements on their original data. No old score is silently presented as a result on corrected training data.

The preparation plan records raw input hashes, split/OCR parameters and software identity. `--resume` accepts only matching inputs and verified intermediate OCR outputs. It rejects changed or partially finalized incompatible artifacts. The resulting `provenance.json` and matrix audit identify output hashes and any missing-file exception. Recomputed OCR can differ from historical OCR or captions.

DocVQA and InfographicVQA preparation commands are listed in [usage.md](usage.md#dataset-preparation). Their downloaded data must follow the official split files. Expected matrix manifests, relative to the chosen data root, are:

| Dataset | Manifest |
|---|---|
| AI2D | `ai2d/model_matrix_v1/manifest.jsonl` |
| DocVQA | `docvqa/prepared_v1/manifest.jsonl` |
| InfographicVQA | `infographicvqa/prepared_v1/manifest.jsonl` |

## Run baselines and matrix training

Use a new output directory for each configuration. Relative `--output-dir` and `--data-root` paths are resolved from the caller's current directory before starting any child process.

```sh
python diagram_vqa/scripts/run_model_matrix.py --dataset ai2d --model random --split test --seed 42 --data-root . --output-dir diagram_vqa/runs/reproduction_random --no-resume
```

The full AI2D Random evaluation uses all 3,088 test questions and 814 unique documents. Repeat Random with seeds 43 and 44. `--max-samples 16` is a smoke check; it receives a distinct status and must use a different output root from a full run. The runner checks for at least 20 GiB of free output space.

For trainable matrix models, validation selects the checkpoint:

```sh
python diagram_vqa/scripts/run_model_matrix.py --dataset ai2d --model clip --split val --seed 42 --data-root . --output-dir diagram_vqa/runs/reproduction_clip --no-resume
```

This trains on train, selects on validation, then evaluates test. `--split test` only evaluates an already completed validation-selected checkpoint in the same experiment root; it cannot start training or select an epoch by test performance. Generic matrix graph architectures are distinct from the presentation's standalone attention models.

`--resume` validates immutable configuration before writing: semantic input rows, seed, sample limit, hyperparameters, implementation and installed package versions. Completed predictions/checkpoints must match stored hashes. Partial training resume checks checkpoint/history consistency and restores Python, NumPy and PyTorch RNG state. Smoke-to-full reuse, incompatible hyperparameters and unverifiable legacy runs are refused. `--no-resume` also refuses a nonempty run directory. CLIP's early-stopping patience is now applied. QLoRA adapter identity is checked and its optimizer checkpoints are no longer automatically deleted.

These checks do not hash every external image, pretrained model or feature-cache byte at every invocation. Preserve those inputs with independent hashes and use fresh caches when changing them. A matching software/configuration identity is necessary but does not promise cross-device bitwise training determinism.

## Run the presentation graph models

The registry now matches all 18 source notebooks: six architectures on three datasets. Register the research environment as a Jupyter kernel:

```sh
python -m ipykernel install --sys-prefix --name python3 --display-name "Diagram VQA reproduction"
python diagram_vqa/scripts/execute_experiment_matrix_notebooks.py --dataset ai2d --architecture sam2_dinov2_dual_branch_evidence_graph --seeds 42 43 44 --output-root diagram_vqa/runs/reproduction_graphs --external-root . --kernel-name python3 --dry-run
```

Check the printed data/cache paths and epoch budget, then remove `--dry-run` to execute. Use `--epochs N` for an explicit new budget. Omitting dataset and architecture plans 54 runs. Source notebooks are preserved: the executor configures copies, sets the seed before RNG initialization, records `notebook_run.json`/`execution.json`, and saves `notebook.executed.ipynb` for each run. Outputs and feature caches are isolated from historical runs. `--resume` requires matching recorded notebook/input identity.

The generator accepts the same dataset/architecture/seeds/output-root parameters if prepared notebooks are needed without execution. Hygiene validation inspects the actual six-architecture schemas and reports missing or incomplete artifacts. It no longer assumes every architecture must train for exactly 40 epochs. Historical attention runs did not save complete per-run hyperparameter configs, so current notebook defaults alone cannot establish exact historical training reproduction.

## Reevaluate a saved graph checkpoint

This requires a trusted checkpoint, its run metadata, prepared node features and the text encoder. Metadata alone is insufficient. To evaluate the saved AI2D Граф с общим отбором связей seed-42 checkpoint into new output files:

```sh
python diagram_vqa/reports/presentation_metrics/evaluate_saved_retrieval.py --project-root diagram_vqa --data-root . --runs-root diagram_vqa/runs --datasets ai2d --architectures sam2_dinov2_dual_branch_evidence_graph --seeds 42 --output-dir diagram_vqa/runs/reproduction_retrieval --local-files-only --device cpu
python diagram_vqa/reports/presentation_metrics/audit_retrieval_matrix.py --project-root diagram_vqa --data-root . --runs-root diagram_vqa/runs --datasets ai2d --architectures sam2_dinov2_dual_branch_evidence_graph --seeds 42 --retrieval-root diagram_vqa/runs/reproduction_retrieval --output diagram_vqa/runs/reproduction_retrieval_audit.json
```

Use a fresh evaluator output directory. `--local-files-only` avoids pretrained-model downloads; missing assets produce a clear failure. `--feature-cache` selects an exact node-feature directory; `--manifest` can override the manifest for one dataset and `--text-encoder` can select a local encoder. Model definitions are found by AST rather than fragile notebook cell numbers. The audit checks only the requested dataset/architecture/seed scope.

This full-test inference was executed in the fresh CPU environment. Bidirectional mean R@1/R@5/R@10 were 0.06693302758717266 / 0.21005660335323548 / 0.31450029280340164, identical to the saved evidence. Directional MRR differs by at most 0.000007711; the cause of this small ranking discrepancy has not been isolated, so exact cross-environment ranking reproduction is not established. The independent audit accepted the checkpoint, run and reported Recall result. This confirms the published Recall values for this saved run; it is not a new training seed.

## Package evidence for another machine

```sh
python diagram_vqa/scripts/export_reproduction_bundle.py --output reproduction-metadata.zip --dry-run
python diagram_vqa/scripts/export_reproduction_bundle.py --output reproduction-metadata.zip
python diagram_vqa/scripts/export_reproduction_bundle.py --verify reproduction-metadata.zip
```

The default archive includes source, compact metrics and available run metadata/predictions. Add `--include-weights` for selected trained checkpoints and `--include-data` for prepared manifests and referenced OCR. Data manifests are normalized in the archive copy to paths relative to the extracted data root; originals stay unchanged. The inventory hashes the archived bytes and records source hashes for transformed manifests. Verification rejects missing, duplicate, unsafe or altered entries. Existing output archives are never overwritten.

Images, pretrained model downloads, node-feature caches and optimizer checkpoints are excluded; obtain or archive them separately and record hashes. A metadata bundle can verify tables but cannot by itself reproduce image inference or training. Dataset distribution rights and access requirements remain those of their original sources. No archive has been published to an external service during this repair.

## Regression checks and remaining limits

```sh
python -m unittest discover -s diagram_vqa/tests -v
python diagram_vqa/scripts/reproduce_results.py --check
python -m pip check
```

All 71 tests passed in a separate copy containing only reproduction-bundle inputs. The tests cover invalid resume, validation-only checkpoint selection, data paths and answer indices, notebook configuration, generated-result integrity and archive portability. Real checks additionally covered OCR on two source images, raw metadata on all 15,501 questions, three full Random seeds, CLIP training/evaluation on a 16-question smoke configuration, and full saved-checkpoint inference. Random Accuracy and Recall matched the historical values exactly; MRR agreed within an absolute tolerance of 1e-12. The copied project also passed a fresh full Random evaluation and matching resume with a separate data root. See the [change record](reproduction_changes_2026-09-08.md) for the final verification results.

The original 36 runs have consistent saved local artifacts, but all 36 were not trained again. Historical incomplete hyperparameter provenance cannot be repaired by inventing settings. Full CUDA/QLoRA training and Docker build/runtime remain unverified here; this host's Docker Linux engine was unavailable. New corrected AI2D training data require new experiments before updating scientific conclusions. The external presentation/preprint and its publication status are separate from these repository changes.
