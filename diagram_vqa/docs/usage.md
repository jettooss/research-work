# Installation, datasets, and interfaces

Commands run from the repository root unless stated otherwise. The [main README](../../README.md#reproducibility) provides the complete clean-clone walkthrough, including official downloads, folder layouts and copyable commands. This page is the compact CLI/API reference. The Docker image covers metric evaluation and the Random CPU baseline; GPU training and VLM inference require the full research environment.

## Environment

The package requires Python 3.11+. The checked reference environment uses Python 3.11.9; the Docker image uses Python 3.11 and CPU PyTorch 2.6.0. PyTorch is imported by the package initializer, including when importing metric helpers.

```sh
python -m venv .venv
```

Activate with `.venv/Scripts/Activate.ps1` in PowerShell or `source .venv/bin/activate` on Linux/macOS. If activation is unavailable, replace `python` below with `.venv/Scripts/python.exe` on Windows or `.venv/bin/python` on Linux/macOS. Reuse an existing environment instead of creating it again.

```sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
python -m pip install -r diagram_vqa/requirements-cpu.txt
python -m pip install -e ./diagram_vqa
```

The [CPU requirements](../requirements-cpu.txt) pin direct dependencies; package metadata also declares runtime requirements. A [complete Windows/Python 3.11 CPU reference lock](../requirements-lock-win-py311-cpu.txt), [graph requirements](../requirements-research.txt) and [optional QLoRA requirements](../requirements-vlm.txt) are provided. The [reproduction guide](reproducibility.md) gives the checked installation order, CUDA 12.4 wheel pair, and external Tesseract/model requirements.

The pinned research dependencies include:

| Component | Observed version | Purpose |
|---|---|---|
| torch | 2.6.0+cu124 | Training and inference |
| torch-geometric | 2.8.0 | GATv2 and graph scripts |
| transformers | 4.49.0 | DINOv2, CLIP, VLM |
| sentence-transformers | 5.3.0 | Text features |
| ultralytics | 8.4.115 | SAM2 |
| peft / accelerate / bitsandbytes | 0.19.1 / 1.14.0 / 0.49.2 | QLoRA |

The complete set was installed in a fresh Windows/Python 3.11 CPU environment and passed dependency/import checks and a GATv2 forward pass; see [environment reference](environment_reference.json). CUDA training and QLoRA generation are separate validation tasks. Interactive notebooks can use the included ipykernel/nbclient; OCR requires the separate [Tesseract executable](https://github.com/tesseract-ocr/tesseract).

## Dataset preparation

Keep datasets and models alongside `diagram_vqa/`. The matrix loaders expect:

| Dataset | Manifest relative to the repository root |
|---|---|
| AI2D | `ai2d/model_matrix_v1/manifest.jsonl` |
| DocVQA | `docvqa/prepared_v1/manifest.jsonl` |
| InfographicVQA | `infographicvqa/prepared_v1/manifest.jsonl` |

Each JSONL row contains `image_path`, `split` (`train`, `val`, `test`), `question`, `sample_id`/`question_id`, and `image_id`. AI2D also needs `options` and `correct_option_text`; the other datasets use `answers`. Relative image paths resolve from the repository root. Update absolute paths when moving manifests between machines or into Linux containers. AI2D uses `ocr_v2_path`; DocVQA uses `ocr_path`; InfographicVQA uses Tesseract and an OCR cache.

Obtain data from [AI2D](https://allenai.org/data/diagrams), [DocVQA](https://www.docvqa.org/datasets/docvqa), and [InfographicVQA](https://www.docvqa.org/datasets/infographicvqa). Preparation scripts convert downloaded data; they do not download the datasets.

```sh
python diagram_vqa/scripts/prepare_ai2d_model_matrix.py --help
python diagram_vqa/scripts/prepare_docvqa_manifest.py --help
python diagram_vqa/scripts/prepare_infographicvqa_manifest.py --help
```

For an unprepared official AI2D archive, first run `prepare_ai2d_raw.py`: it performs OCR, preserves official answer-option indices, writes the hybrid manifest, then runs audited image-level splitting. For an existing hybrid manifest use `prepare_ai2d_model_matrix.py`. The [complete commands and data-version caveats](reproducibility.md#prepare-ai2d-from-the-official-archive) include handling missing files and resuming verified preparation. DocVQA/InfographicVQA preparation accepts `--data-root`, `--output-dir`, and `--splits`. These preparation scripts are not included in the CPU example image.

The AI2D «Граф с общим отбором связей» notebook expects `models/sam2/sam2.1_b.pt`, OCR under `ai2d/prepared_v2/ocr_v2`, and SAM boxes under `model_matrix_cache/sam2_boxes`. DINOv2 uses `facebook/dinov2-base`; the text encoder is configured in the notebook. Local Qwen weights are expected under `models/Qwen2.5-VL-3B-Instruct`. Initial feature extraction can download models and takes longer than evaluation from a prepared cache.

## Run on prepared data

A small Random baseline on 16 AI2D test questions:

```sh
python diagram_vqa/scripts/run_model_matrix.py --dataset ai2d --model random --split test --seed 42 --max-samples 16 --output-dir diagram_vqa/runs/readme_smoke --no-resume
```

The dispatcher resolves data/output paths from the caller's working directory before starting its child process in `diagram_vqa/`. This command writes `config.json`, `metrics.json`, and `predictions.jsonl` under `diagram_vqa/runs/readme_smoke/ai2d/random/seed42/test/`. A limited run has status `available_smoke` and is not comparable with full-test presentation scores. The runner requires at least 20 GiB of free output-disk space even for a small run.

Omit `--max-samples` and use a new output directory for full evaluation. Resume validates input/configuration/software identity before writing; it rejects legacy runs or a full request against smoke artifacts. `--no-resume` also requires a new or empty run directory. For trainable models use `--split val` to train and select the checkpoint on validation, then evaluate test. `--split test` is evaluation-only and requires a completed validation-selected checkpoint; it cannot start training.

## Docker with a dataset

Build the image from the repository root:

```sh
docker build -t diagram-vqa:cpu .
docker run --rm diagram-vqa:cpu
```

The default command evaluates the small synthetic similarity matrix. To run Random on an existing AI2D manifest, mount the data read-only and persist the outputs. In PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path diagram_vqa/runs/docker_cpu | Out-Null
$ai2dData = (Resolve-Path ai2d).Path
$cpuResults = (Resolve-Path diagram_vqa/runs/docker_cpu).Path
docker run --rm --mount "type=bind,source=$ai2dData,target=/workspace/ai2d,readonly" --mount "type=bind,source=$cpuResults,target=/workspace/diagram_vqa/runs" diagram-vqa:cpu python scripts/run_model_matrix.py --dataset ai2d --model random --split test --seed 42 --max-samples 16 --no-resume
```

On Linux/macOS:

```sh
mkdir -p diagram_vqa/runs/docker_cpu
docker run --rm --mount "type=bind,source=$(pwd)/ai2d,target=/workspace/ai2d,readonly" --mount "type=bind,source=$(pwd)/diagram_vqa/runs/docker_cpu,target=/workspace/diagram_vqa/runs" diagram-vqa:cpu python scripts/run_model_matrix.py --dataset ai2d --model random --split test --seed 42 --max-samples 16 --no-resume
```

Results appear at `diagram_vqa/runs/docker_cpu/model_matrix/ai2d/random/seed42/test/`. This baseline reads question/answer metadata without opening images. Image-based approaches additionally need container-valid image/OCR paths and their dependencies. The CPU image contains the baseline/dispatcher scripts and run-identity helper; graph training, OCR extraction, and VLM runners belong to the full environment.

The [.dockerignore](../../.dockerignore) allows only the image's source inputs. Datasets, model weights, `.env` files, run artifacts, and the local virtual environment are excluded from the build context. See Docker's [build-context documentation](https://docs.docker.com/build/concepts/context/#dockerignore-files).

## CLI reference

`python diagram_vqa/scripts/run_model_matrix.py --help`

| Argument | Type / accepted values | Default |
|---|---|---|
| `--dataset` | `ai2d`, `docvqa`, `infographicvqa` | Required |
| `--model` | `random`, `clip`, `siglip`, `ocr_text`, `gatv2_knn`, `graph_transformer`, `sam2_graph_transformer`, `hybrid_gatv2_knn`, `graphcolbert`, `graphcolbert_film`, `qwen25_vl_qlora` | Required |
| `--split` | `val` or `test` | `val` |
| `--seed` | Integer | `42` |
| `--max-samples` | Positive integer for a small run | Unlimited |
| `--data-root` | Root containing dataset/model/cache directories | Repository root |
| `--output-dir` | Path; relative paths resolve from the caller's working directory | Absolute `diagram_vqa/runs/model_matrix` |
| `--resume` / `--no-resume` | Reuse only a verified matching run / require a new run directory | `--resume` |

Use separate directories for different sample sizes and configurations. Validate existing outputs before resuming. The CLI lists the full research model set even when run inside the limited CPU image.

## Presentation graph models

The presentation's «Граф с общим отбором связей» and «Граф с отбором связей по типам» models live in the [experiment notebooks](../notebooks/experiments). They are not separate choices in the matrix CLI; the generic `graph_transformer` does not reproduce those architectures.

Use the notebook executor to create separate configured copies and preserve the source notebooks:

```sh
python diagram_vqa/scripts/execute_experiment_matrix_notebooks.py --dataset ai2d --architecture sam2_dinov2_dual_branch_evidence_graph --seeds 42 43 44 --output-root diagram_vqa/runs/reproduction_graphs --external-root . --kernel-name python3 --dry-run
```

Remove `--dry-run` after checking the printed configuration and installing the research kernel. Each run records the source notebook hash, applied parameters, execution status and executed notebook. All six current architectures on all three datasets are supported. Use the [full recipe](reproducibility.md#run-the-presentation-graph-models) for kernel setup, cache constraints and validation. Historical notebook defaults alone do not reconstruct undocumented historical hyperparameters.

## Python metrics API

After installing the package:

```python
from vqa_retrieval.metrics import recall_at_k_multi_positive

scores = [[0.9, 0.1], [0.4, 0.6], [0.2, 0.8]]
print(recall_at_k_multi_positive(scores, [{0}, {0}, {1}], ks=(1, 5)))
# {1: 0.6666666666666666, 5: 1.0}
```

| Function | Input | Return value |
|---|---|---|
| `recall_at_k_multi_positive(sim, positive_indices, ks=(1, 5, 10))` | Rectangular similarity matrix `[queries, candidates]`; one nonempty set of relevant indices per query | `dict[int, float]`: fraction of queries with at least one hit |
| `mean_reciprocal_rank_multi_positive(sim, positive_indices)` | The same matrix and relevance mapping | `float`: mean reciprocal rank of the first relevant candidate |
| `recall_at_k_from_sim(sim, ks=(1, 5, 10))` | Square matrix `[N, N]` with correct pairs on the diagonal | `dict[int, float]`; suitable only for one-to-one pairs |

Indices are zero-based; higher similarity ranks first. K larger than the candidate count uses all candidates. Empty/ragged matrices, missing positives, out-of-range indices, and K≤0 raise `ValueError`. Ties preserve candidate order.

`vqa_retrieval.model_matrix.retrieval_metrics(similarity, question_doc_ids, document_ids)` accepts a NumPy matrix `[questions, unique_documents]` and identifiers. It returns `question_to_document`, `document_to_question` (each containing `recall_at_k` and `mrr`), and `mean_recall_at_k`. K keys in this wrapper are strings. Every question must refer to a candidate document, and each document needs at least one associated question.

`vqa_retrieval.public_vqa_metrics.public_vqa_score(dataset_name, prediction, gold_answers)` accepts a dataset name, predicted answer string, and sequence of reference strings. It returns `(metric_name: str, score: float)`: normalized exact accuracy for AI2D, or average normalized Levenshtein similarity (ANLS) for DocVQA/InfographicVQA, with distance threshold 0.5. This scores answers separately from retrieval. See the [normalization implementation](../src/vqa_retrieval/public_vqa_metrics.py).

## Interpreting outputs

Check dataset, split, seed, sample count, and selected checkpoint before comparing quality. Presentation retrieval uses a dedicated unique-document evaluation; older graph-training retrieval fields can use different candidate sets. [Results and provenance](results.md) describe the published tables and known source discrepancies.

## Verification notes

The metric example, generated tables, fresh CPU/research environment and focused regression suite were checked after the reproducibility fixes. Baseline and saved-checkpoint checks are detailed in [reproducibility.md](reproducibility.md). Docker's Linux engine remains unavailable on this host, so container build/execution is still unverified; the CPU build context now includes the run-identity helper.
