# Graph Models for Diagram Representation in Retrieval and Ranking Tasks

[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](diagram_vqa/pyproject.toml)
[![Docker: CPU example](https://img.shields.io/badge/Docker-CPU_example-2496ED?logo=docker&logoColor=white)](#docker)
[![Google Slides: presentation](https://img.shields.io/badge/Google_Slides-Presentation-FBBC04?logo=googleslides&logoColor=white)](https://docs.google.com/presentation/d/1a3m6Goh3P5MZFHDC1UjNX7AyD_vt7G5VJsvcX-Syvyc/edit)

Graph-based retrieval and visual question answering for diagrams, documents, and infographics on **AI2D, DocVQA, and InfographicVQA**.

The project studies how explicit text, object, and spatial relationships affect document retrieval, ranking, and answer selection. It compares CLIP, Hybrid GATv2 + kNN, Heterogeneous Graph + Attention, and Dual-Branch + top-k Attention. Additional experiments cover SigLIP, OCR-text, GraphColBERT/FiLM, and Qwen2.5-VL with QLoRA.

The main interfaces are Python scripts and research notebooks. Datasets, model weights, and feature caches are supplied separately. Some reported configurations have fewer than three completed runs; see the evaluation notes below.

## Quick start

### Docker

With Docker running in Linux-container mode, build and run the CPU example from the repository root:

```sh
docker build -t diagram-vqa:cpu .
docker run --rm diagram-vqa:cpu
```

The image runs the [retrieval example](diagram_vqa/examples/retrieval_metrics.py) on three questions and two documents. It requires no dataset download, model weights, or GPU. Python 3.11 and CPU PyTorch are included; graph training and VLM dependencies are installed separately.

```text
R@1: q->doc=66.67%, doc->q=100.00%, mean=83.33%
R@5: q->doc=100.00%, doc->q=100.00%, mean=100.00%
R@10: q->doc=100.00%, doc->q=100.00%, mean=100.00%
MRR q->doc: 0.8333
```

This demonstrates the evaluation API. For a run on prepared AI2D data, see [Docker with a dataset](diagram_vqa/docs/usage.md#docker-with-a-dataset).

### Local installation

Python **3.10+** is required. Create a virtual environment if you do not already have one:

```sh
python -m venv .venv
```

Activate it with `.venv/Scripts/Activate.ps1` in PowerShell or `source .venv/bin/activate` on Linux/macOS, then install the CPU dependencies and package:

```sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
python -m pip install -r diagram_vqa/requirements-cpu.txt
python -m pip install -e ./diagram_vqa
python diagram_vqa/examples/retrieval_metrics.py
```

Use a CUDA-compatible PyTorch build for GPU experiments. The package currently imports PyTorch even when using its metric helpers. Full graph/VLM requirements, dataset preparation, and CLI options are documented in the [usage guide](diagram_vqa/docs/usage.md).

## How it works

1. **Text nodes:** [OCR](https://github.com/tesseract-ocr/tesseract) extracts text and bounding boxes.
2. **Visual nodes:** [SAM2](https://github.com/facebookresearch/sam2) identifies regions; [DINOv2](https://github.com/facebookresearch/dinov2) provides visual embeddings. Each node also carries its type and geometry.
3. **Graph structure:** k-nearest neighbours (kNN) provide proximity-based connections; learned attention selects or weights relationships depending on the architecture.
4. **Retrieval and answers:** models rank documents against a question and evaluate answer selection separately. Dual-Branch combines a base branch with an additional branch of local OCR/SAM evidence.

Baselines include [CLIP](https://github.com/openai/CLIP) and a vision-language model adapted with [QLoRA](https://arxiv.org/abs/2305.14314).

## Results

Results follow the tables in the [research presentation](https://docs.google.com/presentation/d/1a3m6Goh3P5MZFHDC1UjNX7AyD_vt7G5VJsvcX-Syvyc/edit). [Detailed results and provenance](diagram_vqa/docs/results.md) preserve the source snapshot, question-type analysis, resource measurements, and evaluation limitations.

### AI2D answer accuracy

Accuracy is the percentage of correct answers. Current local models use **3,088 test questions**, seeds **42–44**, and mean ± population standard deviation (`ddof=0`). Qwen's archived score has not been revalidated on the current protocol.

| Model | Accuracy, % | Runs |
|---|---:|---:|
| Qwen2.5-VL-3B + QLoRA | 73.19 (archived) | Not revalidated |
| Dual-Branch + top-k Attention | 48.54 ± 0.44 | 3 |
| Heterogeneous Graph + Attention | 47.83 ± 0.44 | 3 |
| Hybrid GATv2 + kNN | 46.24 ± 0.46 | 3 |
| CLIP | 44.52 ± 0.86 | 3 |
| Random | 24.35 ± 0.21 | 3 |

Dual-Branch exceeds CLIP by **4.02 percentage points** using the displayed means.

### Document retrieval

`R@K = (question → document R@K + document → question R@K) / 2`.

Candidates are all unique documents in the dataset's evaluation split. In the reverse direction, any question associated with the document is relevant. A query counts as a hit when at least one relevant candidate occurs in the top K.

| Dataset | Model | R@1, % | R@5, % | R@10, % | n |
|---|---|---:|---:|---:|---:|
| AI2D | CLIP | 4.72 ± 0.29 | 18.46 ± 0.55 | 29.20 ± 1.11 | 3 |
| AI2D | Hybrid GATv2 + kNN | 6.64 ± 0.63 | 20.78 ± 0.45 | 31.82 ± 0.81 | 3 |
| AI2D | Heterogeneous Graph + Attention | 6.30 ± 0.42 | 20.85 ± 0.79 | 31.15 ± 1.35 | 3 |
| AI2D | Dual-Branch + top-k Attention | 6.13 ± 0.42 | 20.28 ± 0.58 | 30.90 ± 0.39 | 3 |
| DocVQA | CLIP | 5.09 ± 0.22 | 14.82 ± 0.81 | 20.98 ± 0.73 | 3 |
| DocVQA | Hybrid GATv2 + kNN | 17.76 | 36.77 | 45.47 | 1 |
| DocVQA | Heterogeneous Graph + Attention | 7.62 ± 0.36 | 20.20 ± 0.35 | 29.52 ± 0.42 | 3 |
| DocVQA | Dual-Branch + top-k Attention | 7.54 ± 0.43 | 20.79 ± 0.48 | 29.50 ± 0.51 | 3 |
| InfographicVQA | CLIP | 11.35 ± 0.08 | 25.76 ± 0.74 | 34.85 ± 0.87 | 3 |
| InfographicVQA | Hybrid GATv2 + kNN | 26.94 | 47.75 | 57.52 | 1 |
| InfographicVQA | Heterogeneous Graph + Attention | 12.92 ± 1.91 | 31.27 ± 2.32 | 41.98 ± 2.85 | 2 |
| InfographicVQA | Dual-Branch + top-k Attention | 14.32 ± 2.16 | 32.14 ± 3.10 | 43.03 ± 3.61 | 3 |

Values are percentages, reported as mean ± population standard deviation. `n` is the number of completed runs from seeds 42–44. Spread is not estimated for `n=1`; missing runs are excluded. Attention does not consistently outperform kNN across metrics, and these averages alone do not establish statistical significance.

The InfographicVQA attention result is **12.92 ± 1.91%, n=2**, matching the presentation table and local audit. The presentation's conclusion still quotes an earlier 11.01%; this discrepancy is documented in the results notes.

## Python API

```python
from vqa_retrieval.metrics import recall_at_k_multi_positive

scores = [[0.9, 0.1], [0.4, 0.6], [0.2, 0.8]]
positives = [{0}, {0}, {1}]
print(recall_at_k_multi_positive(scores, positives, ks=(1, 5)))
# {1: 0.6666666666666666, 5: 1.0}
```

`sim` is a rectangular question-by-candidate similarity matrix; `positive_indices` contains the zero-based relevant candidate indices for each query. `ks` defaults to `(1, 5, 10)`. The result is a `dict[int, float]` with values in [0, 1]. Invalid shapes, missing positives, out-of-range indices, and non-positive K raise `ValueError`.

The [API reference](diagram_vqa/docs/usage.md#python-metrics-api) also covers bidirectional retrieval, mean reciprocal rank (MRR), and answer scoring.

## Project layout

| Path | Purpose |
|---|---|
| [diagram_vqa/src/vqa_retrieval](diagram_vqa/src/vqa_retrieval) | Python package: data, graphs, metrics, VLM |
| [diagram_vqa/scripts](diagram_vqa/scripts) | Preparation, training, evaluation, audits |
| [diagram_vqa/notebooks/experiments](diagram_vqa/notebooks/experiments) | Experiments for all three datasets |
| [diagram_vqa/experiments](diagram_vqa/experiments) | Protocols and configuration |
| [diagram_vqa/reports](diagram_vqa/reports) | Reports and evaluation evidence |
| [diagram_vqa/model_registry](diagram_vqa/model_registry) / [experiment_registry](diagram_vqa/experiment_registry) | Model and experiment catalogues |
| `diagram_vqa/runs/` | Checkpoints, predictions, training histories |
| `ai2d/`, `docvqa/`, `infographicvqa/`, `models/`, `model_matrix_cache/` | External datasets, weights, and caches |

Start with the AI2D notebooks for [Hybrid GATv2](diagram_vqa/notebooks/experiments/ai2d/ai2d_hybrid_v4_multipos_training.ipynb), [Heterogeneous Graph](diagram_vqa/notebooks/experiments/ai2d/ai2d_sam2_dinov2_heterogeneous_balanced_evidence_graph.ipynb), and [Dual-Branch](diagram_vqa/notebooks/experiments/ai2d/ai2d_sam2_dinov2_dual_branch_evidence_graph.ipynb). Corresponding notebooks are available for [DocVQA](diagram_vqa/notebooks/experiments/docvqa) and [InfographicVQA](diagram_vqa/notebooks/experiments/infographicvqa).

The separate [model_matrix_v1 protocol](diagram_vqa/experiments/model_matrix_protocol.json) covers 11 models × 3 datasets with seed 42. Its `available_smoke` results use small samples and must not be mixed with the presentation's multi-run evaluation. Check each run's configuration, metrics, and predictions before comparing results.

## Authors

**Andrey Andreevich Utlyakov** — research author.
**Anastasia Alexandrovna Laushkina** — scientific supervisor.
**Valeria Dmitrievna Volokha** — scientific consultant.

## License

The repository does not currently include a `LICENSE` file or declare a license in `pyproject.toml`. Contact the author for code usage terms. Datasets and third-party models retain their respective licenses.
