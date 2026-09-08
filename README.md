# Graph Models for Diagram Representation in Retrieval and Ranking Tasks

[![Python: 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](diagram_vqa/pyproject.toml)
[![Docker: CPU example](https://img.shields.io/badge/Docker-CPU_example-2496ED?logo=docker&logoColor=white)](#docker)
[![Google Slides: presentation](https://img.shields.io/badge/Google_Slides-Presentation-FBBC04?logo=googleslides&logoColor=white)](https://docs.google.com/presentation/d/1RnsHrDhAxOaTre7VCdBh67QbfDaOExJKicDK9aOw374/edit)

Graph-based retrieval and visual question answering for diagrams, documents, and infographics on **AI2D, DocVQA, and InfographicVQA**.

The project studies how explicit text, object, and spatial relationships affect document retrieval, ranking, and answer selection. It compares CLIP, GATv2 + kNN, «Граф с отбором связей по типам», and «Граф с общим отбором связей». Additional experiments cover SigLIP, OCR-text, GraphColBERT/FiLM, and Qwen2.5-VL with QLoRA.

Model display names follow the [current presentation](https://docs.google.com/presentation/d/1RnsHrDhAxOaTre7VCdBh67QbfDaOExJKicDK9aOw374/edit) verbatim. The [architecture table](#train-graph-models) maps each name to the existing command-line identifier and notebook.

The main interfaces are Python scripts and research notebooks. Datasets, model weights, and feature caches are supplied separately. All 36 principal retrieval runs have completed (seeds 42–44). The detailed [reproducibility instructions below](#reproducibility) cover official downloads, exact folder layouts, environment setup, preparation, new training and saved-result checks.

**Contents:** [Reproducibility](#reproducibility) · [Folders](#clone-the-repository-and-choose-folders) · [Environment](#install-the-environment) · [Datasets](#download-and-prepare-data) · [Model weights](#download-model-weights) · [Baselines](#run-baselines) · [Graph training](#train-graph-models) · [Checkpoint evaluation](#evaluate-graph-checkpoints) · [Result verification](#verify-saved-results) · [Troubleshooting](#resume-archive-and-troubleshoot) · [Method](#how-it-works) · [Results](#results)

## Reproducibility

This section is the complete starting point for a reader with **only a Git clone**. Download the original data, prepare them locally, and run the commands below. No folder from the author's computer is assumed to exist.

| What you want to do | Required inputs | Start here |
|---|---|---|
| Check the numbers printed in this README | Git clone and Python; no datasets or weights | [Verify saved results](#verify-saved-results) |
| Run the evaluation API example | Python CPU environment or Docker | [Install the environment](#install-the-environment) |
| Run a full Random or OCR baseline | Original dataset and prepared manifest; OCR for the OCR baseline | [Download and prepare data](#download-and-prepare-data), then [run baselines](#run-baselines) |
| Train a new graph model | Prepared data, research environment, pretrained encoders; SAM2/DINOv2 for attention graphs | [Download model weights](#download-model-weights), then [train graph models](#train-graph-models) |
| Reevaluate an author's saved attention checkpoint | That checkpoint, its corresponding manifest, feature cache and text encoder | [Evaluate graph checkpoints](#evaluate-graph-checkpoints) |

The repository contains code, notebooks and compact recorded measurements. **Original datasets, pretrained models, trained checkpoints and feature caches are not included in Git.** A public download of the author's complete historical checkpoint/cache archive has not been published. The instructions below support new runs from official inputs; they do not claim that missing historical hyperparameters can be recovered from a table of scores.

### Clone the repository and choose folders

Install [Git](https://git-scm.com/downloads) and [Python 3.11](https://www.python.org/downloads/) first. The checked reference uses Python 3.11.9. Clone into a directory with sufficient free disk space:

```sh
git clone https://github.com/jettooss/research-work.git
cd research-work
git rev-parse HEAD
```

Keep the printed Git revision with your experiment records. All subsequent commands run from this `research-work/` directory, which contains `README.md` and `diagram_vqa/`. Keep this repository root as the working directory throughout the recipe.

The recommended layout puts external data alongside `diagram_vqa/`, in the repository root. These dataset/model directories are already excluded by `.gitignore`. The environment section below creates the empty download/model directories after activating Python.

Set an **absolute** data root once in your terminal. PowerShell:

```powershell
$DATA_ROOT = (Get-Location).Path
```

Bash on Linux:

```bash
DATA_ROOT="$(pwd)"
```

The later commands use `"$DATA_ROOT/..."`, which works in both shells. If the datasets are on another disk, set this variable to that absolute directory instead, for example `D:/research-data` on Windows or `/mnt/research-data` on Linux, and create the same layout there. Keep running code from the Git clone. This avoids moving large data when updating the code.

```text
research-work/                         # Git clone; current working directory
├── README.md
├── .venv-repro/                       # your Python environment
├── diagram_vqa/
│   ├── src/vqa_retrieval/
│   ├── scripts/
│   ├── notebooks/experiments/
│   ├── reports/reproducibility/       # versioned measurements and provenance
│   └── runs/                         # your new experiment outputs
├── downloads/                        # downloaded original archives
├── ai2d/                             # original AI2D plus generated manifests/OCR
├── docvqa/                           # official train/, val/, test/ directories
├── infographicvqa/                   # original InfographicVQA plus manifest
├── models/                           # pretrained weights and optional HF cache
└── model_matrix_cache/               # generated OCR/SAM/feature caches
```

When `$DATA_ROOT` is external, the last six entries belong under that external root. A matrix `--data-root` or notebook `--external-root` points to the **parent of `ai2d/`, `docvqa/` and `infographicvqa/`**, not to an individual dataset directory. Preparation commands instead take the individual dataset directory, as shown below.

### Install the environment

Create and activate the environment in the Git clone. On Windows/PowerShell:

```powershell
py -3.11 -m venv .venv-repro
.\.venv-repro\Scripts\Activate.ps1
python --version
```

On Linux/Bash:

```bash
python3.11 -m venv .venv-repro
source .venv-repro/bin/activate
python --version
```

If PowerShell activation is blocked, use `.\.venv-repro\Scripts\python.exe` instead of `python` in subsequent commands; changing the machine's execution policy is unnecessary. On Linux the equivalent executable is `.venv-repro/bin/python`. Use the selected environment for installation, preparation, training and Jupyter.

Now create the directories under the selected data root. This command works in either activated shell, including when `$DATA_ROOT` is on another disk:

```sh
python -c "import sys; from pathlib import Path; root=Path(sys.argv[1]); [(root/p).mkdir(parents=True, exist_ok=True) for p in ('downloads', 'docvqa', 'infographicvqa', 'models/sam2')]" "$DATA_ROOT"
```

**For Random, metric helpers and the example**, install the minimal CPU environment:

```sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
python -m pip install -r diagram_vqa/requirements-cpu.txt
python -m pip install -e ./diagram_vqa
python -m pip check
python diagram_vqa/examples/retrieval_metrics.py
```

Expected example output; this uses three synthetic questions and two documents:

```text
R@1: q->doc=66.67%, doc->q=100.00%, mean=83.33%
R@5: q->doc=100.00%, doc->q=100.00%, mean=100.00%
R@10: q->doc=100.00%, doc->q=100.00%, mean=100.00%
MRR q->doc: 0.8333
```

**For OCR preparation, CLIP, graph notebooks and saved-graph evaluation**, install the research environment before starting runs:

```sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 torchvision==0.21.0
python -m pip install -r diagram_vqa/requirements-research.txt
python -m pip install -e ./diagram_vqa
python -m pip check
```

Git is needed for the exact OpenAI CLIP source revision in the requirements. `pip install clip` is not a substitute. The [Windows/Python 3.11 CPU lock](diagram_vqa/requirements-lock-win-py311-cpu.txt) is an alternative complete installation snapshot: install it in an empty environment, then install the local package with `pip install -e ./diagram_vqa`.

**For NVIDIA GPU experiments**, create a separate environment and replace the CPU wheel command with:

```sh
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
python -m pip install -r diagram_vqa/requirements-research.txt
python -m pip install -e ./diagram_vqa
python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

These CPU/CUDA wheel pairs follow the [PyTorch 2.6 installation instructions](https://pytorch.org/get-started/previous-versions/#v260). CUDA experiments need a compatible NVIDIA driver; the last command must report `True` before expecting GPU execution. CPU installation and limited training/inference were tested; a new complete CUDA training matrix was not executed. Optional QLoRA dependencies are installed with `python -m pip install -r diagram_vqa/requirements-vlm.txt`.

For OCR, install the **Tesseract executable and English language data** separately. Linux systems using apt can install `tesseract-ocr` and `tesseract-ocr-eng`. For Windows, follow the Windows installer link in the [Tesseract installation documentation](https://tesseract-ocr.github.io/tessdoc/Installation.html), enable English, and add its install directory to PATH. Then open a new terminal, reactivate the environment and reset `$DATA_ROOT`:

```sh
tesseract --version
tesseract --list-langs
```

The language list must include `eng`. Installing `pytesseract` only installs the Python wrapper. AI2D preparation also accepts, for example, `--tesseract-cmd "C:/Program Files/Tesseract-OCR/tesseract.exe"` if PATH is unavailable.

Plan disk space for both archives and extracted data, then OCR/features/checkpoints. DocVQA's three archives total approximately **8.91 GB before extraction**; AI2D is approximately **0.99 GB**. InfographicVQA images and model/cache storage are additional. The matrix runner requires at least **20 GiB free on its output drive**; that check is not an estimate of the full storage requirement. Feature extraction is a substantial first-run computation. The resource table's cached-feature latency does not describe dataset preparation time.

### Download and prepare data

Use the original datasets, not a similarly named converted Hugging Face subset. Each preparation script needs the original images and question annotations; DocVQA also needs the supplied OCR. You can start with AI2D only and add the other datasets later.

| Dataset | Official source | What to download |
|---|---|---|
| AI2D | [AllenAI dataset page](https://prior.allenai.org/projects/diagram-understanding) | [Complete AI2D ZIP](https://ai2-website.s3.amazonaws.com/data/ai2d-all.zip) and [official test IDs](https://s3-us-east-2.amazonaws.com/prior-datasets/ai2d_test_ids.csv) |
| DocVQA | [Dataset page](https://www.docvqa.org/datasets/docvqa), [RRC downloads](https://rrc.cvc.uab.cat/?ch=17&com=downloads) | **Single Page Document VQA / Task 1**: [train](https://datasets.cvc.uab.es/rrc/DocVQA/train.tar.gz), [validation](https://datasets.cvc.uab.es/rrc/DocVQA/val.tar.gz), [test](https://datasets.cvc.uab.es/rrc/DocVQA/test.tar.gz), including images and OCR |
| InfographicVQA | [Dataset page](https://site.docvqa.org/datasets/infographicvqa), [RRC downloads](https://rrc.cvc.uab.cat/?ch=17&com=downloads) | **Infographics VQA / Task 3**: images and question JSONs for train, validation and test |

The relevant RRC challenge is **`ch=17`**; newer DocVQA releases and other tasks have different formats. The RRC site asks users to register/sign in to access its download listings. InfographicVQA archive names behind that login were not independently verified; use the task's current official links and arrange their contents as specified below. If a direct DocVQA link changes, use the official task download page.

**Hosting:** the original data are fetched from their publishers. AI2D's `license.txt`, included in its ZIP, explicitly withholds permission to redistribute the data to third parties. Consequently this project does not publish a Yandex Disk/GitHub mirror of the raw datasets. A project archive can contain the author's code/configuration and permitted artifacts, while the instructions here provide the original data links. Preserve the terms supplied with each dataset.

#### AI2D

Save the ZIP as `$DATA_ROOT/downloads/ai2d-all.zip`. It already contains an `ai2d/` directory, so extract it into **`$DATA_ROOT`**, not into `$DATA_ROOT/ai2d`. For example:

```sh
python -m zipfile -e "$DATA_ROOT/downloads/ai2d-all.zip" "$DATA_ROOT"
```

Save the separately downloaded test-ID CSV as **`$DATA_ROOT/ai2d/ai2d_test_ids.csv`**. The filename is chosen for this recipe; if a browser appends `(1)`, rename the file or supply its exact name to `--official-test-ids`.

The ZIP also creates an unused `__MACOSX/` metadata directory; it is ignored by Git and is not an input to the project.

The expected raw layout is:

```text
ai2d/
├── images/                           # 4,903 PNGs in the checked official ZIP
├── questions/                        # 4,563 files such as 2618.png.json
├── annotations/
├── categories.json
├── README.txt
├── license.txt
└── ai2d_test_ids.csv                  # downloaded separately; 982 IDs
```

First run metadata validation; this command performs no OCR or output writes:

```sh
python diagram_vqa/scripts/prepare_ai2d_raw.py --data-root "$DATA_ROOT/ai2d" --official-test-ids "$DATA_ROOT/ai2d/ai2d_test_ids.csv" --path-root "$DATA_ROOT" --check-inputs
```

**Known upstream inconsistency:** the official ZIP checked on 2026-09-08 omits `images/4325.png` and its question JSON, although the official test CSV includes ID `4325`. Therefore strict validation of that ZIP reports this missing image. Downloading the same ZIP again will not supply it. The project explicitly supports the complete available **question dataset**, while recording that the full image archive is incomplete. After confirming that the error concerns only this document-only image, run:

```sh
python diagram_vqa/scripts/prepare_ai2d_raw.py --data-root "$DATA_ROOT/ai2d" --official-test-ids "$DATA_ROOT/ai2d/ai2d_test_ids.csv" --path-root "$DATA_ROOT" --allow-missing-document-only-test-images --check-inputs
python diagram_vqa/scripts/prepare_ai2d_raw.py --data-root "$DATA_ROOT/ai2d" --official-test-ids "$DATA_ROOT/ai2d/ai2d_test_ids.csv" --path-root "$DATA_ROOT" --allow-missing-document-only-test-images --seed 42 --val-ratio 0.1
```

The second command runs Tesseract and writes both preparation stages. It may take considerable time; `OCR .../... images` reports progress. It does not need author's OCR caches, captions or checkpoints. An absent image referenced by a question is still an error; the exception does not skip arbitrary missing data.

```text
ai2d/
├── prepared_v2/
│   ├── manifest_hybrid.jsonl
│   ├── split_hybrid.json
│   ├── preparation_plan.json
│   ├── provenance.json
│   └── ocr_v2/*.ocr.json
└── model_matrix_v1/
    ├── manifest.jsonl                 # consumed by all matrix/graph recipes
    ├── split.json
    └── audit.json
```

Check `audit.json`: **11,145 train + 1,268 validation + 3,088 test questions**, **814 test documents**, `question_dataset_ready=true`; for the checked upstream archive, `full_archive_ready=false`. The seed42/10% image-level validation split is fixed during preparation, independently of the later model training seed.

To resume interrupted OCR, repeat the same preparation command with `--resume`. Input hashes, parameters, software identity and existing outputs must still match. Existing unrelated/nonempty output directories are protected. For a new data version use a fresh data root, or explicitly choose new `--prepared-dir` and `--output-dir`; the runtime expects the standard paths above, so point it to a root containing that standard layout.

Preparation preserves original empty answer-option slots. This corrects a historical TRAIN label error for `2618.png:2618.png-3`. New training on corrected annotations is a new experiment; the historical scores below are not relabeled as scores on corrected training data.

#### DocVQA

Download **Task 1 / Single Page** archives from the links above into `$DATA_ROOT/downloads/`, retaining `train.tar.gz`, `val.tar.gz`, `test.tar.gz`. The original task bundles include question JSONs, images and Azure OCR JSONs. These checked downloads have a `.tar.gz` suffix but contain plain TAR, so use automatic format detection with **`tar -xf`**, without forcing `-z`:

```sh
tar -xf "$DATA_ROOT/downloads/train.tar.gz" -C "$DATA_ROOT/docvqa"
tar -xf "$DATA_ROOT/downloads/val.tar.gz" -C "$DATA_ROOT/docvqa"
tar -xf "$DATA_ROOT/downloads/test.tar.gz" -C "$DATA_ROOT/docvqa"
```

They contain the split directories. The resulting paths must be:

```text
docvqa/
├── train/
│   ├── train_v1.0.json
│   ├── documents/*.png
│   └── ocr_results/*.json
├── val/
│   ├── val_v1.0.json
│   ├── documents/*.png
│   └── ocr_results/*.json
└── test/
    ├── test_v1.0.json
    ├── documents/*.png
    └── ocr_results/*.json
```

Avoid an extra level such as `docvqa/train/train/`. Image names must match each JSON row's `image` field. Keep `ocr_results/`; empty OCR directories silently remove text evidence from later models.

```sh
python diagram_vqa/scripts/prepare_docvqa_manifest.py --data-root "$DATA_ROOT/docvqa" --output-dir "$DATA_ROOT/docvqa/prepared_v1" --splits train val test
```

This writes `docvqa/prepared_v1/manifest.jsonl` and `split.json`. It references supplied Azure OCR; it neither downloads images nor reruns Tesseract. Inspect `split.json`: the question counts should be **39,463 / 5,349 / 5,188**, `missing_images=0` and `missing_ocr=0` for every split, and all `image_leakage` lists empty. A successful exit code alone does not guarantee that the image/OCR files were found. The preparation script overwrites its target manifest/split, so use a fresh output/data directory when preparing a different version.

#### InfographicVQA

On the RRC download page, sign in and choose **Infographics VQA / Task 3**. Download question annotations and all three image splits. Extract the images into one `infographicsvqa_images/` directory and the question JSONs into `infographicsvqa_qas/`, under the exact parent layout:

```text
infographicvqa/
└── InfographicVQA/
    ├── infographicsvqa_images/
    │   └── <image_local_name from question JSON>
    └── infographicsvqa_qas/
        ├── infographicsVQA_train_v1.0.json
        ├── infographicsVQA_val_v1.0_withQT.json
        └── infographicsVQA_test_v1.0.json
```

Capitalization matters on Linux. If the training JSON is named `infographicVQA_train_v1.0.json` in the downloaded release, rename it to **`infographicsVQA_train_v1.0.json`**: the script expects the extra `s`. Validation may instead use `infographicsVQA_val_v1.0.json`; when both files are present, the script prefers `_withQT`. Preserve each image's exact filename/extension from `image_local_name`, and do not keep separate nested `train/val/test` levels inside `infographicsvqa_images/`.

```sh
python diagram_vqa/scripts/prepare_infographicvqa_manifest.py --data-root "$DATA_ROOT/infographicvqa" --output-dir "$DATA_ROOT/infographicvqa/prepared_v1" --splits train val test
```

Use the absolute `$DATA_ROOT` established earlier. The output is `infographicvqa/prepared_v1/manifest.jsonl` plus `split.json`. Check for **23,946 / 2,801 / 3,288 questions**, zero `missing_images` and empty `image_leakage` lists. The script records missing files without failing and can overwrite an existing output; inspect the report before training. This step does not run OCR. The later matrix CLI and standalone attention notebooks run Tesseract on first use and store OCR in `$DATA_ROOT/model_matrix_cache/ocr/infographicvqa/`. Configured GATv2 + kNN and FiLM notebook copies instead use their isolated `feature_cache_dir/matrix/ocr/infographicvqa/`.

#### Understand the evaluation splits

| Dataset | Train questions | Validation questions | Test questions | Unique test documents | Public test answers |
|---|---:|---:|---:|---:|---|
| AI2D | 11,145 | 1,268 | 3,088 | 814 | Available |
| DocVQA | 39,463 | 5,349 | 5,188 | 1,287 | Hidden |
| InfographicVQA | 23,946 | 2,801 | 3,288 | 579 | Hidden |

The preparation scripts retain the official DocVQA/InfographicVQA split names. They do **not** turn validation into a local test set. The test question-to-image mapping is available, allowing the project's retrieval evaluation. Answer-quality ANLS/Accuracy on those official test sets requires evaluation by the official server. A locally serialized score of zero against empty references must not be interpreted as measured model quality. Use labeled validation for local answer-quality checks; this repository does not automatically submit predictions to the server.

### Download model weights

The Random baseline needs no weights. OCR + TF-IDF needs OCR but no neural encoder. Install the research environment before the remaining commands.

| Experiment | Required pretrained inputs | Location / acquisition |
|---|---|---|
| GATv2 + kNN and GraphColBERT/FiLM notebooks | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Automatically downloaded through SentenceTransformers |
| Standalone SAM/DINO attention graphs | MiniLM, [DINOv2-base](https://huggingface.co/facebook/dinov2-base), [Ultralytics SAM2.1 base](https://docs.ultralytics.com/models/sam-2/) | MiniLM/DINO via Hugging Face; SAM file at `$DATA_ROOT/models/sam2/sam2.1_b.pt` |
| CLIP baseline | [OpenAI CLIP ViT-B/32](https://github.com/openai/CLIP) | `clip.load('ViT-B/32')` downloads weights to the user's `~/.cache/clip/` |
| Optional Qwen QLoRA | [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | Entire model directory at `$DATA_ROOT/models/Qwen2.5-VL-3B-Instruct/` |

To keep Hugging Face downloads under the data root, set `HF_HOME` **before starting Python**. PowerShell:

```powershell
$env:HF_HOME = "$DATA_ROOT/models/hf-cache"
```

Bash:

```bash
export HF_HOME="$DATA_ROOT/models/hf-cache"
```

This setting follows the [Hugging Face cache configuration](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables); it does not change CLIP's separate cache. Optional commands to download encoders in advance:

```sh
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')"
python -c "from transformers import AutoImageProcessor, AutoModel; AutoImageProcessor.from_pretrained('facebook/dinov2-base'); AutoModel.from_pretrained('facebook/dinov2-base')"
```

For SAM graphs, download [sam2.1_b.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_b.pt) and place that exact file in `$DATA_ROOT/models/sam2/`. It is the Ultralytics-format checkpoint used by this project. A differently formatted checkpoint from another SAM repository is not interchangeable by renaming it.

For the optional CLIP and Qwen experiments:

```sh
python -c "import clip; clip.load('ViT-B/32', device='cpu')"
python -c "import sys; from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct', local_dir=sys.argv[1])" "$DATA_ROOT/models/Qwen2.5-VL-3B-Instruct"
```

The [snapshot download](https://huggingface.co/docs/huggingface_hub/guides/download) command saves configuration, tokenizer/processor and weight shards together. Copying only a single weight shard is insufficient. Record model repository revision IDs and file checksums with new experiments: current notebook model IDs do not pin immutable HF revisions.

### Run baselines

First check the full input/output path chain with a 16-question AI2D Random run:

```sh
python diagram_vqa/scripts/run_model_matrix.py --dataset ai2d --model random --split test --seed 42 --max-samples 16 --data-root "$DATA_ROOT" --output-dir diagram_vqa/runs/readme_random_smoke --no-resume
```

Expected output directory:

```text
diagram_vqa/runs/readme_random_smoke/ai2d/random/seed42/test/
├── config.json
├── predictions.jsonl
└── metrics.json                       # status=available_smoke, num_samples=16
```

For the full dataset, remove the sample limit and choose a **different output root**:

```sh
python diagram_vqa/scripts/run_model_matrix.py --dataset ai2d --model random --split test --seed 42 --data-root "$DATA_ROOT" --output-dir diagram_vqa/runs/readme_random_full --no-resume
```

The full AI2D metrics report `available_full`, `num_samples=3088` and `num_documents=814`. Repeat with `--seed 43` and `--seed 44` in the same full output root; their seed subdirectories are separate. On the historical manifest the three accuracies averaged **24.35 ± 0.21%**. A new data/preprocessing version is a new experiment, not an instruction to force this number.

The full OCR baselines run independently for all three datasets:

```sh
python diagram_vqa/scripts/run_model_matrix.py --dataset ai2d --model ocr_text --split test --seed 42 --data-root "$DATA_ROOT" --output-dir diagram_vqa/runs/readme_ocr_full --no-resume
python diagram_vqa/scripts/run_model_matrix.py --dataset docvqa --model ocr_text --split test --seed 42 --data-root "$DATA_ROOT" --output-dir diagram_vqa/runs/readme_ocr_full --no-resume
python diagram_vqa/scripts/run_model_matrix.py --dataset infographicvqa --model ocr_text --split test --seed 42 --data-root "$DATA_ROOT" --output-dir diagram_vqa/runs/readme_ocr_full --no-resume
```

OCR TF-IDF is deterministic; repeated training seeds are unnecessary. On InfographicVQA the first run extracts OCR unless its cache already exists. Outputs follow `OUTPUT/<dataset>/ocr_text/seed42/test/`. Read `metrics.json` for `retrieval.mean_recall_at_k`; for AI2D, `vqa.score` is answer accuracy. For the other datasets heed the hidden-answer restriction above.

To train the CLIP answer/retrieval heads with an explicit budget:

```sh
python diagram_vqa/scripts/train_model_matrix_vision.py --dataset ai2d --model clip --split val --seed 42 --epochs 20 --early-stopping-patience 4 --batch-size 64 --data-root "$DATA_ROOT" --output-dir diagram_vqa/runs/readme_clip_full --no-resume
```

Training uses train, selects the best checkpoint on validation, then evaluates test automatically. The frozen encoder is pretrained; only the task heads are trained. The run saves `val/config.json`, `val/history.json`, `val/checkpoint_best.pt`, `val/checkpoint_last.pt`, `val/metrics.json`, and final `test/metrics.json`/`predictions.jsonl` under `OUTPUT/ai2d/clip/seed42/`. A later evaluation-only invocation is:

```sh
python diagram_vqa/scripts/run_model_matrix.py --dataset ai2d --model clip --split test --seed 42 --data-root "$DATA_ROOT" --output-dir diagram_vqa/runs/readme_clip_full --resume
```

Current trainable **matrix** CLIs restrict training to seed42; do not run a seed43/44 CLIP loop through them. The presentation's historical three-seed CLIP results are retained as evidence. The separate graph notebook executor below supports seeds42–44. Matrix `--split test` cannot start training and will reject missing/unverifiable legacy checkpoints. The dispatcher does not accept `--epochs`; use the backend command above to set the budget.

Optional Qwen + QLoRA training uses the local model folder downloaded earlier and the VLM requirements. Run this only in the GPU environment with sufficient memory:

```sh
python -m pip install -r diagram_vqa/requirements-vlm.txt
python diagram_vqa/scripts/run_model_matrix.py --dataset ai2d --model qwen25_vl_qlora --split val --seed 42 --data-root "$DATA_ROOT" --output-dir diagram_vqa/runs/readme_qwen_full --no-resume
```

The adapter and training state are under `OUTPUT/ai2d/qwen25_vl_qlora/seed42/training/`, with separate evaluation outputs. This is an optional new experiment; the archived Qwen score in the results table has not been revalidated by this recipe.

### Train graph models

For the main presentation models, use the actual experiment notebooks through the executor. Generic matrix `graph_transformer` is a different architecture from the presentation's «Граф с общим отбором связей» / «Граф с отбором связей по типам» models.

Register the activated research environment as a Jupyter kernel:

```sh
python -m ipykernel install --sys-prefix --name python3 --display-name "Diagram VQA reproduction"
```

Plan one architecture on AI2D first:

```sh
python diagram_vqa/scripts/execute_experiment_matrix_notebooks.py --dataset ai2d --architecture sam2_dinov2_dual_branch_evidence_graph --seeds 42 43 44 --external-root "$DATA_ROOT" --output-root diagram_vqa/runs/readme_graphs --kernel-name python3 --dry-run
```

This prints three configurations and does not train or create output directories. Verify the absolute manifest path, seed, epochs and feature-cache location. Execute the same plan by removing `--dry-run`:

```sh
python diagram_vqa/scripts/execute_experiment_matrix_notebooks.py --dataset ai2d --architecture sam2_dinov2_dual_branch_evidence_graph --seeds 42 43 44 --external-root "$DATA_ROOT" --output-root diagram_vqa/runs/readme_graphs --kernel-name python3
```

To run the other two main graph models, substitute:

| Presentation name | Exact `--architecture` value | Current epochs: AI2D / DocVQA / InfographicVQA |
|---|---|---|
| GATv2 + kNN | `hybrid_v4_multipos_training` | 40 / 40 / 20 |
| Граф с отбором связей по типам | `sam2_dinov2_heterogeneous_balanced_evidence_graph` | 20 / 20 / 20 |
| Граф с общим отбором связей | `sam2_dinov2_dual_branch_evidence_graph` | 20 / 20 / 20 |

Change `--dataset` to `docvqa` or `infographicvqa` for those prepared datasets. Omitting only `--dataset` runs the selected architecture on all three datasets. Omitting both selectors schedules **54 runs** (six architectures × three datasets × three seeds); it is not a smoke check. Use `--epochs N` only when deliberately defining a new budget. The listed current notebook defaults are not a reconstruction of undocumented historical settings; for example, historical DocVQA GATv2 + kNN completed at 20 epochs while its current default is 40.

The executor preserves source notebooks and saves configured/executed copies. Each run writes:

```text
diagram_vqa/runs/readme_graphs/<dataset>/<architecture>/seed42_full/
├── notebook_run.json                  # seed, epochs, source/manifest hashes, cache path
├── execution.json                     # running/completed/failed and error if any
├── notebook.executed.ipynb            # parameters, cell outputs and traceback
├── checkpoint_best.pt
└── metrics.json                       # model-specific evaluation schema
```

GATv2 + kNN also writes final `test/metrics.json` and `test/predictions.jsonl`. For attention graphs use the dedicated retrieval evaluator below. In a failed notebook, inspect `execution.json` and the final failing cell before resuming.

No author's node-feature cache is needed for **new training**: the SAM/DINO notebooks precompute features for unique images across train/val/test using the supplied frozen encoders. They save one feature file per image into the isolated location recorded as `feature_cache_dir` in `notebook_run.json`. Missing SAM box caches are generated locally. GATv2 + kNN and FiLM need OCR and MiniLM, not SAM2/DINOv2.

For a fresh run use a fresh data/cache root. Existing SAM boxes affect the feature-extraction path; regenerating OCR/masks/features can change results. Resume identity checks notebook/manifest/settings but does not hash every image, OCR, model or SAM-box file. Keep those external inputs fixed and record their hashes for exact experiment provenance.

### Evaluate graph checkpoints

After training the «Граф с общим отбором связей» model above, evaluate its saved checkpoints on unique test documents. Find the **actual generated** feature directory from seed42; the three seeds of this dataset/architecture share it.

PowerShell:

```powershell
$FEATURE_CACHE = python -c "import json; print(json.load(open('diagram_vqa/runs/readme_graphs/ai2d/sam2_dinov2_dual_branch_evidence_graph/seed42_full/notebook_run.json', encoding='utf-8'))['feature_cache_dir'])"
```

Bash:

```bash
FEATURE_CACHE=$(python -c "import json; print(json.load(open('diagram_vqa/runs/readme_graphs/ai2d/sam2_dinov2_dual_branch_evidence_graph/seed42_full/notebook_run.json', encoding='utf-8'))['feature_cache_dir'])")
```

Then run in either shell:

```sh
python diagram_vqa/reports/presentation_metrics/evaluate_saved_retrieval.py --project-root diagram_vqa --data-root "$DATA_ROOT" --runs-root diagram_vqa/runs/readme_graphs --datasets ai2d --architectures sam2_dinov2_dual_branch_evidence_graph --seeds 42 43 44 --feature-cache "$FEATURE_CACHE" --output-dir diagram_vqa/runs/readme_graph_retrieval --local-files-only --device cpu
python diagram_vqa/reports/presentation_metrics/audit_retrieval_matrix.py --project-root diagram_vqa --data-root "$DATA_ROOT" --runs-root diagram_vqa/runs/readme_graphs --datasets ai2d --architectures sam2_dinov2_dual_branch_evidence_graph --seeds 42 43 44 --retrieval-root diagram_vqa/runs/readme_graph_retrieval --output diagram_vqa/runs/readme_graph_retrieval_audit.json
```

The evaluator writes `OUTPUT/ai2d/<architecture>/seedN_full/retrieval_unique_test.json`; the audit should report `verified_runs=3`, `required_runs=3`, `completed=true`. Results include both directions and their mean R@1/R@5/R@10. For «Граф с отбором связей по типам» or another dataset, change the selections **and read that pair's own `feature_cache_dir`**. New architecture caches are separate, so do not pass one architecture's feature directory to another.

This evaluator supports only **«Граф с общим отбором связей» and «Граф с отбором связей по типам»**. GATv2 + kNN has its own notebook test evaluation. `--local-files-only` requires the MiniLM encoder to be downloaded already; a missing cache produces an error rather than a network download. `--text-encoder` can point to a local encoder directory. `--feature-cache` must contain real generated/restored `{image_id}.pt` files, not an empty directory.

For historical checkpoint evaluation, restore the corresponding original manifest, node features, encoder and `checkpoint_best.pt`/`metrics.json` under `RUNS/<dataset>/<architecture>/seedN_full/` before using these commands. The files must come from the **same run/data version**. The new matrix resume checks intentionally reject old runs lacking verifiable configuration; copying an old checkpoint into a fresh run directory does not make it a valid new training run.

### Verify saved results

To recompute the published table from the compact versioned measurements, without data/model downloads:

```sh
python diagram_vqa/scripts/reproduce_results.py --check
```

Expected summary:

```json
{"accuracy_groups": 5, "retrieval_groups": 15, "saved_full_test_runs": 39, "check": "passed"}
```

This verifies identities, hashes, seeds and aggregation of 36 principal runs plus three deterministic OCR evaluations. Percentages use population standard deviation (`ddof=0`) over seeds42–44. It verifies the marked retrieval tables in this README and `docs/results.md`; it does not perform inference or automatically discover your new run directories.

For newly trained models, inspect the new metrics and scoped audit, retain their configuration/checkpoints/predictions, and establish their data/protocol identity before promoting them into `reports/reproducibility/evidence.json`. Then regenerate marked tables with `python diagram_vqa/scripts/reproduce_results.py --write` and rerun `--check`. Do not use the older `aggregate_model_matrix.py` as a final-test verifier: it reads a different validation-output schema.

For local regression checks:

```sh
python -m unittest discover -s diagram_vqa/tests -v
python -m pip check
```

Recorded validation includes 71 passing tests, three full AI2D Random seeds, a 16-question CLIP train/validation/test smoke run, and full seed42 inference for the AI2D «Граф с общим отбором связей» model. The latter reproduced published Recall exactly; directional MRR differed by at most 0.000007711. See [the machine-readable protocol](diagram_vqa/docs/reproduction_validation.json) and [environment reference](diagram_vqa/docs/environment_reference.json). Full 36-run retraining, CUDA/QLoRA generation and Docker runtime remain unverified; a recipe is not evidence that those runs were executed.

### Resume, archive and troubleshoot

For **matrix backends**, `--resume` requires the same dataset, input manifest, code, package versions, parameters and sample limit. A completed matrix run is checked before reuse; partial matrix training also checks checkpoint/history consistency and restores RNG state. The **notebook executor** checks source/manifest/recorded settings and delegates resume to the embedded notebook implementation; it does not provide the same package-version/RNG restoration guarantees. Use a new output root for changed configurations. Matrix `--no-resume` and a notebook execution without `--resume` refuse nonempty run directories. Never convert a smoke run into a full result by editing `config.json` or `metrics.json`.

To prepare a portable **code/measurement** archive for another machine:

```sh
python diagram_vqa/scripts/export_reproduction_bundle.py --project-root diagram_vqa --data-root "$DATA_ROOT" --output diagram_vqa/runs/reproduction-metadata.zip --dry-run
python diagram_vqa/scripts/export_reproduction_bundle.py --project-root diagram_vqa --data-root "$DATA_ROOT" --output diagram_vqa/runs/reproduction-metadata.zip
python diagram_vqa/scripts/export_reproduction_bundle.py --verify diagram_vqa/runs/reproduction-metadata.zip
```

Its inventory contains SHA256 hashes; source files are not modified and existing archives are not overwritten. The default contains available code, measurement evidence and selected run metadata/predictions. It excludes original images, pretrained models, node-feature caches and optimizer checkpoints. It therefore verifies reported measurements but does not replace the dataset download steps. Distribute only content for which redistribution is permitted; no public project archive URL is currently claimed here.

| Symptom | Check / resolution |
|---|---|
| `No module named ...` | Activate the correct environment and install the appropriate CPU/research/VLM requirements; use `python -m pip` |
| `tesseract is not installed` or missing `eng` | Install the executable and language data; fix PATH or pass AI2D `--tesseract-cmd` |
| AI2D missing `4325` | Confirm the documented upstream document-only exception and use its explicit preparation flag; other missing question images still fail |
| Missing `manifest.jsonl` | Run the preparation script first and point runtime data-root at the parent of dataset directories |
| Missing image/OCR despite successful preparation | Check exact extracted nesting/names and `split.json` counters; DocVQA needs `ocr_results/` |
| Infographic path contains the data root twice | Prepare again using an absolute data-root and a fresh output directory |
| `sam2.1_b.pt` not found | Put the linked Ultralytics checkpoint in `$DATA_ROOT/models/sam2/` |
| Kernel not found | Register the kernel from the same activated environment used for the executor |
| `--split test` needs a validation-selected checkpoint | Train with `--split val` first, or use the documented historical attention evaluator with matching artifacts |
| Resume configuration mismatch | Use the original identical configuration or choose a new output root; do not rewrite saved identity files |
| `Only ... GiB free` | Free space or select an output disk with at least the runner's 20-GiB reserve plus expected outputs |
| CUDA out of memory | Reduce batch/feature-extraction budget in a new experiment; record the changed settings and use a fresh run/cache |
| Different results after rebuilding data | Compare splits, corrected AI2D labels, OCR, model revisions and SAM/feature caches before comparing model scores |
| Official DocVQA/Infographic test ANLS is zero | The test references are hidden; use validation or official server evaluation, not that local placeholder |

### Docker

Docker provides the small **CPU evaluation/Random example**, not the entire graph research environment. With a running Linux-container engine, from the repository root:

```sh
docker build -t diagram-vqa:cpu .
docker run --rm diagram-vqa:cpu
```

The expected output is the synthetic retrieval example shown above. To use prepared AI2D, mount `"$DATA_ROOT/ai2d"` read-only at `/workspace/ai2d` and a writable result directory at `/workspace/diagram_vqa/runs`; see the [platform-specific mount commands](diagram_vqa/docs/usage.md#docker-with-a-dataset). The Docker image deliberately contains no downloaded datasets or model weights. Its build/runtime was not validated on the author's host because the Linux engine was unavailable.


## How it works

1. **Text nodes:** [OCR](https://github.com/tesseract-ocr/tesseract) extracts text and bounding boxes.
2. **Visual nodes:** [SAM2](https://github.com/facebookresearch/sam2) identifies regions; [DINOv2](https://github.com/facebookresearch/dinov2) provides visual embeddings. Each node also carries its type and geometry.
3. **Graph structure:** k-nearest neighbours (kNN) provide proximity-based connections; learned attention selects or weights relationships depending on the architecture.
4. **Retrieval and answers:** models rank documents against a question and evaluate answer selection separately. Граф с общим отбором связей combines a base branch with an additional branch of local OCR/SAM evidence.

Baselines include [CLIP](https://github.com/openai/CLIP) and a vision-language model adapted with [QLoRA](https://arxiv.org/abs/2305.14314).

## Results

Results are generated from versioned full-test evidence. The [research presentation](https://docs.google.com/presentation/d/1RnsHrDhAxOaTre7VCdBh67QbfDaOExJKicDK9aOw374/edit) provides the research narrative; the table below also includes the previously omitted OCR baseline. [Detailed results and provenance](diagram_vqa/docs/results.md) preserve historical snapshots, question-type analysis, resource measurements, and evaluation limitations.

### AI2D answer accuracy

Accuracy is the percentage of correct answers. Current local models use **3,088 test questions**, seeds **42–44**, and mean ± population standard deviation (`ddof=0`). Qwen's archived score has not been revalidated on the current protocol.

| Model | Accuracy, % | Runs |
|---|---:|---:|
| Qwen2.5-VL-3B + QLoRA | 73.19 (archived) | Not revalidated |
| Граф с общим отбором связей | 48.54 ± 0.44 | 3 |
| Граф с отбором связей по типам | 47.83 ± 0.44 | 3 |
| GATv2 + kNN | 46.24 ± 0.46 | 3 |
| CLIP | 44.52 ± 0.86 | 3 |
| Random | 24.35 ± 0.21 | 3 |

Граф с общим отбором связей exceeds CLIP by **4.02 percentage points** using the displayed means.

### Document retrieval

`R@K = (question → document R@K + document → question R@K) / 2`.

Candidates are all unique documents in the dataset's evaluation split. In the reverse direction, any question associated with the document is relevant. A query counts as a hit when at least one relevant candidate occurs in the top K.

<!-- BEGIN GENERATED RETRIEVAL TABLE -->
| Dataset | Model | R@1, % | R@5, % | R@10, % | n |
|---|---|---:|---:|---:|---:|
| AI2D | CLIP | 4.72 ± 0.29 | 18.46 ± 0.55 | 29.20 ± 1.11 | 3 |
| AI2D | GATv2 + kNN | 6.64 ± 0.63 | 20.78 ± 0.45 | 31.82 ± 0.81 | 3 |
| AI2D | Граф с отбором связей по типам | 6.30 ± 0.42 | 20.85 ± 0.79 | 31.15 ± 1.35 | 3 |
| AI2D | Граф с общим отбором связей | 6.13 ± 0.42 | 20.28 ± 0.58 | 30.90 ± 0.39 | 3 |
| AI2D | OCR + TF-IDF | 17.27 | 29.90 | 36.18 | 1 |
| DocVQA | CLIP | 5.09 ± 0.22 | 14.82 ± 0.81 | 20.98 ± 0.73 | 3 |
| DocVQA | GATv2 + kNN | 18.51 ± 0.56 | 36.73 ± 0.30 | 45.80 ± 0.26 | 3 |
| DocVQA | Граф с отбором связей по типам | 7.62 ± 0.36 | 20.20 ± 0.35 | 29.52 ± 0.42 | 3 |
| DocVQA | Граф с общим отбором связей | 7.54 ± 0.43 | 20.79 ± 0.48 | 29.50 ± 0.51 | 3 |
| DocVQA | OCR + TF-IDF | 35.95 | 52.38 | 58.52 | 1 |
| InfographicVQA | CLIP | 11.35 ± 0.08 | 25.76 ± 0.74 | 34.85 ± 0.87 | 3 |
| InfographicVQA | GATv2 + kNN | 24.58 ± 1.73 | 45.92 ± 1.36 | 56.27 ± 0.93 | 3 |
| InfographicVQA | Граф с отбором связей по типам | 13.85 ± 2.04 | 32.70 ± 2.77 | 43.53 ± 3.20 | 3 |
| InfographicVQA | Граф с общим отбором связей | 14.32 ± 2.16 | 32.14 ± 3.10 | 43.03 ± 3.61 | 3 |
| InfographicVQA | OCR + TF-IDF | 55.66 | 73.15 | 78.68 | 1 |
<!-- END GENERATED RETRIEVAL TABLE -->

Values are percentages, reported as mean ± population standard deviation. All graph/CLIP entries use three completed runs, seeds 42–44. OCR + TF-IDF is a deterministic, untrained full-test baseline (one recorded run); its vocabulary is fitted on document OCR and queries are transformed without answer labels. Its saved R@K and MRR were independently recomputed on all three local test sets.

Graph models exceed CLIP on the displayed retrieval means, but **OCR + TF-IDF exceeds every graph on these retrieval metrics**. This comparison concerns document retrieval, not answer accuracy. Attention does not consistently outperform kNN, and means alone do not establish statistical significance. The architectures also differ in features and training budgets, so this table does not isolate the causal contribution of attention.

Recompute and check this table without datasets or model downloads:

```sh
python diagram_vqa/scripts/reproduce_results.py --check
```

The [versioned evidence](diagram_vqa/reports/reproducibility/evidence.json) includes all 36 principal runs plus the three OCR runs. This checks saved measurements and aggregation; fresh inference requires the input data and weights described in the reproduction guide.

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

Start with the AI2D notebooks for [GATv2 + kNN](diagram_vqa/notebooks/experiments/ai2d/ai2d_hybrid_v4_multipos_training.ipynb), [Граф с отбором связей по типам](diagram_vqa/notebooks/experiments/ai2d/ai2d_sam2_dinov2_heterogeneous_balanced_evidence_graph.ipynb), and [Граф с общим отбором связей](diagram_vqa/notebooks/experiments/ai2d/ai2d_sam2_dinov2_dual_branch_evidence_graph.ipynb). Corresponding notebooks are available for [DocVQA](diagram_vqa/notebooks/experiments/docvqa) and [InfographicVQA](diagram_vqa/notebooks/experiments/infographicvqa).

The separate [model_matrix_v1 protocol](diagram_vqa/experiments/model_matrix_protocol.json) covers 11 models × 3 datasets with seed 42. Its `available_smoke` results use small samples and must not be mixed with the presentation's multi-run evaluation. Check each run's configuration, metrics, and predictions before comparing results.

## Authors

**Andrey Andreevich Utlyakov** — research author.
**Anastasia Alexandrovna Laushkina** — scientific supervisor.
**Valeria Dmitrievna Volokha** — scientific consultant.

## License

The repository does not currently include a `LICENSE` file or declare a license in `pyproject.toml`. Contact the author for code usage terms. Datasets and third-party models retain their respective licenses.
