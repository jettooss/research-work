# Графовые модели представления диаграмм для задач поиска и ранжирования

[![Python: 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](diagram_vqa/pyproject.toml)
[![Docker: CPU example](https://img.shields.io/badge/Docker-CPU_example-2496ED?logo=docker&logoColor=white)](#docker)
[![Google Slides: presentation](https://img.shields.io/badge/Google_Slides-Presentation-FBBC04?logo=googleslides&logoColor=white)](https://docs.google.com/presentation/d/1RnsHrDhAxOaTre7VCdBh67QbfDaOExJKicDK9aOw374/edit)

Проект исследует графовый поиск и визуальный ответ на вопросы по диаграммам, документам и инфографике на наборах **AI2D, DocVQA и InfographicVQA**.

Оценивается влияние явных текстовых, объектных и пространственных связей на поиск документов, ранжирование и выбор ответа. Сопоставляются CLIP, GATv2 + kNN, «Граф с отбором связей по типам» и «Граф с общим отбором связей». Дополнительные эксперименты охватывают SigLIP, текст OCR, GraphColBERT/FiLM и Qwen2.5-VL с QLoRA.

Основные интерфейсы проекта — Python-скрипты и исследовательские ноутбуки. Исходные наборы данных, веса моделей, обученные checkpoints и кэши признаков не входят в Git. Завершены 36 основных запусков поиска с seed 42–44.

## Воспроизводимость

Инструкции рассчитаны на чистый клон Git и не предполагают наличия каталогов с компьютера автора.

| Задача | Что требуется |
|---|---|
| Проверить опубликованные числа | Клон Git и Python |
| Запустить пример API метрик | CPU-окружение Python или Docker |
| Запустить Random или OCR baseline | Исходные данные и подготовленный манифест |
| Обучить графовую модель | Подготовленные данные, исследовательское окружение и энкодеры |
| Оценить сохранённый checkpoint | Checkpoint, манифест, кэш признаков и текстовый энкодер |

Публичный архив с полным историческим набором checkpoints и кэшей не опубликован. Описанный протокол поддерживает новые запуски на официальных входных данных, но не восстанавливает отсутствующие исторические гиперпараметры по одной таблице метрик.

### Клонирование и структура каталогов

Установите [Git](https://git-scm.com/downloads) и [Python 3.11](https://www.python.org/downloads/). Проверенное окружение использует Python 3.11.9.

~~~sh
git clone https://github.com/jettooss/research-work.git
cd research-work
git rev-parse HEAD
~~~

Сохраните выведенную ревизию Git вместе с результатами эксперимента. Все последующие команды выполняются из каталога research-work.

Укажите абсолютный корень данных. PowerShell:

~~~powershell
$DATA_ROOT = (Get-Location).Path
~~~

Bash:

~~~bash
DATA_ROOT="$(pwd)"
~~~

При размещении данных на другом диске задайте абсолютный путь, например D:/research-data в Windows или /mnt/research-data в Linux. Код следует запускать из клона Git.

~~~text
research-work/
├── README.md
├── .venv-repro/                       # виртуальное окружение Python
├── diagram_vqa/
│   ├── src/vqa_retrieval/
│   ├── scripts/
│   ├── notebooks/experiments/
│   ├── reports/reproducibility/
│   └── runs/
├── downloads/                         # загруженные архивы
├── ai2d/                              # исходный AI2D и созданные манифесты/OCR
├── docvqa/                            # каталоги train/, val/, test/
├── infographicvqa/                    # исходный InfographicVQA и манифест
├── models/                            # веса и кэш Hugging Face
└── model_matrix_cache/                # кэши OCR/SAM/признаков
~~~

Параметры --data-root и --external-root указывают на родительский каталог ai2d/, docvqa/ и infographicvqa/, а не на отдельный набор данных.

### Установка окружения

Windows/PowerShell:

~~~powershell
py -3.11 -m venv .venv-repro
.\.venv-repro\Scripts\Activate.ps1
python --version
~~~

Linux/Bash:

~~~bash
python3.11 -m venv .venv-repro
source .venv-repro/bin/activate
python --version
~~~

Если активация PowerShell заблокирована, используйте .\.venv-repro\Scripts\python.exe вместо python.

Создайте каталоги в корне данных:

~~~sh
python -c "import sys; from pathlib import Path; root=Path(sys.argv[1]); [(root/p).mkdir(parents=True, exist_ok=True) for p in ('downloads', 'docvqa', 'infographicvqa', 'models/sam2')]" "$DATA_ROOT"
~~~

Для Random, вспомогательных метрик и примера установите CPU-окружение:

~~~sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
python -m pip install -r diagram_vqa/requirements-cpu.txt
python -m pip install -e ./diagram_vqa
python -m pip check
python diagram_vqa/examples/retrieval_metrics.py
~~~

Ожидаемый вывод примера:

~~~text
R@1: q->doc=66.67%, doc->q=100.00%, mean=83.33%
R@5: q->doc=100.00%, doc->q=100.00%, mean=100.00%
R@10: q->doc=100.00%, doc->q=100.00%, mean=100.00%
MRR q->doc: 0.8333
~~~

Для подготовки OCR, CLIP, графовых ноутбуков и оценки сохранённых моделей установите исследовательское окружение:

~~~sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 torchvision==0.21.0
python -m pip install -r diagram_vqa/requirements-research.txt
python -m pip install -e ./diagram_vqa
python -m pip check
~~~

Полный снимок зависимостей для Windows/Python 3.11 доступен в [requirements-lock-win-py311-cpu.txt](diagram_vqa/requirements-lock-win-py311-cpu.txt).

Для NVIDIA GPU создайте отдельное окружение:

~~~sh
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
python -m pip install -r diagram_vqa/requirements-research.txt
python -m pip install -e ./diagram_vqa
python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
~~~

Перед запуском GPU-экспериментов последняя команда должна вывести True. Необязательные зависимости QLoRA:

~~~sh
python -m pip install -r diagram_vqa/requirements-vlm.txt
~~~

Для OCR отдельно установите **Tesseract** и английские языковые данные. В Linux используются пакеты tesseract-ocr и tesseract-ocr-eng. В Windows следуйте [инструкции Tesseract](https://tesseract-ocr.github.io/tessdoc/Installation.html), включите English и добавьте каталог установки в PATH.

~~~sh
tesseract --version
tesseract --list-langs
~~~

Список языков должен содержать eng. При недоступном PATH подготовка AI2D принимает --tesseract-cmd "C:/Program Files/Tesseract-OCR/tesseract.exe".

Три архива DocVQA занимают около **8,91 ГБ до распаковки**, AI2D — около **0,99 ГБ**. Изображения InfographicVQA и хранение моделей/кэшей требуют дополнительного места. Для матричного запуска на диске результатов необходимо не менее **20 GiB** свободного места.

### Загрузка и подготовка данных

Используйте исходные наборы данных, а не одноимённые преобразованные подмножества Hugging Face. Скриптам подготовки требуются исходные изображения и аннотации вопросов; DocVQA также использует поставляемый OCR.

| Набор данных | Официальный источник | Что загрузить |
|---|---|---|
| AI2D | [страница AllenAI](https://prior.allenai.org/projects/diagram-understanding) | [полный AI2D ZIP](https://ai2-website.s3.amazonaws.com/data/ai2d-all.zip) и [официальные test ID](https://s3-us-east-2.amazonaws.com/prior-datasets/ai2d_test_ids.csv) |
| DocVQA | [страница набора](https://www.docvqa.org/datasets/docvqa), [загрузки RRC](https://rrc.cvc.uab.cat/?ch=17&com=downloads) | **Single Page Document VQA / Task 1**: [train](https://datasets.cvc.uab.es/rrc/DocVQA/train.tar.gz), [validation](https://datasets.cvc.uab.es/rrc/DocVQA/val.tar.gz), [test](https://datasets.cvc.uab.es/rrc/DocVQA/test.tar.gz), включая изображения и OCR |
| InfographicVQA | [страница набора](https://site.docvqa.org/datasets/infographicvqa), [загрузки RRC](https://rrc.cvc.uab.cat/?ch=17&com=downloads) | **Infographics VQA / Task 3**: изображения и JSON вопросов для train, validation и test |

Подходящий RRC challenge — ch=17; для доступа к загрузкам требуется регистрация. AI2D не допускает распространения данных третьим лицам, поэтому проект не публикует зеркала сырых данных.

#### AI2D

Сохраните архив как "$DATA_ROOT/downloads/ai2d-all.zip". В нём уже есть каталог ai2d/, поэтому распакуйте его в "$DATA_ROOT":

~~~sh
python -m zipfile -e "$DATA_ROOT/downloads/ai2d-all.zip" "$DATA_ROOT"
~~~

Сохраните отдельный CSV с test ID как "$DATA_ROOT/ai2d/ai2d_test_ids.csv".

~~~text
ai2d/
├── images/                           # 4 903 PNG в проверенном ZIP
├── questions/                        # 4 563 файла
├── annotations/
├── categories.json
├── README.txt
├── license.txt
└── ai2d_test_ids.csv                 # загружен отдельно; 982 ID
~~~

Сначала выполните проверку метаданных:

~~~sh
python diagram_vqa/scripts/prepare_ai2d_raw.py --data-root "$DATA_ROOT/ai2d" --official-test-ids "$DATA_ROOT/ai2d/ai2d_test_ids.csv" --path-root "$DATA_ROOT" --check-inputs
~~~

ZIP, проверенный 2026-09-08, не содержит images/4325.png и JSON вопроса, хотя официальный CSV включает ID 4325. После подтверждения, что отсутствует только это document-only изображение, выполните:

~~~sh
python diagram_vqa/scripts/prepare_ai2d_raw.py --data-root "$DATA_ROOT/ai2d" --official-test-ids "$DATA_ROOT/ai2d/ai2d_test_ids.csv" --path-root "$DATA_ROOT" --allow-missing-document-only-test-images --check-inputs
python diagram_vqa/scripts/prepare_ai2d_raw.py --data-root "$DATA_ROOT/ai2d" --official-test-ids "$DATA_ROOT/ai2d/ai2d_test_ids.csv" --path-root "$DATA_ROOT" --allow-missing-document-only-test-images --seed 42 --val-ratio 0.1
~~~

Вторая команда запускает Tesseract и создаёт обе стадии подготовки.

~~~text
ai2d/
├── prepared_v2/
│   ├── manifest_hybrid.jsonl
│   ├── split_hybrid.json
│   ├── preparation_plan.json
│   ├── provenance.json
│   └── ocr_v2/*.ocr.json
└── model_matrix_v1/
    ├── manifest.jsonl
    ├── split.json
    └── audit.json
~~~

В audit.json должны быть **11 145 train + 1 268 validation + 3 088 test вопросов**, **814 test-документов**, question_dataset_ready=true. Для проверенного архива full_archive_ready=false. Для продолжения прерванного OCR повторите команду с --resume.

#### DocVQA

Скачайте архивы **Task 1 / Single Page** в "$DATA_ROOT/downloads/", сохранив имена train.tar.gz, val.tar.gz и test.tar.gz. Они имеют суффикс .tar.gz, но содержат обычный TAR, поэтому используйте tar -xf:

~~~sh
tar -xf "$DATA_ROOT/downloads/train.tar.gz" -C "$DATA_ROOT/docvqa"
tar -xf "$DATA_ROOT/downloads/val.tar.gz" -C "$DATA_ROOT/docvqa"
tar -xf "$DATA_ROOT/downloads/test.tar.gz" -C "$DATA_ROOT/docvqa"
~~~

~~~text
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
~~~

Имена изображений должны совпадать со значением поля image в JSON. Не создавайте дополнительный уровень docvqa/train/train/ и не удаляйте ocr_results/.

~~~sh
python diagram_vqa/scripts/prepare_docvqa_manifest.py --data-root "$DATA_ROOT/docvqa" --output-dir "$DATA_ROOT/docvqa/prepared_v1" --splits train val test
~~~

Команда создаёт docvqa/prepared_v1/manifest.jsonl и split.json. В split.json ожидаются **39 463 / 5 349 / 5 188 вопросов**, missing_images=0, missing_ocr=0 для каждого split и пустые списки image_leakage.

#### InfographicVQA

На странице RRC выберите **Infographics VQA / Task 3**. Распакуйте изображения в infographicsvqa_images/, а JSON вопросов — в infographicsvqa_qas/:

~~~text
infographicvqa/
└── InfographicVQA/
    ├── infographicsvqa_images/
    │   └── <image_local_name из JSON вопроса>
    └── infographicsvqa_qas/
        ├── infographicsVQA_train_v1.0.json
        ├── infographicsVQA_val_v1.0_withQT.json
        └── infographicsVQA_test_v1.0.json
~~~

В Linux учитывается регистр. Если train JSON назван infographicVQA_train_v1.0.json, переименуйте его в infographicsVQA_train_v1.0.json. Для validation возможен вариант infographicsVQA_val_v1.0.json.

~~~sh
python diagram_vqa/scripts/prepare_infographicvqa_manifest.py --data-root "$DATA_ROOT/infographicvqa" --output-dir "$DATA_ROOT/infographicvqa/prepared_v1" --splits train val test
~~~

Ожидаются **23 946 / 2 801 / 3 288 вопросов**, нулевой missing_images и пустые списки image_leakage. Отчёт необходимо проверить до обучения.

#### Разбиения оценки

| Набор данных | Train-вопросы | Validation-вопросы | Test-вопросы | Уникальные test-документы | Публичные test-ответы |
|---|---:|---:|---:|---:|---|
| AI2D | 11 145 | 1 268 | 3 088 | 814 | Доступны |
| DocVQA | 39 463 | 5 349 | 5 188 | 1 287 | Скрыты |
| InfographicVQA | 23 946 | 2 801 | 3 288 | 579 | Скрыты |

Тестовые ответы DocVQA и InfographicVQA для ANLS/Accuracy скрыты. Локальный ноль по пустым reference нельзя интерпретировать как качество модели; для локальной оценки ответов используйте размеченный validation.

### Загрузка весов моделей

| Эксперимент | Входные данные | Получение |
|---|---|---|
| GATv2 + kNN и GraphColBERT/FiLM | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Автоматически через SentenceTransformers |
| Графы с механизмом внимания | MiniLM, визуальные энкодеры и SAM | Файл SAM находится в "$DATA_ROOT/models/sam2/sam2.1_b.pt" |
| CLIP baseline | [OpenAI CLIP ViT-B/32](https://github.com/openai/CLIP) | clip.load('ViT-B/32') загружает веса в ~/.cache/clip/ |
| Qwen QLoRA | [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | Каталог "$DATA_ROOT/models/Qwen2.5-VL-3B-Instruct/" |

Чтобы хранить загрузки Hugging Face в корне данных, перед запуском Python задайте HF_HOME.

~~~powershell
$env:HF_HOME = "$DATA_ROOT/models/hf-cache"
~~~

~~~bash
export HF_HOME="$DATA_ROOT/models/hf-cache"
~~~

Необязательная предварительная загрузка MiniLM:

~~~sh
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')"
~~~

Скачайте [sam2.1_b.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_b.pt) в "$DATA_ROOT/models/sam2/". Для CLIP и Qwen:

~~~sh
python -c "import clip; clip.load('ViT-B/32', device='cpu')"
python -c "import sys; from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct', local_dir=sys.argv[1])" "$DATA_ROOT/models/Qwen2.5-VL-3B-Instruct"
~~~

Для новых экспериментов фиксируйте идентификаторы ревизий и контрольные суммы файлов моделей.

### Запуск базовых моделей

Проверьте всю цепочку входных и выходных путей на запуске Random AI2D с 16 вопросами:

~~~sh
python diagram_vqa/scripts/run_model_matrix.py --dataset ai2d --model random --split test --seed 42 --max-samples 16 --data-root "$DATA_ROOT" --output-dir diagram_vqa/runs/readme_random_smoke --no-resume
~~~

Ожидаемый каталог:

~~~text
diagram_vqa/runs/readme_random_smoke/ai2d/random/seed42/test/
├── config.json
├── predictions.jsonl
└── metrics.json
~~~

Для полного набора удалите --max-samples и используйте другой корень результатов:

~~~sh
python diagram_vqa/scripts/run_model_matrix.py --dataset ai2d --model random --split test --seed 42 --data-root "$DATA_ROOT" --output-dir diagram_vqa/runs/readme_random_full --no-resume
~~~

Полные метрики AI2D должны содержать available_full, num_samples=3088 и num_documents=814. Повторите запуск с --seed 43 и --seed 44. На историческом манифесте средняя Accuracy равна **24,35 ± 0,21 %**.

Для обучения голов CLIP:

~~~sh
python diagram_vqa/scripts/train_model_matrix_vision.py --dataset ai2d --model clip --split val --seed 42 --epochs 20 --early-stopping-patience 4 --batch-size 64 --data-root "$DATA_ROOT" --output-dir diagram_vqa/runs/readme_clip_full --no-resume
~~~

Обучение использует train, выбирает лучший checkpoint на validation, затем автоматически оценивает test. Повторная оценка без обучения:

~~~sh
python diagram_vqa/scripts/run_model_matrix.py --dataset ai2d --model clip --split test --seed 42 --data-root "$DATA_ROOT" --output-dir diagram_vqa/runs/readme_clip_full --resume
~~~

Необязательное обучение Qwen + QLoRA требует GPU с достаточной памятью:

~~~sh
python -m pip install -r diagram_vqa/requirements-vlm.txt
python diagram_vqa/scripts/run_model_matrix.py --dataset ai2d --model qwen25_vl_qlora --split val --seed 42 --data-root "$DATA_ROOT" --output-dir diagram_vqa/runs/readme_qwen_full --no-resume
~~~

### Обучение графовых моделей

Для основных моделей презентации используйте исходные ноутбуки экспериментов через executor. Универсальный матричный graph_transformer — другая архитектура.

~~~sh
python -m ipykernel install --sys-prefix --name python3 --display-name "Diagram VQA reproduction"
python diagram_vqa/scripts/execute_experiment_matrix_notebooks.py --dataset ai2d --architecture sam2_dinov2_dual_branch_evidence_graph --seeds 42 43 44 --external-root "$DATA_ROOT" --output-root diagram_vqa/runs/readme_graphs --kernel-name python3 --dry-run
~~~

Команда с --dry-run выводит три конфигурации и не создаёт каталоги. Проверьте манифест, seed, число эпох и путь к кэшу признаков. Для запуска удалите --dry-run.

| Название в презентации | Значение --architecture | Эпохи: AI2D / DocVQA / InfographicVQA |
|---|---|---|
| GATv2 + kNN | hybrid_v4_multipos_training | 40 / 40 / 20 |
| Граф с отбором связей по типам | sam2_dinov2_heterogeneous_balanced_evidence_graph | 20 / 20 / 20 |
| Граф с общим отбором связей | sam2_dinov2_dual_branch_evidence_graph | 20 / 20 / 20 |

Для DocVQA или InfographicVQA замените --dataset. Если опустить оба селектора, будет запланировано **54 запуска**: шесть архитектур × три набора × три seed.

Каждый запуск создаёт:

~~~text
diagram_vqa/runs/readme_graphs/<dataset>/<architecture>/seed42_full/
├── notebook_run.json
├── execution.json
├── notebook.executed.ipynb
├── checkpoint_best.pt
└── metrics.json
~~~

Ноутбуки предварительно вычисляют признаки уникальных изображений train/val/test и сохраняют по одному файлу на изображение в feature_cache_dir из notebook_run.json. Для нового эксперимента используйте новый корень данных и кэшей.

### Оценка checkpoint графовой модели

После обучения «Графа с общим отбором связей» получите фактический каталог кэша признаков из seed 42.

~~~powershell
$FEATURE_CACHE = python -c "import json; print(json.load(open('diagram_vqa/runs/readme_graphs/ai2d/sam2_dinov2_dual_branch_evidence_graph/seed42_full/notebook_run.json', encoding='utf-8'))['feature_cache_dir'])"
~~~

~~~sh
python diagram_vqa/reports/presentation_metrics/evaluate_saved_retrieval.py --project-root diagram_vqa --data-root "$DATA_ROOT" --runs-root diagram_vqa/runs/readme_graphs --datasets ai2d --architectures sam2_dinov2_dual_branch_evidence_graph --seeds 42 43 44 --feature-cache "$FEATURE_CACHE" --output-dir diagram_vqa/runs/readme_graph_retrieval --local-files-only --device cpu
python diagram_vqa/reports/presentation_metrics/audit_retrieval_matrix.py --project-root diagram_vqa --data-root "$DATA_ROOT" --runs-root diagram_vqa/runs/readme_graphs --datasets ai2d --architectures sam2_dinov2_dual_branch_evidence_graph --seeds 42 43 44 --retrieval-root diagram_vqa/runs/readme_graph_retrieval --output diagram_vqa/runs/readme_graph_retrieval_audit.json
~~~

Audit должен показать verified_runs=3, required_runs=3 и completed=true. Оценщик поддерживает «Граф с общим отбором связей» и «Граф с отбором связей по типам»; GATv2 + kNN оценивается в собственном ноутбуке.

### Проверка сохранённых результатов

Чтобы пересчитать опубликованные таблицы по компактным версионированным измерениям без загрузки данных и моделей:

~~~sh
python diagram_vqa/scripts/reproduce_results.py --check
~~~

Ожидаемый итог:

~~~json
{"accuracy_groups": 5, "retrieval_groups": 12, "saved_full_test_runs": 36, "check": "passed"}
~~~

Проверяются идентификаторы, хеши, seed и агрегация 36 основных запусков. Проценты используют популяционное стандартное отклонение ddof=0 по seed 42–44.

Для новых результатов сохраните конфигурацию, checkpoints и предсказания, затем обновите свидетельство и таблицы:

~~~sh
python diagram_vqa/scripts/reproduce_results.py --write
python diagram_vqa/scripts/reproduce_results.py --check
python -m unittest discover -s diagram_vqa/tests -v
python -m pip check
~~~

Зафиксированная валидация содержит 71 прошедший тест, три полных seed AI2D Random, smoke-запуск CLIP на 16 вопросах и полный инференс seed 42 для AI2D «Граф с общим отбором связей». Подробности находятся в [машиночитаемом протоколе](diagram_vqa/docs/reproduction_validation.json) и [описании окружения](diagram_vqa/docs/environment_reference.json).

### Возобновление, архив и устранение неполадок

Для матричных backend --resume требует тот же набор данных, манифест, код, версии пакетов, параметры и лимит выборки. Для изменённой конфигурации используйте новый каталог результатов.

Для создания переносимого архива кода и измерений:

~~~sh
python diagram_vqa/scripts/export_reproduction_bundle.py --project-root diagram_vqa --data-root "$DATA_ROOT" --output diagram_vqa/runs/reproduction-metadata.zip --dry-run
python diagram_vqa/scripts/export_reproduction_bundle.py --project-root diagram_vqa --data-root "$DATA_ROOT" --output diagram_vqa/runs/reproduction-metadata.zip
python diagram_vqa/scripts/export_reproduction_bundle.py --verify diagram_vqa/runs/reproduction-metadata.zip
~~~

Архив содержит SHA256-хеши, доступный код, свидетельства измерений и выбранные метаданные/предсказания. Исходные изображения, предобученные модели, кэши признаков и checkpoints оптимизатора исключаются.

| Симптом | Решение |
|---|---|
| No module named ... | Активируйте правильное окружение и установите CPU/research/VLM зависимости через python -m pip |
| tesseract is not installed или отсутствует eng | Установите исполняемый файл и языковые данные; исправьте PATH или передайте --tesseract-cmd |
| В AI2D отсутствует 4325 | Используйте документированное исключение document-only изображения |
| Отсутствует manifest.jsonl | Сначала запустите подготовку и укажите data-root на родительский каталог наборов |
| Не найдены изображение/OCR | Проверьте вложенность, имена и split.json; DocVQA требует ocr_results/ |
| Не найден sam2.1_b.pt | Поместите файл в "$DATA_ROOT/models/sam2/" |
| CUDA out of memory | Уменьшите batch/budget в новом эксперименте и зафиксируйте настройки |

### Docker

Docker предоставляет CPU-пример оценки/Random, а не полное исследовательское окружение графовых моделей.

~~~sh
docker build -t diagram-vqa:cpu .
docker run --rm diagram-vqa:cpu
~~~

Для подготовленного AI2D подключите "$DATA_ROOT/ai2d" только для чтения в /workspace/ai2d и доступный для записи каталог результатов в /workspace/diagram_vqa/runs; см. [команды монтирования для платформ](diagram_vqa/docs/usage.md#docker-with-a-dataset).

## Как работает метод

1. **Текстовые вершины.** [OCR](https://github.com/tesseract-ocr/tesseract) извлекает текст и ограничивающие рамки.
2. **Визуальные вершины.** [SAM2](https://github.com/facebookresearch/sam2) выделяет области изображения; каждая вершина также содержит тип и геометрию.
3. **Структура графа.** k-ближайших соседей (kNN) задают связи по близости, а обучаемое внимание выбирает или взвешивает связи в зависимости от архитектуры.
4. **Поиск и ответы.** Модели ранжируют документы относительно вопроса и отдельно оценивают выбор ответа. «Граф с общим отбором связей» объединяет базовую ветвь и дополнительную ветвь локальных свидетельств OCR/SAM.

К базовым моделям относятся [CLIP](https://github.com/openai/CLIP) и визуально-языковая модель, адаптированная с [QLoRA](https://arxiv.org/abs/2305.14314).

## Результаты

Результаты создаются из версионированных свидетельств полного test. [Исследовательская презентация](https://docs.google.com/presentation/d/1RnsHrDhAxOaTre7VCdBh67QbfDaOExJKicDK9aOw374/edit) содержит изложение работы; [детальные результаты и происхождение](diagram_vqa/docs/results.md) включают исторические снимки, анализ типов вопросов, измерения ресурсов и ограничения оценки.

### Accuracy ответов AI2D

Accuracy — доля верных ответов. Текущие локальные модели используют **3 088 test-вопросов**, seed **42–44** и среднее ± популяционное стандартное отклонение ddof=0.

| Модель | Accuracy, % | Запуски |
|---|---:|---:|
| Qwen2.5-VL-3B + QLoRA | 73.19 (архивный) | Не подтверждён повторно |
| Граф с общим отбором связей | 48.54 ± 0.44 | 3 |
| Граф с отбором связей по типам | 47.83 ± 0.44 | 3 |
| GATv2 + kNN | 46.24 ± 0.46 | 3 |
| CLIP | 44.52 ± 0.86 | 3 |
| Random | 24.35 ± 0.21 | 3 |

Среднее значение «Графа с общим отбором связей» выше CLIP на **4,02 процентного пункта**.

### Поиск документов

R@K = (R@K вопрос → документ + R@K документ → вопрос) / 2.

Кандидаты — все уникальные документы из оценочного split набора данных. В обратном направлении релевантен любой вопрос, связанный с документом. Запрос считается успешным, если хотя бы один релевантный кандидат попадает в первые K позиций.

<!-- BEGIN GENERATED RETRIEVAL TABLE -->
| Набор данных | Модель | R@1, % | R@5, % | R@10, % | n |
|---|---|---:|---:|---:|---:|
| AI2D | CLIP | 4.72 ± 0.29 | 18.46 ± 0.55 | 29.20 ± 1.11 | 3 |
| AI2D | GATv2 + kNN | 6.64 ± 0.63 | 20.78 ± 0.45 | 31.82 ± 0.81 | 3 |
| AI2D | Граф с отбором связей по типам | 6.30 ± 0.42 | 20.85 ± 0.79 | 31.15 ± 1.35 | 3 |
| AI2D | Граф с общим отбором связей | 6.13 ± 0.42 | 20.28 ± 0.58 | 30.90 ± 0.39 | 3 |
| DocVQA | CLIP | 5.09 ± 0.22 | 14.82 ± 0.81 | 20.98 ± 0.73 | 3 |
| DocVQA | GATv2 + kNN | 18.51 ± 0.56 | 36.73 ± 0.30 | 45.80 ± 0.26 | 3 |
| DocVQA | Граф с отбором связей по типам | 7.62 ± 0.36 | 20.20 ± 0.35 | 29.52 ± 0.42 | 3 |
| DocVQA | Граф с общим отбором связей | 7.54 ± 0.43 | 20.79 ± 0.48 | 29.50 ± 0.51 | 3 |
| InfographicVQA | CLIP | 11.35 ± 0.08 | 25.76 ± 0.74 | 34.85 ± 0.87 | 3 |
| InfographicVQA | GATv2 + kNN | 24.58 ± 1.73 | 45.92 ± 1.36 | 56.27 ± 0.93 | 3 |
| InfographicVQA | Граф с отбором связей по типам | 13.85 ± 2.04 | 32.70 ± 2.77 | 43.53 ± 3.20 | 3 |
| InfographicVQA | Граф с общим отбором связей | 14.32 ± 2.16 | 32.14 ± 3.10 | 43.03 ± 3.61 | 3 |
<!-- END GENERATED RETRIEVAL TABLE -->

Значения указаны в процентах как среднее ± популяционное стандартное отклонение; для каждой строки выполнено три запуска с seed 42–44.

Графовые модели превосходят CLIP по приведённым средним значениям поиска. Сравнение относится к поиску документов, а не к Accuracy ответов. Внимание не превосходит kNN стабильно, а одних средних значений недостаточно для установления статистической значимости. Архитектуры также различаются по признакам и бюджету обучения, поэтому таблица не выделяет причинный вклад внимания.

Для проверки результатов:

~~~sh
python diagram_vqa/scripts/reproduce_results.py --check
~~~

[Версионированное свидетельство](diagram_vqa/reports/reproducibility/evidence.json) содержит все 36 основных запусков и три запуска OCR.

## Python API

~~~python
from vqa_retrieval.metrics import recall_at_k_multi_positive

scores = [[0.9, 0.1], [0.4, 0.6], [0.2, 0.8]]
positives = [{0}, {0}, {1}]
print(recall_at_k_multi_positive(scores, positives, ks=(1, 5)))
# {1: 0.6666666666666666, 5: 1.0}
~~~

sim — прямоугольная матрица сходства «вопрос × кандидат»; positive_indices содержит индексы релевантных кандидатов с нуля для каждого запроса. Значение ks по умолчанию — (1, 5, 10). Результат имеет тип dict[int, float] и значения в диапазоне [0, 1]. Некорректные входы приводят к ValueError.

[Справочник API](diagram_vqa/docs/usage.md#python-metrics-api) также описывает двунаправленный поиск, MRR и оценку ответов.

## Структура проекта

| Путь | Назначение |
|---|---|
| [diagram_vqa/src/vqa_retrieval](diagram_vqa/src/vqa_retrieval) | Python-пакет: данные, графы, метрики, VLM |
| [diagram_vqa/scripts](diagram_vqa/scripts) | Подготовка, обучение, оценка и audit |
| [diagram_vqa/notebooks/experiments](diagram_vqa/notebooks/experiments) | Эксперименты для трёх наборов данных |
| [diagram_vqa/experiments](diagram_vqa/experiments) | Протоколы и конфигурация |
| [diagram_vqa/reports](diagram_vqa/reports) | Отчёты и свидетельства оценки |
| [diagram_vqa/model_registry](diagram_vqa/model_registry) / [experiment_registry](diagram_vqa/experiment_registry) | Каталоги моделей и экспериментов |
| diagram_vqa/runs/ | Checkpoints, предсказания и истории обучения |
| ai2d/, docvqa/, infographicvqa/, models/, model_matrix_cache/ | Внешние данные, веса и кэши |

Начните с ноутбука AI2D для [GATv2 + kNN](diagram_vqa/notebooks/experiments/ai2d/ai2d_hybrid_v4_multipos_training.ipynb). Ноутбуки для [DocVQA](diagram_vqa/notebooks/experiments/docvqa) и [InfographicVQA](diagram_vqa/notebooks/experiments/infographicvqa) находятся в соответствующих каталогах.

[Протокол model_matrix_v1](diagram_vqa/experiments/model_matrix_protocol.json) охватывает 11 моделей × 3 набора данных с seed 42. Его результаты available_smoke используют малые выборки и не должны смешиваться с многократной оценкой из презентации.

## Авторы

**Андрей Андреевич Утляков** — автор исследования.

**Анастасия Александровна Лаушкина** — научный руководитель.
**Валерия Дмитриевна Волоха** — научный консультант.

## Лицензия

В репозитории пока нет файла LICENSE, а лицензия не указана в pyproject.toml. По условиям использования кода обращайтесь к автору. Для наборов данных и сторонних моделей действуют их собственные лицензии.
