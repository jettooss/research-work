# AI2D Diagram VQA

Исследовательский проект по **Visual Question Answering для диаграмм** на датасете **AI2D**.  
Цель проекта - сравнить несколько подходов к ответам на вопросы по учебным диаграммам в формате multiple choice:

- графовые модели по OCR-узлам и сегментам диаграмм;
- Graph Transformer и гибридные graph/text retrieval-модели;
- CLIP/SigLIP baseline;
- GraphColBERT с FiLM-фьюжном графовой структуры;
- Qwen2.5-VL с QLoRA;
- LLM-reasoner поверх текстового представления диаграммы.

Проект устроен как рабочая лаборатория: код, ноутбуки, конфиги экспериментов и краткие отчеты лежат в репозитории, а тяжелые данные и модели хранятся рядом с ним.

## Где Что Лежит

| Ресурс | Путь |
|---|---|
| Основной код проекта | `diagram_vqa/` |
| Датасет AI2D | `ai2d/` |
| Локальные модели | `models/` |
| Общие результаты прогонов | `runs/` |
| Виртуальное окружение | `.venv/` |

Ожидаемая локальная модель Qwen:

```text
models/Qwen2.5-VL-3B-Instruct
```

## Структура Проекта

```text
data/
├── ai2d/                         датасет AI2D
├── models/                       локальные веса моделей
├── runs/                         общие артефакты запусков
├── .venv/                        окружение проекта
└── diagram_vqa/
    ├── src/vqa_retrieval/        основной Python-пакет
    ├── vqa_retrieval/            shim для локальных импортов
    ├── scripts/                  CLI-скрипты обучения и оценки
    ├── experiments/              конфиги и запускалки экспериментов
    ├── notebooks/                исследовательские ноутбуки
    ├── runs/                     локальные выходы прогонов
    ├── reports/                  краткие сводки результатов
    ├── model_registry/           каталог чекпойнтов и метрик
    ├── experiment_registry/      каталог семейств экспериментов
    ├── Ultralytics/              конфигурация для SAM
    ├── pyproject.toml
    └── sitecustomize.py          добавляет src/ в PYTHONPATH
```

## Быстрый Старт

Для скриптов удобнее работать из папки `diagram_vqa/`:

```powershell
cd diagram_vqa
```

## Основные Ноутбуки

| Ноутбук | Назначение |
|---|---|
| `clip_vqa_baseline.ipynb` | CLIP/SigLIP zero-shot и fine-tuning baseline |
| `ai2d_gnn_ocr_knn_graph_encoder.ipynb` | kNN-граф по OCR-узлам и GNN-энкодер для retrieval |
| `multi_dataset_gnn_training.ipynb` | обучение GNN-retrieval на нескольких датасетах |
| `ai2d_hybrid_v4_multipos_training.ipynb` | гибрид графа и текста с multi-positive contrastive loss |
| `hybrid_epoch_view_training.ipynb` | обучение гибридной модели с просмотром метрик по эпохам |
| `sam_sam2_graph_nodes.ipynb` | построение графовых узлов из SAM/SAM2-сегментации |
| `train_sam_graph_transformer_models.ipynb` | обучение Graph Transformer на SAM-узлах |
| `qwen_vlm_qlora_training.ipynb` | дообучение Qwen2.5-VL-3B через QLoRA |
| `graphcolbert_film.ipynb` | GraphColBERT с FiLM-фьюжном графовой структуры |
| `ai2d_llm_reasoner.ipynb` | LLM-reasoner: диаграмма -> текст -> рассуждение |

Рекомендуемый порядок знакомства:

1. `clip_vqa_baseline.ipynb` - быстрый baseline.
2. `ai2d_gnn_ocr_knn_graph_encoder.ipynb` - графовый baseline.
3. `qwen_vlm_qlora_training.ipynb` - VLM-подход с лучшим текущим результатом.
4. `graphcolbert_film.ipynb` - retrieval с late interaction.
5. `ai2d_llm_reasoner.ipynb` - reasoning через текстовое описание диаграммы.

## Текущие Результаты

Accuracy на AI2D test:

| Эксперимент | Подход | Метрика |
|---|---|---:|
| SAM2 + Graph Transformer | SAM2 + graph | 0.246 |
| Random baseline | Baseline | 0.250 |
| CLIP zero-shot | Visual baseline | 0.299 |
| CLIP fine-tuned | Visual fine-tuning | 0.376 |
| plain ColBERT (вариант<->OCR) | OCR/text retrieval | 0.290 |
| question-FiLM | FiLM fusion | 0.334 |
| GraphColBERT fusion (v3) | Graph + ColBERT | 0.339 |
| GraphColBERT FiLM (v2) | Graph + FiLM | 0.348 |
| Hybrid GATv2 + kNN | Hybrid graph | 0.391 |
| Graph Transformer | Graph Transformer | 0.396 |
| **ai2d_gnn_ocr_knn_graph encoder** | **OCR+kNN GNN** | **0.5875** |
| LLM-reasoner (CoG-DQA-style) | LLM reasoner | **~0.60-0.66** |
| **Qwen2.5-VL-3B QLoRA (VLM)** | **VLM fine-tuning** | **0.732** |

Вывод: OCR+kNN GNN заметно усиливает графовую ветку и дает лучший подтвержденный non-VLM результат среди локальных graph/retrieval-подходов: **0.5875 accuracy**. При этом VLM fine-tuning остается верхней точкой проекта: **Qwen2.5-VL-3B QLoRA - 0.732 accuracy**.

## SAM И Сегментация

Ноутбуки `sam_sam2_graph_nodes.ipynb` и `train_sam_graph_transformer_models.ipynb` требуют Ultralytics и локальные веса SAM.

Установка:

```powershell
pip install -U ultralytics
```

Ожидаемый путь к весам:

```text
models/sam3/sam3.pt
```

Кеш масок строится один раз и затем переиспользуется:

```text
diagram_vqa/runs/sam_cache_ai2d/
```

## Важные Замечания

- Тяжелые данные, модели и большие результаты не должны попадать в git.
- Большинство экспериментов рассчитаны на локальное окружение с CUDA.
- Ноутбуки обучения могут быть долгими и требовательными к VRAM.
- Для API-based reasoning нужен `.env` с ключами доступа.
- `sitecustomize.py` нужен для удобных локальных импортов; его лучше не удалять.

## Полезные Скрипты

| Скрипт | Что делает |
|---|---|
| `scripts/clip_vqa_baseline.py` | baseline на CLIP/SigLIP |
| `scripts/train_ai2d_graph_experiment.py` | обучение графовых моделей |
| `scripts/train_ai2d_hybrid.py` | обучение гибридных retrieval-моделей |
| `scripts/train_vlm_qlora.py` | QLoRA-дообучение VLM |
| `scripts/evaluate_ai2d_vlm.py` | оценка VLM на AI2D |
| `scripts/probe_vram.py` | проверка доступной VRAM |
| `scripts/validate_notebooks.py` | базовая проверка ноутбуков |

## Коротко

Если нужно быстро понять проект: начните с `diagram_vqa/notebooks/experiments/ai2d/clip_vqa_baseline.ipynb`, затем посмотрите `qwen_vlm_qlora_training.ipynb`.
