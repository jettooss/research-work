from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("ai2d", "infographicvqa", "docvqa")
MODELS = (
    ("00", "random", "Random baseline"),
    ("01", "clip", "CLIP: frozen backbone + PyTorch heads"),
    ("02", "siglip", "SigLIP: frozen backbone + PyTorch heads"),
    ("03", "ocr_text", "OCR text retrieval"),
    ("04", "gatv2_knn", "OCR+kNN GATv2"),
    ("05", "graph_transformer", "Graph Transformer"),
    ("06", "sam2_graph_transformer", "SAM2 + Graph Transformer"),
    ("07", "hybrid_gatv2_knn", "Hybrid GATv2+kNN"),
    ("08", "graphcolbert", "GraphColBERT"),
    ("09", "graphcolbert_film", "GraphColBERT+FiLM"),
    ("10", "qwen25_vl_qlora", "Qwen2.5-VL-3B QLoRA"),
)
GRAPH_MODELS = {
    "gatv2_knn", "graph_transformer", "sam2_graph_transformer",
    "hybrid_gatv2_knn", "graphcolbert", "graphcolbert_film",
}


def cell(kind: str, source: str) -> dict:
    result = {
        "cell_type": kind,
        "id": hashlib.sha1(f"{kind}:{source}".encode()).hexdigest()[:12],
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


SETUP = '''from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path.cwd()
while not (ROOT / "scripts" / "run_model_matrix.py").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("Не найдена корневая папка diagram_vqa")
    ROOT = ROOT.parent
EXTERNAL_ROOT = ROOT.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vqa_retrieval.model_matrix import load_rows, ocr_spans, split_rows

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
print({"root": str(ROOT), "device": str(DEVICE), "torch": torch.__version__})
'''

DATA = '''rows = load_rows(DATASET, EXTERNAL_ROOT)
train_rows = split_rows(rows, "train")
val_rows = split_rows(rows, "val")
test_rows = split_rows(rows, "test")
print({"train_questions": len(train_rows), "val_questions": len(val_rows), "test_questions": len(test_rows)})

sample = train_rows[0]
print("Поля нормализованной записи:", sorted(sample))
print(json.dumps(sample, ensure_ascii=False, indent=2)[:8000])
'''

VISUAL = '''image = Image.open(sample["image_path"]).convert("RGB")
print("image_id:", sample["image_id"], "size:", image.size, "path:", sample["image_path"])
plt.figure(figsize=(12, 8))
plt.imshow(image)
plt.axis("off")
plt.title(sample["question"])
plt.show()

spans = ocr_spans(DATASET, sample, EXTERNAL_ROOT, EXTERNAL_ROOT / "model_matrix_cache/ocr")
print("OCR spans:", len(spans))
print(json.dumps(spans[:10], ensure_ascii=False, indent=2))

overlay = image.copy()
draw = ImageDraw.Draw(overlay)
for span in spans[:80]:
    box = span.get("bbox", [0, 0, 0, 0])
    draw.rectangle(tuple(box), outline="red", width=2)
plt.figure(figsize=(12, 8))
plt.imshow(overlay)
plt.axis("off")
plt.title("OCR-вершины и bounding boxes")
plt.show()
'''

LABELS = '''if DATASET == "ai2d":
    annotation = {
        "question": sample["question"],
        "options": sample.get("options", []),
        "correct_option_idx": sample.get("correct_option_idx"),
        "answer": sample.get("answers", [None])[0],
    }
else:
    annotation = {
        "questionId": sample.get("question_id", sample["sample_id"]),
        "docId": sample.get("image_id"),
        "question": sample["question"],
        "answers": sample.get("answers", []),
    }
print(json.dumps(annotation, ensure_ascii=False, indent=2))
print("Важно: ответы val/test не используются для построения OCR-кандидатов или признаков.")
'''

VISION_MODEL = '''class VisionHeads(nn.Module):
    """Точная структура обучаемых голов из production runner."""
    def __init__(self, input_dim: int, output_dim: int = 192):
        super().__init__()
        self.image = nn.Sequential(nn.Linear(input_dim, 384), nn.GELU(), nn.Linear(384, output_dim))
        self.text = nn.Sequential(nn.Linear(input_dim, 384), nn.GELU(), nn.Linear(384, output_dim))
        self.fusion = nn.Sequential(nn.Linear(output_dim * 2, output_dim), nn.GELU(), nn.Linear(output_dim, output_dim))

    def retrieval(self, images, questions):
        image = F.normalize(self.image(images), dim=-1)
        question = F.normalize(self.text(questions), dim=-1)
        return question @ image.T

    def vqa(self, image, question, candidates):
        image = F.normalize(self.image(image.unsqueeze(0)), dim=-1)
        question = F.normalize(self.text(question.unsqueeze(0)), dim=-1)
        context = F.normalize(self.fusion(torch.cat([image, question], dim=-1)), dim=-1)
        candidates = F.normalize(self.text(candidates), dim=-1)
        return (context @ candidates.T).squeeze(0)

BACKBONE = "openai/clip-vit-base-patch32" if MODEL == "clip" else "google/siglip2-base-patch16-224"
INPUT_DIM = 512 if MODEL == "clip" else 768
model = VisionHeads(INPUT_DIM).to(DEVICE)
print("Frozen backbone:", BACKBONE)
print(model)
print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# Проверка форм входов без загрузки тяжёлого backbone.
n_questions, n_documents, n_candidates = 4, 3, 6
images = torch.randn(n_documents, INPUT_DIM, device=DEVICE)
questions = torch.randn(n_questions, INPUT_DIM, device=DEVICE)
candidates = torch.randn(n_candidates, INPUT_DIM, device=DEVICE)
print("retrieval logits:", model.retrieval(images, questions).shape)
print("VQA logits:", model.vqa(images[0], questions[0], candidates).shape)
'''

GRAPH_MODEL = '''from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, GlobalAttention, TransformerConv

TEXT_DIM, NODE_DIM = 384, 394  # MiniLM text + 10 geometry/type features

class GraphMatrixModel(nn.Module):
    """PyTorch Geometric модель, используемая в общей матрице экспериментов."""
    def __init__(self, variant: str, hidden: int = 192, out_dim: int = 192):
        super().__init__()
        self.variant = variant
        self.input = nn.Linear(NODE_DIM, hidden)
        self.q_proj = nn.Sequential(nn.Linear(TEXT_DIM, hidden), nn.GELU(), nn.Linear(hidden, out_dim))
        self.candidate_proj = nn.Sequential(nn.Linear(TEXT_DIM, hidden), nn.GELU(), nn.Linear(hidden, out_dim))
        conv = TransformerConv if variant in {"graph_transformer", "sam2_graph_transformer"} else GATv2Conv
        self.layer1 = conv(hidden, hidden // 4, heads=4, dropout=0.1)
        self.layer2 = conv(hidden, hidden // 4, heads=4, dropout=0.1)
        self.node_out = nn.Linear(hidden, out_dim)
        gate = nn.Sequential(nn.Linear(out_dim, out_dim // 2), nn.GELU(), nn.Linear(out_dim // 2, 1))
        self.pool = GlobalAttention(gate_nn=gate)
        self.hybrid = nn.Sequential(nn.Linear(out_dim * 2, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))
        self.film = nn.Linear(out_dim, out_dim * 2)

    def encode_nodes(self, batch):
        x = F.gelu(self.input(batch.x))
        x = F.gelu(self.layer1(x, batch.edge_index))
        x = F.gelu(self.layer2(x, batch.edge_index))
        return F.gelu(self.node_out(x))

    def encode(self, batch, questions=None):
        nodes = self.encode_nodes(batch)
        if self.variant == "graphcolbert_film" and questions is not None:
            q = self.q_proj(questions)
            gamma, beta = self.film(q).chunk(2, dim=-1)
            nodes = nodes * (1 + gamma[batch.batch]) + beta[batch.batch]
        pooled = self.pool(nodes, batch.batch)
        return F.normalize(pooled, dim=-1), nodes

    def retrieval_logits(self, batch, questions, question_tokens=None):
        q = F.normalize(self.q_proj(questions), dim=-1)
        docs, nodes = self.encode(batch)
        if self.variant not in {"graphcolbert", "graphcolbert_film"}:
            return q @ docs.T
        token_queries = [F.normalize(self.q_proj(tokens), dim=-1) for tokens in question_tokens]
        gamma, beta = self.film(q).chunk(2, dim=-1)
        columns = []
        for doc_index in range(batch.num_graphs):
            base = nodes[batch.batch == doc_index]
            scores = []
            for query_index, tokens in enumerate(token_queries):
                current = base
                if self.variant == "graphcolbert_film":
                    current = current * (1 + gamma[query_index]) + beta[query_index]
                scores.append((tokens @ F.normalize(current, dim=-1).T).max(dim=1).values.mean())
            columns.append(torch.stack(scores))
        return torch.stack(columns, dim=1)

    def vqa_logits(self, graph, question, candidates):
        batch = Batch.from_data_list([graph]).to(question.device)
        q = F.normalize(self.q_proj(question.unsqueeze(0)), dim=-1)
        doc, nodes = self.encode(batch, question.unsqueeze(0) if self.variant == "graphcolbert_film" else None)
        if self.variant.startswith("graphcolbert"):
            node_context = F.normalize(nodes, dim=-1)
            context = F.normalize((q @ node_context.T).softmax(dim=1) @ node_context + q, dim=-1)
        elif self.variant == "hybrid_gatv2_knn":
            context = F.normalize(self.hybrid(torch.cat([doc, q], dim=-1)), dim=-1)
        else:
            context = F.normalize(doc + q, dim=-1)
        candidate = F.normalize(self.candidate_proj(candidates), dim=-1)
        return (context @ candidate.T).squeeze(0)

def knn_edges(x, k=5):
    if len(x) <= 1:
        return torch.zeros((2, 0), dtype=torch.long)
    distance = torch.cdist(x[:, -10:-8], x[:, -10:-8])
    distance.fill_diagonal_(float("inf"))
    neighbors = distance.topk(min(k, len(x) - 1), largest=False).indices
    source = torch.arange(len(x)).repeat_interleave(neighbors.shape[1])
    return torch.stack([source, neighbors.reshape(-1)])

# Небольшой реальный граф: геометрия берётся из OCR, текстовые embeddings здесь нулевые
# только для демонстрации формы. Production runner использует MiniLM embeddings.
width, height = image.size
nodes = []
for span in spans[:40]:
    x1, y1, x2, y2 = map(float, span.get("bbox", [0, 0, 0, 0]))
    geometry = [(x1+x2)/2/width, (y1+y2)/2/height, (x2-x1)/width, (y2-y1)/height,
                max(1, (x2-x1)*(y2-y1))/(width*height), x1/width, y1/height, x2/width, y2/height, 1.0]
    nodes.append(torch.cat([torch.zeros(TEXT_DIM), torch.tensor(geometry)]))
if not nodes:
    nodes = [torch.zeros(NODE_DIM)]
x = torch.stack(nodes).float()
graph = Data(x=x, edge_index=knn_edges(x))
model = GraphMatrixModel(MODEL).to(DEVICE)
batch = Batch.from_data_list([graph]).to(DEVICE)
question = torch.randn(1, TEXT_DIM, device=DEVICE)
tokens = [torch.randn(4, TEXT_DIM, device=DEVICE)]
print(model)
print("nodes:", x.shape, "edges:", graph.edge_index.shape)
print("retrieval logits:", model.retrieval_logits(batch, question, tokens).shape)
print("trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
'''

RANDOM_MODEL = '''import random

class RandomBaseline:
    """Не nn.Module: baseline не обучается и не имеет checkpoint."""
    def __init__(self, seed=42):
        self.rng = random.Random(seed)

    def answer(self, row, train_answers):
        if DATASET == "ai2d":
            options = row.get("options", [])
            return self.rng.choice(options) if options else ""
        return self.rng.choice(train_answers) if train_answers else ""

baseline = RandomBaseline(42)
train_answers = [a for row in train_rows for a in row.get("answers", []) if a]
print("prediction:", baseline.answer(sample, train_answers))
print("Параметров: 0; обучение: отсутствует")
'''

OCR_MODEL = '''from sklearn.feature_extraction.text import TfidfVectorizer

class OCRTextRetriever:
    """Не nn.Module: TF-IDF индекс строится детерминированно и не обучает нейросеть."""
    def fit_documents(self, document_texts):
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=100_000)
        self.documents = self.vectorizer.fit_transform(document_texts)
        return self

    def scores(self, questions):
        return (self.vectorizer.transform(questions) @ self.documents.T).toarray()

demo_docs = [" ".join(s.get("text", "") for s in spans), "second example document"]
retriever = OCRTextRetriever().fit_documents(demo_docs)
print("question→document scores:", retriever.scores([sample["question"]]))
print("Параметров: 0; checkpoint не создаётся")
'''

QWEN_MODEL = '''# Qwen является torch.nn.Module; базовые веса загружаются один раз и не копируются.
# LOAD_QWEN=False оставляет учебный notebook лёгким и безопасным для открытия.
LOAD_QWEN = False
BASE_MODEL = EXTERNAL_ROOT / "models/Qwen2.5-VL-3B-Instruct"

from transformers import BitsAndBytesConfig
from peft import LoraConfig

quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
)
lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
print(quantization)
print(lora)

if LOAD_QWEN:
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import get_peft_model, prepare_model_for_kbit_training
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True, use_fast=False)
    base = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL, trust_remote_code=True, device_map="auto",
        quantization_config=quantization,
        torch_dtype=quantization.bnb_4bit_compute_dtype,
    )
    model = get_peft_model(prepare_model_for_kbit_training(base), lora)
    model.print_trainable_parameters()
'''

LOSS = '''def multipositive_retrieval_loss(logits, document_ids, temperature=0.07):
    positive = torch.tensor(
        [[left == right for right in document_ids] for left in document_ids],
        device=logits.device,
    )
    scaled = logits / temperature
    forward = -(torch.logsumexp(scaled.masked_fill(~positive, -torch.inf), 1) - torch.logsumexp(scaled, 1)).mean()
    reverse = -(torch.logsumexp(scaled.T.masked_fill(~positive.T, -torch.inf), 1) - torch.logsumexp(scaled.T, 1)).mean()
    return 0.5 * (forward + reverse)

print("VQA loss: cross_entropy для AI2D/span target; retrieval loss: multi-positive contrastive.")
print("ANLS/accuracy и Recall@K/MRR публикуются раздельно.")
'''

RUN = '''# Полный запуск выключен, чтобы просмотр notebook случайно не начал многодневное обучение.
RUN_TRAINING = False
SEEDS = [42]
commands = [
    [sys.executable, str(ROOT / "scripts/run_model_matrix.py"),
     "--dataset", DATASET, "--model", MODEL, "--split", "val",
     "--seed", str(seed), "--output-dir", str(ROOT / "runs/model_matrix"), "--resume"]
    for seed in SEEDS
]
for command in commands:
    print(subprocess.list2cmdline(command))
    if RUN_TRAINING:
        subprocess.run(command, cwd=ROOT, check=True)
'''


def architecture(model: str) -> str:
    if model == "random":
        return RANDOM_MODEL
    if model == "ocr_text":
        return OCR_MODEL
    if model in {"clip", "siglip"}:
        return VISION_MODEL
    if model in GRAPH_MODELS:
        return GRAPH_MODEL
    return QWEN_MODEL


def notebook(dataset: str, model: str, title: str) -> dict:
    header = f'''# {title} — {dataset}

Чистый учебный notebook: данные, разметка, входы модели и сама архитектура на PyTorch.

Он не заменяет выполняемый эксперимент в `notebooks/model_matrix`; полный runner остаётся единым источником истины. Ячейки тяжёлого обучения по умолчанию выключены.
'''
    constants = f'DATASET = "{dataset}"\nMODEL = "{model}"\nprint({{"dataset": DATASET, "model": MODEL}})\n'
    cells = [
        cell("markdown", header),
        cell("markdown", "## 1. Окружение"), cell("code", SETUP), cell("code", constants),
        cell("markdown", "## 2. Как выглядят данные"), cell("code", DATA),
        cell("markdown", "## 3. Изображение и OCR-разметка"), cell("code", VISUAL),
        cell("markdown", "## 4. Целевая разметка ответа"), cell("code", LABELS),
        cell("markdown", "## 5. Модель / baseline"), cell("code", architecture(model)),
    ]
    if model not in {"random", "ocr_text", "qwen25_vl_qlora"}:
        cells += [cell("markdown", "## 6. Функции потерь"), cell("code", LOSS)]
    cells += [
        cell("markdown", "## 7. Воспроизводимый полный запуск"), cell("code", RUN),
        cell("markdown", "## 8. Где сохраняются результаты\n\n"
             "`runs/model_matrix/<dataset>/<model>/seed<seed>/{val,test}`: config, history, predictions, metrics и лучший checkpoint/adapter."),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "model_matrix_clean": {"dataset": dataset, "model": model, "training_enabled": False},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 33 clean, readable PyTorch model notebooks.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "notebooks/model_matrix_clean")
    args = parser.parse_args()
    paths = []
    for dataset in DATASETS:
        folder = args.output_root / dataset
        folder.mkdir(parents=True, exist_ok=True)
        for prefix, model, title in MODELS:
            path = folder / f"{prefix}_{model}.ipynb"
            path.write_text(json.dumps(notebook(dataset, model, title), ensure_ascii=False, indent=1), encoding="utf-8")
            paths.append(str(path.resolve()))
    index = {
        "purpose": "Clean educational notebooks with data, labels, and visible PyTorch model code",
        "count": len(paths), "datasets": list(DATASETS), "models": [m for _, m, _ in MODELS],
        "notebooks": paths,
    }
    (args.output_root / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"count": len(paths), "output_root": str(args.output_root.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
