"""
Graph building utilities extracted from i2d_gnn_encoder_ocr_knn.ipynb.

Supports two OCR sources:
  - Azure OCR JSON (pre-computed, used for DocVQA)
  - Tesseract (fallback for AI2D, InfographicVQA, etc.)
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, GlobalAttention, global_mean_pool
from torchvision import transforms
import timm
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Node / geometry helpers
# ---------------------------------------------------------------------------

@dataclass
class Node:
    bbox: Tuple[int, int, int, int]
    kind: str  # "shape" | "text"
    text: str = ""


def bbox_center(b: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


# ---------------------------------------------------------------------------
# OCR parsing
# ---------------------------------------------------------------------------

def parse_azure_ocr(ocr_path: str) -> List[Node]:
    """Parse a DocVQA-style Azure OCR JSON file into Node objects.

    The bounding box format is 8 coordinates (4 corners, clockwise):
        [x1,y1, x2,y2, x3,y3, x4,y4]
    We convert to axis-aligned bbox: (min_x, min_y, max_x, max_y).
    """
    with open(ocr_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes: List[Node] = []
    for page in data.get("recognitionResults", []):
        for line in page.get("lines", []):
            for word in line.get("words", []):
                text = (word.get("text") or "").strip()
                if not text:
                    continue
                bb = word.get("boundingBox", [])
                if len(bb) < 8:
                    continue
                xs = bb[0::2]
                ys = bb[1::2]
                x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
                nodes.append(Node(bbox=(x1, y1, x2, y2), kind="text", text=text))
    return nodes


def detect_shapes_opencv(
    img_bgr: np.ndarray,
    min_area: int = 300,
    max_nodes: int = 80,
) -> List[Tuple[int, int, int, int]]:
    """Detect bounding boxes of visual objects via connected components."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 31, 7,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area:
            continue
        boxes.append((x, y, x + w, y + h))

    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return boxes[:max_nodes]


def detect_text_ocr(
    img_bgr: np.ndarray,
    lang: str = "eng",
    conf_th: int = 60,
) -> List[Node]:
    """Run Tesseract OCR and return text nodes."""
    try:
        import pytesseract
    except ImportError as e:
        raise ImportError("Install pytesseract (pip) and tesseract binary.") from e

    # Set Tesseract path on Windows if not already configured
    import os
    if os.name == "nt" and pytesseract.pytesseract.tesseract_cmd == "tesseract":
        default_win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(default_win_path):
            pytesseract.pytesseract.tesseract_cmd = default_win_path

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_data(rgb, lang=lang, output_type=pytesseract.Output.DICT)

    nodes: List[Node] = []
    for i in range(len(data["text"])):
        text = (data["text"][i] or "").strip()
        conf = int(float(data["conf"][i])) if data["conf"][i] != "-1" else -1
        if conf < conf_th or not text:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        nodes.append(Node(bbox=(x, y, x + w, y + h), kind="text", text=text))
    return nodes


# ---------------------------------------------------------------------------
# Graph edges
# ---------------------------------------------------------------------------

def build_edges_knn(nodes: List[Node], k: int = 4) -> torch.Tensor:
    """Build an undirected kNN graph over node centers."""
    centers = np.array([bbox_center(n.bbox) for n in nodes], dtype=np.float32)
    if len(centers) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    dists = np.sqrt(((centers[:, None] - centers[None]) ** 2).sum(-1))
    np.fill_diagonal(dists, np.inf)

    edges = []
    for i in range(len(nodes)):
        for j in np.argsort(dists[i])[:k]:
            edges.append((i, int(j)))
            edges.append((int(j), i))

    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


# ---------------------------------------------------------------------------
# Node featurizer
# ---------------------------------------------------------------------------

class NodeFeaturizer:
    """Extracts per-node features: vision crop + text embedding + geometry."""

    def __init__(
        self,
        device: str = "cpu",
        vision_model_name: str = "vit_base_patch16_224",
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.device = torch.device(device)
        self.vision_model_name = vision_model_name
        self.text_model_name = text_model_name

        self.vision = timm.create_model(vision_model_name, pretrained=True, num_classes=0)
        self.vision.eval().to(self.device)
        self.vision_dim = self.vision.num_features

        self.text_enc = SentenceTransformer(text_model_name, device=str(self.device))
        self.text_dim = self.text_enc.get_sentence_embedding_dimension()

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    @torch.no_grad()
    def _visual_emb(self, pil_img: Image.Image) -> torch.Tensor:
        x = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        return self.vision(x).squeeze(0).detach().cpu()

    @torch.no_grad()
    def _text_emb(self, text: str) -> torch.Tensor:
        if not text:
            return torch.zeros(self.text_dim, dtype=torch.float32)
        emb = self.text_enc.encode([text], convert_to_tensor=True, normalize_embeddings=False)
        return emb.squeeze(0).detach().cpu()

    def _geom_feat(self, bbox: Tuple[int, int, int, int], W: int, H: int) -> torch.Tensor:
        x1, y1, x2, y2 = bbox
        cx, cy = bbox_center(bbox)
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        return torch.tensor([
            cx / W, cy / H, w / W, h / H, (w * h) / (W * H),
            x1 / W, y1 / H, x2 / W, y2 / H,
        ], dtype=torch.float32)

    def extract(self, pil_img: Image.Image, nodes: List[Node]) -> torch.Tensor:
        W, H = pil_img.size
        feats = []
        for n in nodes:
            crop = pil_img.crop(n.bbox)
            v = self._visual_emb(crop)
            t = self._text_emb(n.text if n.kind == "text" else "")
            g = self._geom_feat(n.bbox, W, H)
            kind_flag = torch.tensor([1.0 if n.kind == "text" else 0.0])
            feats.append(torch.cat([v, t, g, kind_flag]))
        if not feats:
            return torch.zeros((1, self.vision_dim + self.text_dim + 9 + 1))
        return torch.stack(feats)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(
    image_path: str,
    featurizer: NodeFeaturizer,
    ocr_path: Optional[str] = None,
    ocr_source: str = "auto",   # "auto" | "azure" | "tesseract" | "none"
    ocr_lang: str = "eng",
    min_area: int = 300,
    max_nodes: int = 80,
    ocr_conf: int = 60,
    knn_k: int = 4,
) -> Data:
    """Build a PyG Data graph from a document/diagram image.

    ocr_source:
      "auto"      — use Azure OCR if ocr_path given, else Tesseract
      "azure"     — use Azure OCR (requires ocr_path)
      "tesseract" — always run Tesseract
      "none"      — shape nodes only, no OCR
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise RuntimeError(f"cv2.imread failed: {image_path}")
    pil_img = Image.open(image_path).convert("RGB")

    shapes = detect_shapes_opencv(img_bgr, min_area=min_area, max_nodes=max_nodes)
    shape_nodes = [Node(bbox=b, kind="shape") for b in shapes]

    effective_source = ocr_source
    if ocr_source == "auto":
        effective_source = "azure" if ocr_path else "tesseract"

    if effective_source == "azure":
        if not ocr_path:
            raise ValueError("ocr_source='azure' requires ocr_path")
        text_nodes = parse_azure_ocr(ocr_path)
    elif effective_source == "tesseract":
        text_nodes = detect_text_ocr(img_bgr, lang=ocr_lang, conf_th=ocr_conf)
    else:  # "none"
        text_nodes = []

    nodes = shape_nodes + text_nodes
    edge_index = build_edges_knn(nodes, k=knn_k)
    x = featurizer.extract(pil_img, nodes)
    return Data(x=x, edge_index=edge_index)


# ---------------------------------------------------------------------------
# Graph encoder
# ---------------------------------------------------------------------------

class GraphEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        out_dim: int = 256,
        num_heads: int = 4,
        use_attn_pool: bool = True,
    ):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.gnn1 = GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=0.1)
        self.gnn2 = GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=0.1)
        self.out = nn.Linear(hidden_dim, out_dim)
        self.use_attn_pool = use_attn_pool
        gate_nn = nn.Sequential(
            nn.Linear(out_dim, max(1, out_dim // 2)),
            nn.GELU(),
            nn.Linear(max(1, out_dim // 2), 1),
        )
        self.pool = GlobalAttention(gate_nn=gate_nn)

    def forward(self, batch):
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        x = F.gelu(self.proj(x))
        x = F.gelu(self.gnn1(x, edge_index))
        x = F.gelu(self.gnn2(x, edge_index))
        x = F.gelu(self.out(x))
        return self.pool(x, batch_idx) if self.use_attn_pool else global_mean_pool(x, batch_idx)


# ---------------------------------------------------------------------------
# Feature cache (graph + text)
# ---------------------------------------------------------------------------

class FeatureCache:
    def __init__(
        self,
        cache_dir: str | Path,
        signature: str,
        enabled: bool = True,
        max_mem_graphs: int = 2048,
        max_mem_texts: int = 8192,
    ) -> None:
        self.enabled = enabled
        self.signature = str(signature)
        self.max_mem_graphs = int(max_mem_graphs)
        self.max_mem_texts = int(max_mem_texts)
        self.root = Path(cache_dir)
        self.graph_dir = self.root / self.signature / "graphs"
        self.text_dir = self.root / self.signature / "texts"
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self.text_dir.mkdir(parents=True, exist_ok=True)
        self._mem_graph: OrderedDict[str, Data] = OrderedDict()
        self._mem_text: OrderedDict[str, torch.Tensor] = OrderedDict()

    @staticmethod
    def _remember(cache: OrderedDict, key: str, value, max_items: int) -> None:
        if max_items <= 0:
            return
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > max_items:
            cache.popitem(last=False)

    @staticmethod
    def _sha1(s: str) -> str:
        return hashlib.sha1(s.encode("utf-8")).hexdigest()

    @staticmethod
    def make_signature(
        vision_model_name: str,
        text_model_name: str,
        ocr_source: str,
        ocr_lang: str,
        min_area: int,
        max_nodes: int,
        ocr_conf: int,
        knn_k: int,
    ) -> str:
        payload = {
            "vision_model_name": vision_model_name,
            "text_model_name": text_model_name,
            "ocr_source": ocr_source,
            "ocr_lang": ocr_lang,
            "min_area": int(min_area),
            "max_nodes": int(max_nodes),
            "ocr_conf": int(ocr_conf),
            "knn_k": int(knn_k),
            "cache_version": 2,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def _load_pt(self, path: Path):
        import inspect
        if "weights_only" in inspect.signature(torch.load).parameters:
            return torch.load(path, map_location="cpu", weights_only=False)
        return torch.load(path, map_location="cpu")

    def get_graph(
        self,
        image_path: str,
        featurizer: NodeFeaturizer,
        ocr_path: Optional[str] = None,
        ocr_source: str = "auto",
        ocr_lang: str = "eng",
        min_area: int = 300,
        max_nodes: int = 80,
        ocr_conf: int = 60,
        knn_k: int = 4,
    ) -> Data:
        if not self.enabled:
            return build_graph(image_path, featurizer, ocr_path=ocr_path,
                               ocr_source=ocr_source, ocr_lang=ocr_lang,
                               min_area=min_area, max_nodes=max_nodes,
                               ocr_conf=ocr_conf, knn_k=knn_k)

        p = Path(image_path)
        try:
            mtime = p.stat().st_mtime_ns
            resolved = str(p.resolve())
        except FileNotFoundError:
            mtime, resolved = 0, str(p)

        key = self._sha1(
            f"{resolved}|{mtime}|{ocr_path or ''}|{ocr_source}|"
            f"{ocr_lang}|{min_area}|{max_nodes}|{ocr_conf}|{knn_k}|{self.signature}"
        )

        if key in self._mem_graph:
            self._mem_graph.move_to_end(key)
            return self._mem_graph[key]

        fpath = self.graph_dir / f"{key}.pt"
        if fpath.exists():
            try:
                g = self._load_pt(fpath)
            except (EOFError, OSError, RuntimeError):
                fpath.unlink(missing_ok=True)
                g = build_graph(image_path, featurizer, ocr_path=ocr_path,
                                ocr_source=ocr_source, ocr_lang=ocr_lang,
                                min_area=min_area, max_nodes=max_nodes,
                                ocr_conf=ocr_conf, knn_k=knn_k)
                torch.save(g, fpath)
        else:
            g = build_graph(image_path, featurizer, ocr_path=ocr_path,
                            ocr_source=ocr_source, ocr_lang=ocr_lang,
                            min_area=min_area, max_nodes=max_nodes,
                            ocr_conf=ocr_conf, knn_k=knn_k)
            torch.save(g, fpath)

        self._remember(self._mem_graph, key, g, self.max_mem_graphs)
        return g

    def get_text_batch(
        self,
        texts: List[str],
        text_encoder,
        normalize: bool = False,
    ) -> torch.Tensor:
        if not self.enabled:
            with torch.no_grad():
                embs = text_encoder.encode(texts, convert_to_tensor=True,
                                           normalize_embeddings=normalize)
            return embs.detach().cpu() if isinstance(embs, torch.Tensor) else torch.tensor(embs)

        order_keys: List[str] = []
        missing_texts: List[str] = []
        missing_keys: List[str] = []

        for t in texts:
            t = t or ""
            key = self._sha1(f"{t}|{self.signature}")
            order_keys.append(key)
            if key in self._mem_text:
                self._mem_text.move_to_end(key)
                continue
            fpath = self.text_dir / f"{key}.pt"
            if fpath.exists():
                self._remember(self._mem_text, key, self._load_pt(fpath), self.max_mem_texts)
            else:
                missing_texts.append(t)
                missing_keys.append(key)

        if missing_texts:
            with torch.no_grad():
                embs = text_encoder.encode(missing_texts, convert_to_tensor=True,
                                           normalize_embeddings=normalize)
            embs = embs.detach().cpu() if isinstance(embs, torch.Tensor) else torch.tensor(embs)
            for k, e in zip(missing_keys, embs):
                e = e.contiguous()
                self._remember(self._mem_text, k, e, self.max_mem_texts)
                torch.save(e, self.text_dir / f"{k}.pt")

        out = []
        for key in order_keys:
            if key in self._mem_text:
                self._mem_text.move_to_end(key)
                out.append(self._mem_text[key])
            else:
                value = self._load_pt(self.text_dir / f"{key}.pt")
                self._remember(self._mem_text, key, value, self.max_mem_texts)
                out.append(value)
        return torch.stack(out)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def contrastive_loss(z_img: torch.Tensor, z_txt: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    logits = (z_img @ z_txt.t()) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2.0


@torch.no_grad()
def recall_at_k(sim: torch.Tensor, ks=(1, 5, 10)) -> Dict[int, float]:
    """Recall@K where ground truth is diagonal (pair i ↔ i)."""
    ranks = sim.argsort(dim=1, descending=True)
    gt = torch.arange(sim.size(0), device=sim.device).unsqueeze(1)
    return {k: (ranks[:, :k] == gt).any(dim=1).float().mean().item() for k in ks}
