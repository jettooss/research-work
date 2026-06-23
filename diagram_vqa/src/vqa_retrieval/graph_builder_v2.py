from __future__ import annotations

import hashlib
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, GlobalAttention, global_mean_pool
from torchvision import transforms


@dataclass
class NodeV2:
    bbox: Tuple[int, int, int, int]
    kind: str  # "shape" | "text"
    text: str = ""
    conf: float = 0.0


EDGE_TYPE_TO_ID = {
    "knn": 0,
    "text_near_shape": 1,
    "contains": 2,
    "left_right": 3,
    "above_below": 4,
    "label_to_object": 5,
}


def bbox_center_v2(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def detect_shapes_opencv_v2(
    img_bgr: np.ndarray,
    min_area: int = 300,
    max_nodes: int = 80,
) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[Tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < min_area:
            continue
        boxes.append((x, y, x + w, y + h))

    boxes.sort(key=lambda bbox: (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), reverse=True)
    return boxes[:max_nodes]


def parse_ocr_v2_json(
    ocr_path: str | Path,
    level: str = "line",
    min_conf: float = 0.0,
) -> List[NodeV2]:
    ocr_path = Path(ocr_path)
    with ocr_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if level not in {"line", "word"}:
        raise ValueError(f"Unsupported OCR level: {level}")

    key = "lines" if level == "line" else "words"
    nodes: list[NodeV2] = []
    for item in payload.get(key, []):
        text = str(item.get("text", "")).strip()
        conf = float(item.get("conf", 0.0))
        bbox_raw = item.get("bbox", [])
        if not text or conf < min_conf or len(bbox_raw) != 4:
            continue
        bbox = tuple(int(v) for v in bbox_raw)
        nodes.append(NodeV2(bbox=bbox, kind="text", text=text, conf=conf))
    return nodes


def build_edges_knn_v2(nodes: List[NodeV2], k: int = 4) -> torch.Tensor:
    centers = np.array([bbox_center_v2(node.bbox) for node in nodes], dtype=np.float32)
    if len(centers) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    dists = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(dists, np.inf)

    edges: list[tuple[int, int]] = []
    for idx in range(len(nodes)):
        for nbr in np.argsort(dists[idx])[:k]:
            edges.append((idx, int(nbr)))
            edges.append((int(nbr), idx))

    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_typed_edges_v2(nodes: List[NodeV2], k: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    edges: list[tuple[int, int]] = []
    edge_types: list[int] = []

    def add_edge(src: int, dst: int, edge_type: str) -> None:
        if src == dst:
            return
        edges.append((src, dst))
        edge_types.append(EDGE_TYPE_TO_ID[edge_type])

    centers = [bbox_center_v2(node.bbox) for node in nodes]
    if nodes:
        center_array = np.array(centers, dtype=np.float32)
        dists = np.sqrt(((center_array[:, None, :] - center_array[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(dists, np.inf)
        for idx in range(len(nodes)):
            for nbr in np.argsort(dists[idx])[:k]:
                add_edge(idx, int(nbr), "knn")
                add_edge(int(nbr), idx, "knn")

    for left_idx, left in enumerate(nodes):
        for right_idx, right in enumerate(nodes):
            if left_idx == right_idx:
                continue
            relation = _spatial_relation_type(left.bbox, right.bbox)
            if relation is not None:
                add_edge(left_idx, right_idx, relation)
            if _bbox_contains(left.bbox, right.bbox):
                add_edge(left_idx, right_idx, "contains")

    text_indices = [idx for idx, node in enumerate(nodes) if node.kind == "text"]
    shape_indices = [idx for idx, node in enumerate(nodes) if node.kind == "shape"]
    for text_idx in text_indices:
        nearest_shape = _nearest_node_idx(text_idx, shape_indices, centers)
        if nearest_shape is not None:
            add_edge(text_idx, nearest_shape, "text_near_shape")
            add_edge(nearest_shape, text_idx, "text_near_shape")
        if _looks_like_label(nodes[text_idx].text):
            label_target = _nearest_node_idx(text_idx, shape_indices, centers)
            if label_target is not None:
                add_edge(text_idx, label_target, "label_to_object")

    if not edges:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous(), torch.tensor(edge_types, dtype=torch.long)


def _bbox_contains(outer: Tuple[int, int, int, int], inner: Tuple[int, int, int, int], margin: int = 2) -> bool:
    ox1, oy1, ox2, oy2 = outer
    ix1, iy1, ix2, iy2 = inner
    return ox1 - margin <= ix1 and oy1 - margin <= iy1 and ox2 + margin >= ix2 and oy2 + margin >= iy2


def _spatial_relation_type(
    left: Tuple[int, int, int, int],
    right: Tuple[int, int, int, int],
    min_gap: int = 8,
) -> Optional[str]:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    if lx2 + min_gap < rx1 or rx2 + min_gap < lx1:
        return "left_right"
    if ly2 + min_gap < ry1 or ry2 + min_gap < ly1:
        return "above_below"
    return None


def _nearest_node_idx(
    source_idx: int,
    candidate_indices: List[int],
    centers: List[Tuple[float, float]],
) -> Optional[int]:
    if not candidate_indices:
        return None
    sx, sy = centers[source_idx]
    return min(candidate_indices, key=lambda idx: (centers[idx][0] - sx) ** 2 + (centers[idx][1] - sy) ** 2)


def _looks_like_label(text: str) -> bool:
    value = str(text or "").strip()
    return bool(re.fullmatch(r"[A-Za-z0-9]", value))


class NodeFeaturizerV2:
    def __init__(
        self,
        device: str = "cpu",
        vision_model_name: str = "vit_base_patch16_224",
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
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

    def _geom_feat(self, bbox: Tuple[int, int, int, int], width: int, height: int) -> torch.Tensor:
        x1, y1, x2, y2 = bbox
        cx, cy = bbox_center_v2(bbox)
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        return torch.tensor([
            cx / width,
            cy / height,
            w / width,
            h / height,
            (w * h) / (width * height),
            x1 / width,
            y1 / height,
            x2 / width,
            y2 / height,
            max(0.0, min(1.0, float(w) / max(1.0, float(h)))),
            max(0.0, min(1.0, float(h) / max(1.0, float(w)))),
            1.0 if x1 <= 2 or y1 <= 2 else 0.0,
        ], dtype=torch.float32)

    def extract(self, pil_img: Image.Image, nodes: List[NodeV2]) -> torch.Tensor:
        width, height = pil_img.size
        feats = []
        for node in nodes:
            crop = pil_img.crop(node.bbox)
            visual = self._visual_emb(crop)
            text = self._text_emb(node.text if node.kind == "text" else "")
            geom = self._geom_feat(node.bbox, width, height)
            kind_flag = torch.tensor([1.0 if node.kind == "text" else 0.0], dtype=torch.float32)
            conf_feat = torch.tensor([max(0.0, min(1.0, node.conf / 100.0))], dtype=torch.float32)
            feats.append(torch.cat([visual, text, geom, kind_flag, conf_feat]))
        if not feats:
            return torch.zeros((1, self.vision_dim + self.text_dim + 12 + 1 + 1), dtype=torch.float32)
        return torch.stack(feats)


def build_graph_v2(
    image_path: str | Path,
    featurizer: NodeFeaturizerV2,
    ocr_path: Optional[str | Path] = None,
    ocr_level: str = "line",
    min_area: int = 300,
    max_shape_nodes: int = 80,
    min_text_conf: float = 35.0,
    knn_k: int = 4,
    include_shapes: bool = True,
) -> Data:
    image_path = str(Path(image_path))
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise RuntimeError(f"cv2.imread failed: {image_path}")

    pil_img = Image.open(image_path).convert("RGB")

    shape_nodes: list[NodeV2] = []
    if include_shapes:
        shape_nodes = [
            NodeV2(bbox=bbox, kind="shape")
            for bbox in detect_shapes_opencv_v2(img_bgr, min_area=min_area, max_nodes=max_shape_nodes)
        ]

    text_nodes: list[NodeV2] = []
    if ocr_path:
        text_nodes = parse_ocr_v2_json(ocr_path, level=ocr_level, min_conf=min_text_conf)

    nodes = shape_nodes + text_nodes
    edge_index, edge_type = build_typed_edges_v2(nodes, k=knn_k)
    x = featurizer.extract(pil_img, nodes)
    return Data(x=x, edge_index=edge_index, edge_type=edge_type)


class GraphEncoderV2(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        out_dim: int = 256,
        num_heads: int = 4,
        use_attn_pool: bool = True,
    ) -> None:
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

    def forward(self, batch) -> torch.Tensor:
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        x = F.gelu(self.proj(x))
        x = F.gelu(self.gnn1(x, edge_index))
        x = F.gelu(self.gnn2(x, edge_index))
        x = F.gelu(self.out(x))
        return self.pool(x, batch_idx) if self.use_attn_pool else global_mean_pool(x, batch_idx)


class FeatureCacheV2:
    def __init__(self, cache_dir: str | Path, signature: str, enabled: bool = True) -> None:
        self.enabled = enabled
        self.signature = str(signature)
        self.root = Path(cache_dir)
        self.graph_dir = self.root / self.signature / "graphs"
        self.text_dir = self.root / self.signature / "texts"
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self.text_dir.mkdir(parents=True, exist_ok=True)
        self._mem_graph: Dict[str, Data] = {}
        self._mem_text: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _sha1(value: str) -> str:
        return hashlib.sha1(value.encode("utf-8")).hexdigest()

    @staticmethod
    def make_signature(
        vision_model_name: str,
        text_model_name: str,
        ocr_level: str,
        min_area: int,
        max_shape_nodes: int,
        min_text_conf: float,
        knn_k: int,
        include_shapes: bool,
    ) -> str:
        payload = {
            "vision_model_name": vision_model_name,
            "text_model_name": text_model_name,
            "ocr_level": ocr_level,
            "min_area": int(min_area),
            "max_shape_nodes": int(max_shape_nodes),
            "min_text_conf": float(min_text_conf),
            "knn_k": int(knn_k),
            "include_shapes": bool(include_shapes),
            "cache_version": 2,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def _load_pt(self, path: Path):
        if "weights_only" in inspect.signature(torch.load).parameters:
            return torch.load(path, map_location="cpu", weights_only=False)
        return torch.load(path, map_location="cpu")

    def get_graph(
        self,
        image_path: str | Path,
        featurizer: NodeFeaturizerV2,
        ocr_path: Optional[str | Path] = None,
        ocr_level: str = "line",
        min_area: int = 300,
        max_shape_nodes: int = 80,
        min_text_conf: float = 35.0,
        knn_k: int = 4,
        include_shapes: bool = True,
    ) -> Data:
        if not self.enabled:
            return build_graph_v2(
                image_path=image_path,
                featurizer=featurizer,
                ocr_path=ocr_path,
                ocr_level=ocr_level,
                min_area=min_area,
                max_shape_nodes=max_shape_nodes,
                min_text_conf=min_text_conf,
                knn_k=knn_k,
                include_shapes=include_shapes,
            )

        image_path = Path(image_path)
        ocr_file = Path(ocr_path) if ocr_path else None

        try:
            image_mtime = image_path.stat().st_mtime_ns
            image_resolved = str(image_path.resolve())
        except FileNotFoundError:
            image_mtime = 0
            image_resolved = str(image_path)

        if ocr_file is not None:
            try:
                ocr_mtime = ocr_file.stat().st_mtime_ns
                ocr_resolved = str(ocr_file.resolve())
            except FileNotFoundError:
                ocr_mtime = 0
                ocr_resolved = str(ocr_file)
        else:
            ocr_mtime = 0
            ocr_resolved = ""

        key = self._sha1(
            "|".join([
                image_resolved,
                str(image_mtime),
                ocr_resolved,
                str(ocr_mtime),
                ocr_level,
                str(min_area),
                str(max_shape_nodes),
                str(min_text_conf),
                str(knn_k),
                str(include_shapes),
                self.signature,
            ])
        )

        if key in self._mem_graph:
            return self._mem_graph[key]

        fpath = self.graph_dir / f"{key}.pt"
        if fpath.exists():
            graph = self._load_pt(fpath)
        else:
            graph = build_graph_v2(
                image_path=image_path,
                featurizer=featurizer,
                ocr_path=ocr_path,
                ocr_level=ocr_level,
                min_area=min_area,
                max_shape_nodes=max_shape_nodes,
                min_text_conf=min_text_conf,
                knn_k=knn_k,
                include_shapes=include_shapes,
            )
            torch.save(graph, fpath)

        self._mem_graph[key] = graph
        return graph

    def get_text_batch(
        self,
        texts: List[str],
        text_encoder,
        normalize: bool = False,
    ) -> torch.Tensor:
        if not self.enabled:
            with torch.no_grad():
                embs = text_encoder.encode(texts, convert_to_tensor=True, normalize_embeddings=normalize)
            return embs.detach().cpu() if isinstance(embs, torch.Tensor) else torch.tensor(embs)

        order_keys: list[str] = []
        missing_texts: list[str] = []
        missing_keys: list[str] = []

        for text in texts:
            text = text or ""
            key = self._sha1(f"{text}|{self.signature}")
            order_keys.append(key)
            if key in self._mem_text:
                continue
            fpath = self.text_dir / f"{key}.pt"
            if fpath.exists():
                self._mem_text[key] = self._load_pt(fpath)
            else:
                missing_texts.append(text)
                missing_keys.append(key)

        if missing_texts:
            with torch.no_grad():
                embs = text_encoder.encode(missing_texts, convert_to_tensor=True, normalize_embeddings=normalize)
            embs = embs.detach().cpu() if isinstance(embs, torch.Tensor) else torch.tensor(embs)
            for key, emb in zip(missing_keys, embs):
                emb = emb.contiguous()
                self._mem_text[key] = emb
                torch.save(emb, self.text_dir / f"{key}.pt")

        return torch.stack([self._mem_text[key] for key in order_keys])


def resolve_ocr_v2_path(
    image_path: str | Path,
    image_root: str | Path,
    ocr_root: str | Path,
    suffix: str = ".ocr.json",
) -> Path:
    image_path = Path(image_path)
    image_root = Path(image_root)
    ocr_root = Path(ocr_root)
    relative = image_path.relative_to(image_root)
    return ocr_root / relative.with_suffix(suffix)
