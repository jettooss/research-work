from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import cv2
import torch
from PIL import Image
from torch_geometric.data import Data

from .graph_builder import Node, detect_shapes_opencv, parse_azure_ocr


class CompactDocVqaGraphCache:
    """OCR/text/geometry-only view of the legacy DocVQA feature cache.

    Existing legacy graphs are converted without recomputing embeddings. Cache
    misses use batched text encoding and deliberately skip ViT crop features,
    which are zeroed in every controlled DocVQA ablation anyway.
    """

    def __init__(self, root: str | Path, legacy_root: str | Path, legacy_signature: str, text_encoder) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.legacy_graphs = Path(legacy_root) / legacy_signature / "graphs"
        self.legacy_signature = legacy_signature
        self.text_encoder = text_encoder
        self._memory: dict[str, Data] = {}

    @staticmethod
    def _load(path: Path) -> Data:
        kwargs = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(torch.load).parameters:
            kwargs["weights_only"] = False
        return torch.load(path, **kwargs)

    @staticmethod
    def _compact_key(image_path: str | Path) -> str:
        return hashlib.sha1(str(Path(image_path).resolve()).encode("utf-8")).hexdigest()

    def _legacy_path(self, image_path: str | Path, ocr_path: str | Path) -> Path:
        image = Path(image_path)
        raw = (
            f"{image.resolve()}|{image.stat().st_mtime_ns}|{ocr_path}|auto|"
            f"eng|300|80|60|4|{self.legacy_signature}"
        )
        return self.legacy_graphs / f"{hashlib.sha1(raw.encode('utf-8')).hexdigest()}.pt"

    def get(self, image_path: str | Path, ocr_path: str | Path) -> Data:
        key = self._compact_key(image_path)
        if key in self._memory:
            return self._memory[key]
        compact_path = self.root / f"{key}.pt"
        if compact_path.exists():
            graph = self._load(compact_path)
        else:
            legacy_path = self._legacy_path(image_path, ocr_path)
            if legacy_path.exists():
                legacy = self._load(legacy_path)
                graph = Data(x=legacy.x[:, 768:].to(torch.float16).contiguous())
            else:
                graph = self._build(image_path, ocr_path)
            torch.save(graph, compact_path)
        graph = Data(x=graph.x.float())
        if len(self._memory) >= 128:
            self._memory.pop(next(iter(self._memory)))
        self._memory[key] = graph
        return graph

    def _build(self, image_path: str | Path, ocr_path: str | Path) -> Data:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise RuntimeError(f"cv2.imread failed: {image_path}")
        width, height = Image.open(image_path).size
        shapes = [Node(bbox=box, kind="shape") for box in detect_shapes_opencv(image_bgr, min_area=300, max_nodes=80)]
        text_nodes = parse_azure_ocr(str(ocr_path))
        nodes = shapes + text_nodes
        text_values = [node.text for node in nodes if node.kind == "text"]
        if text_values:
            encoded = self.text_encoder.encode(text_values, batch_size=128, convert_to_tensor=True, normalize_embeddings=False)
            encoded = encoded.detach().cpu().float()
        else:
            encoded = torch.empty((0, 384), dtype=torch.float32)
        text_index = 0
        features = []
        for node in nodes:
            if node.kind == "text":
                text = encoded[text_index]
                text_index += 1
            else:
                text = torch.zeros(384, dtype=torch.float32)
            x1, y1, x2, y2 = node.bbox
            node_width, node_height = max(1, x2 - x1), max(1, y2 - y1)
            geom = torch.tensor(
                [
                    (x1 + x2) / 2 / width,
                    (y1 + y2) / 2 / height,
                    node_width / width,
                    node_height / height,
                    (node_width * node_height) / (width * height),
                    x1 / width,
                    y1 / height,
                    x2 / width,
                    y2 / height,
                ],
                dtype=torch.float32,
            )
            kind = torch.tensor([1.0 if node.kind == "text" else 0.0])
            features.append(torch.cat([text, geom, kind]))
        if not features:
            features = [torch.zeros(394, dtype=torch.float32)]
        return Data(x=torch.stack(features).to(torch.float16))
