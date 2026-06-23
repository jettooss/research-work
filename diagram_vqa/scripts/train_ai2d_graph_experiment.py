from __future__ import annotations

import argparse
import json
import os
import random
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, GPSConv, TransformerConv, global_mean_pool
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runs" / "ultralytics_config"))

from vqa_retrieval.ai2d_hybrid import (  # noqa: E402
    Ai2dHybridDataset,
    compute_option_logits,
    contrastive_loss,
    load_manifest_hybrid,
    load_split_payload,
    make_hybrid_collate_fn,
    retrieval_metrics_from_embeddings,
    resolve_sample_file_paths,
    select_samples_for_split,
)
from vqa_retrieval.graph_builder_v2 import (  # noqa: E402
    EDGE_TYPE_TO_ID,
    FeatureCacheV2,
    GraphEncoderV2,
    NodeFeaturizerV2,
    NodeV2,
    build_typed_edges_v2,
    detect_shapes_opencv_v2,
    parse_ocr_v2_json,
)


@dataclass(frozen=True)
class SamSegment:
    bbox: tuple[int, int, int, int]
    area: int
    predicted_iou: float = 0.0
    stability_score: float = 0.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_limit_samples(samples: list, limit: Optional[int], seed: int) -> list:
    if limit is None or limit <= 0 or len(samples) <= limit:
        return samples
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    return [samples[i] for i in sorted(indices[:limit])]


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class ProgressJsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def write(self, event: str, **payload: Any) -> None:
        row = {"event": event, **payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, ax2 - ax1) * max(1, ay2 - ay1)
    area_b = max(1, bx2 - bx1) * max(1, by2 - by1)
    return float(inter / max(1, area_a + area_b - inter))


def dedupe_segments(segments: list[SamSegment], iou_threshold: float, max_segments: int) -> list[SamSegment]:
    ordered = sorted(segments, key=lambda item: (item.predicted_iou, item.stability_score, item.area), reverse=True)
    kept: list[SamSegment] = []
    for segment in ordered:
        if all(bbox_iou(segment.bbox, prev.bbox) < iou_threshold for prev in kept):
            kept.append(segment)
        if len(kept) >= max_segments:
            break
    return kept


def opencv_segments(
    image_path: str | Path,
    min_area: int,
    max_segments: int,
) -> list[SamSegment]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"cv2.imread failed: {image_path}")
    segments = [
        SamSegment(
            bbox=tuple(int(v) for v in bbox),
            area=int(max(1, bbox[2] - bbox[0]) * max(1, bbox[3] - bbox[1])),
            predicted_iou=0.0,
            stability_score=0.0,
        )
        for bbox in detect_shapes_opencv_v2(image, min_area=min_area, max_nodes=max_segments)
    ]
    return segments


def mask_to_bbox(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


_SAM3_PREDICTOR_CACHE: dict[tuple[str, float, bool], Any] = {}
_SAM2_MODEL_CACHE: dict[tuple[str, bool], Any] = {}


def _get_ultralytics_sam2_model(model_name: str | Path, half: bool):
    model_value = str(model_name)
    key = (model_value, bool(half))
    if key in _SAM2_MODEL_CACHE:
        return _SAM2_MODEL_CACHE[key]
    try:
        from ultralytics import SAM
    except ImportError as exc:
        raise ImportError("Ultralytics SAM2 needs ultralytics. Install with: pip install -U ultralytics") from exc
    model = SAM(model_value)
    _SAM2_MODEL_CACHE[key] = model
    return model


def _get_ultralytics_sam3_predictor(model_path: Path, conf: float, half: bool):
    model_value = str(model_path)
    key = (model_value, float(conf), bool(half))
    if key in _SAM3_PREDICTOR_CACHE:
        return _SAM3_PREDICTOR_CACHE[key]
    try:
        from ultralytics.models.sam import SAM3SemanticPredictor
    except ImportError as exc:
        raise ImportError(
            "Ultralytics SAM 3 needs ultralytics>=8.3.237. Install with: pip install -U ultralytics"
        ) from exc
    overrides = dict(
        conf=float(conf),
        task="segment",
        mode="predict",
        model=model_value,
        half=bool(half),
        save=False,
        verbose=False,
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)
    _SAM3_PREDICTOR_CACHE[key] = predictor
    return predictor


def _segments_from_ultralytics_results(results: Any, min_area: int, max_segments: int) -> list[SamSegment]:
    if results is None:
        return []
    if not isinstance(results, (list, tuple)):
        results = [results]
    out: list[SamSegment] = []
    for result in results:
        boxes_obj = getattr(result, "boxes", None)
        masks_obj = getattr(result, "masks", None)
        boxes_xyxy = None
        confs = None
        if boxes_obj is not None and getattr(boxes_obj, "xyxy", None) is not None:
            boxes_xyxy = boxes_obj.xyxy.detach().cpu().numpy()
            if getattr(boxes_obj, "conf", None) is not None:
                confs = boxes_obj.conf.detach().cpu().numpy()
        masks = None
        if masks_obj is not None and getattr(masks_obj, "data", None) is not None:
            masks = masks_obj.data.detach().cpu().numpy()

        n = 0
        if boxes_xyxy is not None:
            n = len(boxes_xyxy)
        elif masks is not None:
            n = len(masks)

        for idx in range(n):
            if boxes_xyxy is not None:
                x1, y1, x2, y2 = [int(round(float(v))) for v in boxes_xyxy[idx].tolist()]
                bbox = (x1, y1, x2, y2)
            elif masks is not None:
                bbox = mask_to_bbox(masks[idx])
                if bbox is None:
                    continue
            else:
                continue
            area = int(max(1, bbox[2] - bbox[0]) * max(1, bbox[3] - bbox[1]))
            if masks is not None and idx < len(masks):
                area = int(max(area, float((masks[idx] > 0).sum())))
            if area < min_area:
                continue
            score = float(confs[idx]) if confs is not None and idx < len(confs) else 0.0
            out.append(SamSegment(bbox=bbox, area=area, predicted_iou=score, stability_score=score))
    return sorted(out, key=lambda item: (item.predicted_iou, item.area), reverse=True)[:max_segments]


def sam_segments(
    image_path: str | Path,
    backend: str,
    checkpoint: Optional[Path],
    device: str,
    min_area: int,
    max_segments: int,
    dedupe_iou: float,
    sam3_model: Optional[Path] = None,
    sam3_text_prompts: tuple[str, ...] = ("diagram",),
    sam3_conf: float = 0.25,
    sam3_half: bool = True,
    sam3_allow_download: bool = True,
    sam2_model: str | Path = EXTERNAL_ROOT / "models" / "sam2" / "sam2.1_b.pt",
    sam2_conf: float = 0.25,
    sam2_imgsz: int = 1024,
    sam2_half: bool = True,
) -> tuple[list[SamSegment], str]:
    if backend == "opencv_fallback":
        return opencv_segments(image_path, min_area=min_area, max_segments=max_segments), backend

    if backend in {"ultralytics_sam3", "sam3"}:
        model_path = Path(sam3_model or checkpoint or "sam3.pt")
        if not model_path.exists():
            if sam3_allow_download:
                print(f"[SAM3] Missing local model={model_path}; asking Ultralytics to resolve sam3.pt.")
                model_path = Path("sam3.pt")
            else:
                print(f"[SAM3] Missing model={model_path}; using OpenCV fallback.")
                return opencv_segments(image_path, min_area=min_area, max_segments=max_segments), "opencv_fallback"
        try:
            predictor = _get_ultralytics_sam3_predictor(model_path, conf=sam3_conf, half=sam3_half)
            predictor.set_image(str(image_path))
            prompts = [str(x).strip() for x in sam3_text_prompts if str(x).strip()] or ["diagram"]
            results = predictor(text=prompts)
            out = _segments_from_ultralytics_results(results, min_area=min_area, max_segments=max_segments)
            return dedupe_segments(out, iou_threshold=dedupe_iou, max_segments=max_segments), "ultralytics_sam3"
        except Exception as exc:
            print(f"[SAM3] Ultralytics SAM 3 failed ({type(exc).__name__}: {exc}); using OpenCV fallback.")
            return opencv_segments(image_path, min_area=min_area, max_segments=max_segments), "opencv_fallback"

    if backend.startswith("sam_"):
        if checkpoint is None or not checkpoint.exists():
            print(f"[SAM] Missing checkpoint={checkpoint}; using OpenCV fallback.")
            return opencv_segments(image_path, min_area=min_area, max_segments=max_segments), "opencv_fallback"
        try:
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        except ImportError:
            print("[SAM] segment-anything is not installed; using OpenCV fallback.")
            return opencv_segments(image_path, min_area=min_area, max_segments=max_segments), "opencv_fallback"

        model_type = backend.replace("sam_", "")
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
        sam.to(device=device)
        generator = SamAutomaticMaskGenerator(sam)
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise RuntimeError(f"cv2.imread failed: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        masks = generator.generate(image_rgb)
        out: list[SamSegment] = []
        for mask in masks:
            bbox = mask_to_bbox(mask.get("segmentation"))
            if bbox is None:
                continue
            area = int(mask.get("area", max(1, bbox[2] - bbox[0]) * max(1, bbox[3] - bbox[1])))
            if area < min_area:
                continue
            out.append(
                SamSegment(
                    bbox=bbox,
                    area=area,
                    predicted_iou=float(mask.get("predicted_iou", 0.0)),
                    stability_score=float(mask.get("stability_score", 0.0)),
                )
            )
        return dedupe_segments(out, iou_threshold=dedupe_iou, max_segments=max_segments), backend

    if backend in {"ultralytics_sam2", "sam2", "sam2.1"}:
        try:
            model = _get_ultralytics_sam2_model(sam2_model, half=sam2_half)
            results = model(
                str(image_path),
                conf=float(sam2_conf),
                imgsz=int(sam2_imgsz),
                retina_masks=True,
                verbose=False,
            )
            out = _segments_from_ultralytics_results(results, min_area=min_area, max_segments=max_segments)
            return dedupe_segments(out, iou_threshold=dedupe_iou, max_segments=max_segments), "ultralytics_sam2"
        except Exception as exc:
            print(f"[SAM2] Ultralytics SAM2 failed ({type(exc).__name__}: {exc}); using OpenCV fallback.")
            return opencv_segments(image_path, min_area=min_area, max_segments=max_segments), "opencv_fallback"

    raise ValueError(f"Unsupported SAM backend: {backend}")


def sam_cache_path(cache_dir: Path, image_id: str) -> Path:
    return cache_dir / f"{image_id}.sam_segments.json"


def load_sam_cache(cache_dir: Path, image_id: str, image_path: str | Path) -> Optional[list[SamSegment]]:
    path = sam_cache_path(cache_dir, image_id)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        if int(payload.get("image_mtime_ns", -1)) != int(Path(image_path).stat().st_mtime_ns):
            return None
    except FileNotFoundError:
        return None
    return [
        SamSegment(
            bbox=tuple(int(v) for v in item["bbox"]),
            area=int(item.get("area", 0)),
            predicted_iou=float(item.get("predicted_iou", 0.0)),
            stability_score=float(item.get("stability_score", 0.0)),
        )
        for item in payload.get("segments", [])
        if len(item.get("bbox", [])) == 4
    ]


def save_sam_cache(
    cache_dir: Path,
    image_id: str,
    image_path: str | Path,
    backend: str,
    checkpoint: Optional[Path],
    segments: list[SamSegment],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_path": str(image_path),
        "image_id": str(image_id),
        "sam_backend": backend,
        "checkpoint": str(checkpoint) if checkpoint else "",
        "image_mtime_ns": int(Path(image_path).stat().st_mtime_ns),
        "segments": [
            {
                "bbox": list(segment.bbox),
                "area": int(segment.area),
                "predicted_iou": float(segment.predicted_iou),
                "stability_score": float(segment.stability_score),
            }
            for segment in segments
        ],
    }
    sam_cache_path(cache_dir, image_id).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_or_load_sam_segments(sample, args) -> list[SamSegment]:
    cached = load_sam_cache(args.sam_cache_dir, sample.image_id, sample.image_path)
    if cached is not None:
        return cached

    if args.sam_cache_missing == "skip":
        segments = opencv_segments(sample.image_path, min_area=args.extract_min_area, max_segments=args.extract_max_nodes)
        return segments

    segments, used_backend = sam_segments(
        image_path=sample.image_path,
        backend=args.sam_backend,
        checkpoint=args.sam_checkpoint,
        device=args.device,
        min_area=args.extract_min_area,
        max_segments=args.extract_max_nodes,
        dedupe_iou=args.sam_dedupe_iou,
        sam3_model=args.sam3_model,
        sam3_text_prompts=tuple(args.sam3_text_prompts),
        sam3_conf=args.sam3_conf,
        sam3_half=not args.sam3_disable_half,
        sam3_allow_download=not args.sam3_disable_download,
        sam2_model=args.sam2_model,
        sam2_conf=args.sam2_conf,
        sam2_imgsz=args.sam2_imgsz,
        sam2_half=not args.sam2_disable_half,
    )
    save_sam_cache(
        cache_dir=args.sam_cache_dir,
        image_id=sample.image_id,
        image_path=sample.image_path,
        backend=used_backend,
        checkpoint=args.sam_checkpoint,
        segments=segments,
    )
    return segments


def sam_nodes_to_graph(sample, segments: list[SamSegment], featurizer: NodeFeaturizerV2, args) -> Data:
    image_path = Path(sample.image_path)
    pil_img = Image.open(image_path).convert("RGB")
    shape_nodes = [NodeV2(bbox=segment.bbox, kind="shape", conf=segment.stability_score * 100.0) for segment in segments]
    text_nodes: list[NodeV2] = []
    if sample.ocr_v2_path:
        text_nodes = parse_ocr_v2_json(sample.ocr_v2_path, level=args.ocr_level, min_conf=args.extract_min_text_conf)
    nodes = shape_nodes + text_nodes
    edge_index, edge_type = build_typed_edges_v2(nodes, k=args.extract_knn_k)
    x = featurizer.extract(pil_img, nodes)
    return Data(x=x, edge_index=edge_index, edge_type=edge_type)


def build_sam_cache_for_samples(samples: list, args) -> None:
    for sample in tqdm(samples, desc="[sam-cache]"):
        _ = build_or_load_sam_segments(sample, args)


class GraphTransformerEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_heads: int,
        num_layers: int,
        edge_type_count: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.node_proj = nn.Linear(in_dim, hidden_dim)
        self.edge_emb = nn.Embedding(edge_type_count, hidden_dim)
        self.layers = nn.ModuleList(
            [
                TransformerConv(
                    hidden_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    edge_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, batch: Batch) -> torch.Tensor:
        x = F.gelu(self.node_proj(batch.x))
        edge_type = getattr(batch, "edge_type", None)
        if edge_type is None or edge_type.numel() == 0:
            edge_attr = x.new_zeros((batch.edge_index.size(1), self.edge_emb.embedding_dim))
        else:
            edge_attr = self.edge_emb(edge_type.clamp_min(0).clamp_max(self.edge_emb.num_embeddings - 1))
        for conv, norm in zip(self.layers, self.norms):
            residual = x
            x = conv(x, batch.edge_index, edge_attr)
            x = norm(F.gelu(x) + residual)
        x = F.gelu(self.out(x))
        return global_mean_pool(x, batch.batch)


class GPSGraphEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_heads: int, num_layers: int) -> None:
        super().__init__()
        self.node_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                GPSConv(
                    hidden_dim,
                    conv=TransformerConv(hidden_dim, hidden_dim // num_heads, heads=num_heads),
                    heads=num_heads,
                )
                for _ in range(num_layers)
            ]
        )
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, batch: Batch) -> torch.Tensor:
        x = F.gelu(self.node_proj(batch.x))
        for layer in self.layers:
            x = F.gelu(layer(x, batch.edge_index, batch.batch))
        x = F.gelu(self.out(x))
        return global_mean_pool(x, batch.batch)


class LateFusionGraphModel(nn.Module):
    def __init__(self, opencv_encoder: nn.Module, sam_encoder: nn.Module, out_dim: int) -> None:
        super().__init__()
        self.opencv_encoder = opencv_encoder
        self.sam_encoder = sam_encoder
        self.fusion = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, opencv_batch: Batch, sam_batch: Batch) -> torch.Tensor:
        z_opencv = self.opencv_encoder(opencv_batch)
        z_sam = self.sam_encoder(sam_batch)
        return self.fusion(torch.cat([z_opencv, z_sam], dim=1))


def make_encoder(args, in_dim: int) -> nn.Module:
    if args.encoder == "gatv2":
        return GraphEncoderV2(
            in_dim=in_dim,
            hidden_dim=args.hidden_dim,
            out_dim=args.out_dim,
            num_heads=args.num_heads,
            use_attn_pool=not args.disable_attn_pool,
        )
    if args.encoder == "transformer":
        return GraphTransformerEncoder(
            in_dim=in_dim,
            hidden_dim=args.hidden_dim,
            out_dim=args.out_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            edge_type_count=len(EDGE_TYPE_TO_ID),
            dropout=args.dropout,
        )
    if args.encoder == "gps":
        return GPSGraphEncoder(
            in_dim=in_dim,
            hidden_dim=args.hidden_dim,
            out_dim=args.out_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
        )
    raise ValueError(f"Unsupported encoder: {args.encoder}")


def build_cache_signature(args, featurizer: NodeFeaturizerV2, graph_mode: str) -> str:
    return FeatureCacheV2.make_signature(
        vision_model_name=featurizer.vision_model_name,
        text_model_name=featurizer.text_model_name,
        ocr_level=args.ocr_level,
        min_area=args.extract_min_area,
        max_shape_nodes=args.extract_max_nodes,
        min_text_conf=args.extract_min_text_conf,
        knn_k=args.extract_knn_k,
        include_shapes=True,
    ) + f"_{graph_mode}"


def build_graph_batches(batch: dict[str, Any], samples_by_id: dict[str, Any], cache, featurizer, args, device):
    if args.graph_mode in {"opencv_only", "opencv_sam_late"}:
        opencv_graphs = [
            cache.get_graph(
                image_path=image_path,
                featurizer=featurizer,
                ocr_path=ocr_path,
                ocr_level=args.ocr_level,
                min_area=args.extract_min_area,
                max_shape_nodes=args.extract_max_nodes,
                min_text_conf=args.extract_min_text_conf,
                knn_k=args.extract_knn_k,
                include_shapes=True,
            )
            for image_path, ocr_path in zip(batch["image_paths"], batch["ocr_paths"])
        ]
        opencv_batch = Batch.from_data_list(opencv_graphs).to(device)
    else:
        opencv_batch = None

    if args.graph_mode in {"sam_only", "opencv_sam_late"}:
        sam_graphs = []
        for sample_id in batch["sample_ids"]:
            sample = samples_by_id[sample_id]
            segments = build_or_load_sam_segments(sample, args)
            sam_graphs.append(sam_nodes_to_graph(sample, segments, featurizer, args))
        sam_batch = Batch.from_data_list(sam_graphs).to(device)
    else:
        sam_batch = None

    return opencv_batch, sam_batch


def encode_question_embeddings(batch, featurizer: NodeFeaturizerV2, cache: FeatureCacheV2, device: torch.device):
    return cache.get_text_batch(batch["question_texts"], featurizer.text_enc, normalize=False).to(device)


def forward_graph_model(model, opencv_batch, sam_batch, args) -> torch.Tensor:
    if args.graph_mode == "opencv_only":
        return model(opencv_batch)
    if args.graph_mode == "sam_only":
        return model(sam_batch)
    if args.graph_mode == "opencv_sam_late":
        return model(opencv_batch, sam_batch)
    raise ValueError(f"Unsupported graph_mode: {args.graph_mode}")


def evaluate_split(loader, model, text_proj, featurizer, cache, samples_by_id, args, device, split_name):
    model.eval()
    text_proj.eval()
    all_img_emb: list[torch.Tensor] = []
    all_q_emb: list[torch.Tensor] = []
    ordered_sample_ids: list[str] = []
    ordered_image_ids: list[str] = []
    vqa_predictions: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"[eval:{split_name}]", leave=False):
            opencv_batch, sam_batch = build_graph_batches(batch, samples_by_id, cache, featurizer, args, device)
            q_emb = encode_question_embeddings(batch, featurizer, cache, device)
            z_img = F.normalize(forward_graph_model(model, opencv_batch, sam_batch, args), dim=1)
            z_q = F.normalize(text_proj(q_emb), dim=1)
            logits = compute_option_logits(
                z_img=z_img,
                option_texts=batch["option_texts"],
                option_mask=batch["option_mask"],
                text_encoder=featurizer.text_enc,
                text_proj=text_proj,
                cache=cache,
                temperature=args.temperature,
            )
            targets = batch["correct_indices"].to(device)
            preds = logits.argmax(dim=1)

            all_img_emb.append(z_img.detach().cpu())
            all_q_emb.append(z_q.detach().cpu())
            ordered_sample_ids.extend(batch["sample_ids"])
            ordered_image_ids.extend(batch["image_ids"])
            for i, sample_id in enumerate(batch["sample_ids"]):
                options = batch["options"][i]
                pred_idx = int(preds[i].item())
                gold_idx = int(targets[i].item())
                vqa_predictions.append(
                    {
                        "sample_id": sample_id,
                        "image_id": batch["image_ids"][i],
                        "question": batch["questions"][i],
                        "pred_option_idx": pred_idx,
                        "pred_option_text": options[pred_idx] if 0 <= pred_idx < len(options) else "",
                        "gold_option_idx": gold_idx,
                        "gold_option_text": options[gold_idx] if 0 <= gold_idx < len(options) else "",
                        "is_correct": bool(pred_idx == gold_idx),
                    }
                )

    z_img = torch.cat(all_img_emb, dim=0)
    z_q = torch.cat(all_q_emb, dim=0)
    retrieval = retrieval_metrics_from_embeddings(z_img=z_img, z_txt=z_q, ks=(1, 5, 10), image_ids=ordered_image_ids)
    vqa_acc = float(torch.tensor([x["is_correct"] for x in vqa_predictions], dtype=torch.float32).mean().item())
    composite = 0.5 * float(retrieval["mean"][10]) + 0.5 * vqa_acc
    sim = retrieval["sim"]
    top_k = min(10, sim.size(1))
    top_idx = sim.topk(k=top_k, dim=1).indices
    retrieval_predictions = [
        {
            "sample_id": ordered_sample_ids[row_idx],
            "image_id": ordered_image_ids[row_idx],
            "top10_image_ids": [ordered_image_ids[int(col_idx)] for col_idx in top_idx[row_idx].tolist()],
            "hit@10": bool(ordered_image_ids[row_idx] in [ordered_image_ids[int(col_idx)] for col_idx in top_idx[row_idx].tolist()]),
        }
        for row_idx in range(sim.size(0))
    ]
    metrics = {
        "split": split_name,
        "i2t": {str(k): float(v) for k, v in retrieval["i2t"].items()},
        "t2i": {str(k): float(v) for k, v in retrieval["t2i"].items()},
        "mean": {str(k): float(v) for k, v in retrieval["mean"].items()},
        "vqa_acc": vqa_acc,
        "composite": composite,
    }
    return {"metrics": metrics, "vqa_predictions": vqa_predictions, "retrieval_predictions": retrieval_predictions}


def train_epoch(loader, model, text_proj, featurizer, cache, samples_by_id, optimizer, scaler, args, device, stage):
    model.train()
    text_proj.train()
    total_loss = total_ret = total_vqa = 0.0
    n_steps = 0
    use_amp = scaler.is_enabled()
    autocast_ctx = torch.cuda.amp.autocast if use_amp else nullcontext
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader, desc=f"[train:{stage}]", leave=False), start=1):
        opencv_batch, sam_batch = build_graph_batches(batch, samples_by_id, cache, featurizer, args, device)
        q_emb = encode_question_embeddings(batch, featurizer, cache, device)
        with autocast_ctx():
            z_img = F.normalize(forward_graph_model(model, opencv_batch, sam_batch, args), dim=1)
            z_q = F.normalize(text_proj(q_emb), dim=1)
            loss_ret = contrastive_loss(z_img, z_q, temperature=args.temperature, group_ids=batch["image_ids"])
            if stage == "stage2":
                logits = compute_option_logits(
                    z_img=z_img,
                    option_texts=batch["option_texts"],
                    option_mask=batch["option_mask"],
                    text_encoder=featurizer.text_enc,
                    text_proj=text_proj,
                    cache=cache,
                    temperature=args.temperature,
                )
                loss_vqa = F.cross_entropy(logits, batch["correct_indices"].to(device))
                loss = args.lambda_ret * loss_ret + args.lambda_vqa * loss_vqa
            else:
                loss_vqa = z_img.new_tensor(0.0)
                loss = loss_ret
            loss = loss / args.grad_accum_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % args.grad_accum_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(text_proj.parameters()), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(text_proj.parameters()), args.grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.item() * args.grad_accum_steps)
        total_ret += float(loss_ret.item())
        total_vqa += float(loss_vqa.item())
        n_steps += 1

    if n_steps % args.grad_accum_steps != 0:
        if use_amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(text_proj.parameters()), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(text_proj.parameters()), args.grad_clip)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return {
        "loss": total_loss / max(1, n_steps),
        "loss_ret": total_ret / max(1, n_steps),
        "loss_vqa": total_vqa / max(1, n_steps),
    }


def save_checkpoint(path, model, text_proj, optimizer, featurizer, args, epoch_idx, stage, val_metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch_idx,
            "stage": stage,
            "model_state_dict": model.state_dict(),
            "text_proj_state_dict": text_proj.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": {
                "graph_mode": args.graph_mode,
                "encoder": args.encoder,
                "in_dim": args.in_dim,
                "hidden_dim": args.hidden_dim,
                "out_dim": args.out_dim,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "vision_model_name": featurizer.vision_model_name,
                "text_model_name": featurizer.text_model_name,
                "sam_cache_dir": str(args.sam_cache_dir),
                "sam_backend": args.sam_backend,
            },
            "val_metrics": val_metrics,
        },
        path,
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AI2D graph experiments: SAM nodes, typed edges, Graph Transformer.")
    parser.add_argument("--manifest", type=Path, default=EXTERNAL_ROOT / "ai2d" / "prepared_v2" / "manifest_hybrid.jsonl")
    parser.add_argument("--split-json", type=Path, default=EXTERNAL_ROOT / "ai2d" / "prepared_v2" / "split_hybrid.json")
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / "ai2d_graph_experiment")
    parser.add_argument("--progress-jsonl", type=Path, default=None)
    parser.add_argument("--graph-mode", choices=["opencv_only", "sam_only", "opencv_sam_late"], default="opencv_only")
    parser.add_argument("--encoder", choices=["gatv2", "transformer", "gps"], default="transformer")
    parser.add_argument("--sam-cache-only", action="store_true")
    parser.add_argument("--sam-cache-dir", type=Path, default=Path("runs") / "sam_cache_ai2d")
    parser.add_argument("--sam-cache-missing", choices=["build", "skip"], default="build")
    parser.add_argument(
        "--sam-backend",
        default="ultralytics_sam3",
        help="ultralytics_sam3, ultralytics_sam2, sam_vit_b, sam_vit_l, sam_vit_h, or opencv_fallback.",
    )
    parser.add_argument("--sam-checkpoint", type=Path, default=EXTERNAL_ROOT / "models" / "sam" / "sam_vit_b_01ec64.pth")
    parser.add_argument("--sam-dedupe-iou", type=float, default=0.85)
    parser.add_argument("--sam3-model", type=Path, default=EXTERNAL_ROOT / "models" / "sam3" / "sam3.pt")
    parser.add_argument(
        "--sam3-text-prompts",
        nargs="+",
        default=["diagram", "arrow", "line", "text label", "circle", "rectangle", "object"],
    )
    parser.add_argument("--sam3-conf", type=float, default=0.25)
    parser.add_argument("--sam3-disable-half", action="store_true")
    parser.add_argument("--sam3-disable-download", action="store_true")
    parser.add_argument("--sam2-model", default=str(EXTERNAL_ROOT / "models" / "sam2" / "sam2.1_b.pt"))
    parser.add_argument("--sam2-conf", type=float, default=0.25)
    parser.add_argument("--sam2-imgsz", type=int, default=1024)
    parser.add_argument("--sam2-disable-half", action="store_true")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--epochs-stage1", type=int, default=8)
    parser.add_argument("--epochs-stage2", type=int, default=25)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--out-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--disable-attn-pool", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda-ret", type=float, default=0.6)
    parser.add_argument("--lambda-vqa", type=float, default=0.4)
    parser.add_argument("--use-caption-context", action="store_true")
    parser.add_argument("--extract-min-area", type=int, default=300)
    parser.add_argument("--extract-max-nodes", type=int, default=80)
    parser.add_argument("--extract-min-text-conf", type=float, default=35.0)
    parser.add_argument("--extract-knn-k", type=int, default=4)
    parser.add_argument("--ocr-level", choices=["line", "word"], default="line")
    parser.add_argument("--cache-dir", type=Path, default=EXTERNAL_ROOT / "ai2d" / "_cache_graph_transformer")
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--disable-amp", action="store_true")
    return parser.parse_args(argv)


def make_args(**overrides: Any) -> argparse.Namespace:
    """Create training args for notebook/function use.

    Example:
        args = make_args(graph_mode="sam_only", sam_cache_only=True)
        run_experiment(args)
    """
    args = parse_args([])
    for key, value in overrides.items():
        attr = key.replace("-", "_")
        if not hasattr(args, attr):
            raise ValueError(f"Unknown training argument: {key}")
        setattr(args, attr, value)
    return args


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    device = torch.device(args.device)

    samples = resolve_sample_file_paths(load_manifest_hybrid(args.manifest), roots=[Path.cwd(), ROOT, EXTERNAL_ROOT])
    split_payload = load_split_payload(args.split_json)
    train_samples = maybe_limit_samples(select_samples_for_split(samples, "train", split_payload), args.max_train_samples, args.seed)
    val_samples = maybe_limit_samples(select_samples_for_split(samples, "val", split_payload), args.max_val_samples, args.seed + 1)
    test_samples = maybe_limit_samples(select_samples_for_split(samples, "test", split_payload), args.max_test_samples, args.seed + 2)
    all_used_samples = train_samples + val_samples + test_samples
    samples_by_id = {sample.sample_id: sample for sample in all_used_samples}

    print(f"[INFO] graph_mode={args.graph_mode} encoder={args.encoder}")
    print(f"[INFO] train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")
    if not train_samples or not val_samples or not test_samples:
        raise SystemExit("Empty split detected. Check manifest/split json or --max-*-samples.")

    if args.sam_cache_only:
        build_sam_cache_for_samples(all_used_samples, args)
        print(f"[INFO] SAM cache saved under: {args.sam_cache_dir}")
        return {
            "mode": "sam_cache_only",
            "sam_cache_dir": str(args.sam_cache_dir),
            "num_samples": len(all_used_samples),
        }

    collate_fn = make_hybrid_collate_fn(use_caption_context=args.use_caption_context)
    train_loader = DataLoader(
        Ai2dHybridDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    eval_kwargs = {
        "batch_size": args.eval_batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "collate_fn": collate_fn,
    }
    val_loader = DataLoader(Ai2dHybridDataset(val_samples), **eval_kwargs)
    test_loader = DataLoader(Ai2dHybridDataset(test_samples), **eval_kwargs)

    featurizer = NodeFeaturizerV2(device=str(device))
    args.in_dim = featurizer.vision_dim + featurizer.text_dim + 12 + 1 + 1
    cache = FeatureCacheV2(
        cache_dir=args.cache_dir,
        signature=build_cache_signature(args, featurizer, args.graph_mode),
        enabled=not args.disable_cache,
    )

    if args.graph_mode == "opencv_sam_late":
        model = LateFusionGraphModel(
            opencv_encoder=make_encoder(args, args.in_dim),
            sam_encoder=make_encoder(args, args.in_dim),
            out_dim=args.out_dim,
        ).to(device)
    else:
        model = make_encoder(args, args.in_dim).to(device)
    text_proj = nn.Sequential(
        nn.Linear(featurizer.text_dim, args.hidden_dim),
        nn.GELU(),
        nn.Linear(args.hidden_dim, args.out_dim),
    ).to(device)

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(text_proj.parameters()), lr=args.lr)
    use_amp = (device.type == "cuda") and (not args.disable_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "checkpoint_best.pt"
    metrics_path = args.output_dir / "metrics.json"
    progress_path = args.progress_jsonl or (args.output_dir / "progress.jsonl")
    progress = ProgressJsonlLogger(progress_path)
    progress.write(
        "train_begin",
        graph_mode=args.graph_mode,
        encoder=args.encoder,
        epochs_stage1=int(args.epochs_stage1),
        epochs_stage2=int(args.epochs_stage2),
        train_size=len(train_samples),
        val_size=len(val_samples),
        test_size=len(test_samples),
    )
    print(f"[INFO] Training progress will be saved to {progress_path}")
    history: list[dict[str, Any]] = []
    best_composite = float("-inf")
    best_epoch = -1
    no_improve = 0
    global_epoch = 0
    stop_training = False

    for stage_name, stage_epochs in [("stage1", args.epochs_stage1), ("stage2", args.epochs_stage2)]:
        for _ in range(max(0, stage_epochs)):
            global_epoch += 1
            train_stats = train_epoch(
                train_loader, model, text_proj, featurizer, cache, samples_by_id, optimizer, scaler, args, device, stage_name
            )
            val_result = evaluate_split(val_loader, model, text_proj, featurizer, cache, samples_by_id, args, device, "val")
            val_metrics = val_result["metrics"]
            history.append({"epoch": global_epoch, "stage": stage_name, "train": train_stats, "val": val_metrics})
            progress.write(
                "epoch",
                epoch=global_epoch,
                stage=stage_name,
                train=train_stats,
                val=val_metrics,
                best_epoch=best_epoch,
                best_composite=best_composite if best_composite != float("-inf") else None,
            )
            print(
                f"[E{global_epoch} {stage_name}] loss={train_stats['loss']:.4f} "
                f"ret={train_stats['loss_ret']:.4f} vqa={train_stats['loss_vqa']:.4f} | "
                f"val MeanR@10={val_metrics['mean']['10']:.4f} "
                f"val VQA={val_metrics['vqa_acc']:.4f} "
                f"val Composite={val_metrics['composite']:.4f}"
            )
            if val_metrics["composite"] > best_composite:
                best_composite = float(val_metrics["composite"])
                best_epoch = global_epoch
                no_improve = 0
                save_checkpoint(checkpoint_path, model, text_proj, optimizer, featurizer, args, global_epoch, stage_name, val_metrics)
                progress.write(
                    "new_best",
                    epoch=global_epoch,
                    stage=stage_name,
                    best_composite=best_composite,
                    val=val_metrics,
                    checkpoint=str(checkpoint_path),
                )
                print(f"[INFO] New best checkpoint at epoch {global_epoch} (composite={best_composite:.4f})")
            else:
                no_improve += 1
                if no_improve >= args.early_stopping_patience:
                    print(f"[INFO] Early stopping: no improvement for {no_improve} eval steps.")
                    progress.write(
                        "early_stop",
                        epoch=global_epoch,
                        stage=stage_name,
                        no_improve=no_improve,
                        best_epoch=best_epoch,
                        best_composite=best_composite,
                    )
                    stop_training = True
                    break
        if stop_training:
            break

    if not checkpoint_path.exists():
        raise RuntimeError("Best checkpoint was not saved.")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    text_proj.load_state_dict(checkpoint["text_proj_state_dict"])
    test_result = evaluate_split(test_loader, model, text_proj, featurizer, cache, samples_by_id, args, device, "test")
    progress.write("test", best_epoch=best_epoch, best_composite=best_composite, test=test_result["metrics"])
    write_jsonl(test_result["vqa_predictions"], args.output_dir / "test_vqa_predictions.jsonl")
    write_jsonl(test_result["retrieval_predictions"], args.output_dir / "test_retrieval_predictions.jsonl")

    payload = {
        "best_epoch": best_epoch,
        "best_composite": best_composite,
        "train_history": history,
        "test": test_result["metrics"],
        "config": {
            "manifest": str(args.manifest),
            "split_json": str(args.split_json),
            "graph_mode": args.graph_mode,
            "encoder": args.encoder,
            "sam_cache_dir": str(args.sam_cache_dir),
            "sam_backend": args.sam_backend,
            "epochs_stage1": args.epochs_stage1,
            "epochs_stage2": args.epochs_stage2,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "lambda_ret": args.lambda_ret,
            "lambda_vqa": args.lambda_vqa,
            "progress_jsonl": str(progress_path),
        },
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    progress.write(
        "train_end",
        best_epoch=best_epoch,
        best_composite=best_composite,
        test_composite=test_result["metrics"]["composite"],
        metrics_path=str(metrics_path),
    )
    print(f"[INFO] Best epoch: {best_epoch}")
    print(f"[INFO] Best composite: {best_composite:.4f}")
    print(f"[INFO] Test composite: {test_result['metrics']['composite']:.4f}")
    print(f"[INFO] Artifacts saved under: {args.output_dir}")
    print(f"[INFO] Progress log saved to: {progress_path}")
    return payload


def run_experiment_from_kwargs(**overrides: Any) -> dict[str, Any]:
    """Notebook-friendly wrapper around the full training pipeline."""
    return run_experiment(make_args(**overrides))


def main(argv: Optional[list[str]] = None) -> None:
    run_experiment(parse_args(argv))


if __name__ == "__main__":
    main()
