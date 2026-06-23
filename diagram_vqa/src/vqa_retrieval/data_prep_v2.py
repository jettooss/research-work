from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

from vqa_retrieval.caption_v1 import extract_json_object, normalize_caption_payload
from vqa_retrieval.caption_v1 import generate_image_caption, save_caption_result
from vqa_retrieval.caption_v2_local import (
    LocalCaptionResult,
    generate_local_image_caption,
    save_local_caption_result,
)
from vqa_retrieval.datasets import (
    Ai2dRetrievalDataset,
    DocVQARetrievalDataset,
    InfographicVQARetrievalDataset,
    RetrievalSample,
)
from vqa_retrieval.graph_builder_v2 import resolve_ocr_v2_path
from vqa_retrieval.ocr_v2 import run_tesseract_ocr_v2, save_ocr_result


@dataclass(frozen=True)
class PreparedSampleV2:
    dataset_name: str
    split: str
    sample_id: str
    image_path: str
    question: str
    answers: tuple[str, ...]
    text: str
    short_description: Optional[str]
    source_ocr_path: Optional[str]
    ocr_v2_path: Optional[str]
    caption_v1_path: Optional[str]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["answers"] = list(self.answers)
        return payload


def load_dataset_samples_v2(
    dataset_name: str,
    data_root: str | Path,
    split: str = "train",
) -> list[RetrievalSample]:
    data_root = Path(data_root)
    key = dataset_name.strip().lower()
    if key == "ai2d":
        return list(Ai2dRetrievalDataset(data_root))
    if key == "docvqa":
        return list(DocVQARetrievalDataset(data_root, split=split))
    if key == "infographicvqa":
        return list(InfographicVQARetrievalDataset(data_root, split=split))
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def infer_image_root_v2(dataset_name: str, data_root: str | Path, split: str = "train") -> Path:
    data_root = Path(data_root)
    key = dataset_name.strip().lower()
    if key == "ai2d":
        return data_root / "images"
    if key == "docvqa":
        return data_root / split
    if key == "infographicvqa":
        return data_root / "InfographicVQA" / "infographicsvqa_images"
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def infer_default_output_root_v2(dataset_name: str, data_root: str | Path, split: str = "train") -> Path:
    data_root = Path(data_root)
    key = dataset_name.strip().lower()
    if key == "ai2d":
        return data_root / "prepared_v2"
    if key == "docvqa":
        return data_root / split / "prepared_v2"
    if key == "infographicvqa":
        return data_root / "InfographicVQA" / f"prepared_v2_{split}"
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _caption_output_path(image_path: Path, image_root: Path, caption_root: Path) -> Path:
    relative = image_path.relative_to(image_root)
    return caption_root / relative.with_suffix(".caption.json")


def build_manifest_entries_v2(
    samples: Iterable[RetrievalSample],
    image_root: str | Path,
    ocr_root: Optional[str | Path] = None,
    caption_root: Optional[str | Path] = None,
) -> Iterator[PreparedSampleV2]:
    image_root = Path(image_root)
    ocr_root = Path(ocr_root) if ocr_root is not None else None
    caption_root = Path(caption_root) if caption_root is not None else None

    for sample in samples:
        image_path = Path(sample.image_path)
        ocr_v2_path = str(resolve_ocr_v2_path(image_path, image_root, ocr_root)) if ocr_root else None
        caption_v1_path = str(_caption_output_path(image_path, image_root, caption_root)) if caption_root else None
        short_description = None
        if caption_v1_path and Path(caption_v1_path).exists():
            payload = json.loads(Path(caption_v1_path).read_text(encoding="utf-8"))
            short_description = str(payload.get("short_description", "")).strip() or None
        yield PreparedSampleV2(
            dataset_name=sample.dataset_name,
            split=sample.split,
            sample_id=sample.sample_id,
            image_path=str(image_path),
            question=sample.question,
            answers=sample.answers,
            text=sample.text,
            short_description=short_description,
            source_ocr_path=sample.ocr_path,
            ocr_v2_path=ocr_v2_path,
            caption_v1_path=caption_v1_path,
        )


def precompute_ocr_for_images_v2(
    image_paths: Iterable[str | Path],
    image_root: str | Path,
    ocr_root: str | Path,
    lang: str = "eng",
    min_conf: float = 35.0,
    overwrite: bool = False,
) -> list[Path]:
    image_root = Path(image_root)
    ocr_root = Path(ocr_root)
    outputs: list[Path] = []
    for image_path in image_paths:
        image_path = Path(image_path)
        output_path = resolve_ocr_v2_path(image_path, image_root, ocr_root)
        if output_path.exists() and not overwrite:
            outputs.append(output_path)
            continue
        result = run_tesseract_ocr_v2(image_path=image_path, lang=lang, min_conf=min_conf)
        outputs.append(save_ocr_result(result, output_path))
    return outputs


def precompute_captions_for_images_v2(
    image_paths: Iterable[str | Path],
    image_root: str | Path,
    caption_root: str | Path,
    backend: str,
    credentials: Optional[str] = None,
    model: str = "GigaChat-2-Max",
    model_path: Optional[str | Path] = None,
    use_4bit: bool = True,
    overwrite: bool = False,
) -> list[Path]:
    image_root = Path(image_root)
    caption_root = Path(caption_root)
    backend = backend.strip().lower()
    outputs: list[Path] = []
    for image_path in image_paths:
        image_path = Path(image_path)
        output_path = _caption_output_path(image_path, image_root, caption_root)
        if output_path.exists() and not overwrite:
            outputs.append(output_path)
            continue
        if backend == "gigachat":
            if not credentials:
                raise ValueError("credentials are required for gigachat backend")
            result = generate_image_caption(
                image_path=image_path,
                credentials=credentials,
                model=model,
            )
            outputs.append(save_caption_result(result, output_path))
        elif backend == "qwen":
            if not model_path:
                raise ValueError("model_path is required for qwen backend")
            try:
                result = generate_local_image_caption(
                    image_path=image_path,
                    model_path=model_path,
                    use_4bit=use_4bit,
                )
            except Exception as exc:
                print(f"[WARN] Qwen caption failed for {image_path}: {exc}")
                result = LocalCaptionResult(
                    image_path=str(image_path),
                    model_path=str(model_path),
                    visual_type="",
                    short_description="Caption generation failed.",
                    summary=f"Caption generation failed: {type(exc).__name__}",
                    main_objects=(),
                    relationships=(),
                    important_text=(),
                    raw_response=str(exc),
                )
            outputs.append(save_local_caption_result(result, output_path))
        else:
            raise ValueError(f"Unsupported caption backend: {backend}")
    return outputs


def write_manifest_jsonl_v2(entries: Iterable[PreparedSampleV2], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
    return output_path
