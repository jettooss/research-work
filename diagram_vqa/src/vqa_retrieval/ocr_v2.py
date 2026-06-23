from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import cv2
import numpy as np


_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class OcrWord:
    text: str
    conf: float
    bbox: tuple[int, int, int, int]
    source_variant: str


@dataclass(frozen=True)
class OcrLine:
    text: str
    conf: float
    bbox: tuple[int, int, int, int]
    words: tuple[OcrWord, ...]


@dataclass(frozen=True)
class OcrResult:
    image_path: str
    image_size: tuple[int, int]
    engine: str
    language: str
    variant: str
    words: tuple[OcrWord, ...]
    lines: tuple[OcrLine, ...]

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "image_size": list(self.image_size),
            "engine": self.engine,
            "language": self.language,
            "variant": self.variant,
            "words": [asdict(word) for word in self.words],
            "lines": [
                {
                    "text": line.text,
                    "conf": line.conf,
                    "bbox": list(line.bbox),
                    "words": [asdict(word) for word in line.words],
                }
                for line in self.lines
            ],
        }


@dataclass(frozen=True)
class TesseractVariant:
    name: str
    psm: int
    oem: int = 3
    upscale: float = 2.0
    threshold: str = "adaptive"
    invert: bool = False

    @property
    def config(self) -> str:
        return f"--oem {self.oem} --psm {self.psm}"


DEFAULT_TESSERACT_VARIANTS: tuple[TesseractVariant, ...] = (
    TesseractVariant(name="sparse-adaptive", psm=11, upscale=2.0, threshold="adaptive"),
    TesseractVariant(name="block-otsu", psm=6, upscale=2.0, threshold="otsu"),
    TesseractVariant(name="sparse-otsu", psm=12, upscale=1.75, threshold="otsu"),
)


def _ensure_tesseract():
    try:
        import pytesseract
    except ImportError as exc:
        raise ImportError(
            "pytesseract is required for ocr_v2. Install it with '.\\.venv\\Scripts\\pip install pytesseract'."
        ) from exc

    if pytesseract.pytesseract.tesseract_cmd == "tesseract":
        win_default = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if win_default.exists():
            pytesseract.pytesseract.tesseract_cmd = str(win_default)
    return pytesseract


def normalize_ocr_text(text: str) -> str:
    text = text.replace("\n", " ").replace("\t", " ")
    text = text.replace("|", "I")
    text = text.replace("’", "'").replace("`", "'")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _clip_bbox(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def preprocess_for_ocr(image_bgr: np.ndarray, variant: TesseractVariant) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    if variant.upscale and variant.upscale != 1.0:
        gray = cv2.resize(gray, None, fx=variant.upscale, fy=variant.upscale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    if variant.threshold == "adaptive":
        proc = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            15,
        )
    elif variant.threshold == "otsu":
        _, proc = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        proc = gray

    if variant.invert:
        proc = 255 - proc

    return proc


def _extract_words_from_tesseract(
    image_bgr: np.ndarray,
    image_path: str,
    lang: str,
    min_conf: float,
    variant: TesseractVariant,
) -> list[OcrWord]:
    pytesseract = _ensure_tesseract()
    processed = preprocess_for_ocr(image_bgr, variant)
    data = pytesseract.image_to_data(
        processed,
        lang=lang,
        config=variant.config,
        output_type=pytesseract.Output.DICT,
    )

    scale_x = image_bgr.shape[1] / processed.shape[1]
    scale_y = image_bgr.shape[0] / processed.shape[0]
    words: list[OcrWord] = []
    for idx, raw_text in enumerate(data["text"]):
        text = normalize_ocr_text(raw_text or "")
        raw_conf = data["conf"][idx]
        conf = float(raw_conf) if raw_conf != "-1" else -1.0
        if conf < min_conf or not text:
            continue

        x = int(float(data["left"][idx]) * scale_x)
        y = int(float(data["top"][idx]) * scale_y)
        w = max(1, int(float(data["width"][idx]) * scale_x))
        h = max(1, int(float(data["height"][idx]) * scale_y))
        bbox = _clip_bbox(x, y, x + w, y + h, image_bgr.shape[1], image_bgr.shape[0])
        words.append(OcrWord(text=text, conf=conf, bbox=bbox, source_variant=variant.name))
    return words


def _word_score(words: Sequence[OcrWord]) -> float:
    if not words:
        return 0.0
    return sum(max(0.0, word.conf) * max(1, len(word.text)) for word in words)


def _line_gap_threshold(words: Sequence[OcrWord]) -> float:
    if not words:
        return 18.0
    heights = sorted(word.bbox[3] - word.bbox[1] for word in words)
    median_height = heights[len(heights) // 2]
    return max(12.0, median_height * 0.8)


def merge_words_into_lines(words: Sequence[OcrWord]) -> list[OcrLine]:
    if not words:
        return []

    words_sorted = sorted(words, key=lambda word: (word.bbox[1], word.bbox[0]))
    groups: list[list[OcrWord]] = []
    gap_th = _line_gap_threshold(words_sorted)

    for word in words_sorted:
        placed = False
        word_center_y = 0.5 * (word.bbox[1] + word.bbox[3])
        for group in groups:
            group_center_y = sum(0.5 * (w.bbox[1] + w.bbox[3]) for w in group) / len(group)
            if abs(word_center_y - group_center_y) <= gap_th:
                group.append(word)
                placed = True
                break
        if not placed:
            groups.append([word])

    lines: list[OcrLine] = []
    for group in groups:
        group.sort(key=lambda word: word.bbox[0])
        text = normalize_ocr_text(" ".join(word.text for word in group))
        if not text:
            continue
        x1 = min(word.bbox[0] for word in group)
        y1 = min(word.bbox[1] for word in group)
        x2 = max(word.bbox[2] for word in group)
        y2 = max(word.bbox[3] for word in group)
        conf = sum(word.conf for word in group) / len(group)
        lines.append(OcrLine(text=text, conf=conf, bbox=(x1, y1, x2, y2), words=tuple(group)))

    return sorted(lines, key=lambda line: (line.bbox[1], line.bbox[0]))


def run_tesseract_ocr_v2(
    image_path: str | Path,
    lang: str = "eng",
    min_conf: float = 35.0,
    variants: Optional[Iterable[TesseractVariant]] = None,
) -> OcrResult:
    image_path = str(Path(image_path))
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise RuntimeError(f"cv2.imread failed: {image_path}")

    variants = tuple(variants or DEFAULT_TESSERACT_VARIANTS)
    if not variants:
        raise ValueError("At least one OCR variant is required")

    best_variant = variants[0]
    best_words = _extract_words_from_tesseract(image_bgr, image_path, lang, min_conf, best_variant)
    best_score = _word_score(best_words)

    for variant in variants[1:]:
        words = _extract_words_from_tesseract(image_bgr, image_path, lang, min_conf, variant)
        score = _word_score(words)
        if score > best_score:
            best_variant = variant
            best_words = words
            best_score = score

    lines = merge_words_into_lines(best_words)
    return OcrResult(
        image_path=image_path,
        image_size=(image_bgr.shape[1], image_bgr.shape[0]),
        engine="tesseract_v2",
        language=lang,
        variant=best_variant.name,
        words=tuple(best_words),
        lines=tuple(lines),
    )


def save_ocr_result(result: OcrResult, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, ensure_ascii=False, indent=2)
    return output_path

