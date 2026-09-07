from __future__ import annotations

import base64
import json
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from vqa_retrieval.ocr_v2 import OcrResult, run_tesseract_ocr_v2

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover
    PeftModel = None


TASK_MODES = {"qa", "ocr", "all"}
STATUS_OK = "ok"
STATUS_LOW_CONFIDENCE = "low_confidence"
STATUS_FAILED = "failed"


@dataclass(frozen=True)
class LineItem:
    line_index: int
    text: str
    bbox: tuple[int, int, int, int]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bbox"] = list(self.bbox)
        return payload


@dataclass(frozen=True)
class AnalyzeResponse:
    answer: str
    evidence_lines: tuple[LineItem, ...]
    extracted_text: tuple[LineItem, ...]
    final_confidence: float
    status: str
    raw_response: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "evidence_lines": [item.to_dict() for item in self.evidence_lines],
            "extracted_text": [item.to_dict() for item in self.extracted_text],
            "final_confidence": self.final_confidence,
            "status": self.status,
            "raw_response": self.raw_response,
        }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split()).lower()


def _extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Model output is empty")

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    fence_matches = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    for block in fence_matches:
        block = block.strip()
        if not block.startswith("{"):
            continue
        try:
            payload = json.loads(block)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            continue

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Model output does not contain a JSON object")
    return json.loads(text[start : end + 1])


def ocr_result_to_lines(ocr_result: OcrResult, max_lines: Optional[int] = None) -> list[LineItem]:
    lines: list[LineItem] = []
    for idx, line in enumerate(ocr_result.lines):
        lines.append(
            LineItem(
                line_index=idx,
                text=str(line.text).strip(),
                bbox=tuple(int(v) for v in line.bbox),
                confidence=_clamp01(float(line.conf) / 100.0),
            )
        )
    if max_lines is not None and max_lines > 0:
        return lines[: max_lines]
    return lines


def _build_schema_template() -> str:
    return json.dumps(
        {
            "answer": "short string answer in English",
            "evidence_lines": [
                {"line_index": 0, "text": "exact OCR line text", "bbox": [0, 0, 10, 10], "confidence": 0.9}
            ],
            "extracted_text": [
                {"line_index": 0, "text": "exact OCR line text", "bbox": [0, 0, 10, 10], "confidence": 0.9}
            ],
            "final_confidence": 0.9,
            "status": "ok",
        },
        ensure_ascii=False,
        indent=2,
    )


def build_analysis_prompt(question: str, ocr_lines: Sequence[LineItem], task_mode: str = "all") -> str:
    task_mode = task_mode.strip().lower()
    if task_mode not in TASK_MODES:
        raise ValueError(f"Unsupported task_mode={task_mode}. Expected one of {sorted(TASK_MODES)}")

    lines_block = "\n".join(
        f"{line.line_index}: text='{line.text}' bbox={list(line.bbox)} conf={line.confidence:.3f}"
        for line in ocr_lines
    )
    if not lines_block:
        lines_block = "NO_OCR_LINES"

    task_note = {
        "qa": "Answer the question using OCR lines. Keep extracted_text minimal.",
        "ocr": "Prioritize extracted_text; answer may be concise summary.",
        "all": "Answer the question and provide supporting extracted_text and evidence lines.",
    }[task_mode]

    return (
        "You are a vision-language assistant.\n"
        "Return STRICT JSON only.\n"
        f"Schema:\n{_build_schema_template()}\n\n"
        "Rules:\n"
        "- Use only OCR lines provided below for line_index/text/bbox.\n"
        "- evidence_lines must be a subset of OCR lines that support the answer.\n"
        "- extracted_text must contain OCR-derived lines relevant to the task.\n"
        "- status must be one of: ok, low_confidence, failed.\n"
        "- If evidence is weak, set status=low_confidence and lower final_confidence.\n"
        "- If impossible to answer, set status=failed and answer=''.\n"
        "- Output must be valid JSON.\n\n"
        f"Task mode: {task_mode}. {task_note}\n"
        f"Question: {question.strip()}\n\n"
        f"OCR lines:\n{lines_block}\n"
    )


def build_json_repair_prompt(raw_output: str) -> str:
    return (
        "Convert the following content into STRICT JSON matching this schema exactly.\n"
        f"Schema:\n{_build_schema_template()}\n\n"
        "Rules:\n"
        "- Keep existing information when possible.\n"
        "- If something is unknown, use empty fields and status='failed'.\n"
        "- Output JSON only.\n\n"
        f"Content:\n{raw_output}\n"
    )


def _find_best_line_index(text: str, lines: Sequence[LineItem]) -> Optional[int]:
    needle = _normalize_text(text)
    if not needle:
        return None
    for line in lines:
        if needle == _normalize_text(line.text):
            return line.line_index
    for line in lines:
        if needle in _normalize_text(line.text) or _normalize_text(line.text) in needle:
            return line.line_index
    return None


def _normalize_line_ref(item: Any, lines_by_idx: dict[int, LineItem]) -> Optional[LineItem]:
    if not isinstance(item, dict):
        return None

    line_index = item.get("line_index")
    if isinstance(line_index, bool):
        line_index = None
    if isinstance(line_index, (int, float)):
        line_index = int(line_index)
    else:
        line_index = None

    if line_index is None:
        line_index = _find_best_line_index(str(item.get("text", "")), list(lines_by_idx.values()))
    if line_index is None or line_index not in lines_by_idx:
        return None
    return lines_by_idx[line_index]


def validate_and_normalize_payload(
    payload: dict[str, Any],
    ocr_lines: Sequence[LineItem],
    low_conf_threshold: float = 0.35,
) -> AnalyzeResponse:
    lines_by_idx = {line.line_index: line for line in ocr_lines}
    answer = str(payload.get("answer", "")).strip()

    evidence_items: list[LineItem] = []
    for item in payload.get("evidence_lines", []) if isinstance(payload.get("evidence_lines"), list) else []:
        line = _normalize_line_ref(item, lines_by_idx)
        if line is not None and line not in evidence_items:
            evidence_items.append(line)

    extracted_items: list[LineItem] = []
    extracted_raw = payload.get("extracted_text", [])
    if isinstance(extracted_raw, list):
        for item in extracted_raw:
            line = _normalize_line_ref(item, lines_by_idx)
            if line is not None and line not in extracted_items:
                extracted_items.append(line)

    if not extracted_items:
        extracted_items = list(ocr_lines)

    if not evidence_items and answer:
        maybe_idx = _find_best_line_index(answer, ocr_lines)
        if maybe_idx is not None and maybe_idx in lines_by_idx:
            evidence_items.append(lines_by_idx[maybe_idx])

    parsed_conf = payload.get("final_confidence", None)
    if isinstance(parsed_conf, (int, float)):
        final_conf = _clamp01(float(parsed_conf))
    elif evidence_items:
        final_conf = float(sum(x.confidence for x in evidence_items) / max(1, len(evidence_items)))
    elif answer:
        final_conf = 0.25
    else:
        final_conf = 0.0

    status = str(payload.get("status", "")).strip().lower()
    if status not in {STATUS_OK, STATUS_LOW_CONFIDENCE, STATUS_FAILED}:
        if not answer:
            status = STATUS_FAILED
        elif final_conf < low_conf_threshold:
            status = STATUS_LOW_CONFIDENCE
        else:
            status = STATUS_OK

    if not answer and status != STATUS_FAILED:
        status = STATUS_FAILED
    if answer and final_conf < low_conf_threshold and status == STATUS_OK:
        status = STATUS_LOW_CONFIDENCE

    return AnalyzeResponse(
        answer=answer,
        evidence_lines=tuple(evidence_items),
        extracted_text=tuple(extracted_items),
        final_confidence=final_conf,
        status=status,
        raw_response="",
    )


class QwenVlmRunner:
    def __init__(
        self,
        model_path: str | Path,
        adapter_path: Optional[str | Path] = None,
        use_4bit: bool = True,
        device_mode: str = "auto",
        max_memory: Optional[dict[Any, str]] = None,
        offload_folder: Optional[str | Path] = None,
        low_cpu_mem_usage: bool = True,
        max_pixels: int | None = None,
    ) -> None:
        if device_mode not in {"auto", "balanced", "balanced_low_0", "sequential", "cpu"}:
            raise ValueError(f"Unsupported device_mode={device_mode}")
        self.model_path = str(Path(model_path))
        self.adapter_path = str(Path(adapter_path)) if adapter_path is not None else None
        self.use_4bit = bool(use_4bit)
        self.device_mode = device_mode
        self.max_pixels = max_pixels
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True, use_fast=False)

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": device_mode if device_mode != "cpu" else "cpu",
            "low_cpu_mem_usage": bool(low_cpu_mem_usage),
        }
        if max_memory is not None:
            model_kwargs["max_memory"] = max_memory
        if offload_folder is not None:
            offload_path = Path(offload_folder)
            offload_path.mkdir(parents=True, exist_ok=True)
            model_kwargs["offload_folder"] = str(offload_path)
            model_kwargs["offload_state_dict"] = True
        if use_4bit and device_mode != "cpu":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() and device_mode == "auto" else torch.float32

        self.model = AutoModelForImageTextToText.from_pretrained(self.model_path, **model_kwargs)
        if self.adapter_path is not None:
            if PeftModel is None:
                raise ImportError("peft is required to load adapter_path")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        self.model.eval()

    def generate(self, image_path: str | Path, prompt: str, max_new_tokens: int = 512) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": str(Path(image_path).resolve()),
                        **({"max_pixels": int(self.max_pixels)} if self.max_pixels is not None else {}),
                    },
                    {"type": "text", "text": str(prompt)},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                remove_invalid_values=True,
                renormalize_logits=True,
            )
        generated_ids_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


def _default_generator_factory(
    model_path: str | Path,
    adapter_path: Optional[str | Path],
    use_4bit: bool,
    device_mode: str,
) -> Callable[[str | Path, str, int], str]:
    runner = QwenVlmRunner(
        model_path=model_path,
        adapter_path=adapter_path,
        use_4bit=use_4bit,
        device_mode=device_mode,
    )
    return runner.generate


def analyze_image_with_vlm(
    image_path: str | Path,
    question: str,
    task_mode: str = "all",
    model_path: str | Path = Path("models") / "Qwen2.5-VL-3B-Instruct",
    adapter_path: Optional[str | Path] = None,
    use_4bit: bool = True,
    device_mode: str = "auto",
    max_new_tokens: int = 512,
    ocr_lang: str = "eng",
    ocr_min_conf: float = 35.0,
    max_ocr_lines: int = 60,
    low_conf_threshold: float = 0.35,
    retry_on_invalid_json: bool = True,
    generator: Optional[Callable[[str | Path, str, int], str]] = None,
) -> AnalyzeResponse:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    ocr_result = run_tesseract_ocr_v2(image_path=image_path, lang=ocr_lang, min_conf=ocr_min_conf)
    ocr_lines = ocr_result_to_lines(ocr_result, max_lines=max_ocr_lines)

    generate = generator or _default_generator_factory(
        model_path=model_path,
        adapter_path=adapter_path,
        use_4bit=use_4bit,
        device_mode=device_mode,
    )
    prompt = build_analysis_prompt(question=question, ocr_lines=ocr_lines, task_mode=task_mode)
    raw_output = generate(image_path, prompt, max_new_tokens)

    parsed_payload: Optional[dict[str, Any]] = None
    try:
        parsed_payload = _extract_json_object(raw_output)
    except Exception:
        parsed_payload = None

    if parsed_payload is None and retry_on_invalid_json:
        repair_prompt = build_json_repair_prompt(raw_output)
        repaired_output = generate(image_path, repair_prompt, max_new_tokens)
        try:
            parsed_payload = _extract_json_object(repaired_output)
            raw_output = repaired_output
        except Exception:
            parsed_payload = None
            raw_output = f"{raw_output}\n\n[REPAIR_ATTEMPT]\n{repaired_output}"

    if parsed_payload is None:
        response = AnalyzeResponse(
            answer="",
            evidence_lines=tuple(),
            extracted_text=tuple(ocr_lines),
            final_confidence=0.0,
            status=STATUS_FAILED,
            raw_response=raw_output,
        )
        return response

    normalized = validate_and_normalize_payload(parsed_payload, ocr_lines=ocr_lines, low_conf_threshold=low_conf_threshold)
    return AnalyzeResponse(
        answer=normalized.answer,
        evidence_lines=normalized.evidence_lines,
        extracted_text=normalized.extracted_text,
        final_confidence=normalized.final_confidence,
        status=normalized.status,
        raw_response=raw_output,
    )


def decode_base64_image_to_temp_file(image_base64: str, suffix: str = ".png") -> Path:
    data = base64.b64decode(image_base64.encode("utf-8"))
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        return Path(tmp.name)
