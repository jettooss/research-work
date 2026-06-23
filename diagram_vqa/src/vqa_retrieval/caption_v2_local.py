from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)


DEFAULT_LOCAL_PROMPT = """
Analyze this image and describe it in English.

Return JSON only:
{
  "visual_type": "...",
  "short_description": "...",
  "summary": "...",
  "main_objects": ["..."],
  "relationships": ["..."],
  "important_text": ["..."]
}

Rules:
- Be concise and accurate.
- short_description must be one short sentence about what is happening or shown.
- Focus on semantic meaning, not full OCR transcription.
- Include only the most important visible text.
- Do not invent unseen details.
- If the image is unclear, say so in the summary.
- Output only valid JSON.
""".strip()


@dataclass(frozen=True)
class LocalCaptionResult:
    image_path: str
    model_path: str
    visual_type: str
    short_description: str
    summary: str
    main_objects: tuple[str, ...]
    relationships: tuple[str, ...]
    important_text: tuple[str, ...]
    raw_response: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Model response is empty")

    # Try strict JSON first.
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    # Try fenced code blocks and generic embedded JSON object extraction.
    candidates: list[str] = []
    fence_matches = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    for block in fence_matches:
        block = block.strip()
        if block.startswith("{"):
            candidates.append(block)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            continue

    raise ValueError("Model response does not contain a valid JSON object")


def _strip_markdown_fences(text: str) -> str:
    stripped = (text or "").strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _extract_quoted_field(text: str, field_name: str) -> str:
    pattern = rf'"{re.escape(field_name)}"\s*:\s*"((?:\\.|[^"\\])*)"'
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        return ""
    value = match.group(1)
    # Decode escaped sequences safely.
    return bytes(value, "utf-8").decode("unicode_escape").strip()


def _extract_array_field_loose(text: str, field_name: str) -> list[str]:
    start_match = re.search(rf'"{re.escape(field_name)}"\s*:\s*\[', text)
    if not start_match:
        return []

    tail = text[start_match.end():]
    items: list[str] = []
    token_chars: list[str] = []
    in_str = False
    escaped = False

    for ch in tail:
        if in_str:
            if escaped:
                token_chars.append(ch)
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                token = "".join(token_chars).strip()
                token_chars.clear()
                in_str = False
                if token:
                    try:
                        token = bytes(token, "utf-8").decode("unicode_escape").strip()
                    except UnicodeDecodeError:
                        token = token.strip()
                    if token:
                        items.append(token)
                continue
            token_chars.append(ch)
            continue

        if ch == "]":
            break
        if ch == '"':
            in_str = True
            token_chars.clear()

    return items


def _fallback_payload_from_text(text: str) -> dict[str, Any]:
    stripped = _strip_markdown_fences(text)
    visual_type = _extract_quoted_field(stripped, "visual_type")
    short_description = _extract_quoted_field(stripped, "short_description")
    summary = _extract_quoted_field(stripped, "summary")
    main_objects = _extract_array_field_loose(stripped, "main_objects")
    relationships = _extract_array_field_loose(stripped, "relationships")
    important_text = _extract_array_field_loose(stripped, "important_text")

    raw = " ".join(stripped.split())
    if not raw:
        raw = "Image description is unavailable."

    if not short_description:
        short_description = summary or raw[:200].strip()
    if not summary:
        summary = raw

    return {
        "visual_type": visual_type,
        "short_description": short_description,
        "summary": summary,
        "main_objects": main_objects,
        "relationships": relationships,
        "important_text": important_text,
    }


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    def _as_list_str(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    summary = str(payload.get("summary", "")).strip()
    return {
        "visual_type": str(payload.get("visual_type", "")).strip(),
        "short_description": str(payload.get("short_description") or summary).strip(),
        "summary": summary,
        "main_objects": _as_list_str(payload.get("main_objects")),
        "relationships": _as_list_str(payload.get("relationships")),
        "important_text": _as_list_str(payload.get("important_text")),
    }


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=2)
def _load_processor(model_path: str):
    return AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )


@lru_cache(maxsize=4)
def _load_model(model_path: str, use_4bit: bool, device_mode: str = "auto"):
    if device_mode not in {"auto", "cpu"}:
        raise ValueError(f"Unsupported device_mode: {device_mode}")

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto" if device_mode == "auto" else "cpu",
    }
    if use_4bit and device_mode != "cpu":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["dtype"] = torch.float16
    else:
        kwargs["dtype"] = torch.float16 if (device_mode == "auto" and _device() == "cuda") else torch.float32

    return AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)


def _is_cuda_generation_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "cuda error" in message
        or "device-side assert" in message
        or "cublas" in message
        or "cudnn" in message
    )


def _generate_output_text(
    image_path: Path,
    model_path: str,
    processor,
    prompt: str,
    max_new_tokens: int,
    use_4bit: bool,
    device_mode: str,
) -> str:
    model = _load_model(model_path, use_4bit=use_4bit, device_mode=device_mode)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path.resolve())},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    model_device = next(model.parameters()).device
    inputs = {key: value.to(model_device) if hasattr(value, "to") else value for key, value in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def generate_local_image_caption(
    image_path: str | Path,
    model_path: str | Path,
    prompt: str = DEFAULT_LOCAL_PROMPT,
    max_new_tokens: int = 256,
    use_4bit: bool = True,
) -> LocalCaptionResult:
    image_path = Path(image_path)
    model_path = str(Path(model_path))
    processor = _load_processor(model_path)
    try:
        output_text = _generate_output_text(
            image_path=image_path,
            model_path=model_path,
            processor=processor,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            use_4bit=use_4bit,
            device_mode="auto",
        )
    except RuntimeError as exc:
        if not _is_cuda_generation_error(exc):
            raise
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass
        output_text = _generate_output_text(
            image_path=image_path,
            model_path=model_path,
            processor=processor,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            use_4bit=False,
            device_mode="cpu",
        )

    try:
        parsed_payload = _extract_json_object(output_text)
    except ValueError:
        parsed_payload = _fallback_payload_from_text(output_text)
    normalized = _normalize_payload(parsed_payload)
    return LocalCaptionResult(
        image_path=str(image_path),
        model_path=model_path,
        visual_type=normalized["visual_type"],
        short_description=normalized["short_description"],
        summary=normalized["summary"],
        main_objects=tuple(normalized["main_objects"]),
        relationships=tuple(normalized["relationships"]),
        important_text=tuple(normalized["important_text"]),
        raw_response=output_text,
    )


def save_local_caption_result(result: LocalCaptionResult, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, ensure_ascii=False, indent=2)
    return output_path
