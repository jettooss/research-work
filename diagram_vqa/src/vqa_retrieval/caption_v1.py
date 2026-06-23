from __future__ import annotations

import json
import mimetypes
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole


DEFAULT_PROMPT = """
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
- Focus on semantic meaning, not full OCR transcription.
- short_description must be a brief 1-sentence description of what is happening or shown in the image.
- Do not invent unseen details.
- If the image is unclear, say so in the summary.
- Output only valid JSON.
""".strip()


@dataclass(frozen=True)
class CaptionResult:
    image_path: str
    model: str
    visual_type: str
    short_description: str
    summary: str
    main_objects: tuple[str, ...]
    relationships: tuple[str, ...]
    important_text: tuple[str, ...]
    raw_response: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Model response does not contain a JSON object")
    return json.loads(text[start : end + 1])


def normalize_caption_payload(payload: dict[str, Any]) -> dict[str, Any]:
    def _as_list_str(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    return {
        "visual_type": str(payload.get("visual_type", "")).strip(),
        "short_description": str(
            payload.get("short_description") or payload.get("summary", "")
        ).strip(),
        "summary": str(payload.get("summary", "")).strip(),
        "main_objects": _as_list_str(payload.get("main_objects")),
        "relationships": _as_list_str(payload.get("relationships")),
        "important_text": _as_list_str(payload.get("important_text")),
    }


def generate_image_caption(
    image_path: str | Path,
    credentials: str | None = None,
    model: str = "GigaChat-2-Max",
    prompt: str = DEFAULT_PROMPT,
    timeout: float = 120,
    verify_ssl_certs: bool = False,
) -> CaptionResult:
    image_path = Path(image_path)
    credentials = credentials or os.environ.get("GIGACHAT_CREDENTIALS")
    if not credentials:
        raise ValueError("GIGACHAT_CREDENTIALS is required")

    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
    file_bytes = image_path.read_bytes()

    with GigaChat(
        credentials=credentials,
        model=model,
        verify_ssl_certs=verify_ssl_certs,
        timeout=timeout,
    ) as giga_chat_model:
        uploaded = giga_chat_model.upload_file((image_path.name, file_bytes, mime_type), purpose="general")
        payload = Chat(messages=[
            Messages(
                role=MessagesRole.SYSTEM,
                content="You are an expert in visual understanding of diagrams, infographics, and scientific illustrations. Return only valid JSON in English. Include a short_description field with one brief sentence describing what is happening or shown in the image.",
            ),
            Messages(
                role=MessagesRole.USER,
                content=prompt,
                attachments=[uploaded.id_],
            ),
        ])
        response = giga_chat_model.chat(payload)
        raw_text = response.choices[0].message.content

    normalized = normalize_caption_payload(extract_json_object(raw_text))
    return CaptionResult(
        image_path=str(image_path),
        model=model,
        visual_type=normalized["visual_type"],
        short_description=normalized["short_description"],
        summary=normalized["summary"],
        main_objects=tuple(normalized["main_objects"]),
        relationships=tuple(normalized["relationships"]),
        important_text=tuple(normalized["important_text"]),
        raw_response=raw_text,
    )


def save_caption_result(result: CaptionResult, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, ensure_ascii=False, indent=2)
    return output_path
