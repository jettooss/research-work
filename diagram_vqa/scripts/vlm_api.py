from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from vqa_retrieval.vlm_service import (
    AnalyzeResponse,
    analyze_image_with_vlm,
    decode_base64_image_to_temp_file,
)


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = ROOT.parent


class AnalyzeRequest(BaseModel):
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    image_bytes: Optional[str] = None
    question: str = Field(min_length=1)
    task_mode: Literal["qa", "ocr", "all"] = "all"
    max_new_tokens: int = Field(default=512, ge=64, le=2048)

    @model_validator(mode="after")
    def _check_image_source(self):
        if not self.image_path and not self.image_base64 and not self.image_bytes:
            raise ValueError("Either image_path or image_base64/image_bytes is required")
        return self


class AnalyzeApiResponse(BaseModel):
    answer: str
    evidence_lines: list[dict]
    extracted_text: list[dict]
    final_confidence: float
    status: Literal["ok", "low_confidence", "failed"]
    raw_response: str


def _response_to_api(response: AnalyzeResponse) -> AnalyzeApiResponse:
    payload = response.to_dict()
    return AnalyzeApiResponse(**payload)


def create_app() -> FastAPI:
    app = FastAPI(title="VLM Analyze API", version="1.0.0")

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.post("/vlm/analyze", response_model=AnalyzeApiResponse)
    def analyze(request: AnalyzeRequest):
        temp_path: Optional[Path] = None
        try:
            if request.image_path:
                image_path = Path(request.image_path)
            else:
                encoded = request.image_base64 or request.image_bytes or ""
                temp_path = decode_base64_image_to_temp_file(encoded, suffix=".png")
                image_path = temp_path

            response = analyze_image_with_vlm(
                image_path=image_path,
                question=request.question,
                task_mode=request.task_mode,
                model_path=EXTERNAL_ROOT / "models" / "Qwen2.5-VL-3B-Instruct",
                max_new_tokens=request.max_new_tokens,
            )
            return _response_to_api(response)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Analyze failed: {type(exc).__name__}: {exc}") from exc
        finally:
            if temp_path is not None:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("scripts.vlm_api:app", host="0.0.0.0", port=8000, reload=False)
