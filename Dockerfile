# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace/diagram_vqa
COPY diagram_vqa/requirements-cpu.txt ./requirements-cpu.txt
RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 \
    && python -m pip install -r requirements-cpu.txt

COPY diagram_vqa/pyproject.toml ./pyproject.toml
COPY diagram_vqa/src ./src
COPY diagram_vqa/examples ./examples
COPY diagram_vqa/scripts/run_model_matrix.py diagram_vqa/scripts/run_model_matrix_baseline.py ./scripts/
RUN python -m pip install --no-deps .

CMD ["python", "examples/retrieval_metrics.py"]
