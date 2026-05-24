# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    HOME=/home/user

ENV PATH="/home/user/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-por \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

USER user
WORKDIR /home/user

RUN pip install --no-cache-dir --upgrade pip

COPY --chown=user pyproject.toml ./
COPY --chown=user README.md ./
COPY --chown=user app ./app

RUN pip install --no-cache-dir --user .

RUN find /home/user/app -type d -name __pycache__ -prune -exec rm -rf {} + && \
    python - <<'PY'
import importlib

for module in ("app.schemas", "app.api.routes", "app.main"):
    importlib.import_module(module)
    print(f"import ok: {module}")
PY

RUN mkdir -p /home/user/models && \
    wget --tries=3 --timeout=60 -O /home/user/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
    "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf?download=true"

ENV LLM_MODEL_PATH=/home/user/models/qwen2.5-1.5b-instruct-q4_k_m.gguf
ENV USE_LOCAL_LLM=true
ENV SUMMARIZATION_MAX_NEW_TOKENS=80
ENV JUSTIFICATION_MAX_NEW_TOKENS=80
ENV PYTHONPATH="/home/user:/home/user/app"

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
