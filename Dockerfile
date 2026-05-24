# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    HOME=/home/user

ENV PATH="/home/user/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-por git wget \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user

RUN pip install --no-cache-dir --upgrade pip
COPY --chown=user pyproject.toml ./
RUN pip install --no-cache-dir --user .

# Cria os caminhos necessários e o link simbólico para o modelo
RUN mkdir -p /home/user/models /app/models && \
    wget -O /home/user/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
    https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf && \
    ln -s /home/user/models/qwen2.5-1.5b-instruct-q4_k_m.gguf /app/models/qwen2.5-1.5b-instruct-q4_k_m.gguf

ENV LLM_MODEL_PATH=/app/models/qwen2.5-1.5b-instruct-q4_k_m.gguf

# Copia o código para dentro de uma pasta chamada 'app' na raiz, 
# mas garantindo que o diretório atual seja o pai
COPY --chown=user app ./app

# A chave aqui: PYTHONPATH aponta para a pasta onde a pasta 'app' reside
ENV PYTHONPATH="/home/user"

EXPOSE 8000 7860

# Comando ajustado: rodar o uvicorn a partir do módulo 'app.main'
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]