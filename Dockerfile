# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    HOME=/home/user

# Adiciona os binários locais ao PATH do sistema
ENV PATH="/home/user/.local/bin:${PATH}"

# Instala dependências do sistema obrigatórias
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

# Cria e define o usuário padrão do Hugging Face
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user

RUN pip install --no-cache-dir --upgrade pip

# Copia apenas as definições de dependências para instalar os pacotes limpos
COPY --chown=user pyproject.toml ./

# Instala as dependências externas usando o próprio pyproject de forma isolada
RUN pip install --no-cache-dir --user .

# Cria a estrutura e baixa o modelo GGUF local
RUN mkdir -p /home/user/models && \
    wget -O /home/user/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
    https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf

ENV LLM_MODEL_PATH=/home/user/models/qwen2.5-1.5b-instruct-q4_k_m.gguf

# COPIA COMPLETA DA PASTA LOCAL DO PROJETO
# Isso garante que o Python leia os arquivos e subpastas em tempo real
COPY --chown=user app ./app

EXPOSE 8000
EXPOSE 7860

# Configura o PYTHONPATH apontando diretamente para o diretório pai (/home/user)
# Isso faz com que 'from app.models' busque fisicamente em /home/user/app/models
ENV PYTHONPATH="/home/user"

CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]