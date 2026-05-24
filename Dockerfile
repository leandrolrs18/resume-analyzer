# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    LLM_MODEL_PATH=/home/user/app/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
    HOME=/home/user

# Instala dependências do sistema
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

# Cria o usuário sem privilégios exigido pelo Hugging Face (UID 1000)
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

# Copia os arquivos de definição de dependências com permissão para o usuário
COPY --chown=user pyproject.toml README.md ./

RUN pip install --upgrade pip

# Instala o projeto e as dependências
RUN pip install --no-cache-dir .

# Cria a pasta para o modelo e baixa o Qwen2.5 automaticamente
RUN mkdir -p /home/user/app/models && \
    wget -O /home/user/app/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
    https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf

# Copia o código da aplicação e o frontend estático
COPY --chown=user app ./app

# O Hugging Face Spaces exige EXCLUSIVAMENTE a porta 7860
EXPOSE 7860

# Inicia o Uvicorn na porta correta exigida pela nuvem
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]