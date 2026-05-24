# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    LLM_MODEL_PATH=/home/user/app/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
    HOME=/home/user

# Instala dependências do sistema: compiladores, bibliotecas de imagem e Tesseract OCR
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

# Copia os ficheiros de definição de dependências com permissão para o usuário
COPY --chown=user pyproject.toml README.md ./

RUN pip install --upgrade pip

# Instala o projeto e as dependências
RUN pip install --no-cache-dir .

# Cria a pasta para o modelo e faz o download direto do Qwen2.5-1.5B (GGUF) automaticamente
RUN mkdir -p /home/user/app/models && \
    wget -O /home/user/app/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
    https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf

# Copia o código da aplicação e o frontend estático
COPY --chown=user app ./app

# Expõe as portas padrão de desenvolvimento e de produção da nuvem
EXPOSE 8000
EXPOSE 7860

# O PULO DO GATO: Se a variável PORT existir (Hugging Face injeta 7860), ele usa.
# Se rodar local no Mac, ele cai no fallback e usa a porta 8000 padrão do Docker Compose.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]