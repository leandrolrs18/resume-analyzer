# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    HOME=/home/user

# Garante que os binários instalados via --user sejam encontrados no PATH global
ENV PATH="/home/user/.local/bin:${PATH}"

# Instala dependências do sistema necessárias para compilação, manipulação de imagem e OCR
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

# Cria o usuário sem privilégios exigido pela infraestrutura do Hugging Face (UID 1000)
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user

# Copia os arquivos de definição do pacote com as permissões corretas para o usuário local
COPY --chown=user pyproject.toml README.md ./

RUN pip install --no-cache-dir --upgrade pip

# Instala as dependências do projeto no escopo do usuário local (--user)
RUN pip install --no-cache-dir --user .

# Cria a pasta para o modelo de IA e realiza o download direto do Qwen2.5-1.5B (GGUF) automaticamente
RUN mkdir -p /home/user/models && \
    wget -O /home/user/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
    https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf

# Configura o caminho exato onde o serviço LLM vai buscar o arquivo .gguf baixado acima
ENV LLM_MODEL_PATH=/home/user/models/qwen2.5-1.5b-instruct-q4_k_m.gguf

# Copia a pasta de código da aplicação e arquivos estáticos (Frontend) para a raiz do container do usuário
COPY --chown=user app ./app

# Expõe as portas de desenvolvimento (8000) e de produção do Hugging Face (7860)
EXPOSE 8000
EXPOSE 7860

# Configura o PYTHONPATH explicitamente na raiz do usuário para resolver 'from app.models ...' de forma absoluta
ENV PYTHONPATH="/home/user"

# Inicia o servidor uvicorn chamando o módulo direto do Python e aceitando a porta dinâmica injetada na nuvem
CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]