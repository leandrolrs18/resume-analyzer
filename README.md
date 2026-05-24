# Teddy Open Finance - Backend IA Resume Analyzer

API completa em Python 3.11 para análise stateless de currículos com OCR, ranking em memória, auditoria e observabilidade.

## Visão geral

Esta aplicação recebe múltiplos currículos em PDF, PNG e JPG/JPEG, extrai texto sem persistir os arquivos enviados, gera sumários curtos, responde queries semânticas de recrutamento e retorna um ranking com score, justificativas e citações dos trechos realmente usados.

Principais garantias do desafio:

- Não persiste arquivos recebidos.
- Usa extração nativa em PDFs quando houver texto disponível.
- Faz fallback para OCR com Tesseract apenas quando necessário.
- Calcula ranking lexical/BM25 em memória, sem salvar vetores ou embeddings.
- Limita as respostas ao texto extraído dos currículos.
- Registra auditoria e expõe métricas Prometheus.

## Arquitetura

```text
app/
  api/               Endpoints FastAPI
  core/              Configuração, segurança, logging e exceções
  models/            Schemas Pydantic
  observability/     Middleware, métricas e correlação de request
  rag/               Chunking de texto em memória
  repositories/      Persistência de logs em MongoDB
  services/          OCR, extração, sumarização, ranking, healthcheck
tests/
  unit/              Testes unitários
  contract/          Testes de contrato com httpx
```

### Fluxo de processamento

1. Upload multipart de múltiplos currículos.
2. Validação de MIME type, extensão, quantidade e tamanho.
3. PDFs com texto nativo são extraídos via PyMuPDF.
4. PDFs escaneados e imagens passam por Tesseract OCR.
5. O texto é dividido em chunks de aproximadamente 500 caracteres com overlap de 50.
6. Os chunks são ranqueados em memória com BM25 lexical contra a query.
7. O modelo local Hugging Face gera sumários e justificativas usando apenas o texto extraído.
8. O resultado é auditado no MongoDB.
9. Métricas e logs estruturados ficam disponíveis para operação.

## Stack

- Python 3.11
- FastAPI
- Tesseract OCR
- PyMuPDF
- Hugging Face Transformers
- MongoDB
- Prometheus
- Docker
- pytest
- httpx
- black
- ruff

## Decisões técnicas

### OCR e extração

- PDFs nativos: `PyMuPDF` com `page.get_text("text")`
- PDFs escaneados: renderização de páginas em memória e OCR com `Tesseract`
- Imagens: OCR direto em memória com Pillow + Tesseract

Essa estratégia reduz latência, evita custo computacional desnecessário e melhora precisão quando o PDF já contém texto pesquisável.

### Ranking sem persistência

- Chunks: 500 caracteres
- Overlap: 50 caracteres
- Busca: BM25 lexical em memória
- Persistência: nenhuma no disco

Os chunks e pontuações existem somente durante a requisição e são descartados ao final, o que favorece privacidade e simplifica a operação para a escala do desafio.

### Geração com LLM local

O projeto usa um modelo Hugging Face local configurável por variável de ambiente. O default está definido como:

- `Qwen/Qwen2.5-0.5B-Instruct`

Também é compatível com variantes leves do Phi/Qwen, desde que apontadas em `HF_MODEL`.

## Endpoints

### `POST /analyze`

`multipart/form-data`

Campos:

- `files`: múltiplos arquivos
- `query`: opcional
- `request_id`: obrigatório
- `user_id`: obrigatório

Sem `query`, a API retorna apenas sumários:

```json
{
  "request_id": "abc",
  "results": [
    {
      "candidate": "Maria",
      "summary": "Senior backend engineer with Python, AWS, FastAPI and distributed systems experience..."
    }
  ]
}
```

Com `query`, retorna ranking:

```json
{
  "request_id": "abc",
  "query": "Python AWS",
  "results": [
    {
      "candidate": "Maria",
      "score": 0.92,
      "summary": "Senior backend engineer...",
      "justification": "Strong overlap with Python and AWS based on the extracted evidence.",
      "citations": [
        {
          "chunk_id": "Maria-0",
          "text": "5 years with AWS and Python"
        }
      ]
    }
  ]
}
```

### `GET /healthz`

```json
{
  "status": "ok"
}
```

### `GET /metrics`

Exposto por `prometheus-fastapi-instrumentator`, com métricas customizadas:

- `requests_total`
- `request_latency_seconds`
- `ocr_failures_total`
- `llm_failures_total`

### `GET /logs/{request_id}`

Retorna os registros da collection `audit_logs`.

## MongoDB

Collection utilizada:

- `audit_logs`

Campos salvos:

- `request_id`
- `user_id`
- `timestamp`
- `query`
- `result`
- `latency_ms`
- `costs`
- `status`

## Observabilidade

### JSON logging

Todos os logs são emitidos em JSON com:

- timestamp
- level
- logger
- message
- request_id

### Correlação por request

Um middleware injeta ou reaproveita `x-request-id` em toda requisição.

### Métricas

- Contador total de requisições
- Histograma de latência
- Falhas de OCR
- Falhas de LLM

### Exceções

Existe tratamento global para erros de negócio e erros inesperados.

## Segurança

- Validação de MIME type
- Validação de extensão
- Limite de tamanho por upload
- Limite de quantidade de arquivos
- Limite de páginas por documento
- Não persistência dos arquivos recebidos
- Processamento em memória
- Temporários descartados imediatamente

## Performance

Meta do desafio:

- até 10 currículos
- até 15 páginas por currículo
- latência alvo de até 20 segundos

Otimizações implementadas:

- OCR paralelo por página para PDFs escaneados
- ranking lexical em memória, sem carga de modelo de embeddings
- bypass completo do OCR quando o PDF já tem texto nativo
- Tesseract no lugar de OCR neural pesado para reduzir instalação e latência em CPU

## Como rodar localmente

### 1. Configurar ambiente

```bash
cp .env.example .env
```

### 2. Subir com Docker Compose

```bash
docker compose up --build
```

O container instala `tesseract-ocr`, `tesseract-ocr-eng` e `tesseract-ocr-por` para OCR em inglês e português.

Serviços:

- API: [http://localhost:8000](http://localhost:8000)
- MongoDB: `mongodb://localhost:27017`

### 3. Rodar sem Docker

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Exemplos curl

### Sumários

```bash
curl -X POST http://localhost:8000/analyze \
  -F "request_id=req-001" \
  -F "user_id=recruiter-123" \
  -F "files=@./samples/maria.pdf"
```

### Ranking semântico

```bash
curl -X POST http://localhost:8000/analyze \
  -F "request_id=req-002" \
  -F "user_id=recruiter-123" \
  -F "query=Python AWS FastAPI" \
  -F "files=@./samples/maria.pdf" \
  -F "files=@./samples/joao.png"
```

### Logs

```bash
curl http://localhost:8000/logs/req-002
```

### Métricas

```bash
curl http://localhost:8000/metrics
```

## Testes

Rodar toda a suíte:

```bash
pytest
```

O projeto também gera coverage da pasta `app` via `pytest-cov`.

Lint e formatação:

```bash
black --check .
ruff check .
```

## CI com GitHub Actions

Pipeline configurado para:

- instalar dependências
- executar `black --check`
- executar `ruff check`
- executar `pytest`
- validar `docker build`

## RAG: como evita alucinação

O sistema reduz alucinação por design:

- a resposta é sempre condicionada ao texto extraído do currículo
- as justificativas são geradas apenas com base nos chunks retornados pelo ranking em memória
- toda resposta de ranking inclui `citations`
- quando a evidência for fraca, o prompt instrui o modelo a dizer isso explicitamente

## Why BM25 instead of a vector database?

Para este desafio stateless, BM25 em memória foi escolhido no lugar de embeddings
ou banco vetorial dedicado pelos seguintes motivos:

- baixa escala do problema
- simplicidade operacional
- menor latência e menos dependências pesadas em CPU
- privacidade, já que arquivos, vetores e embeddings não são persistidos

Em um cenário de produção com múltiplas requisições concorrentes, histórico persistente e busca semântica distribuída, embeddings e um banco vetorial dedicado poderiam passar a fazer sentido.

## Tradeoffs

- O uso de um modelo local leve reduz custo e dependência externa, mas pode ter qualidade inferior a modelos maiores.
- BM25 é rápido e simples, mas pode capturar menos semântica que embeddings em consultas muito abstratas.
- OCR em CPU é suficiente para a escala pedida, mas pode ser o principal gargalo sob carga maior.

## Future improvements

- Redis cache para resultados de consultas repetidas, se a regra de negócio permitir estado
- Celery/background workers para OCR pesado e filas de processamento
- Kubernetes para autoscaling e operação multiambiente
- Vertex AI migration para variantes gerenciadas de modelos
- busca semântica opcional para cenários em que o requisito stateless não exista

## Estrutura de ambiente

Variáveis disponíveis em `.env.example`:

- `MONGO_URI`
- `MONGO_DB`
- `HF_MODEL`
- `LOG_LEVEL`
