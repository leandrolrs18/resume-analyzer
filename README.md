---
title: Analisador de Currículos com IA
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Analisador de Currículos com IA

## Introdução

Este projeto é uma API stateless para triagem de currículos. A aplicação recebe múltiplos
currículos em PDF, PNG, JPG ou JPEG, extrai texto com leitura nativa de PDF e OCR, gera sumários
curtos e responde perguntas de recrutamento com ranking, score, justificativas e citações.

O desenho central combina recuperação em memória com um LLM local. Em vez de salvar documentos,
currículos ou vetores, cada requisição envia os arquivos e a pergunta ao mesmo tempo. A API
processa tudo em tempo real, encontra as evidências mais relevantes, aciona o LLM para sintetizar
a resposta e registra apenas logs de auditoria com metadados e resultado.

## Por que triagem de currículos?

Times de recrutamento lidam com muitos currículos escritos em linguagem natural. Uma busca
simples por palavras-chave costuma falhar quando candidatos descrevem experiências equivalentes
com termos diferentes, quando as evidências aparecem fora da seção de habilidades, ou quando a
vaga exige análise contextual.

Este projeto busca reduzir esse problema usando recuperação sobre trechos extraídos dos
currículos. O resultado é uma resposta mais útil para o recrutador e auditável por meio de
citações.

## Por que RAG?

Arquiteturas RAG ajudam a tornar respostas de LLM mais confiáveis porque condicionam a geração em
um contexto recuperado. Neste projeto, a recuperação acontece somente dentro da requisição atual:

- **Extração nativa e OCR:** PDFs com texto selecionável são lidos diretamente; PDFs escaneados e
  imagens usam OCR.
- **Parsing estruturado em memória:** um section splitter simples, com spaCy NER para nome quando
  disponível, separa formação, experiência, skills, certificações, projetos e idiomas em um JSON
  canônico temporário.
- **Views semânticas:** o JSON estruturado gera representações textuais específicas para busca.
- **Chunking em memória:** o texto bruto e as views estruturadas são divididos em trechos menores.
- **Recuperação em memória:** os trechos podem ser ranqueados por BM25, embeddings em memória
  ou modo híbrido.
- **Geração baseada em evidências:** o LLM local recebe apenas as principais evidências.
- **Citações:** cada candidato ranqueado retorna os trechos usados como base.

Não há banco vetorial e os arquivos enviados não são persistidos.

## Demonstração

Link da aplicação:

> TODO: colar link do deploy aqui.

Tela inicial:

> TODO: colar print aqui.

Exemplo de resposta com descrição de vaga:

> TODO: colar print aqui.

Exemplo de resposta com pergunta específica de recrutamento:

> TODO: colar print aqui.

Painel de logs e métricas:

> TODO: colar print aqui.

## Descrição do sistema

### 1. Fluxo da requisição

A API é stateless. O usuário envia currículos, `request_id`, `user_id` e uma `query` opcional na
mesma chamada `POST /analyze`.

Quando a query não é enviada, a API retorna um sumário curto por currículo. Quando a query é
enviada, a API retorna candidatos ranqueados com score normalizado, sumário, justificativa e
citações.

```mermaid
flowchart LR
    A["POST /analyze<br/>files + query + request_id + user_id"] --> B["Validação<br/>tipo, tamanho e quantidade"]
    B --> C["Extração PDF/imagem<br/>PyMuPDF + Tesseract por+eng"]
    C --> D["Section splitter + spaCy NER<br/>JSON simples em memória"]
    D --> E["Views semânticas<br/>perfil, formação, experiência, skills"]
    E --> F["Chunking em memória<br/>texto bruto + views"]
    F --> G["Ranking BM25 / embeddings<br/>em memória"]
    G --> H["Principais evidências"]
    H --> I["LLM final<br/>Qwen local ou Groq"]
    I --> J["Resposta<br/>ranking, score, sumário, justificativa e citações"]
    J --> K["audit_logs<br/>metadados e resultado"]
```

### 2. Recuperação e geração

Antes do ranking, o parser cria um JSON simples em memória com campos como `education`,
`experience`, `skills`, `certifications`, `projects` e `languages`. Ele usa títulos de seção em
português e inglês, e spaCy NER apenas para apoiar a identificação de nome quando o modelo está
instalado. Esse JSON não é persistido; ele serve para gerar views semânticas, por exemplo
"formação", "competências" e "experiência".

O ranqueador pode operar em três modos: `bm25`, `embedding` ou `hybrid`. O modo híbrido combina
BM25, que preserva termos literais como tecnologias, instituições e certificações, com embeddings
em memória, que melhoram perguntas abertas e semânticas. Os vetores são criados somente durante a
requisição e não são persistidos. O LLM entra depois da recuperação, recebendo um prompt curto com
as principais evidências. Caso o LLM falhe ou devolva JSON inválido, o sistema usa fallback baseado
no texto recuperado.

### 3. Estratégia anti-alucinação

A API reduz alucinação com as seguintes camadas:

- O prompt instrui o LLM a usar somente as evidências fornecidas.
- Justificativas são vinculadas às citações recuperadas.
- O fallback de justificativa é extrativo.
- Trechos com muitos dados de contato são penalizados e contatos são redigidos nas citações.

Isso não é uma prova matemática contra alucinação, mas torna a resposta auditável e baseada no
texto extraído.

### 4. Segurança e privacidade

- Arquivos de currículo não são salvos.
- Vetores não são salvos.
- Os uploads são processados em memória.
- Os formatos aceitos são PDF, PNG, JPG e JPEG.
- Há limites de tamanho, quantidade de arquivos e páginas por documento.
- PDFs criptografados, protegidos por senha ou com arquivos embutidos são rejeitados.
- A auditoria salva metadados e resultado da requisição, não os arquivos originais.

## Stack técnica

- **Linguagem:** Python 3.11
- **API:** FastAPI
- **OCR:** Tesseract OCR com pacotes de português e inglês
- **PDF:** PyMuPDF
- **Recuperação:** BM25, embeddings em memória ou modo híbrido
- **LLM:** Qwen2.5 GGUF local via llama.cpp
- **Banco:** MongoDB para auditoria
- **Métricas:** formato Prometheus
- **Infra:** Docker e Docker Compose

## API

### `POST /analyze`

Campos `multipart/form-data`:

- `files`: um ou mais currículos PDF, PNG, JPG ou JPEG
- `request_id`: identificador da requisição
- `user_id`: identificador do usuário
- `query`: pergunta ou requisito de recrutamento opcional

Resposta sem `query`:

```json
{
  "request_id": "req-001",
  "query": null,
  "results": [
    {
      "candidate": "Maria Silva",
      "summary": "Engenheira backend com experiência em Python, FastAPI e AWS."
    }
  ]
}
```

Resposta com `query`:

```json
{
  "request_id": "req-002",
  "query": "backend Python FastAPI Docker",
  "results": [
    {
      "rank": 1,
      "candidate": "Maria Silva",
      "score": 0.93,
      "summary": "Engenheira backend com experiência em Python e AWS.",
      "justification": "A candidata apresenta evidências diretas de Python, FastAPI e Docker.",
      "citations": [
        {
          "chunk_id": "Maria Silva-2",
          "text": "APIs com Python, FastAPI, Docker e AWS."
        }
      ]
    }
  ]
}
```

### `GET /healthz`

Retorna o estado da API.

### `GET /metrics`

Retorna métricas no padrão Prometheus, incluindo:

- `requests_total`
- `request_latency_seconds`
- `ocr_failures_total`
- `llm_failures_total`

### `GET /logs/{request_id}`

Retorna os registros de auditoria salvos no MongoDB para uma requisição.

## Instalação e execução

### 1. Clonar o projeto

```bash
git clone <url-do-repositorio>
cd teddy
```

### 2. Configurar variáveis de ambiente

Crie um arquivo `.env` na raiz:

```env
MONGO_URI=mongodb://mongodb:27017
MONGO_DB=resume-analyzer
USE_LOCAL_LLM=true
LOG_LEVEL=INFO
GROQ_API_KEY=
GROQ_MODEL=llama-3.3-70b-versatile
```

`GROQ_API_KEY` é opcional. Quando configurada, o front permite usar a opção
`Groq Llama 3.3 70B`; quando ausente, o sistema continua funcionando com o modelo local.
Não coloque chaves reais no repositório.

Ao selecionar Groq, somente as evidências ranqueadas e a pergunta são enviadas para a
API externa de LLM. Para execução 100% privada dentro do container, use o modelo local.

### 3. Rodar com Docker Compose

Para subir a API e suas dependências:

```bash
docker compose up -d --build api
```

Endereços úteis:

- Frontend: `http://localhost:8000`
- Frontend em inglês: `http://localhost:8000/en`
- Swagger: `http://localhost:8000/docs`
- Métricas: `http://localhost:8000/metrics`

### 4. Testar com curl

```bash
curl --max-time 60 -sS \
  -X POST http://localhost:8000/analyze \
  -F 'request_id=req-demo-001' \
  -F 'user_id=recrutador-demo' \
  -F 'query=backend Python FastAPI Docker AWS' \
  -F 'files=@/caminho/para/curriculo.pdf;type=application/pdf'
```

### 5. Consultar auditoria

```bash
curl http://localhost:8000/logs/req-demo-001
```

## Testes e qualidade

Rodar testes:

```bash
.venv/bin/pytest
```

Rodar lint e checagem de formatação:

```bash
.venv/bin/ruff check app tests
.venv/bin/black --check app tests
```

Cobertura atual de qualidade:

- Testes unitários de OCR, ranking, citações, chunking, serviço de LLM e sumarização
- Contract tests com pytest/httpx
- Ruff
- Black
- Métricas verificáveis em `/metrics`

## Limites

Limites padrão:

- Até 10 arquivos por requisição
- Até 10 MB por arquivo
- Até 15 páginas por documento
- Formatos aceitos: PDF, PNG, JPG, JPEG

Meta de latência:

- A meta é responder em até 20 segundos para casos práticos.
- PDFs nativos são mais rápidos do que PDFs escaneados.
- PDFs escaneados grandes dependem do tempo de OCR.
- No macOS, Docker executa o modelo GGUF em CPU, sem aceleração Metal.

## Escalabilidade

O desenho atual prioriza conformidade com o desafio e privacidade:

- Processamento stateless
- Sem persistência de arquivos
- Sem persistência de vetores
- Recuperação em memória
- LLM local

Evoluções naturais para produção:

- Workers separados para OCR de PDFs escaneados
- Fila para lotes pesados
- Réplicas horizontais da API
- MongoDB externo com política de retenção
- Endpoint de LLM gerenciado para menor latência
- Validação pós-geração mais rígida para reforçar anti-alucinação

## Agradecimento

Inspirado em pipelines RAG para triagem de currículos, adaptado para um desafio backend stateless
em que arquivos, currículos e vetores não podem ser persistidos.
