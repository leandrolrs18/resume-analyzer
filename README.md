---
title: AI Resume Analyzer
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# AI Resume Analyzer

API stateless de alta performance para análise automatizada de currículos com extração de texto nativo, suporte a OCR, inteligência artificial local e auditoria estruturada.

---

## 🚀 Visão Geral

Esta aplicação recebe múltiplos currículos nos formatos PDF, PNG e JPEG, realiza a extração do texto sem persistir nenhum arquivo físico e utiliza inteligência artificial local para gerar sumários executivos ou realizar o ranqueamento dos candidatos com base em critérios de recrutamento.

### 🛡️ Garantias e Regras de Negócio
* **Segurança e Privacidade:** Os arquivos são processados estritamente em memória e descartados após a requisição.
* **Extração Híbrida:** Extrai o texto nativo de PDFs sempre que disponível. Faz fallback automático para OCR com Tesseract apenas em imagens ou PDFs escaneados.
* **Alinhamento Contextual:** O modelo de IA é blindado para gerar resumos e justificativas baseando-se **apenas** nas informações reais extraídas dos documentos, mitigando alucinações.
* **Busca Lexical Baseada em Evidências:** Utiliza o algoritmo BM25 em memória para ranquear os trechos mais relevantes frente à busca, anexando citações diretas no retorno da API.

---

## 🛠️ Stack Técnica

* **Core:** Python 3.11, FastAPI, Pydantic v2
* **Processamento de Documentos:** PyMuPDF (extração nativa) & Tesseract OCR
* **Inteligência Artificial:** Hugging Face Transformers / LLM GGUF local
* **Persistência de Auditoria:** MongoDB
* **Observabilidade:** Prometheus Metrics & JSON Structured Logging

---

## 🏁 Como Rodar o Projeto Localmente

### 1. Configurar as Variáveis de Ambiente
Crie um arquivo `.env` na raiz do projeto com as seguintes chaves:
```env
MONGO_URI=mongodb://mongodb:27017
MONGO_DB=resume-analyzer
USE_LOCAL_LLM=true
LOG_LEVEL=INFO