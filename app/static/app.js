const form = document.querySelector("#analyze-form");
const filesInput = document.querySelector("#files");
const fileList = document.querySelector("#file-list");
const clearButton = document.querySelector("#clear-button");
const results = document.querySelector("#results");
const apiStatus = document.querySelector("#api-status");
const latency = document.querySelector("#latency");
const requestIdLabel = document.querySelector("#request-id");
const resultCount = document.querySelector("#result-count");
const jsonToggle = document.querySelector("#json-toggle");
const jsonOutput = document.querySelector("#json-output");
const diagnosticsToggle = document.querySelector("#diagnostics-toggle");
const diagnosticsPanel = document.querySelector("#diagnostics-panel");
const diagnosticsClose = document.querySelector("#diagnostics-close");
const refreshMetrics = document.querySelector("#refresh-metrics");
const refreshLogs = document.querySelector("#refresh-logs");
const metricsOutput = document.querySelector("#metrics-output");
const logsOutput = document.querySelector("#logs-output");
let selectedFiles = [];
let lastPayload = null;

const locale = window.location.pathname.startsWith("/en") ? "en" : "pt";
const copy = {
  pt: {
    addResumeFirst: "Anexe pelo menos um currículo e execute uma análise.",
    analyze: "Analisar",
    analyzing: "Analisando",
    apiUnstable: "Instável",
    apiOnline: "Online",
    apiOffline: "Offline",
    noCandidates: "Nenhum candidato retornado.",
    noFiles: "Nenhum arquivo selecionado",
    oneCandidate: "candidato",
    manyCandidates: "candidatos",
    process: "Processando currículos...",
    remove: "Remover",
    selectFile: "Selecione pelo menos um currículo antes de analisar.",
    summary: "Sumário",
    justification: "Justificativa",
    citations: "Citações",
    noSummary: "Nenhum sumário retornado.",
    apiError: "A API não conseguiu analisar os currículos.",
    networkError: "Não foi possível acessar a API.",
    controlPanel: "Painel de Controle",
    productEyebrow: "Análise de currículos",
    uploadTitle: "Anexar currículos",
    uploadHelp: "Selecione vários arquivos PDF, PNG, JPG ou JPEG",
    clear: "Limpar",
    requestMode: "Modo da requisição",
    requestModeCopy:
      "Os arquivos e a pergunta são enviados juntos no POST e processados sem salvar currículos, arquivos ou vetores.",
    heroEyebrow: "Triagem com IA local",
    heroTitle: "Encontre os currículos mais aderentes",
    heroCopy:
      "Anexe vários currículos, faça uma pergunta de recrutamento e revise o ranking, os sumários, as justificativas e as citações retornadas pela API.",
    featureCallout: "Resposta a queries com ranking, score, justificativas e citações",
    results: "Resultados",
    showJson: "Ver JSON",
    ragModel: "Modelo RAG",
    llmModel: "Modelo LLM",
    singleOption: "Apenas uma opção disponível",
    userId: "ID do usuário",
    queryLabel: "Pergunta de recrutamento",
    api: "API",
    latency: "Latência",
    request: "Requisição",
    diagnosticsToggle: "Logs e métricas",
    diagnosticsEyebrow: "Observabilidade",
    diagnosticsTitle: "Logs e métricas",
    closeDiagnostics: "Fechar painel",
    metricsTitle: "Métricas Prometheus",
    auditTitle: "Auditoria da requisição",
    refresh: "Atualizar",
    loading: "Carregando...",
    noRequestForLogs: "Execute uma análise primeiro para consultar /logs/{request_id}.",
    diagnosticsError: "Não foi possível carregar os dados.",
  },
  en: {
    addResumeFirst: "Attach at least one resume and run an analysis.",
    analyze: "Analyze",
    analyzing: "Analyzing",
    apiUnstable: "Unstable",
    apiOnline: "Online",
    apiOffline: "Offline",
    noCandidates: "No candidates returned.",
    noFiles: "No file selected",
    oneCandidate: "candidate",
    manyCandidates: "candidates",
    process: "Processing resumes...",
    remove: "Remove",
    selectFile: "Select at least one resume before analyzing.",
    summary: "Summary",
    justification: "Justification",
    citations: "Citations",
    noSummary: "No summary returned.",
    apiError: "The API could not analyze the resumes.",
    networkError: "Could not reach the API.",
    controlPanel: "Control Panel",
    productEyebrow: "Resume analysis",
    uploadTitle: "Upload resumes",
    uploadHelp: "Select multiple PDF, PNG, JPG or JPEG files",
    clear: "Clear",
    requestMode: "Request mode",
    requestModeCopy:
      "Files and the query are sent together in the POST request and processed without saving resumes, files or vectors.",
    heroEyebrow: "Local AI screening",
    heroTitle: "Find the most relevant resumes",
    heroCopy:
      "Attach multiple resumes, ask a recruiting question, and review ranking, summaries, justifications and citations returned by the API.",
    featureCallout: "Query responses with ranking, score, justifications and citations",
    results: "Results",
    showJson: "View JSON",
    ragModel: "RAG model",
    llmModel: "LLM model",
    singleOption: "Only one option available",
    userId: "User ID",
    queryLabel: "Recruiting question",
    api: "API",
    latency: "Latency",
    request: "Request",
    diagnosticsToggle: "Logs and metrics",
    diagnosticsEyebrow: "Observability",
    diagnosticsTitle: "Logs and metrics",
    closeDiagnostics: "Close panel",
    metricsTitle: "Prometheus metrics",
    auditTitle: "Request audit",
    refresh: "Refresh",
    loading: "Loading...",
    noRequestForLogs: "Run an analysis first to query /logs/{request_id}.",
    diagnosticsError: "Could not load data.",
  },
}[locale];

function applyLocale() {
  document.documentElement.lang = locale === "en" ? "en" : "pt-BR";
  document.title =
    locale === "en" ? "Intelligent Resume Screening" : "Triagem Inteligente de Currículos";
  document.querySelectorAll("[data-i18n]").forEach((element) => {
    element.textContent = copy[element.dataset.i18n] || element.textContent;
  });
  document.querySelectorAll("[data-i18n-aria]").forEach((element) => {
    element.setAttribute("aria-label", copy[element.dataset.i18nAria] || "");
  });
  form.elements.query.placeholder =
    locale === "en"
      ? "E.g.: backend Python FastAPI Docker AWS..."
      : "Ex.: backend Python FastAPI Docker AWS...";
}

function relevantMetrics(text) {
  return text
    .split("\n")
    .filter((line) =>
      /^(# (HELP|TYPE) (requests_total|request_latency_seconds|ocr_failures_total|llm_failures_total)|requests_total|request_latency_seconds|ocr_failures_total|llm_failures_total)/.test(
        line,
      ),
    )
    .join("\n");
}

async function loadMetrics() {
  metricsOutput.textContent = copy.loading;
  try {
    const response = await fetch("/metrics");
    const text = await response.text();
    metricsOutput.textContent = relevantMetrics(text) || text.slice(0, 4000);
  } catch {
    metricsOutput.textContent = copy.diagnosticsError;
  }
}

async function loadLogs() {
  const id = requestIdLabel.textContent.trim();
  if (!id || id === "-") {
    logsOutput.textContent = copy.noRequestForLogs;
    return;
  }
  logsOutput.textContent = copy.loading;
  try {
    const response = await fetch(`/logs/${encodeURIComponent(id)}`);
    const payload = await response.json();
    logsOutput.textContent = JSON.stringify(payload, null, 2);
  } catch {
    logsOutput.textContent = copy.diagnosticsError;
  }
}

function openDiagnostics() {
  diagnosticsPanel.classList.add("is-open");
  diagnosticsPanel.setAttribute("aria-hidden", "false");
  loadMetrics();
  loadLogs();
}

function closeDiagnostics() {
  diagnosticsPanel.classList.remove("is-open");
  diagnosticsPanel.setAttribute("aria-hidden", "true");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function requestId() {
  return `ui-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
}

function updateFileList() {
  if (!selectedFiles.length) {
    fileList.textContent = copy.noFiles;
    return;
  }
  fileList.innerHTML = selectedFiles
    .map(
      (file, index) => `
        <div class="file-pill">
          <span>${escapeHtml(file.name)}</span>
          <button type="button" class="remove-file" data-index="${index}" aria-label="${copy.remove} ${escapeHtml(file.name)}">×</button>
        </div>
      `,
    )
    .join("");
}

function addFiles(files) {
  const knownFiles = new Set(selectedFiles.map((file) => `${file.name}:${file.size}`));
  files.forEach((file) => {
    const key = `${file.name}:${file.size}`;
    if (!knownFiles.has(key)) {
      selectedFiles.push(file);
      knownFiles.add(key);
    }
  });
  filesInput.value = "";
  updateFileList();
}

async function checkHealth() {
  try {
    const response = await fetch("/healthz");
    apiStatus.textContent = response.ok ? copy.apiOnline : copy.apiUnstable;
  } catch {
    apiStatus.textContent = copy.apiOffline;
  }
}

function renderEmpty(message) {
  results.className = "results empty-state";
  results.innerHTML = `<p>${escapeHtml(message)}</p>`;
  resultCount.textContent = `0 ${copy.manyCandidates}`;
}

function renderError(message) {
  results.className = "results empty-state error";
  results.innerHTML = `<p>${escapeHtml(message)}</p>`;
}

function renderResults(payload) {
  lastPayload = payload;
  updateJsonOutput();
  const items = payload.results || [];
  resultCount.textContent = `${items.length} ${
    items.length === 1 ? copy.oneCandidate : copy.manyCandidates
  }`;
  requestIdLabel.textContent = payload.request_id || "-";

  if (!items.length) {
    renderEmpty(copy.noCandidates);
    return;
  }

  results.className = "results";
  results.innerHTML = items
    .map((item) => {
      const score =
        typeof item.score === "number"
          ? `<div class="score">${Math.round(item.score * 100)}%</div>`
          : "";
      const justification = item.justification
        ? `<p class="meta-title">${copy.justification}</p><div class="justification">${escapeHtml(item.justification)}</div>`
        : "";
      const citations = Array.isArray(item.citations)
        ? item.citations
            .map(
              (citation) =>
                `<p class="citation">${escapeHtml(citation.text || "")}</p>`,
            )
            .join("")
        : "";
      const citationBlock = citations ? `<p class="meta-title">${copy.citations}</p>${citations}` : "";

      return `
        <article class="result-card">
          <div class="result-topline">
            <h3>${item.rank ? `${item.rank}. ` : ""}${escapeHtml(item.candidate || "Candidato")}</h3>
            ${score}
          </div>
          <p class="meta-title">${copy.summary}</p>
          <div class="summary">${escapeHtml(item.summary || copy.noSummary)}</div>
          ${justification}
          ${citationBlock}
        </article>
      `;
    })
    .join("");
}

function updateJsonOutput() {
  if (!jsonOutput) {
    return;
  }
  jsonOutput.textContent = lastPayload ? JSON.stringify(lastPayload, null, 2) : "";
}

filesInput.addEventListener("change", () => {
  addFiles(Array.from(filesInput.files || []));
});

fileList.addEventListener("click", (event) => {
  const button = event.target.closest(".remove-file");
  if (!button) {
    return;
  }
  selectedFiles = selectedFiles.filter((_, index) => index !== Number(button.dataset.index));
  updateFileList();
});

clearButton.addEventListener("click", () => {
  form.reset();
  selectedFiles = [];
  lastPayload = null;
  updateJsonOutput();
  updateFileList();
  latency.textContent = "-";
  requestIdLabel.textContent = "-";
  renderEmpty(copy.addResumeFirst);
});

jsonToggle.addEventListener("change", () => {
  const showJson = jsonToggle.checked;
  jsonOutput.hidden = !showJson;
  results.hidden = showJson;
  updateJsonOutput();
});

diagnosticsToggle.addEventListener("click", openDiagnostics);
diagnosticsClose.addEventListener("click", closeDiagnostics);
refreshMetrics.addEventListener("click", loadMetrics);
refreshLogs.addEventListener("click", loadLogs);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!selectedFiles.length) {
    renderError(copy.selectFile);
    return;
  }

  const submitButton = form.querySelector("button[type='submit']");
  submitButton.disabled = true;
  submitButton.textContent = copy.analyzing;
  lastPayload = null;
  updateJsonOutput();
  renderEmpty(copy.process);

  const id = requestId();
  const startedAt = performance.now();
  const data = new FormData();
  data.append("request_id", id);
  data.append("user_id", form.elements.userId.value.trim() || "recrutador-demo");
  const query = form.elements.query.value.trim();
  if (query) {
    data.append("query", query);
  }
  selectedFiles.forEach((file) => data.append("files", file));

  try {
    const response = await fetch("/analyze", {
      method: "POST",
      body: data,
    });
    const elapsed = performance.now() - startedAt;
    latency.textContent = `${Math.round(elapsed)} ms`;
    requestIdLabel.textContent = id;

    const payload = await response.json();
    if (!response.ok) {
      renderError(payload.detail || copy.apiError);
      return;
    }
    renderResults(payload);
  } catch {
    renderError(copy.networkError);
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = copy.analyze;
  }
});

applyLocale();
renderEmpty(copy.addResumeFirst);
checkHealth();
