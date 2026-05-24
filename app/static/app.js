const form = document.querySelector("#analyze-form");
const filesInput = document.querySelector("#files");
const fileList = document.querySelector("#file-list");
const clearButton = document.querySelector("#clear-button");
const results = document.querySelector("#results");
const apiStatus = document.querySelector("#api-status");
const latency = document.querySelector("#latency");
const requestIdLabel = document.querySelector("#request-id");
const resultCount = document.querySelector("#result-count");
let selectedFiles = [];

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
    fileList.textContent = "Nenhum arquivo selecionado";
    return;
  }
  fileList.innerHTML = selectedFiles
    .map(
      (file, index) => `
        <div class="file-pill">
          <span>${escapeHtml(file.name)}</span>
          <button type="button" class="remove-file" data-index="${index}" aria-label="Remover ${escapeHtml(file.name)}">×</button>
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
    apiStatus.textContent = response.ok ? "Online" : "Instável";
  } catch {
    apiStatus.textContent = "Offline";
  }
}

function renderEmpty(message) {
  results.className = "results empty-state";
  results.innerHTML = `<p>${escapeHtml(message)}</p>`;
  resultCount.textContent = "0 candidatos";
}

function renderError(message) {
  results.className = "results empty-state error";
  results.innerHTML = `<p>${escapeHtml(message)}</p>`;
}

function renderResults(payload) {
  const items = payload.results || [];
  resultCount.textContent = `${items.length} candidato${items.length === 1 ? "" : "s"}`;
  requestIdLabel.textContent = payload.request_id || "-";

  if (!items.length) {
    renderEmpty("Nenhum candidato retornado.");
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
        ? `<p class="meta-title">Justificativa</p><div class="justification">${escapeHtml(item.justification)}</div>`
        : "";
      const citations = Array.isArray(item.citations)
        ? item.citations
            .map(
              (citation) =>
                `<p class="citation">${escapeHtml(citation.text || "")}</p>`,
            )
            .join("")
        : "";
      const citationBlock = citations ? `<p class="meta-title">Citações</p>${citations}` : "";

      return `
        <article class="result-card">
          <div class="result-topline">
            <h3>${item.rank ? `${item.rank}. ` : ""}${escapeHtml(item.candidate || "Candidato")}</h3>
            ${score}
          </div>
          <p class="meta-title">Sumário</p>
          <div class="summary">${escapeHtml(item.summary || "Nenhum sumário retornado.")}</div>
          ${justification}
          ${citationBlock}
        </article>
      `;
    })
    .join("");
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
  updateFileList();
  latency.textContent = "-";
  requestIdLabel.textContent = "-";
  renderEmpty("Anexe pelo menos um currículo e execute uma análise.");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!selectedFiles.length) {
    renderError("Selecione pelo menos um currículo antes de analisar.");
    return;
  }

  const submitButton = form.querySelector("button[type='submit']");
  submitButton.disabled = true;
  submitButton.textContent = "Analisando";
  renderEmpty("Processando currículos...");

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
      renderError(payload.detail || "A API não conseguiu analisar os currículos.");
      return;
    }
    renderResults(payload);
  } catch {
    renderError("Não foi possível acessar a API.");
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Analisar";
  }
});

checkHealth();
