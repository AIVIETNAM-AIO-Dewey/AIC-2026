/**
 * AIC-2026 Multimodal Retrieval Studio — Frontend Logic
 * KIS-only no-fusion retrieval UI.
 */

// ──────────────────────────────────────────────────────────────────────────────
// App State
// ──────────────────────────────────────────────────────────────────────────────
import { createIcons, ExternalLink, Maximize, Pause, Play, RotateCcw, RotateCw } from "lucide";
import { createYouTubeVideoView } from "./src/youtube-video-view.ts";

const state = {
  taskType: "KIS",
  queryMode: "auto", // "auto", "edit", "direct"
  useGemini: true,
  sessionId: null,
  keyframesRoot: "/Users/macbookpro/Downloads/AIC-HCM-BATCH-1/AIC_HCM_BATCH_1/artifacts/keyframes",
  parsedQuery: null,
  modalityResults: {},
  activeModality: "all",
  searchResults: [],
  activeInspectorItem: null,
  activeVideoKeyframes: [],
  selectedSubmission: "",
  selectedSubmissionItem: null,
  activeBBoxObjects: [],
  inspectorMediaMode: "keyframe",
};

const SEARCH_POOL_SIZE = 100;
const FILMSTRIP_WINDOW_RADIUS = 12;
let parseAbortController = null;
let searchAbortController = null;
let inspectorAbortController = null;
let filmstripAbortController = null;
let parseRequestId = 0;
let searchRequestId = 0;
let inspectorRequestId = 0;
let filmstripRequestId = 0;
let resultsRenderId = 0;
let toastTimer = null;
let lastInspectorFocus = null;
let lastExportFocus = null;

// ──────────────────────────────────────────────────────────────────────────────
// DOM Elements
// ──────────────────────────────────────────────────────────────────────────────
const el = {
  taskBtns: /** @type {NodeListOf<HTMLButtonElement>} */ (document.querySelectorAll(".task-btn")),
  geminiToggle: document.getElementById("btn-gemini-toggle"),
  geminiText: document.querySelector("#btn-gemini-toggle .btn-text"),
  modeTabs: /** @type {NodeListOf<HTMLButtonElement>} */ (document.querySelectorAll(".mode-tab")),
  
  colOriginalQuery: document.getElementById("col-original-query"),
  colParsedJson: document.getElementById("col-parsed-json"),
  inputQuery: /** @type {HTMLTextAreaElement} */ (document.getElementById("input-query")),
  jsonEditor: /** @type {HTMLTextAreaElement} */ (document.getElementById("json-editor")),
  
  btnRunQuery: /** @type {HTMLButtonElement} */ (document.getElementById("btn-run-query")),
  btnRunLabel: document.getElementById("btn-run-label"),
  btnExecuteJson: /** @type {HTMLButtonElement} */ (document.getElementById("btn-execute-json")),
  btnFormatJson: document.getElementById("btn-format-json"),
  
  timingBadge: document.getElementById("timing-badge"),
  sessionBadge: document.getElementById("session-badge"),
  statusPill: document.querySelector(".status-pill"),
  serverStatusText: document.getElementById("server-status-text"),
  resultsGrid: document.getElementById("results-grid"),
  resultsCount: document.getElementById("results-count-badge"),
  selectTopK: /** @type {HTMLSelectElement} */ (document.getElementById("select-top-k")),
  
  modalityTabs: /** @type {NodeListOf<HTMLButtonElement>} */ (document.querySelectorAll(".modality-tab")),
  modalityQuerySummary: document.getElementById("modality-query-summary"),
  
  // Submission & Export
  submissionInput: /** @type {HTMLInputElement} */ (document.getElementById("submission-input")),
  btnCopySubmission: document.getElementById("btn-copy-submission"),
  btnExportCsv: document.getElementById("btn-export-csv"),
  btnClearSubmission: document.getElementById("btn-clear-submission"),
  toast: document.getElementById("toast"),

  // 100-Row Export Modal
  exportModal: document.getElementById("export-modal"),
  btnCloseExportModal: document.getElementById("btn-close-export-modal"),
  btnCancelExportModal: document.getElementById("btn-cancel-export-modal"),
  btnDownloadCsvAction: /** @type {HTMLButtonElement} */ (document.getElementById("btn-download-csv-action")),
  exportQueryId: /** @type {HTMLInputElement} */ (document.getElementById("export-query-id")),
  exportRow1Preview: document.getElementById("export-row1-preview"),
  
  // Inspector Modal
  modal: document.getElementById("inspector-modal"),
  inspectorImg: /** @type {HTMLImageElement} */ (document.getElementById("inspector-img")),
  inspectorCanvas: /** @type {HTMLCanvasElement} */ (document.getElementById("inspector-canvas")),
  inspectorPlaceholder: document.getElementById("inspector-img-placeholder"),
  placeholderText: document.getElementById("placeholder-text"),
  keyframeView: document.getElementById("keyframe-view"),
  videoView: document.getElementById("video-view"),
  btnViewKeyframe: document.getElementById("btn-view-keyframe"),
  btnViewVideo: document.getElementById("btn-view-video"),
  keyframeControls: document.getElementById("keyframe-visual-controls"),
  mappingStatus: document.getElementById("mapping-status"),
  chkBBoxes: /** @type {HTMLInputElement} */ (document.getElementById("chk-show-bboxes")),
  
  inspVideoId: document.getElementById("insp-video-id"),
  inspKeyframeN: document.getElementById("insp-keyframe-n"),
  inspFrameIdx: document.getElementById("insp-frame-idx"),
  inspScoreRank: document.getElementById("insp-score-rank"),
  inspMatchedTime: document.getElementById("insp-matched-time"),
  inspVideoSource: document.getElementById("insp-video-source"),
  inspMappingOffset: document.getElementById("insp-mapping-offset"),
  inspAsrText: document.getElementById("insp-asr-text"),
  inspDamText: document.getElementById("insp-dam-text"),
  inspOcrText: document.getElementById("insp-ocr-text"),
  inspObjectsList: document.getElementById("insp-objects-list"),
  
  btnToggleInSubmission: document.getElementById("btn-toggle-in-submission"),
  btnCloseInspector: document.getElementById("btn-close-inspector"),
  
  filmstripScroll: document.getElementById("filmstrip-scroll"),
  filmstripCount: document.getElementById("filmstrip-count"),
  btnFilmstripPrev: document.getElementById("btn-filmstrip-prev"),
  btnFilmstripNext: document.getElementById("btn-filmstrip-next"),
};

// ──────────────────────────────────────────────────────────────────────────────
// Initialization
// ──────────────────────────────────────────────────────────────────────────────
function refreshVideoIcons() {
  createIcons({ icons: { ExternalLink, Maximize, Pause, Play, RotateCcw, RotateCw } });
}

function setMappingStatus(label, status = "pending") {
  el.mappingStatus.textContent = label;
  el.mappingStatus.classList.toggle("ready", status === "ready");
  el.mappingStatus.classList.toggle("error", status === "error");
}

function toVideoFrame(item) {
  return {
    videoId: item.video_id,
    ptsTimeS: Number(item.pts_time_s || 0),
    posterPath: getImageUrl(item),
  };
}

const videoController = createYouTubeVideoView({
  onSourceChange: (label) => {
    el.inspVideoSource.textContent = label;
  },
  onStatusChange: setMappingStatus,
  onToast: showToast,
  refreshIcons: refreshVideoIcons,
});

async function initApp() {
  bindEvents();
  setInspectorMediaMode("keyframe");
  refreshVideoIcons();
  await loadServerConfig();
  updateQueryModeUI();
  if (typeof ResizeObserver !== "undefined") {
    const resizeObserver = new ResizeObserver(() => {
      if (!el.modal.classList.contains("hidden")) drawBBoxesOnCanvas();
    });
    resizeObserver.observe(el.keyframeView);
  }
}

async function loadServerConfig() {
  try {
    const res = await fetch("/api/config");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const cfg = await res.json();
    if (cfg.keyframes_root) state.keyframesRoot = cfg.keyframes_root;
    setServerStatus("No fusion / no reranking", "ready");
  } catch (err) {
    console.warn("Could not load /api/config:", err);
    setServerStatus("Server configuration unavailable", "error");
  }
}

function setServerStatus(label, status = "ready") {
  if (el.serverStatusText) el.serverStatusText.textContent = label;
  if (!el.statusPill) return;
  el.statusPill.classList.toggle("ready", status === "ready");
  el.statusPill.classList.toggle("error", status === "error");
  el.statusPill.classList.toggle("pending", status === "pending");
}

// ──────────────────────────────────────────────────────────────────────────────
// Event Listeners
// ──────────────────────────────────────────────────────────────────────────────
function bindEvents() {
  // Task selector
  el.taskBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      el.taskBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.taskType = btn.dataset.task;
      syncJsonEditorWithState();
    });
  });

  // Gemini toggle
  el.geminiToggle.addEventListener("click", () => {
    state.useGemini = !state.useGemini;
    el.geminiToggle.classList.toggle("active", state.useGemini);
    el.geminiText.textContent = state.useGemini ? "Gemini 3.6 ON" : "Local Qwen 2.5";
  });

  // Query mode tabs
  el.modeTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      el.modeTabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      state.queryMode = tab.dataset.mode;
      updateQueryModeUI();
    });
  });

  // Primary Run Buttons
  el.btnRunQuery.addEventListener("click", handleRunQueryClick);
  el.btnExecuteJson.addEventListener("click", () => void handleExecuteJsonClick());
  el.btnFormatJson.addEventListener("click", formatJsonEditor);

  // Keyboard shortcut Ctrl+Enter / Cmd+Enter on textareas
  el.inputQuery.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      void handleRunQueryClick();
    }
  });

  el.jsonEditor.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      void handleExecuteJsonClick();
    }
  });

  el.modalityTabs.forEach((tab) => {
    tab.addEventListener("click", () => setActiveModality(tab.dataset.modality));
  });

  // Top-k dropdown
  el.selectTopK.addEventListener("change", () => {
    renderModalityResults();
  });

  // Submission bar
  el.btnCopySubmission.addEventListener("click", copySubmissionToClipboard);
  if (el.btnExportCsv) el.btnExportCsv.addEventListener("click", openExportModal);
  if (el.btnCloseExportModal) el.btnCloseExportModal.addEventListener("click", closeExportModal);
  if (el.btnCancelExportModal) el.btnCancelExportModal.addEventListener("click", closeExportModal);
  if (el.btnDownloadCsvAction) el.btnDownloadCsvAction.addEventListener("click", executeDownload100Csv);
  if (el.exportQueryId) el.exportQueryId.addEventListener("input", updateExportPreview);
  if (el.exportModal) {
    el.exportModal.addEventListener("click", (event) => {
      if (event.target === el.exportModal) closeExportModal();
    });
  }

  el.btnClearSubmission.addEventListener("click", () => {
    clearSubmissionSelection();
  });

  // Inspector modal
  el.btnCloseInspector.addEventListener("click", closeInspector);
  el.btnViewKeyframe.addEventListener("click", () => setInspectorMediaMode("keyframe"));
  el.btnViewVideo.addEventListener("click", () => setInspectorMediaMode("video"));
  el.btnToggleInSubmission.addEventListener("click", toggleCurrentInSubmission);
  el.chkBBoxes.addEventListener("change", drawBBoxesOnCanvas);
  el.modal.addEventListener("click", (event) => {
    if (event.target === el.modal) closeInspector();
  });

  el.btnFilmstripPrev.addEventListener("click", () => {
    el.filmstripScroll.scrollBy({ left: -300, behavior: "smooth" });
  });
  el.btnFilmstripNext.addEventListener("click", () => {
    el.filmstripScroll.scrollBy({ left: 300, behavior: "smooth" });
  });

  // Global Keydown (Escape, ArrowLeft, ArrowRight, Enter)
  document.addEventListener("keydown", (e) => {
    if (!el.exportModal.classList.contains("hidden")) {
      if (e.key === "Tab") keepFocusInside(el.exportModal, e);
      if (e.key === "Escape") {
        e.preventDefault();
        closeExportModal();
      }
      return;
    }

    if (el.modal.classList.contains("hidden")) return;
    if (e.key === "Tab") keepFocusInside(el.modal, e);
    const targetIsInteractive = isInteractiveTarget(e.target);

    if (e.key === "Escape") {
      e.preventDefault();
      closeInspector();
    } else if (e.key === "ArrowLeft" && !targetIsInteractive) {
      e.preventDefault();
      navigateFilmstrip(-1);
    } else if (e.key === "ArrowRight" && !targetIsInteractive) {
      e.preventDefault();
      navigateFilmstrip(1);
    } else if (e.key === "Enter" && !targetIsInteractive && !e.repeat) {
      e.preventDefault();
      toggleCurrentInSubmission();
    }
  });
}

function keepFocusInside(container, event) {
  const focusable = [...container.querySelectorAll(
    "button:not([disabled]), input:not([disabled]), textarea:not([disabled]), select:not([disabled]), a[href], [tabindex]:not([tabindex='-1'])",
  )].filter((node) => !node.closest(".hidden") && node.getClientRects().length > 0);
  if (!focusable.length) return;
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  if (!container.contains(document.activeElement) || !focusable.includes(document.activeElement)) {
    event.preventDefault();
    /** @type {HTMLElement} */ (event.shiftKey ? last : first).focus();
    return;
  }
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
}

function setBackgroundInert(activeModal, isInert) {
  [...document.body.children].forEach((child) => {
    if (child !== activeModal && child instanceof HTMLElement) child.inert = isInert;
  });
}

function isInteractiveTarget(target) {
  return target instanceof Element && Boolean(
    target.closest("button, input, textarea, select, a, [role='button'], [contenteditable='true']"),
  );
}

function clearSubmissionSelection() {
  state.selectedSubmission = "";
  state.selectedSubmissionItem = null;
  el.submissionInput.value = "No keyframe selected";
  updateInspectorSubmitBtn();
}

function setSearchBusy(isBusy) {
  el.btnRunQuery.disabled = isBusy;
  el.btnExecuteJson.disabled = isBusy;
  el.selectTopK.disabled = isBusy;
  el.btnRunQuery.setAttribute("aria-busy", String(isBusy));
}

// ──────────────────────────────────────────────────────────────────────────────
// Query Mode Handling
// ──────────────────────────────────────────────────────────────────────────────
function updateQueryModeUI() {
  if (state.queryMode === "auto") {
    el.colOriginalQuery.style.display = "flex";
    el.colParsedJson.style.display = "flex";
    el.inputQuery.disabled = false;
    el.jsonEditor.readOnly = true;
    el.btnRunLabel.textContent = "Parse & Search";
    el.btnExecuteJson.style.display = "none";
  } else if (state.queryMode === "edit") {
    el.colOriginalQuery.style.display = "flex";
    el.colParsedJson.style.display = "flex";
    el.inputQuery.disabled = false;
    el.jsonEditor.readOnly = false;
    el.btnRunLabel.textContent = "Parse Query";
    el.btnExecuteJson.style.display = "inline-flex";
  } else if (state.queryMode === "direct") {
    el.colOriginalQuery.style.display = "none";
    el.colParsedJson.style.display = "flex";
    el.colParsedJson.style.gridColumn = "1 / -1";
    el.jsonEditor.readOnly = false;
    el.btnExecuteJson.style.display = "inline-flex";
  }

  if (state.queryMode !== "direct") {
    el.colParsedJson.style.gridColumn = "auto";
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Independent Modality View
// ──────────────────────────────────────────────────────────────────────────────
const MODALITY_ORDER = ["siglip", "dam", "ocr", "asr"];

function setActiveModality(modality) {
  if (modality !== "all" && !MODALITY_ORDER.includes(modality)) return;
  state.activeModality = modality;
  el.modalityTabs.forEach((tab) => {
    const active = tab.dataset.modality === modality;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  renderModalityResults();
}

function formatPoolQuery(pool) {
  if (!pool) return "";
  return Array.isArray(pool.query) ? pool.query.join(" · ") : String(pool.query || "");
}

function updateModalityQuerySummary() {
  if (!Object.keys(state.modalityResults).length) return;
  if (state.activeModality === "all") {
    el.modalityQuerySummary.innerHTML = MODALITY_ORDER.map((modality) => {
      const pool = state.modalityResults[modality];
      if (!pool) return "";
      const query = pool.status === "ok" ? formatPoolQuery(pool) : pool.reason;
      return `<strong>${escapeHtml(pool.display_name)}</strong>: <code>${escapeHtml(query)}</code> · ${escapeHtml(pool.score_type)}`;
    }).filter(Boolean).join("<br>");
    return;
  }
  const pool = state.modalityResults[state.activeModality];
  if (!pool) return;
  const query = pool.status === "ok" ? formatPoolQuery(pool) : pool.reason;
  el.modalityQuerySummary.innerHTML = `<strong>${escapeHtml(pool.display_name)}</strong> · query source <code>${escapeHtml(pool.query_source)}</code> · query <code>${escapeHtml(query)}</code><br>${escapeHtml(pool.score_description)}`;
}

function formatJsonEditor() {
  try {
    const data = JSON.parse(el.jsonEditor.value);
    el.jsonEditor.value = JSON.stringify(data, null, 2);
  } catch (err) {
    showToast("Invalid JSON: " + err.message, "error");
  }
}

function syncJsonEditorWithState() {
  try {
    const data = JSON.parse(el.jsonEditor.value);
    data.task_type = state.taskType;
    el.jsonEditor.value = JSON.stringify(data, null, 2);
  } catch (_) {}
}

// ──────────────────────────────────────────────────────────────────────────────
// API Calls: Parse & Search
// ──────────────────────────────────────────────────────────────────────────────
async function handleRunQueryClick() {
  const query = el.inputQuery.value.trim();
  if (!query) {
    showToast("Please enter a query text.", "error");
    return;
  }

  parseAbortController?.abort();
  searchAbortController?.abort();
  searchRequestId += 1;
  parseAbortController = new AbortController();
  const requestId = ++parseRequestId;
  setSearchBusy(true);
  clearSubmissionSelection();
  state.modalityResults = {};
  state.searchResults = [];
  el.resultsCount.textContent = "Parsing…";
  el.resultsGrid.innerHTML = `<div class="empty-placeholder" aria-live="polite"><div class="empty-icon">⏳</div><div class="empty-title">Parsing query</div><div class="empty-desc">Previous results were cleared to prevent stale selections.</div></div>`;
  el.timingBadge.textContent = "Parsing query...";
  setServerStatus("Parsing query…", "pending");

  try {
    const parseRes = await fetch("/api/parse", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: query,
        task_type: state.taskType,
        engine: state.useGemini ? "gemini" : "qwen",
      }),
      signal: parseAbortController.signal,
    });

    if (!parseRes.ok) throw new Error("Parse failed: " + parseRes.statusText);
    const parseData = await parseRes.json();
    if (requestId !== parseRequestId) return;
    state.parsedQuery = parseData.parsed_query;

    el.jsonEditor.value = JSON.stringify(state.parsedQuery, null, 2);
    if (state.queryMode === "auto") {
      await handleExecuteJsonClick();
    } else {
      el.timingBadge.textContent = `Parsed in ${parseData.execution_time_ms}ms (Ready to edit)`;
      setServerStatus("Query parsed", "ready");
    }
  } catch (err) {
    if (err.name === "AbortError") return;
    console.error(err);
    el.timingBadge.textContent = "Error parsing";
    setServerStatus("Query parse failed", "error");
    showToast("Error: " + err.message, "error");
  } finally {
    if (requestId === parseRequestId) setSearchBusy(false);
  }
}

async function handleExecuteJsonClick() {
  let parsedJson;
  try {
    parsedJson = JSON.parse(el.jsonEditor.value);
  } catch (e) {
    showToast("Invalid JSON syntax: " + e.message, "error");
    return;
  }

  delete parsedJson.weights;
  parsedJson.task_type = state.taskType;

  searchAbortController?.abort();
  searchAbortController = new AbortController();
  const requestId = ++searchRequestId;
  setSearchBusy(true);
  clearSubmissionSelection();
  if (!el.modal.classList.contains("hidden")) closeInspector();
  state.modalityResults = {};
  state.searchResults = [];
  el.resultsCount.textContent = "Searching…";
  el.resultsGrid.innerHTML = `<div class="empty-placeholder" aria-live="polite"><div class="empty-icon">⏳</div><div class="empty-title">Searching independent pools</div><div class="empty-desc">Results from the previous query were cleared to avoid stale submissions.</div></div>`;
  el.timingBadge.textContent = "Searching 247,956 frames in four independent pools...";
  setServerStatus("Searching…", "pending");

  try {
    const res = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        parsed_query: parsedJson,
        session_id: state.sessionId,
        // Keep a complete, verified candidate reservoir; the selector only controls rendering.
        top_k: SEARCH_POOL_SIZE,
      }),
      signal: searchAbortController.signal,
    });

    if (!res.ok) throw new Error("Search failed: " + res.statusText);
    const data = await res.json();
    if (requestId !== searchRequestId) return;

    state.sessionId = data.session_id;
    state.modalityResults = data.modality_results || {};

    el.timingBadge.textContent = `Four independent pools completed in ${data.execution_time_ms}ms`;
    el.sessionBadge.textContent = `Session: ${data.session_id.slice(0, 8)}...`;
    el.sessionBadge.classList.remove("hidden");
    setServerStatus("No fusion / no reranking", "ready");

    renderModalityResults();
  } catch (err) {
    if (err.name === "AbortError") return;
    console.error(err);
    el.timingBadge.textContent = "Search error";
    state.modalityResults = {};
    state.searchResults = [];
    renderModalityResults();
    setServerStatus("Search failed", "error");
    showToast("Search failed: " + err.message, "error");
  } finally {
    if (requestId === searchRequestId) setSearchBusy(false);
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Raw Modality Result Rendering
// ──────────────────────────────────────────────────────────────────────────────
function getImageUrl(item) {
  const vid = item.video_id;
  const relpath = String(item.image_relpath || "").replace(/^\/+/, "");
  const filename = relpath ? relpath.split("/").pop() : `${String(item.frame_idx || 0).padStart(8, "0")}.jpg`;
  if (window.location.protocol === "file:") {
    return `file://${state.keyframesRoot}/${vid}/${filename}`;
  }
  return `/keyframes/${encodeURIComponent(vid)}/${encodeURIComponent(filename)}`;
}

function getAllSearchCandidates() {
  const pools = MODALITY_ORDER.map((modality) => state.modalityResults[modality]?.results || []);
  const maxLength = pools.reduce((max, results) => Math.max(max, results.length), 0);
  const seen = new Set();
  const candidates = [];

  // Round-robin preserves representation from every independent modality.
  for (let index = 0; index < maxLength; index += 1) {
    pools.forEach((results) => {
      const item = results[index];
      if (!item) return;
      const key = `${item.video_id}:${item.frame_idx}`;
      if (seen.has(key)) return;
      seen.add(key);
      candidates.push(item);
    });
  }
  return candidates;
}

function renderModalityResults() {
  const renderId = ++resultsRenderId;
  el.resultsGrid.innerHTML = "";
  const limit = parseInt(el.selectTopK.value) || 20;
  updateModalityQuerySummary();

  if (!Object.keys(state.modalityResults).length) {
    state.searchResults = [];
    el.resultsCount.textContent = "0 independent results";
    el.resultsGrid.innerHTML = `
      <div class="empty-placeholder">
        <div class="empty-icon">🔍</div>
        <div class="empty-title">Ready for Independent Search</div>
        <div class="empty-desc">Run one query to produce separate SigLIP, DAM, OCR, and ASR rankings.</div>
      </div>`;
    return;
  }

  state.searchResults = getAllSearchCandidates();

  if (state.activeModality === "all") {
    const total = MODALITY_ORDER.reduce((sum, modality) => {
      const pool = state.modalityResults[modality];
      return sum + Math.min(pool?.results?.length || 0, limit);
    }, 0);
    el.resultsCount.textContent = `${total} results across separate pools`;
    MODALITY_ORDER.forEach((modality) => {
      const pool = state.modalityResults[modality];
      if (pool) renderPoolSection(pool, limit, renderId);
    });
    return;
  }

  const pool = state.modalityResults[state.activeModality];
  const list = (pool?.results || []).slice(0, limit);
  el.resultsCount.textContent = `${list.length} ${pool?.display_name || state.activeModality} results`;
  if (pool?.status !== "ok" || list.length === 0) {
    el.resultsGrid.innerHTML = `
      <div class="empty-placeholder">
        <div class="empty-icon">🔍</div>
        <div class="empty-title">No Results in This Pool</div>
        <div class="empty-desc">${escapeHtml(pool?.reason || "The independent search returned no matches.")}</div>
      </div>`;
    return;
  }
  renderStandardCards(list, el.resultsGrid, state.activeModality, renderId);
}

function renderPoolSection(pool, limit, renderId) {
  const section = document.createElement("section");
  section.className = "modality-pool-section";
  const query = pool.status === "ok" ? formatPoolQuery(pool) : pool.reason;
  const list = (pool.results || []).slice(0, limit);
  section.innerHTML = `
    <div class="modality-pool-header">
      <div>
        <div class="modality-pool-title">${escapeHtml(pool.display_name)}</div>
        <div class="modality-pool-meta">${escapeHtml(pool.score_type)} · ${escapeHtml(query)}</div>
      </div>
      <span class="count-badge">${list.length} shown of ${pool.result_count} · ${pool.execution_time_ms}ms</span>
    </div>`;
  const grid = document.createElement("div");
  grid.className = "modality-pool-grid";
  if (pool.status !== "ok" || list.length === 0) {
    grid.innerHTML = `<div class="pool-status-message">${escapeHtml(pool.reason || "No matches returned.")}</div>`;
  } else {
    renderStandardCards(list, grid, pool.modality, renderId);
  }
  section.appendChild(grid);
  el.resultsGrid.appendChild(section);
}

function resultEvidence(item, modality) {
  if (modality === "asr") return item.transcript || item.asr_transcript || "No speech text";
  if (modality === "ocr") {
    const matches = (item.matched_keywords || []).join(", ");
    return `${matches ? `Matched: ${matches} · ` : ""}${item.ocr_text || "No OCR text"}`;
  }
  if (modality === "dam") {
    const subjects = (item.subject_scores || [])
      .map((entry) => `${entry.subject}: ${Number(entry.cosine).toFixed(4)}`)
      .join(" · ");
    return subjects || item.dam_summary || "No DAM evidence";
  }
  return item.dam_summary || "Full-frame image/text cosine similarity";
}

function renderStandardCards(list, container = el.resultsGrid, modality = state.activeModality, renderId = resultsRenderId) {
  const appendRange = (start, end) => {
    if (renderId !== resultsRenderId) return;
    const fragment = document.createDocumentFragment();
    for (let idx = start; idx < end; idx += 1) {
      const item = list[idx];
    const rank = item.rank || idx + 1;
    const score = Number(item.score ?? 0.0);
    const ptsTime = Number(item.pts_time_s);
    const timeS = Number.isFinite(ptsTime) ? ptsTime.toFixed(1) + "s" : "-";
    const imgUrl = getImageUrl(item);
    const evidence = resultEvidence(item, modality);
    const scoreType = item.score_type || "raw_score";
    item.retrieval_modality = modality;

    const card = document.createElement("div");
    card.className = "candidate-card";
    card.dataset.index = idx;
    card.tabIndex = 0;
    card.setAttribute("role", "button");
    card.setAttribute("aria-label", `Open ${item.video_id}, frame ${item.frame_idx}, rank ${rank}`);

    card.innerHTML = `
      <div class="card-media">
        <img src="${imgUrl}" alt="Keyframe from ${escapeHtml(item.video_id)}" loading="lazy" decoding="async">
        <span class="card-rank-badge">#${rank}</span>
        <span class="card-modality-badge">${escapeHtml(modality.toUpperCase())}</span>
        <span class="card-time-badge">${timeS}</span>
      </div>
      <div class="card-body">
        <div class="card-title-row">
          <span class="card-vid-name">${item.video_id} : ${item.frame_idx}</span>
          <div class="card-scores">
            <span class="score-final" title="${escapeHtml(scoreType)}">${score.toFixed(4)}</span>
          </div>
        </div>
        <div class="card-speech-snippet">${escapeHtml(evidence)}</div>
        <div class="card-tags-row">
          <span class="pill-tag active">${escapeHtml(scoreType)}</span>
          <span class="pill-tag">raw rank #${rank}</span>
        </div>
      </div>`;

    const image = card.querySelector("img");
    image.addEventListener("error", () => {
      image.hidden = true;
      image.parentElement.classList.add("img-fallback");
    }, { once: true });
    const openCard = () => void openStandardInspector(item);
    card.addEventListener("click", openCard);
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openCard();
      }
    });
      fragment.appendChild(card);
    }
    container.appendChild(fragment);
  };

  if (list.length <= 20) {
    appendRange(0, list.length);
    return;
  }

  let nextIndex = 0;
  const appendNextChunk = () => {
    if (renderId !== resultsRenderId || !container.isConnected) return;
    const end = Math.min(list.length, nextIndex + 10);
    appendRange(nextIndex, end);
    nextIndex = end;
    if (nextIndex < list.length) requestAnimationFrame(appendNextChunk);
  };
  appendRange(0, 10);
  nextIndex = 10;
  if (nextIndex < list.length) requestAnimationFrame(appendNextChunk);
}


// ──────────────────────────────────────────────────────────────────────────────
// Frame Inspector Modal
// ──────────────────────────────────────────────────────────────────────────────

function setInspectorMediaMode(mode) {
  state.inspectorMediaMode = mode;
  const showVideo = mode === "video";
  el.btnViewKeyframe.classList.toggle("active", !showVideo);
  el.btnViewVideo.classList.toggle("active", showVideo);
  el.btnViewKeyframe.setAttribute("aria-selected", String(!showVideo));
  el.btnViewVideo.setAttribute("aria-selected", String(showVideo));
  el.keyframeView.classList.toggle("hidden", showVideo);
  el.videoView.classList.toggle("hidden", !showVideo);
  el.keyframeControls.classList.toggle("hidden", showVideo);
  if (showVideo) void videoController.activate().catch(() => {});
  else videoController.deactivate();
}

// KIS frame inspector
async function openStandardInspector(item, preserveMediaMode = false) {
  if (el.modal.classList.contains("hidden")) lastInspectorFocus = document.activeElement;
  state.activeInspectorItem = item;
  if (!preserveMediaMode) setInspectorMediaMode("keyframe");
  el.modal.classList.remove("hidden");
  setBackgroundInert(el.modal, true);

  populateInspectorCommon(item);
  void loadFilmstrip(item.video_id, item.keyframe_n);
  if (!preserveMediaMode) {
    requestAnimationFrame(() => {
      /** @type {HTMLElement | null} */ (el.modal.querySelector(".inspector-card"))?.focus({ preventScroll: true });
    });
  }
}

function populateInspectorCommon(item) {
  videoController.deactivate();
  el.inspVideoId.textContent = item.video_id || "-";
  el.inspKeyframeN.textContent = String(item.keyframe_n ?? 1).padStart(3, "0");
  const itemPtsTime = Number(item.pts_time_s);
  const itemTimeLabel = Number.isFinite(itemPtsTime) ? `${itemPtsTime.toFixed(1)}s` : "-";
  el.inspFrameIdx.textContent = `${item.frame_idx ?? 0} (${itemTimeLabel})`;
  
  const rank = item.rank || 1;
  const score = Number(item.score ?? 0.0);
  const scoreType = item.score_type || "metadata frame";
  el.inspScoreRank.textContent = `${score.toFixed(4)} • #${rank} • ${scoreType}`;

  const ptsTime = Number.isFinite(itemPtsTime) ? itemPtsTime : 0;
  const minutes = Math.floor(ptsTime / 60);
  const seconds = Math.floor(ptsTime % 60);
  el.inspMatchedTime.textContent = `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}.${Math.floor((ptsTime % 1) * 10)}`;
  el.inspMappingOffset.textContent = "0.0s";

  inspectorAbortController?.abort();
  inspectorAbortController = new AbortController();
  const requestId = ++inspectorRequestId;
  state.activeBBoxObjects = [];
  renderMatchedObjectsList([]);
  drawBBoxesOnCanvas();

  const imgUrl = getImageUrl(item);
  videoController.setFrame(toVideoFrame(item));
  el.inspectorImg.onerror = () => {
    if (requestId !== inspectorRequestId) return;
    el.inspectorPlaceholder.classList.remove("hidden");
    const filename = String(item.image_relpath || "").split("/").pop() || `${String(item.frame_idx || 0).padStart(8, "0")}.jpg`;
    el.placeholderText.textContent = `${item.video_id} / ${filename}`;
  };
  el.inspectorImg.onload = () => {
    if (requestId !== inspectorRequestId) return;
    el.inspectorPlaceholder.classList.add("hidden");
    drawBBoxesOnCanvas();
  };
  el.inspectorImg.src = imgUrl;

  el.inspAsrText.textContent = item.asr_transcript || "(No speech / silent frame)";
  el.inspDamText.textContent = item.dam_summary || "(No visual description available)";
  if (el.inspOcrText) {
    el.inspOcrText.textContent = item.ocr_text || "(No text detected on screen)";
  }

  if (state.inspectorMediaMode === "video") {
    void videoController.activate().catch(() => {});
  }

  updateInspectorSubmitBtn();

  // Load detailed DAM bounding boxes & metadata from API
  fetch(`/api/keyframe/${encodeURIComponent(item.video_id)}/${encodeURIComponent(item.keyframe_n)}`, {
    signal: inspectorAbortController.signal,
  })
    .then((r) => (r.ok ? r.json() : null))
    .then((data) => {
      if (data && requestId === inspectorRequestId && state.activeInspectorItem) {
        const active = state.activeInspectorItem;
        if (active.video_id !== item.video_id) return;
        state.activeBBoxObjects = data.dam_objects || [];
        if (data.macro_audio_transcript) {
          el.inspAsrText.textContent = data.macro_audio_transcript;
        }
        if (el.inspOcrText && data.keyframe && data.keyframe.ocr_text) {
          el.inspOcrText.textContent = data.keyframe.ocr_text;
        }
        if (data.keyframe?.dam_summary_en) {
          el.inspDamText.textContent = data.keyframe.dam_summary_en;
        }
        renderMatchedObjectsList(state.activeBBoxObjects);
        drawBBoxesOnCanvas();
      }
    })
    .catch((error) => {
      if (error.name !== "AbortError") console.warn("Keyframe detail load failed:", error);
    });
}

function renderMatchedObjectsList(objects) {
  el.inspObjectsList.innerHTML = "";
  if (!objects || objects.length === 0) {
    el.inspObjectsList.innerHTML = `<span style="font-size:11px;color:var(--text-dim);">No distinct DAM objects detected</span>`;
    return;
  }

  objects.slice(0, 5).forEach((obj) => {
    const row = document.createElement("div");
    row.className = "object-item";
    row.innerHTML = `
      <span class="obj-name">${obj.class_entity || 'Object'}</span>
      <span class="obj-score">${obj.description_en ? obj.description_en.slice(0, 45) + '...' : ''}</span>`;
    el.inspObjectsList.appendChild(row);
  });
}

function drawBBoxesOnCanvas() {
  const canvas = el.inspectorCanvas;
  const ctx = canvas.getContext("2d");
  const img = el.inspectorImg;

  const cssWidth = img.clientWidth;
  const cssHeight = img.clientHeight;
  if (!cssWidth || !cssHeight) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const renderWidth = Math.round(cssWidth * dpr);
  const renderHeight = Math.round(cssHeight * dpr);
  if (canvas.width !== renderWidth || canvas.height !== renderHeight) {
    canvas.width = renderWidth;
    canvas.height = renderHeight;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);

  if (!el.chkBBoxes.checked || !state.activeBBoxObjects.length) return;

  const colors = ["#38bdf8", "#34d399", "#f59e0b", "#ec4899", "#818cf8"];

  state.activeBBoxObjects.forEach((obj, idx) => {
    const normalizedBox = obj.bbox_yxyx_norm || obj.bbox;
    const pixelBox = obj.bbox_xyxy_px;
    if ((!normalizedBox || normalizedBox.length < 4) && (!pixelBox || pixelBox.length < 4)) return;
    
    const b = normalizedBox || [];
    let x1;
    let y1;
    let x2;
    let y2;
    if (pixelBox?.length >= 4 && img.naturalWidth && img.naturalHeight) {
      x1 = (pixelBox[0] / img.naturalWidth) * cssWidth;
      y1 = (pixelBox[1] / img.naturalHeight) * cssHeight;
      x2 = (pixelBox[2] / img.naturalWidth) * cssWidth;
      y2 = (pixelBox[3] / img.naturalHeight) * cssHeight;
    } else {
      // DAM's public contract is normalized [y1, x1, y2, x2].
      y1 = Number(b[0]) * cssHeight;
      x1 = Number(b[1]) * cssWidth;
      y2 = Number(b[2]) * cssHeight;
      x2 = Number(b[3]) * cssWidth;
    }

    if (![x1, y1, x2, y2].every(Number.isFinite) || x2 <= x1 || y2 <= y1) return;
    x1 = Math.max(0, Math.min(cssWidth, x1));
    x2 = Math.max(0, Math.min(cssWidth, x2));
    y1 = Math.max(0, Math.min(cssHeight, y1));
    y2 = Math.max(0, Math.min(cssHeight, y2));

    const w = x2 - x1;
    const h = y2 - y1;
    const color = colors[idx % colors.length];

    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.strokeRect(x1, y1, w, h);

    const label = obj.class_entity || "Object";
    ctx.fillStyle = color;
    ctx.font = "bold 11px Inter, sans-serif";
    const textWidth = ctx.measureText(label).width;
    ctx.fillRect(x1, Math.max(0, y1 - 18), textWidth + 8, 18);

    ctx.fillStyle = "#000";
    ctx.fillText(label, x1 + 4, Math.max(13, y1 - 4));
  });
}

// Filmstrip Loader with a small virtual window around the active frame.
async function loadFilmstrip(videoId, currentKeyframeN) {
  if (!state.activeVideoKeyframes.length || state.activeVideoKeyframes[0].video_id !== videoId) {
    filmstripAbortController?.abort();
    filmstripAbortController = new AbortController();
    const requestId = ++filmstripRequestId;
    state.activeVideoKeyframes = [];
    el.filmstripScroll.innerHTML = `<div class="filmstrip-loading" aria-live="polite">Loading timeline…</div>`;
    try {
      const res = await fetch(`/api/video/${encodeURIComponent(videoId)}/keyframes`, {
        signal: filmstripAbortController.signal,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (requestId !== filmstripRequestId || state.activeInspectorItem?.video_id !== videoId) return;
      state.activeVideoKeyframes = data.keyframes || [];
    } catch (err) {
      if (err.name === "AbortError") return;
      console.warn("Filmstrip load error:", err);
      el.filmstripScroll.innerHTML = `<div class="filmstrip-loading error">Timeline unavailable</div>`;
      return;
    }
  }

  renderFilmstripWindow(currentKeyframeN);
}

function renderFilmstripWindow(currentKeyframeN) {
  const keyframes = state.activeVideoKeyframes;
  el.filmstripScroll.innerHTML = "";
  el.filmstripCount.textContent = `${keyframes.length} keyframes`;
  if (!keyframes.length) return;

  const activeIndex = Math.max(0, keyframes.findIndex((kf) => kf.keyframe_n === currentKeyframeN));
  const start = Math.max(0, activeIndex - FILMSTRIP_WINDOW_RADIUS);
  const end = Math.min(keyframes.length, activeIndex + FILMSTRIP_WINDOW_RADIUS + 1);
  const fragment = document.createDocumentFragment();

  if (start > 0) fragment.appendChild(createFilmstripJump(keyframes, start, -1));
  keyframes.slice(start, end).forEach((kf) => fragment.appendChild(createFilmstripItem(kf, currentKeyframeN)));
  if (end < keyframes.length) fragment.appendChild(createFilmstripJump(keyframes, end, 1));
  el.filmstripScroll.appendChild(fragment);

  const activeElement = el.filmstripScroll.querySelector(".filmstrip-item.active");
  activeElement?.scrollIntoView({ behavior: "auto", inline: "center", block: "nearest" });
}

function createFilmstripJump(keyframes, edgeIndex, direction) {
  const jump = document.createElement("button");
  jump.type = "button";
  jump.className = "filmstrip-jump";
  const remaining = direction < 0 ? edgeIndex : keyframes.length - edgeIndex;
  jump.textContent = direction < 0 ? `← ${remaining} earlier` : `${remaining} later →`;
  const targetIndex = direction < 0
    ? Math.max(0, edgeIndex - FILMSTRIP_WINDOW_RADIUS)
    : Math.min(keyframes.length - 1, edgeIndex + FILMSTRIP_WINDOW_RADIUS - 1);
  jump.addEventListener("click", () => selectFilmstripKeyframe(keyframes[targetIndex]));
  return jump;
}

function createFilmstripItem(kf, currentKeyframeN) {
  const isActive = kf.keyframe_n === currentKeyframeN;
  const item = document.createElement("div");
  item.className = "filmstrip-item" + (isActive ? " active" : "");
  item.dataset.keyframeN = kf.keyframe_n;
  item.tabIndex = 0;
  item.setAttribute("role", "button");
  item.setAttribute("aria-label", `Open keyframe ${kf.keyframe_n}, frame ${kf.frame_idx}`);

  const image = document.createElement("img");
  image.src = getImageUrl(kf);
  image.alt = "";
  image.loading = "lazy";
  image.decoding = "async";
  image.addEventListener("error", () => {
    image.hidden = true;
    item.classList.add("img-fallback");
  }, { once: true });
  const label = document.createElement("span");
  label.className = "filmstrip-lbl";
  label.textContent = String(kf.keyframe_n).padStart(3, "0");
  item.append(image, label);

  item.addEventListener("click", () => selectFilmstripKeyframe(kf));
  item.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      selectFilmstripKeyframe(kf);
    }
  });
  return item;
}

function selectFilmstripKeyframe(kf) {
  const retainFilmstripFocus = document.activeElement?.classList.contains("filmstrip-item");
  void openStandardInspector(kf, true);
  if (retainFilmstripFocus) {
    requestAnimationFrame(() => {
      /** @type {HTMLElement | null} */ (el.filmstripScroll.querySelector(".filmstrip-item.active"))
        ?.focus({ preventScroll: true });
    });
  }
}

function navigateFilmstrip(step) {
  if (!state.activeVideoKeyframes.length || !state.activeInspectorItem) return;
  const currN = parseInt(el.inspKeyframeN.textContent) || 1;
  const currIdx = state.activeVideoKeyframes.findIndex((k) => k.keyframe_n === currN);
  if (currIdx === -1) return;

  const nextIdx = currIdx + step;
  if (nextIdx >= 0 && nextIdx < state.activeVideoKeyframes.length) {
    selectFilmstripKeyframe(state.activeVideoKeyframes[nextIdx]);
  }
}

function closeInspector() {
  inspectorAbortController?.abort();
  filmstripAbortController?.abort();
  inspectorRequestId += 1;
  filmstripRequestId += 1;
  videoController.deactivate();
  setInspectorMediaMode("keyframe");
  el.modal.classList.add("hidden");
  setBackgroundInert(el.modal, false);
  el.filmstripScroll.replaceChildren();
  el.filmstripCount.textContent = "0 frames";
  el.inspectorImg.removeAttribute("src");
  state.activeVideoKeyframes = [];
  state.activeBBoxObjects = [];
  state.activeInspectorItem = null;
  drawBBoxesOnCanvas();
  if (lastInspectorFocus instanceof HTMLElement) lastInspectorFocus.focus();
  lastInspectorFocus = null;
}

function toggleCurrentInSubmission() {
  if (!state.activeInspectorItem) return;
  const item = state.activeInspectorItem;
  const subStr = item.submission_string || `${item.video_id}, ${item.frame_idx}`;

  if (state.selectedSubmission === subStr) {
    clearSubmissionSelection();
  } else {
    state.selectedSubmission = subStr;
    state.selectedSubmissionItem = item;
    el.submissionInput.value = subStr;
    showToast(`Added ${subStr} to submission!`);
  }

  updateInspectorSubmitBtn();
}

function updateInspectorSubmitBtn() {
  if (!state.activeInspectorItem) return;
  const item = state.activeInspectorItem;
  const subStr = item.submission_string || `${item.video_id}, ${item.frame_idx}`;
  const isSelected = state.selectedSubmission === subStr;

  el.btnToggleInSubmission.classList.toggle("in-submit", isSelected);
  el.btnToggleInSubmission.innerHTML = isSelected
    ? "<span>✓ In submission</span>"
    : "<span>+ Add to submission</span>";
}

function copySubmissionToClipboard() {
  const val = el.submissionInput.value.trim();
  if (!val || val === "No keyframe selected") {
    showToast("Please select a keyframe first.", "error");
    return;
  }
  navigator.clipboard.writeText(val).then(() => {
    showToast("📋 Copied to clipboard: " + val);
  }).catch(() => {
    showToast("Copy failed, please copy manually.", "error");
  });
}

function showToast(msg, type = "info") {
  if (toastTimer) window.clearTimeout(toastTimer);
  el.toast.textContent = msg;
  el.toast.classList.toggle("error", type === "error");
  el.toast.classList.toggle("success", type === "success");
  el.toast.classList.remove("hidden");
  toastTimer = window.setTimeout(() => {
    el.toast.classList.add("hidden");
    toastTimer = null;
  }, type === "error" ? 5000 : 3000);
}

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// ──────────────────────────────────────────────────────────────────────────────
// 100-Candidate Submission CSV Exporter
// ──────────────────────────────────────────────────────────────────────────────
function openExportModal() {
  if (!state.searchResults || state.searchResults.length === 0) {
    showToast("Please run a search query first.", "error");
    return;
  }
  if (!state.selectedSubmissionItem) {
    showToast("Select the keyframe you want as Row 1 before exporting.", "error");
    return;
  }
  lastExportFocus = document.activeElement;
  updateExportPreview();
  el.exportModal.classList.remove("hidden");
  setBackgroundInert(el.exportModal, true);
  requestAnimationFrame(() => el.exportQueryId.focus());
}

function closeExportModal() {
  el.exportModal.classList.add("hidden");
  setBackgroundInert(el.exportModal, false);
  if (lastExportFocus instanceof HTMLElement) lastExportFocus.focus();
  lastExportFocus = null;
}

function updateExportPreview() {
  const qId = el.exportQueryId ? el.exportQueryId.value.trim() || "1" : "1";
  if (el.exportQueryId) {
    el.exportQueryId.setCustomValidity(isValidQueryId(qId)
      ? ""
      : "Use only letters, numbers, underscores, and hyphens.");
  }
  const rows = generateValidatedSubmissionRows(qId);
  if (rows.length > 0 && el.exportRow1Preview) {
    el.exportRow1Preview.textContent = `${rows[0]} · ${rows.length}/100 verified rows ready`;
  }
  if (el.btnDownloadCsvAction) {
    el.btnDownloadCsvAction.disabled = rows.length !== 100 || !isValidQueryId(qId);
    const label = el.btnDownloadCsvAction.querySelector("span");
    if (label) label.textContent = rows.length === 100 ? "⚡ Download CSV (100 Rows)" : `Need ${100 - rows.length} more verified rows`;
  }
}

function isValidQueryId(queryId) {
  return /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(queryId);
}

function csvCell(value) {
  const text = String(value ?? "");
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function generateValidatedSubmissionRows(queryId) {
  if (!isValidQueryId(queryId) || !state.selectedSubmissionItem) return [];
  const selected = state.selectedSubmissionItem;
  const rows = [];
  const seen = new Set();

  const addCandidate = (candidate) => {
    if (!candidate?.video_id || !Number.isInteger(Number(candidate.frame_idx)) || rows.length >= 100) return;
    const frameIndex = Number(candidate.frame_idx);
    if (frameIndex < 0) return;
    const key = `${candidate.video_id}:${frameIndex}`;
    if (seen.has(key)) return;
    seen.add(key);
    rows.push([queryId, candidate.video_id, frameIndex].map(csvCell).join(","));
  };

  // Row 1 is always the explicit human selection, never the last inspected frame.
  addCandidate(selected);

  // Prefer real temporal neighbours only when canonical keyframe metadata is loaded.
  if (state.activeVideoKeyframes[0]?.video_id === selected.video_id) {
    const selectedIndex = state.activeVideoKeyframes.findIndex((keyframe) => (
      keyframe.frame_idx === selected.frame_idx || keyframe.keyframe_n === selected.keyframe_n
    ));
    if (selectedIndex >= 0) {
      [-1, 1, -2, 2, -3, 3, -4, 4].forEach((offset) => {
        const neighbour = state.activeVideoKeyframes[selectedIndex + offset];
        if (neighbour) addCandidate(neighbour);
      });
    }
  }

  state.searchResults.forEach(addCandidate);
  return rows;
}

function executeDownload100Csv() {
  const qId = el.exportQueryId ? el.exportQueryId.value.trim() || "1" : "1";
  if (!isValidQueryId(qId)) {
    showToast("Query ID may only contain letters, numbers, underscores, and hyphens.", "error");
    el.exportQueryId.focus();
    return;
  }
  const rows = generateValidatedSubmissionRows(qId);
  
  if (rows.length !== 100) {
    showToast(`Only ${rows.length} verified unique candidates are available; no fabricated rows were exported.`, "error");
    return;
  }

  const csvContent = rows.join("\n") + "\n";
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  
  const link = document.createElement("a");
  link.setAttribute("href", url);
  link.setAttribute("download", `submission_query_${qId}_${state.taskType.toLowerCase()}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);

  closeExportModal();
  showToast(`📥 Exported ${rows.length} verified submission rows for Query ${qId}.`, "success");
}

// ──────────────────────────────────────────────────────────────────────────────
// Run App
// ──────────────────────────────────────────────────────────────────────────────
if (document.readyState === "loading") {
  window.addEventListener("DOMContentLoaded", initApp, { once: true });
} else {
  void initApp();
}
