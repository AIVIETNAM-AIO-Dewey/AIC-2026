/**
 * AIC-2026 Multimodal Retrieval Studio — Frontend Logic
 * Supports KIS, VQA with answer editing, and TRAKE Monotonic Sequence Editor.
 */

// ──────────────────────────────────────────────────────────────────────────────
// App State
// ──────────────────────────────────────────────────────────────────────────────
const state = {
  taskType: "KIS",
  queryMode: "auto", // "auto", "edit", "direct"
  useGemini: true,
  sessionId: null,
  keyframesRoot: "/Users/khoale/Downloads/AIC_Challenger/data/keyframes",
  parsedQuery: null,
  searchResults: [],
  activeInspectorItem: null,
  activeVideoKeyframes: [],
  selectedSubmission: "",
  activeBBoxObjects: [],
  activeTrakeEventIndex: 0,
};

// ──────────────────────────────────────────────────────────────────────────────
// DOM Elements
// ──────────────────────────────────────────────────────────────────────────────
const el = {
  taskBtns: document.querySelectorAll(".task-btn"),
  geminiToggle: document.getElementById("btn-gemini-toggle"),
  geminiText: document.querySelector("#btn-gemini-toggle .btn-text"),
  modeTabs: document.querySelectorAll(".mode-tab"),
  
  colOriginalQuery: document.getElementById("col-original-query"),
  colParsedJson: document.getElementById("col-parsed-json"),
  inputQuery: document.getElementById("input-query"),
  jsonEditor: document.getElementById("json-editor"),
  
  btnRunQuery: document.getElementById("btn-run-query"),
  btnRunLabel: document.getElementById("btn-run-label"),
  btnExecuteJson: document.getElementById("btn-execute-json"),
  btnFormatJson: document.getElementById("btn-format-json"),
  
  btnReFuse: document.getElementById("btn-re-fuse"),
  btnReRank: document.getElementById("btn-re-rank"),
  
  timingBadge: document.getElementById("timing-badge"),
  sessionBadge: document.getElementById("session-badge"),
  resultsGrid: document.getElementById("results-grid"),
  resultsCount: document.getElementById("results-count-badge"),
  selectTopK: document.getElementById("select-top-k"),
  
  // Sliders
  sliderVis: document.getElementById("slider-w-vis"),
  sliderDam: document.getElementById("slider-w-dam"),
  sliderAsr: document.getElementById("slider-w-asr"),
  sliderOcr: document.getElementById("slider-w-ocr"),
  valVis: document.getElementById("val-w-vis"),
  valDam: document.getElementById("val-w-dam"),
  valAsr: document.getElementById("val-w-asr"),
  valOcr: document.getElementById("val-w-ocr"),
  
  // Submission & Export
  submissionInput: document.getElementById("submission-input"),
  btnCopySubmission: document.getElementById("btn-copy-submission"),
  btnExportCsv: document.getElementById("btn-export-csv"),
  btnClearSubmission: document.getElementById("btn-clear-submission"),
  toast: document.getElementById("toast"),

  // 100-Row Export Modal
  exportModal: document.getElementById("export-modal"),
  btnCloseExportModal: document.getElementById("btn-close-export-modal"),
  btnCancelExportModal: document.getElementById("btn-cancel-export-modal"),
  btnDownloadCsvAction: document.getElementById("btn-download-csv-action"),
  exportQueryId: document.getElementById("export-query-id"),
  exportRow1Preview: document.getElementById("export-row1-preview"),
  
  // Inspector Modal
  modal: document.getElementById("inspector-modal"),
  inspectorImg: document.getElementById("inspector-img"),
  inspectorCanvas: document.getElementById("inspector-canvas"),
  inspectorPlaceholder: document.getElementById("inspector-img-placeholder"),
  placeholderText: document.getElementById("placeholder-text"),
  chkBBoxes: document.getElementById("chk-show-bboxes"),
  
  inspVideoId: document.getElementById("insp-video-id"),
  inspKeyframeN: document.getElementById("insp-keyframe-n"),
  inspFrameIdx: document.getElementById("insp-frame-idx"),
  inspScoreRank: document.getElementById("insp-score-rank"),
  inspAsrText: document.getElementById("insp-asr-text"),
  inspDamText: document.getElementById("insp-dam-text"),
  inspOcrText: document.getElementById("insp-ocr-text"),
  inspObjectsList: document.getElementById("insp-objects-list"),
  
  // TRAKE and VQA Inspector Additions
  trakeTabsBar: document.getElementById("insp-trake-event-tabs"),
  trakeTabsList: document.getElementById("trake-tabs-list"),
  inspVqaBox: document.getElementById("insp-vqa-box"),
  inspVqaInput: document.getElementById("insp-vqa-input"),
  
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
async function initApp() {
  bindEvents();
  await loadServerConfig();
  updateQueryModeUI();
}

async function loadServerConfig() {
  try {
    const res = await fetch("/api/config");
    if (res.ok) {
      const cfg = await res.json();
      if (cfg.keyframes_root) state.keyframesRoot = cfg.keyframes_root;
      if (cfg.default_weights) {
        setSliderValues(cfg.default_weights);
      }
    }
  } catch (err) {
    console.warn("Could not load /api/config:", err);
  }
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
  el.btnExecuteJson.addEventListener("click", () => handleExecuteJsonClick(true));
  el.btnFormatJson.addEventListener("click", formatJsonEditor);

  // Keyboard shortcut Ctrl+Enter / Cmd+Enter on textareas
  el.inputQuery.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      handleRunQueryClick();
    }
  });

  el.jsonEditor.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      handleExecuteJsonClick(true);
    }
  });

  // Sliders input
  [el.sliderVis, el.sliderDam, el.sliderAsr, el.sliderOcr].forEach((s) => {
    s.addEventListener("input", handleSliderChange);
  });

  // Instant Re-Fuse and Re-Rank
  el.btnReFuse.addEventListener("click", () => handleCachedReFuse(false));
  el.btnReRank.addEventListener("click", () => handleCachedReFuse(true));

  // Top-k dropdown
  el.selectTopK.addEventListener("change", () => {
    renderResults(state.searchResults);
  });

  // Submission bar
  el.btnCopySubmission.addEventListener("click", copySubmissionToClipboard);
  if (el.btnExportCsv) el.btnExportCsv.addEventListener("click", openExportModal);
  if (el.btnCloseExportModal) el.btnCloseExportModal.addEventListener("click", closeExportModal);
  if (el.btnCancelExportModal) el.btnCancelExportModal.addEventListener("click", closeExportModal);
  if (el.btnDownloadCsvAction) el.btnDownloadCsvAction.addEventListener("click", executeDownload100Csv);
  if (el.exportQueryId) el.exportQueryId.addEventListener("input", updateExportPreview);

  el.btnClearSubmission.addEventListener("click", () => {
    state.selectedSubmission = "";
    el.submissionInput.value = "No keyframe selected";
    updateInspectorSubmitBtn();
  });

  // Inspector modal
  el.btnCloseInspector.addEventListener("click", closeInspector);
  el.btnToggleInSubmission.addEventListener("click", toggleCurrentInSubmission);
  el.chkBBoxes.addEventListener("change", drawBBoxesOnCanvas);

  // VQA Answer Input Edit
  el.inspVqaInput.addEventListener("input", (e) => {
    if (!state.activeInspectorItem) return;
    const ans = e.target.value.trim();
    state.activeInspectorItem.vqa_answer = ans;
    const vid = state.activeInspectorItem.video_id;
    const fIdx = state.activeInspectorItem.frame_idx;
    state.activeInspectorItem.submission_string = `${vid}, ${fIdx}, "${ans}"`;
    
    if (state.selectedSubmission) {
      state.selectedSubmission = state.activeInspectorItem.submission_string;
      el.submissionInput.value = state.selectedSubmission;
    }
  });

  el.btnFilmstripPrev.addEventListener("click", () => {
    el.filmstripScroll.scrollBy({ left: -300, behavior: "smooth" });
  });
  el.btnFilmstripNext.addEventListener("click", () => {
    el.filmstripScroll.scrollBy({ left: 300, behavior: "smooth" });
  });

  // Global Keydown (Escape, ArrowLeft, ArrowRight, Enter)
  document.addEventListener("keydown", (e) => {
    if (el.modal.classList.contains("hidden")) return;

    if (e.key === "Escape") {
      closeInspector();
    } else if (e.key === "ArrowLeft" && !e.target.matches("input, textarea")) {
      navigateFilmstrip(-1);
    } else if (e.key === "ArrowRight" && !e.target.matches("input, textarea")) {
      navigateFilmstrip(1);
    } else if (e.key === "Enter" && !e.target.matches("textarea, input")) {
      e.preventDefault();
      toggleCurrentInSubmission();
    }
  });
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
// Sliders & Weight Logic
// ──────────────────────────────────────────────────────────────────────────────
function getSliderWeights() {
  return {
    vis: parseFloat(el.sliderVis.value),
    dam: parseFloat(el.sliderDam.value),
    asr: parseFloat(el.sliderAsr.value),
    ocr: parseFloat(el.sliderOcr.value),
  };
}

function setSliderValues(weights) {
  if (!weights) return;
  if (weights.vis !== undefined) {
    el.sliderVis.value = weights.vis;
    el.valVis.textContent = parseFloat(weights.vis).toFixed(2);
  }
  if (weights.dam !== undefined) {
    el.sliderDam.value = weights.dam;
    el.valDam.textContent = parseFloat(weights.dam).toFixed(2);
  }
  if (weights.asr !== undefined) {
    el.sliderAsr.value = weights.asr;
    el.valAsr.textContent = parseFloat(weights.asr).toFixed(2);
  }
  if (weights.ocr !== undefined) {
    el.sliderOcr.value = weights.ocr;
    el.valOcr.textContent = parseFloat(weights.ocr).toFixed(2);
  }
}

function handleSliderChange() {
  const w = getSliderWeights();
  el.valVis.textContent = w.vis.toFixed(2);
  el.valDam.textContent = w.dam.toFixed(2);
  el.valAsr.textContent = w.asr.toFixed(2);
  el.valOcr.textContent = w.ocr.toFixed(2);

  try {
    const data = JSON.parse(el.jsonEditor.value);
    data.weights = w;
    el.jsonEditor.value = JSON.stringify(data, null, 2);
  } catch (_) {}
}

function formatJsonEditor() {
  try {
    const data = JSON.parse(el.jsonEditor.value);
    el.jsonEditor.value = JSON.stringify(data, null, 2);
  } catch (err) {
    showToast("Invalid JSON: " + err.message);
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
    showToast("Please enter a query text.");
    return;
  }

  el.timingBadge.textContent = "Parsing query...";

  try {
    const parseRes = await fetch("/api/parse", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: query,
        task_type: state.taskType,
        engine: state.useGemini ? "gemini" : "qwen",
      }),
    });

    if (!parseRes.ok) throw new Error("Parse failed: " + parseRes.statusText);
    const parseData = await parseRes.json();
    state.parsedQuery = parseData.parsed_query;

    el.jsonEditor.value = JSON.stringify(state.parsedQuery, null, 2);
    if (state.parsedQuery.weights) {
      setSliderValues(state.parsedQuery.weights);
    }

    if (state.queryMode === "auto") {
      await handleExecuteJsonClick(true);
    } else {
      el.timingBadge.textContent = `Parsed in ${parseData.execution_time_ms}ms (Ready to edit)`;
    }
  } catch (err) {
    console.error(err);
    el.timingBadge.textContent = "Error parsing";
    showToast("Error: " + err.message);
  }
}

async function handleExecuteJsonClick(runStage2 = true) {
  let parsedJson;
  try {
    parsedJson = JSON.parse(el.jsonEditor.value);
  } catch (e) {
    showToast("Invalid JSON syntax: " + e.message);
    return;
  }

  parsedJson.weights = getSliderWeights();
  parsedJson.task_type = state.taskType;

  el.timingBadge.textContent = "Searching 177k keyframes...";

  try {
    const res = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        parsed_query: parsedJson,
        session_id: state.sessionId,
        top_k_pool: 300,
        top_k_rerank: 50,
        final_top_k: parseInt(el.selectTopK.value) || 20,
        run_stage2: runStage2,
      }),
    });

    if (!res.ok) throw new Error("Search failed: " + res.statusText);
    const data = await res.json();

    state.sessionId = data.session_id;
    state.searchResults = data.results || [];

    el.timingBadge.textContent = `Completed in ${data.execution_time_ms}ms`;
    el.sessionBadge.textContent = `Session: ${data.session_id.slice(0, 8)}...`;
    el.sessionBadge.classList.remove("hidden");

    renderResults(state.searchResults);
  } catch (err) {
    console.error(err);
    el.timingBadge.textContent = "Search error";
    showToast("Search failed: " + err.message);
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Instant Cached Re-Fuse (< 5ms)
// ──────────────────────────────────────────────────────────────────────────────
async function handleCachedReFuse(runStage2 = false) {
  if (!state.sessionId) {
    return handleExecuteJsonClick(runStage2);
  }

  const weights = getSliderWeights();
  el.timingBadge.textContent = runStage2 ? "Re-ranking top 50..." : "Instant re-fusing...";

  try {
    const res = await fetch("/api/search/cached", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: state.sessionId,
        weights: weights,
        top_k_pool: 300,
        top_k_rerank: 50,
        final_top_k: parseInt(el.selectTopK.value) || 20,
        run_stage2: runStage2,
      }),
    });

    if (!res.ok) {
      return handleExecuteJsonClick(runStage2);
    }

    const data = await res.json();
    state.searchResults = data.results || [];
    el.timingBadge.textContent = `⚡ Re-fused in ${data.execution_time_ms}ms (Cached)`;

    renderResults(state.searchResults);
  } catch (err) {
    console.error(err);
    showToast("Re-fuse failed: " + err.message);
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Results Grid Rendering (KIS, VQA, and TRAKE Sequences)
// ──────────────────────────────────────────────────────────────────────────────
function getImageUrl(item) {
  const vid = item.video_id;
  const kn = String(item.keyframe_n || 1).padStart(3, "0");
  
  if (window.location.protocol === "file:") {
    return `file://${state.keyframesRoot}/${vid}/${kn}.jpg`;
  }
  return `/keyframes/${vid}/${kn}.jpg`;
}

function renderResults(results) {
  el.resultsGrid.innerHTML = "";
  const limit = parseInt(el.selectTopK.value) || 20;
  const list = results.slice(0, limit);

  el.resultsCount.textContent = `${list.length} candidate items`;

  if (list.length === 0) {
    el.resultsGrid.innerHTML = `
      <div class="empty-placeholder">
        <div class="empty-icon">🔍</div>
        <div class="empty-title">No Candidates Found</div>
        <div class="empty-desc">Try lowering threshold or adjusting modality weights.</div>
      </div>`;
    return;
  }

  // Detect if results are TRAKE sequence format
  const isTrake = state.taskType === "TRAKE" || (list[0] && Array.isArray(list[0].matched_frames));

  if (isTrake) {
    renderTrakeSequences(list);
  } else {
    renderStandardCards(list);
  }
}

// Standard KIS & VQA Cards
function renderStandardCards(list) {
  list.forEach((item, idx) => {
    const rank = item.final_rank || item.rank || idx + 1;
    const score = item.final_score || item.stage1_score || 0.0;
    const timeS = item.pts_time_s ? item.pts_time_s.toFixed(1) + "s" : "-";
    const imgUrl = getImageUrl(item);

    const card = document.createElement("div");
    card.className = "candidate-card";
    card.dataset.index = idx;

    const speechTxt = item.asr_transcript && item.asr_transcript !== "[Silent Frame]"
      ? `🎙️ "${item.asr_transcript}"`
      : "(No speech / background audio)";

    const vqaBadge = (state.taskType === "VQA" || item.vqa_answer)
      ? `<div class="vqa-answer-badge">💡 Answer: "${escapeHtml(item.vqa_answer || 'N/A')}"</div>`
      : "";

    card.innerHTML = `
      <div class="card-media">
        <img src="${imgUrl}" alt="${item.video_id}" onerror="this.onerror=null; this.src=''; this.parentElement.classList.add('img-fallback');">
        <span class="card-rank-badge">#${rank}</span>
        <span class="card-time-badge">${timeS}</span>
      </div>
      <div class="card-body">
        <div class="card-title-row">
          <span class="card-vid-name">${item.video_id} : ${item.frame_idx}</span>
          <div class="card-scores">
            <span class="score-final">${(score * 100).toFixed(1)}%</span>
          </div>
        </div>
        ${vqaBadge}
        <div class="card-speech-snippet">${escapeHtml(speechTxt)}</div>
        <div class="card-tags-row">
          <span class="pill-tag ${item.rank_vis ? 'active' : ''}">👁️ Vis ${item.rank_vis ? '#' + item.rank_vis : '-'}</span>
          <span class="pill-tag ${item.rank_dam ? 'active' : ''}">📦 DAM ${item.rank_dam ? '#' + item.rank_dam : '-'}</span>
          <span class="pill-tag ${item.rank_asr ? 'active' : ''}">🎙️ ASR ${item.rank_asr ? '#' + item.rank_asr : '-'}</span>
        </div>
      </div>`;

    card.addEventListener("click", () => openStandardInspector(item));
    el.resultsGrid.appendChild(card);
  });
}

// TRAKE Horizontal Multi-Event Sequence Cards
function renderTrakeSequences(list) {
  list.forEach((seq, idx) => {
    const rank = seq.rank || idx + 1;
    const fScore = seq.final_score || seq.sequence_score || 0.0;
    const dpScore = seq.dp_score || seq.sequence_score || 0.0;
    const narrScore = seq.narrative_score || dpScore;
    const frames = seq.matched_frames || [];
    const timestamps = seq.timestamps || [];
    const dossiers = seq.event_dossiers || [];

    const isMono = frames.every((f, i) => i === 0 || f > frames[i - 1]);

    const card = document.createElement("div");
    card.className = "trake-sequence-card";

    // Build event slots HTML
    let eventsHtml = "";
    frames.forEach((f, evIdx) => {
      const timeVal = timestamps[evIdx] ? timestamps[evIdx].toFixed(1) + "s" : "";
      const d = dossiers[evIdx] || {};
      const kn = d.keyframe_n || (evIdx + 1);
      const imgUrl = getImageUrl({ video_id: seq.video_id, keyframe_n: kn });

      eventsHtml += `
        <div class="trake-event-slot">
          <div class="event-slot-header">
            <span class="event-num-pill">E${evIdx + 1}</span>
            <span class="event-time-txt">${timeVal} (Frame ${f})</span>
          </div>
          <div class="event-thumb-wrap">
            <img src="${imgUrl}" alt="E${evIdx + 1}" onerror="this.onerror=null; this.src=''; this.parentElement.style.background='#1e293b';">
          </div>
        </div>`;

      if (evIdx < frames.length - 1) {
        eventsHtml += `<span class="event-arrow-separator">➔</span>`;
      }
    });

    card.innerHTML = `
      <div class="trake-card-header">
        <div class="trake-header-left">
          <span class="trake-rank-pill">#${rank}</span>
          <span class="trake-video-name">${seq.video_id}</span>
          <span class="trake-mono-badge">${isMono ? '✅ Strict Monotonic' : '⚠️ Order Warning'}</span>
        </div>
        <div class="card-scores">
          <span style="font-size:12px;color:var(--text-dim);">DP: ${(dpScore * 100).toFixed(1)}% | Narr: ${(narrScore * 100).toFixed(1)}%</span>
          <span class="score-final" style="font-size:14px;margin-left:8px;">${(fScore * 100).toFixed(1)}%</span>
        </div>
      </div>
      <div class="trake-events-strip">${eventsHtml}</div>
      <div style="font-size:11.5px;color:var(--text-dim);display:flex;justify-content:space-between;">
        <span>Click sequence to open Interactive Monotonic Editor</span>
        <span style="font-family:var(--font-mono);color:var(--accent-cyan);">${seq.submission_string || ''}</span>
      </div>`;

    card.addEventListener("click", () => openTrakeInspector(seq, 0));
    el.resultsGrid.appendChild(card);
  });
}

// ──────────────────────────────────────────────────────────────────────────────
// Frame Inspector Modal
// ──────────────────────────────────────────────────────────────────────────────

// Standard Inspector (KIS & VQA)
async function openStandardInspector(item) {
  state.activeInspectorItem = item;
  el.modal.classList.remove("hidden");
  el.trakeTabsBar.classList.add("hidden");

  // VQA Answer Input
  if (state.taskType === "VQA" || item.vqa_answer) {
    el.inspVqaBox.classList.remove("hidden");
    el.inspVqaInput.value = item.vqa_answer || "";
  } else {
    el.inspVqaBox.classList.add("hidden");
  }

  populateInspectorCommon(item);
  loadFilmstrip(item.video_id, item.keyframe_n, null, null);
}

// TRAKE Monotonic Sequence Inspector
async function openTrakeInspector(sequence, activeEventIndex = 0) {
  state.activeInspectorItem = sequence;
  state.activeTrakeEventIndex = activeEventIndex;
  el.modal.classList.remove("hidden");
  el.trakeTabsBar.classList.remove("hidden");
  el.inspVqaBox.classList.add("hidden");

  renderTrakeTabs(sequence);

  // Load all keyframes for this video first if not already loaded
  if (!state.activeVideoKeyframes.length || state.activeVideoKeyframes[0].video_id !== sequence.video_id) {
    try {
      const res = await fetch(`/api/video/${sequence.video_id}/keyframes`);
      if (res.ok) {
        const data = await res.json();
        state.activeVideoKeyframes = data.keyframes || [];
      }
    } catch (err) {
      console.warn("Could not load video keyframes:", err);
    }
  }

  displayTrakeActiveEvent(sequence, activeEventIndex);
}

function renderTrakeTabs(sequence) {
  el.trakeTabsList.innerHTML = "";
  const frames = sequence.matched_frames || [];

  frames.forEach((f, idx) => {
    const btn = document.createElement("button");
    btn.className = "trake-tab-btn" + (idx === state.activeTrakeEventIndex ? " active" : "");
    btn.textContent = `E${idx + 1}: Frame ${f}`;
    btn.addEventListener("click", () => {
      state.activeTrakeEventIndex = idx;
      renderTrakeTabs(sequence);
      displayTrakeActiveEvent(sequence, idx);
    });
    el.trakeTabsList.appendChild(btn);
  });
}

function displayTrakeActiveEvent(sequence, eventIdx) {
  const frames = sequence.matched_frames || [];
  const currentF = frames[eventIdx];
  
  // Find keyframe item in active video keyframes
  let targetKf = state.activeVideoKeyframes.find((k) => k.frame_idx === currentF);
  if (!targetKf) {
    targetKf = {
      video_id: sequence.video_id,
      frame_idx: currentF,
      keyframe_n: eventIdx + 1,
      pts_time_s: (sequence.timestamps && sequence.timestamps[eventIdx]) || 0.0,
      score: sequence.final_score || 0.0,
    };
  }

  populateInspectorCommon(targetKf);

  // Calculate strict monotonic boundaries for this event slot:
  // min_frame = previous event's frame + 1 (or 0 if event 1)
  // max_frame = next event's frame - 1 (or Infinity if last event)
  const minFrame = eventIdx > 0 ? frames[eventIdx - 1] + 1 : 0;
  const maxFrame = eventIdx < frames.length - 1 ? frames[eventIdx + 1] - 1 : Infinity;

  loadFilmstrip(sequence.video_id, targetKf.keyframe_n, minFrame, maxFrame);
}

function populateInspectorCommon(item) {
  el.inspVideoId.textContent = item.video_id || "-";
  el.inspKeyframeN.textContent = String(item.keyframe_n || 1).padStart(3, "0");
  el.inspFrameIdx.textContent = `${item.frame_idx || 0} (${item.pts_time_s ? item.pts_time_s.toFixed(1) + 's' : '-'})`;
  
  const rank = item.final_rank || item.rank || 1;
  const score = item.final_score || item.stage1_score || 0.0;
  el.inspScoreRank.textContent = `${score.toFixed(4)} • #${rank}`;

  const imgUrl = getImageUrl(item);
  el.inspectorImg.src = imgUrl;
  el.inspectorImg.onerror = () => {
    el.inspectorPlaceholder.classList.remove("hidden");
    el.placeholderText.textContent = `${item.video_id} / ${String(item.keyframe_n || 1).padStart(3, "0")}.jpg`;
  };
  el.inspectorImg.onload = () => {
    el.inspectorPlaceholder.classList.add("hidden");
    drawBBoxesOnCanvas();
  };

  el.inspAsrText.textContent = item.asr_transcript || "(No speech / silent frame)";
  el.inspDamText.textContent = item.dam_summary || "(No visual description available)";
  if (el.inspOcrText) {
    el.inspOcrText.textContent = item.ocr_text || "(No text detected on screen)";
  }

  updateInspectorSubmitBtn();

  // Load detailed DAM bounding boxes & metadata from API
  fetch(`/api/keyframe/${item.video_id}/${item.keyframe_n}`)
    .then((r) => (r.ok ? r.json() : null))
    .then((data) => {
      if (data) {
        state.activeBBoxObjects = data.dam_objects || [];
        if (data.macro_audio_transcript) {
          el.inspAsrText.textContent = data.macro_audio_transcript;
        }
        if (el.inspOcrText && data.keyframe && data.keyframe.ocr_text) {
          el.inspOcrText.textContent = data.keyframe.ocr_text;
        }
        renderMatchedObjectsList(state.activeBBoxObjects);
        drawBBoxesOnCanvas();
      }
    })
    .catch(() => {});
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

  canvas.width = img.clientWidth || 800;
  canvas.height = img.clientHeight || 450;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!el.chkBBoxes.checked || !state.activeBBoxObjects.length) return;

  const colors = ["#38bdf8", "#34d399", "#f59e0b", "#ec4899", "#818cf8"];

  state.activeBBoxObjects.forEach((obj, idx) => {
    if (!obj.bbox || obj.bbox.length < 4) return;
    
    const b = obj.bbox;
    let x1, y1, x2, y2;
    if (b[0] < b[2] && b[1] < b[3]) {
      x1 = b[0] * canvas.width;
      y1 = b[1] * canvas.height;
      x2 = b[2] * canvas.width;
      y2 = b[3] * canvas.height;
    } else {
      y1 = b[0] * canvas.height;
      x1 = b[1] * canvas.width;
      y2 = b[2] * canvas.height;
      x2 = b[3] * canvas.width;
    }

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

// Filmstrip Loader with Dynamic Monotonic Boundary Enforcement
async function loadFilmstrip(videoId, currentKeyframeN, minFrame = null, maxFrame = null) {
  el.filmstripScroll.innerHTML = "";
  
  if (!state.activeVideoKeyframes.length || state.activeVideoKeyframes[0].video_id !== videoId) {
    try {
      const res = await fetch(`/api/video/${videoId}/keyframes`);
      if (res.ok) {
        const data = await res.json();
        state.activeVideoKeyframes = data.keyframes || [];
      }
    } catch (err) {
      console.warn("Filmstrip load error:", err);
      return;
    }
  }

  el.filmstripCount.textContent = `${state.activeVideoKeyframes.length} keyframes`;

  state.activeVideoKeyframes.forEach((kf) => {
    const isLocked = (minFrame !== null && kf.frame_idx < minFrame) || (maxFrame !== null && kf.frame_idx > maxFrame);
    const isActive = kf.keyframe_n === currentKeyframeN;

    const item = document.createElement("div");
    item.className = "filmstrip-item" + (isActive ? " active" : "") + (isLocked ? " locked" : "");
    item.dataset.keyframeN = kf.keyframe_n;

    const imgUrl = getImageUrl(kf);
    item.innerHTML = `
      <img src="${imgUrl}" alt="${kf.keyframe_n}" onerror="this.onerror=null; this.src=''; this.parentElement.style.background='#1e293b';">
      <span class="filmstrip-lbl">${String(kf.keyframe_n).padStart(3, "0")}</span>`;

    if (!isLocked) {
      item.addEventListener("click", () => {
        if (state.activeInspectorItem && Array.isArray(state.activeInspectorItem.matched_frames)) {
          // In TRAKE mode: update this event slot's selected frame!
          const seq = state.activeInspectorItem;
          const evIdx = state.activeTrakeEventIndex;
          seq.matched_frames[evIdx] = kf.frame_idx;
          
          if (seq.event_dossiers && seq.event_dossiers[evIdx]) {
            seq.event_dossiers[evIdx].keyframe_n = kf.keyframe_n;
            seq.event_dossiers[evIdx].frame_idx = kf.frame_idx;
            seq.event_dossiers[evIdx].pts_time_s = kf.pts_time_s;
            seq.event_dossiers[evIdx].asr_transcript = kf.asr_transcript_vi || "";
          }
          seq.submission_string = `${seq.video_id}, ${seq.matched_frames.join(', ')}`;
          
          if (state.selectedSubmission) {
            state.selectedSubmission = seq.submission_string;
            el.submissionInput.value = seq.submission_string;
          }

          // Re-render tabs and refresh sequence card + live preview!
          renderTrakeTabs(seq);
          displayTrakeActiveEvent(seq, evIdx);
          renderResults(state.searchResults);
          showToast(`Event E${evIdx + 1} updated to frame ${kf.frame_idx}!`);
        } else {
          // Standard mode
          openStandardInspector(kf);
        }
      });
    }

    el.filmstripScroll.appendChild(item);
  });

  // Auto-scroll to active item
  const activeEl = el.filmstripScroll.querySelector(".filmstrip-item.active");
  if (activeEl) {
    activeEl.scrollIntoView({ behavior: "smooth", inline: "center", block: "nearest" });
  }
}

function navigateFilmstrip(step) {
  if (!state.activeVideoKeyframes.length || !state.activeInspectorItem) return;
  const currN = parseInt(el.inspKeyframeN.textContent) || 1;
  const currIdx = state.activeVideoKeyframes.findIndex((k) => k.keyframe_n === currN);
  if (currIdx === -1) return;

  const nextIdx = currIdx + step;
  if (nextIdx >= 0 && nextIdx < state.activeVideoKeyframes.length) {
    const nextKf = state.activeVideoKeyframes[nextIdx];
    if (state.activeInspectorItem && Array.isArray(state.activeInspectorItem.matched_frames)) {
      displayTrakeActiveEvent(state.activeInspectorItem, state.activeTrakeEventIndex);
    } else {
      openStandardInspector(nextKf);
    }
  }
}

function closeInspector() {
  el.modal.classList.add("hidden");
}

function toggleCurrentInSubmission() {
  if (!state.activeInspectorItem) return;
  const item = state.activeInspectorItem;
  const subStr = item.submission_string || `${item.video_id}, ${item.frame_idx}`;

  if (state.selectedSubmission === subStr) {
    state.selectedSubmission = "";
    el.submissionInput.value = "No keyframe selected";
  } else {
    state.selectedSubmission = subStr;
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
    showToast("Please select a keyframe first.");
    return;
  }
  navigator.clipboard.writeText(val).then(() => {
    showToast("📋 Copied to clipboard: " + val);
  }).catch(() => {
    showToast("Copy failed, please copy manually.");
  });
}

function showToast(msg) {
  el.toast.textContent = msg;
  el.toast.classList.remove("hidden");
  setTimeout(() => {
    el.toast.classList.add("hidden");
  }, 2200);
}

function escapeHtml(str) {
  return (str || "")
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
    showToast("Please run a search query first.");
    return;
  }
  updateExportPreview();
  el.exportModal.classList.remove("hidden");
}

function closeExportModal() {
  el.exportModal.classList.add("hidden");
}

function updateExportPreview() {
  const qId = el.exportQueryId ? el.exportQueryId.value.trim() || "1" : "1";
  const rows = generate100SubmissionRows(qId);
  if (rows.length > 0 && el.exportRow1Preview) {
    el.exportRow1Preview.textContent = rows[0];
  }
}

function generate100SubmissionRows(queryId) {
  const rows = [];
  const task = state.taskType;
  const results = state.searchResults || [];
  
  if (task === "KIS") {
    const topItem = state.activeInspectorItem || results[0];
    if (!topItem) return [];

    const topVid = topItem.video_id;
    const topFrame = topItem.frame_idx;
    const topKn = topItem.keyframe_n || 1;

    // Row 1: Human pick
    rows.push(`${queryId},${topVid},${topFrame}`);
    const seen = new Set([`${topVid}_${topFrame}`]);

    // Tier 2 (Rows 2–8): Adjacent frames of top video (±1, ±2, ±3)
    if (state.activeVideoKeyframes && state.activeVideoKeyframes.length > 0 && state.activeVideoKeyframes[0].video_id === topVid) {
      const currIdx = state.activeVideoKeyframes.findIndex(k => k.frame_idx === topFrame || k.keyframe_n === topKn);
      if (currIdx !== -1) {
        const offsets = [-1, 1, -2, 2, -3, 3, -4, 4];
        for (const off of offsets) {
          const target = currIdx + off;
          if (target >= 0 && target < state.activeVideoKeyframes.length && rows.length < 10) {
            const kf = state.activeVideoKeyframes[target];
            const key = `${topVid}_${kf.frame_idx}`;
            if (!seen.has(key)) {
              seen.add(key);
              rows.push(`${queryId},${topVid},${kf.frame_idx}`);
            }
          }
        }
      }
    }

    // Tier 3 (Rows 9–25): Top 2 frames from secondary suspect videos
    const secondaryVideos = new Set();
    results.forEach(cand => {
      if (cand.video_id !== topVid) secondaryVideos.add(cand.video_id);
    });

    for (const vid of secondaryVideos) {
      if (rows.length >= 30) break;
      const vidCands = results.filter(c => c.video_id === vid);
      for (const c of vidCands.slice(0, 2)) {
        const key = `${c.video_id}_${c.frame_idx}`;
        if (!seen.has(key) && rows.length < 30) {
          seen.add(key);
          rows.push(`${queryId},${c.video_id},${c.frame_idx}`);
        }
      }
    }

    // Tier 4 (Rows 26–100): Remaining candidates from searchResults
    for (const cand of results) {
      const key = `${cand.video_id}_${cand.frame_idx}`;
      if (!seen.has(key) && rows.length < 100) {
        seen.add(key);
        rows.push(`${queryId},${cand.video_id},${cand.frame_idx}`);
      }
    }

    // Fill to 100 if needed
    let ptr = 0;
    while (rows.length < 100 && ptr < results.length) {
      const cand = results[ptr++];
      const f = cand.frame_idx;
      for (const delta of [-30, 30, -60, 60, -90, 90]) {
        const fakeF = Math.max(1, f + delta);
        const key = `${cand.video_id}_${fakeF}`;
        if (!seen.has(key) && rows.length < 100) {
          seen.add(key);
          rows.push(`${queryId},${cand.video_id},${fakeF}`);
        }
      }
    }
  } else if (task === "VQA") {
    const topItem = state.activeInspectorItem || results[0];
    if (!topItem) return [];

    const topVid = topItem.video_id;
    const topFrame = topItem.frame_idx;
    const topKn = topItem.keyframe_n || 1;
    
    // User-edited answer (or model answer if not edited)
    const chosenAnswer = (el.inspVqaInput && el.inspVqaInput.value.trim()) || topItem.vqa_answer || "màu xanh";
    const formattedAns = chosenAnswer.replace(/"/g, '""');

    // Row 1: Human pick + answer
    rows.push(`${queryId},${topVid},${topFrame},"${formattedAns}"`);
    const seen = new Set([`${topVid}_${topFrame}`]);

    // Tier 2 (Rows 2–8): Adjacent frames of top video with user's answer
    if (state.activeVideoKeyframes && state.activeVideoKeyframes.length > 0 && state.activeVideoKeyframes[0].video_id === topVid) {
      const currIdx = state.activeVideoKeyframes.findIndex(k => k.frame_idx === topFrame || k.keyframe_n === topKn);
      if (currIdx !== -1) {
        const offsets = [-1, 1, -2, 2, -3, 3, -4, 4];
        for (const off of offsets) {
          const target = currIdx + off;
          if (target >= 0 && target < state.activeVideoKeyframes.length && rows.length < 10) {
            const kf = state.activeVideoKeyframes[target];
            const key = `${topVid}_${kf.frame_idx}`;
            if (!seen.has(key)) {
              seen.add(key);
              rows.push(`${queryId},${topVid},${kf.frame_idx},"${formattedAns}"`);
            }
          }
        }
      }
    }

    // Tier 3 & 4 (Rows 9–100): Secondary candidate frames ALL using user's answer
    for (const cand of results) {
      const key = `${cand.video_id}_${cand.frame_idx}`;
      if (!seen.has(key) && rows.length < 100) {
        seen.add(key);
        rows.push(`${queryId},${cand.video_id},${cand.frame_idx},"${formattedAns}"`);
      }
    }

    let ptr = 0;
    while (rows.length < 100 && ptr < results.length) {
      const cand = results[ptr++];
      const f = cand.frame_idx;
      for (const delta of [-30, 30, -60, 60, -90, 90]) {
        const fakeF = Math.max(1, f + delta);
        const key = `${cand.video_id}_${fakeF}`;
        if (!seen.has(key) && rows.length < 100) {
          seen.add(key);
          rows.push(`${queryId},${cand.video_id},${fakeF},"${formattedAns}"`);
        }
      }
    }
  } else if (task === "TRAKE") {
    const topSeq = state.activeInspectorItem || results[0];
    if (!topSeq || !Array.isArray(topSeq.matched_frames)) return [];

    const topVid = topSeq.video_id;
    const topFrames = [...topSeq.matched_frames];

    // Row 1: Human refined sequence
    rows.push(`${queryId},${topVid},${topFrames.join(",")}`);
    const seen = new Set([`${topVid}_${topFrames.join("_")}`]);

    // Tier 2 (Rows 2–20): Monotonic perturbations of top sequence
    for (let slot = 0; slot < topFrames.length; slot++) {
      for (const delta of [-30, 30, -60, 60, -90, 90]) {
        const candidateFrames = [...topFrames];
        candidateFrames[slot] += delta;
        
        let isMonotonic = true;
        for (let i = 0; i < candidateFrames.length - 1; i++) {
          if (candidateFrames[i] >= candidateFrames[i + 1] || candidateFrames[i] <= 0) {
            isMonotonic = false;
            break;
          }
        }
        if (isMonotonic) {
          const key = `${topVid}_${candidateFrames.join("_")}`;
          if (!seen.has(key) && rows.length < 25) {
            seen.add(key);
            rows.push(`${queryId},${topVid},${candidateFrames.join(",")}`);
          }
        }
      }
    }

    // Tier 3 & 4 (Rows 21–100): Other candidate sequences from results
    for (const seq of results) {
      if (!seq.matched_frames) continue;
      const key = `${seq.video_id}_${seq.matched_frames.join("_")}`;
      if (!seen.has(key) && rows.length < 100) {
        seen.add(key);
        rows.push(`${queryId},${seq.video_id},${seq.matched_frames.join(",")}`);
      }
    }

    // Fill to 100 with valid variations of other sequences
    for (const seq of results) {
      if (rows.length >= 100) break;
      if (!seq.matched_frames) continue;
      for (let slot = 0; slot < seq.matched_frames.length; slot++) {
        for (const delta of [-30, 30, -60, 60]) {
          const candidateFrames = [...seq.matched_frames];
          candidateFrames[slot] += delta;
          let isMonotonic = true;
          for (let i = 0; i < candidateFrames.length - 1; i++) {
            if (candidateFrames[i] >= candidateFrames[i + 1] || candidateFrames[i] <= 0) {
              isMonotonic = false;
              break;
            }
          }
          if (isMonotonic) {
            const key = `${seq.video_id}_${candidateFrames.join("_")}`;
            if (!seen.has(key) && rows.length < 100) {
              seen.add(key);
              rows.push(`${queryId},${seq.video_id},${candidateFrames.join(",")}`);
            }
          }
        }
      }
    }
  }

  return rows.slice(0, 100);
}

function executeDownload100Csv() {
  const qId = el.exportQueryId ? el.exportQueryId.value.trim() || "1" : "1";
  const rows = generate100SubmissionRows(qId);
  
  if (rows.length === 0) {
    showToast("No candidates available to export.");
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
  showToast(`📥 Successfully exported ${rows.length} submission rows for Query ${qId}!`);
}

// ──────────────────────────────────────────────────────────────────────────────
// Run App
// ──────────────────────────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", initApp);
