/**
 * AIC-2026 Multimodal Retrieval Studio — Frontend Logic
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
};

// ──────────────────────────────────────────────────────────────────────────────
// DOM Elements
// ──────────────────────────────────────────────────────────────────────────────
const el = {
  taskBtns: document.querySelectorAll(".task-btn"),
  geminiToggle: document.getElementById("btn-gemini-toggle"),
  geminiDot: document.querySelector("#btn-gemini-toggle .indicator-dot"),
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
  
  // Submission
  submissionInput: document.getElementById("submission-input"),
  btnCopySubmission: document.getElementById("btn-copy-submission"),
  btnClearSubmission: document.getElementById("btn-clear-submission"),
  toast: document.getElementById("toast"),
  
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
  el.btnClearSubmission.addEventListener("click", () => {
    state.selectedSubmission = "";
    el.submissionInput.value = "No keyframe selected";
    updateInspectorSubmitBtn();
  });

  // Inspector modal
  el.btnCloseInspector.addEventListener("click", closeInspector);
  el.btnToggleInSubmission.addEventListener("click", toggleCurrentInSubmission);
  el.chkBBoxes.addEventListener("change", drawBBoxesOnCanvas);

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
    } else if (e.key === "ArrowLeft") {
      navigateFilmstrip(-1);
    } else if (e.key === "ArrowRight") {
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

  // Sync with JSON editor if it has valid JSON
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
    // 1. Call Parse endpoint
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

    // Update JSON editor and sliders
    el.jsonEditor.value = JSON.stringify(state.parsedQuery, null, 2);
    if (state.parsedQuery.weights) {
      setSliderValues(state.parsedQuery.weights);
    }

    // In Auto-Run mode, immediately execute search
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

  // Inject current sliders into parsedJson
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
    // If no session exists yet, run full search
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
      // If session expired or not found, fallback to full search
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
// Results Grid Rendering
// ──────────────────────────────────────────────────────────────────────────────
function getImageUrl(item) {
  const vid = item.video_id;
  const kn = String(item.keyframe_n || 1).padStart(3, "0");
  
  // Prefer file:// protocol if loaded locally, fallback to /keyframes/
  if (window.location.protocol === "file:") {
    return `file://${state.keyframesRoot}/${vid}/${kn}.jpg`;
  }
  return `/keyframes/${vid}/${kn}.jpg`;
}

function renderResults(results) {
  el.resultsGrid.innerHTML = "";
  const limit = parseInt(el.selectTopK.value) || 20;
  const list = results.slice(0, limit);

  el.resultsCount.textContent = `${list.length} candidate keyframes`;

  if (list.length === 0) {
    el.resultsGrid.innerHTML = `
      <div class="empty-placeholder">
        <div class="empty-icon">🔍</div>
        <div class="empty-title">No Candidates Found</div>
        <div class="empty-desc">Try lowering threshold or adjusting modality weights.</div>
      </div>`;
    return;
  }

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
        <div class="card-speech-snippet">${escapeHtml(speechTxt)}</div>
        <div class="card-tags-row">
          <span class="pill-tag ${item.rank_vis ? 'active' : ''}">👁️ Vis ${item.rank_vis ? '#' + item.rank_vis : '-'}</span>
          <span class="pill-tag ${item.rank_dam ? 'active' : ''}">📦 DAM ${item.rank_dam ? '#' + item.rank_dam : '-'}</span>
          <span class="pill-tag ${item.rank_asr ? 'active' : ''}">🎙️ ASR ${item.rank_asr ? '#' + item.rank_asr : '-'}</span>
        </div>
      </div>`;

    card.addEventListener("click", () => openInspector(item));
    el.resultsGrid.appendChild(card);
  });
}

// ──────────────────────────────────────────────────────────────────────────────
// Frame Inspector Modal
// ──────────────────────────────────────────────────────────────────────────────
async function openInspector(item) {
  state.activeInspectorItem = item;
  el.modal.classList.remove("hidden");

  // Populate basic card metadata
  el.inspVideoId.textContent = item.video_id || "-";
  el.inspKeyframeN.textContent = String(item.keyframe_n || 1).padStart(3, "0");
  el.inspFrameIdx.textContent = `${item.frame_idx || 0} (${item.pts_time_s ? item.pts_time_s.toFixed(1) + 's' : '-'})`;
  
  const rank = item.final_rank || item.rank || 1;
  const score = item.final_score || item.stage1_score || 0.0;
  el.inspScoreRank.textContent = `${score.toFixed(4)} • #${rank}`;

  // Image & Canvas setup
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

  // Populate text descriptions
  el.inspAsrText.textContent = item.asr_transcript || "(No speech / silent frame)";
  el.inspDamText.textContent = item.dam_summary || "(No visual description available)";

  updateInspectorSubmitBtn();

  // Load detailed DAM bounding boxes & macro audio from API
  try {
    const res = await fetch(`/api/keyframe/${item.video_id}/${item.keyframe_n}`);
    if (res.ok) {
      const data = await res.json();
      state.activeBBoxObjects = data.dam_objects || [];
      if (data.macro_audio_transcript) {
        el.inspAsrText.textContent = data.macro_audio_transcript;
      }
      renderMatchedObjectsList(state.activeBBoxObjects);
      drawBBoxesOnCanvas();
    }
  } catch (err) {
    console.warn("Could not fetch detailed keyframe metadata:", err);
  }

  // Load filmstrip for this video
  loadFilmstrip(item.video_id, item.keyframe_n);
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
    
    // Normalized bbox format: [ymin, xmin, ymax, xmax] or [x1, y1, x2, y2]
    const b = obj.bbox;
    let x1, y1, x2, y2;
    if (b[0] < b[2] && b[1] < b[3]) {
      // standard [x1, y1, x2, y2]
      x1 = b[0] * canvas.width;
      y1 = b[1] * canvas.height;
      x2 = b[2] * canvas.width;
      y2 = b[3] * canvas.height;
    } else {
      // [ymin, xmin, ymax, xmax]
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

    // Label
    const label = obj.class_entity || "Object";
    ctx.fillStyle = color;
    ctx.font = "bold 11px Inter, sans-serif";
    const textWidth = ctx.measureText(label).width;
    ctx.fillRect(x1, Math.max(0, y1 - 18), textWidth + 8, 18);

    ctx.fillStyle = "#000";
    ctx.fillText(label, x1 + 4, Math.max(13, y1 - 4));
  });
}

async function loadFilmstrip(videoId, currentKeyframeN) {
  el.filmstripScroll.innerHTML = "";
  try {
    const res = await fetch(`/api/video/${videoId}/keyframes`);
    if (!res.ok) return;

    const data = await res.json();
    state.activeVideoKeyframes = data.keyframes || [];
    el.filmstripCount.textContent = `${state.activeVideoKeyframes.length} keyframes`;

    state.activeVideoKeyframes.forEach((kf) => {
      const item = document.createElement("div");
      item.className = "filmstrip-item" + (kf.keyframe_n === currentKeyframeN ? " active" : "");
      item.dataset.keyframeN = kf.keyframe_n;

      const imgUrl = getImageUrl(kf);
      item.innerHTML = `
        <img src="${imgUrl}" alt="${kf.keyframe_n}" onerror="this.onerror=null; this.src=''; this.parentElement.style.background='#1e293b';">
        <span class="filmstrip-lbl">${String(kf.keyframe_n).padStart(3, "0")}</span>`;

      item.addEventListener("click", () => {
        openInspector(kf);
      });

      el.filmstripScroll.appendChild(item);
    });

    // Auto-scroll to active item
    const activeEl = el.filmstripScroll.querySelector(".filmstrip-item.active");
    if (activeEl) {
      activeEl.scrollIntoView({ behavior: "smooth", inline: "center", block: "nearest" });
    }
  } catch (err) {
    console.warn("Filmstrip load error:", err);
  }
}

function navigateFilmstrip(step) {
  if (!state.activeVideoKeyframes.length || !state.activeInspectorItem) return;
  const currN = state.activeInspectorItem.keyframe_n;
  const currIdx = state.activeVideoKeyframes.findIndex((k) => k.keyframe_n === currN);
  if (currIdx === -1) return;

  const nextIdx = currIdx + step;
  if (nextIdx >= 0 && nextIdx < state.activeVideoKeyframes.length) {
    openInspector(state.activeVideoKeyframes[nextIdx]);
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
// Run App
// ──────────────────────────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", initApp);
