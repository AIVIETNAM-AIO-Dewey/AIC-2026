// AIC-2026 Multimodal Video Retrieval Engine Frontend Client

let currentTask = "KIS";
let currentParsedQuery = null;
let currentResults = [];

// DOM Elements
const taskTabs = document.querySelectorAll(".tab-btn");
const rawQueryInput = document.getElementById("rawQueryInput");
const parseBtn = document.getElementById("parseBtn");
const searchBtn = document.getElementById("searchBtn");

const queryInspector = document.getElementById("queryInspector");
const toggleInspector = document.getElementById("toggleInspector");
const inspectorBody = document.getElementById("inspectorBody");
const inspectorChevron = document.getElementById("inspectorChevron");

const inputSceneEn = document.getElementById("inputSceneEn");
const inputObjectsEn = document.getElementById("inputObjectsEn");
const inputSpeechVi = document.getElementById("inputSpeechVi");
const inputOcrText = document.getElementById("inputOcrText");

const sliderVis = document.getElementById("sliderVis");
const sliderDam = document.getElementById("sliderDam");
const sliderAsr = document.getElementById("sliderAsr");
const sliderOcr = document.getElementById("sliderOcr");

const valVis = document.getElementById("valVis");
const valDam = document.getElementById("valDam");
const valAsr = document.getElementById("valAsr");
const valOcr = document.getElementById("valOcr");

const resultsSection = document.querySelector(".results-section");
const resultsHeader = document.getElementById("resultsHeader");
const resultsGrid = document.getElementById("resultsGrid");
const emptyState = document.getElementById("emptyState");
const loadingOverlay = document.getElementById("loadingOverlay");
const latencyBadge = document.getElementById("latencyBadge");
const resultsCountBadge = document.getElementById("resultsCountBadge");

const vqaBanner = document.getElementById("vqaBanner");
const vqaAnswerText = document.getElementById("vqaAnswerText");
const vqaEvidenceSource = document.getElementById("vqaEvidenceSource");
const toast = document.getElementById("toast");

// --- Event Listeners ---

// 1. Task Tabs Switcher
taskTabs.forEach(tab => {
  tab.addEventListener("click", () => {
    taskTabs.forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    currentTask = tab.getAttribute("data-task");

    if (currentTask === "VQA") {
      rawQueryInput.placeholder = "Enter VQA Question (e.g., 'Người đàn ông mặc áo sơ mi màu gì?')...";
    } else if (currentTask === "TRAKE") {
      rawQueryInput.placeholder = "Enter multi-event sequence (e.g., 'Đầu tiên xe rẽ trái, sau đó người phụ nữ bước vào khách sạn')...";
    } else {
      rawQueryInput.placeholder = "Type search query in Vietnamese, English, or Vietlish (e.g., 'người đàn ông mặc áo vest xanh trong trường quay')...";
    }
  });
});

// 2. Sliders live sync
sliderVis.addEventListener("input", (e) => valVis.innerText = `${e.target.value}%`);
sliderDam.addEventListener("input", (e) => valDam.innerText = `${e.target.value}%`);
sliderAsr.addEventListener("input", (e) => valAsr.innerText = `${e.target.value}%`);
sliderOcr.addEventListener("input", (e) => valOcr.innerText = `${e.target.value}%`);

// 3. Toggle Inspector Accordion
toggleInspector.addEventListener("click", () => {
  const isOpen = inspectorBody.style.display !== "none";
  inspectorBody.style.display = isOpen ? "none" : "flex";
  inspectorChevron.style.transform = isOpen ? "rotate(0deg)" : "rotate(180deg)";
});

// 4. Enter Key Trigger Search
rawQueryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    executeFullSearch();
  }
});

parseBtn.addEventListener("click", parseQuery);
searchBtn.addEventListener("click", executeFullSearch);

// --- Core API Functions ---

async function parseQuery() {
  const query = rawQueryInput.value.trim();
  if (!query) return;

  showLoading("Parsing & Decomposing Sub-Queries with LLM...");

  try {
    const res = await fetch("/api/parse_query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: query, task_type: currentTask }),
    });

    if (!res.ok) throw new Error("Parse request failed");
    currentParsedQuery = await res.json();
    populateInspector(currentParsedQuery);
    hideLoading();

    // Ensure inspector is open so user can inspect
    inspectorBody.style.display = "flex";
    inspectorChevron.style.transform = "rotate(180deg)";
  } catch (err) {
    hideLoading();
    showToast("Query Parsing Error: " + err.message);
  }
}

function populateInspector(parsed) {
  inputSceneEn.value = parsed.global_scene_en || "";
  inputObjectsEn.value = (parsed.objects_en || []).join(", ");
  inputSpeechVi.value = parsed.speech_vi || "";
  inputOcrText.value = (parsed.ocr_keywords || []).join(", ");

  const w = parsed.weights || {};
  sliderVis.value = Math.round((w.vis || 0.45) * 100);
  sliderDam.value = Math.round((w.dam || 0.40) * 100);
  sliderAsr.value = Math.round((w.asr || 0.15) * 100);
  sliderOcr.value = Math.round((w.ocr || 0.00) * 100);

  valVis.innerText = `${sliderVis.value}%`;
  valDam.innerText = `${sliderDam.value}%`;
  valAsr.innerText = `${sliderAsr.value}%`;
  valOcr.innerText = `${sliderOcr.value}%`;
}

function collectInspectorPayload() {
  const rawText = rawQueryInput.value.trim();
  const sceneEn = inputSceneEn.value.trim();
  const objs = inputObjectsEn.value.split(",").map(s => s.trim()).filter(Boolean);
  const speech = inputSpeechVi.value.trim();
  const ocr = inputOcrText.value.split(",").map(s => s.trim()).filter(Boolean);

  const total = (+sliderVis.value) + (+sliderDam.value) + (+sliderAsr.value) + (+sliderOcr.value) || 100;
  const weights = {
    vis: (+sliderVis.value) / total,
    dam: (+sliderDam.value) / total,
    asr: (+sliderAsr.value) / total,
    ocr: (+sliderOcr.value) / total,
  };

  return {
    task_type: currentTask,
    language: "vi",
    original_query: rawText,
    global_scene_en: sceneEn || rawText,
    objects_en: objs.length ? objs : [rawText],
    ocr_keywords: ocr,
    speech_vi: speech,
    is_temporal_trake: currentTask === "TRAKE",
    trake_events: (currentParsedQuery && currentParsedQuery.trake_events) || [],
    vqa_question: currentTask === "VQA" ? rawText : "",
    weights: weights,
  };
}

async function executeFullSearch() {
  const rawQuery = rawQueryInput.value.trim();
  if (!rawQuery) return;

  // Auto-populate inspector if empty
  if (!inputSceneEn.value && !inputObjectsEn.value) {
    await parseQuery();
  }

  const payload = collectInspectorPayload();
  showLoading("Executing 4-Channel Parallel Search & Cross-Attention Re-Ranking...");

  try {
    const res = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ parsed_query: payload, top_k: 50 }),
    });

    if (!res.ok) throw new Error("Search request failed");
    const data = await res.json();
    currentResults = data.results || [];

    renderResults(data);
    hideLoading();
  } catch (err) {
    hideLoading();
    showToast("Search Error: " + err.message);
  }
}

// --- Render Search Results & Canvas Overlays ---

function renderResults(data) {
  emptyState.style.display = "none";
  resultsHeader.style.display = "flex";
  latencyBadge.innerText = `⚡ ${data.execution_time_ms.toFixed(1)} ms`;
  resultsCountBadge.innerText = `${data.results.length} Candidates`;

  // Render VQA banner if VQA mode
  if (data.task_type === "VQA" && data.results.length > 0 && data.results[0].vqa_answer) {
    vqaBanner.style.display = "flex";
    vqaAnswerText.innerText = data.results[0].vqa_answer;
    vqaEvidenceSource.innerText = `Evidence from Rank #1: ${data.results[0].video_id} (Frame ${data.results[0].keyframe_n} / idx ${data.results[0].frame_idx})`;
  } else {
    vqaBanner.style.display = "none";
  }

  resultsGrid.innerHTML = "";

  data.results.forEach((item, idx) => {
    const card = document.createElement("div");
    card.className = "result-card glassmorphism";

    const isTop1 = item.rank === 1;
    const rankClass = isTop1 ? "rank-badge top-1" : "rank-badge";

    // Build Evidence Items HTML
    const objListHtml = (item.best_matching_objects || [])
      .map(o => `<li><strong>${o.class_entity}</strong> (${(o.score).toFixed(2)}): <em>${o.description_en}</em></li>`)
      .join("");

    const speechHtml = item.speech_evidence
      ? `<div class="evidence-item"><strong>🎙️ Audio Speech [${item.speech_evidence.start_s.toFixed(1)}s -> ${item.speech_evidence.end_s.toFixed(1)}s]:</strong><p>"${item.speech_evidence.transcript_raw}"</p></div>`
      : `<div class="evidence-item" style="color: var(--text-muted)">🎙️ Silence / Music Scene</div>`;

    card.innerHTML = `
      <div class="card-media">
        <span class="${rankClass}">#${item.rank}</span>
        <span class="score-badge">Score: ${(item.final_score).toFixed(3)}</span>
        
        <img 
          class="card-image" 
          src="/${item.image_relpath}" 
          alt="${item.video_id} Frame ${item.keyframe_n}" 
          id="img_${idx}"
          onerror="handleImageFallback(this, '${item.video_id}', ${item.keyframe_n})"
        />
        <canvas class="card-canvas" id="canvas_${idx}"></canvas>
      </div>

      <div class="card-content">
        <div class="card-title-row">
          <div class="video-ref">${item.video_id} • Frame #${item.keyframe_n}</div>
          <div class="time-ref">idx: ${item.frame_idx} (${item.pts_time_s.toFixed(2)}s)</div>
        </div>

        <div class="submission-bar">
          <span class="submission-text">${item.submission_string}</span>
          <button class="copy-btn" onclick="copyToClipboard('${item.submission_string}')">📋 Copy Code</button>
        </div>

        <div class="evidence-drawer">
          <div class="evidence-toggle" onclick="toggleEvidence(${idx})">
            <span>🔍 Inspect Evidence (${item.best_matching_objects.length} Objects)</span>
            <span id="ev_chevron_${idx}">▼</span>
          </div>

          <div class="evidence-body" id="ev_body_${idx}">
            <div class="evidence-item">
              <strong>🖼️ Visual Cosine:</strong> ${(item.visual_similarity).toFixed(3)} | 
              <strong>🧠 BGE-Rerank:</strong> ${(item.stage2_rerank_score).toFixed(3)}
            </div>
            ${item.best_matching_objects.length ? `<div class="evidence-item"><strong>DAM Objects:</strong><ul style="padding-left:16px">${objListHtml}</ul></div>` : ""}
            ${speechHtml}
            ${item.ocr_text ? `<div class="evidence-item"><strong>🔤 Screen Text:</strong> ${item.ocr_text}</div>` : ""}
          </div>
        </div>
      </div>
    `;

    resultsGrid.appendChild(card);

    // Setup Canvas Bounding Box Overlays
    const imgEl = card.querySelector(`#img_${idx}`);
    const canvasEl = card.querySelector(`#canvas_${idx}`);

    if (imgEl && canvasEl && item.best_matching_objects && item.best_matching_objects.length > 0) {
      imgEl.onload = () => {
        drawBoundingBoxes(canvasEl, imgEl, item.best_matching_objects);
      };
    }
  });
}

// --- Canvas Bounding Box Renderer ---

function drawBoundingBoxes(canvas, img, objects) {
  const rect = img.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const colors = [
    { stroke: "#00F2FE", fill: "rgba(0, 242, 254, 0.18)" },
    { stroke: "#8B5CF6", fill: "rgba(139, 92, 246, 0.18)" },
    { stroke: "#10B981", fill: "rgba(16, 185, 129, 0.18)" },
  ];

  objects.forEach((obj, i) => {
    if (!obj.bbox || obj.bbox.length < 4) return;
    const [ymin, xmin, ymax, xmax] = obj.bbox;
    const color = colors[i % colors.length];

    const x = xmin * canvas.width;
    const y = ymin * canvas.height;
    const w = (xmax - xmin) * canvas.width;
    const h = (ymax - ymin) * canvas.height;

    // Translucent Bounding Box Fill
    ctx.fillStyle = color.fill;
    ctx.fillRect(x, y, w, h);

    // Crisp Border
    ctx.strokeStyle = color.stroke;
    ctx.lineWidth = 2.5;
    ctx.strokeRect(x, y, w, h);

    // Label Tag
    const label = `${obj.class_entity} ${(obj.score).toFixed(2)}`;
    ctx.font = "bold 11px Plus Jakarta Sans, sans-serif";
    const textWidth = ctx.measureText(label).width;

    ctx.fillStyle = color.stroke;
    ctx.fillRect(x, Math.max(0, y - 18), textWidth + 8, 18);

    ctx.fillStyle = "#070A10";
    ctx.fillText(label, x + 4, Math.max(12, y - 5));
  });
}

function handleImageFallback(img, videoId, keyframeN) {
  const container = img.parentElement;
  img.style.display = "none";

  const fallback = document.createElement("div");
  fallback.className = "image-fallback";
  fallback.innerHTML = `
    <div style="font-size: 1.8rem">🎬</div>
    <div style="font-weight: 600; color: var(--text-secondary)">${videoId} • Frame #${keyframeN}</div>
    <div class="fallback-badge">Image on Storage Server</div>
  `;
  container.appendChild(fallback);
}

// --- Toggle Evidence Accordion ---

window.toggleEvidence = function(idx) {
  const body = document.getElementById(`ev_body_${idx}`);
  const chevron = document.getElementById(`ev_chevron_${idx}`);
  if (body) {
    const isOpen = body.classList.contains("open");
    if (isOpen) {
      body.classList.remove("open");
      chevron.style.transform = "rotate(0deg)";
    } else {
      body.classList.add("open");
      chevron.style.transform = "rotate(180deg)";
    }
  }
};

// --- Copy to Clipboard Toast ---

window.copyToClipboard = function(text) {
  navigator.clipboard.writeText(text).then(() => {
    showToast(`Copied: "${text}"`);
  });
};

document.getElementById("copyAllSubmissionsBtn")?.addEventListener("click", () => {
  if (!currentResults.length) return;
  const list = currentResults.slice(0, 10).map(r => r.submission_string).join("\n");
  copyToClipboard(list);
});

function showToast(msg) {
  toast.innerText = msg;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 2500);
}

function showLoading(msg) {
  document.getElementById("loadingStatusText").innerText = msg;
  loadingOverlay.style.display = "flex";
}

function hideLoading() {
  loadingOverlay.style.display = "none";
}
