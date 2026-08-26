import { createIcons, ExternalLink, Maximize, Pause, Play, RotateCcw, RotateCw } from "lucide";
import { createYouTubeVideoView, type VideoFrameMapping } from "./youtube-video-view";

type VisualMode = "keyframe" | "video";

interface DemoObject {
  classEntity: string;
  description: string;
  bbox?: [number, number, number, number];
}

interface DemoFrame {
  videoId: string;
  keyframeN: number;
  frameIdx: number;
  ptsTimeS: number;
  score: number;
  imagePath: string;
  ocrText: string;
  damSummary: string;
  objects: DemoObject[];
}

const VIDEO_DURATION_S = 1262;

const filmstripFrames: DemoFrame[] = [
  {
    videoId: "L21_V001",
    keyframeN: 1,
    frameIdx: 4,
    ptsTimeS: 0.1333,
    score: 0.947,
    imagePath: "/data/keyframe/L21_V001/00000004.jpg",
    ocrText: "71 06:30:11 giay",
    damSummary: "A city skyline rises above a calm river at the beginning of the morning news broadcast.",
    objects: [
      { classEntity: "Skyline", description: "Tall buildings beside the river", bbox: [0, 0, 1, 0.615] },
      { classEntity: "River", description: "Calm water reflecting the skyline", bbox: [0, 0.646, 1, 0.997] },
    ],
  },
  {
    videoId: "L21_V001",
    keyframeN: 2,
    frameIdx: 31,
    ptsTimeS: 1.0333,
    score: 0.921,
    imagePath: "/data/keyframe/L21_V001/00000031.jpg",
    ocrText: "06:30:12",
    damSummary: "The opening aerial view continues over the city and river.",
    objects: [],
  },
  {
    videoId: "L21_V001",
    keyframeN: 101,
    frameIdx: 4948,
    ptsTimeS: 164.9333,
    score: 0.905,
    imagePath: "/data/keyframe/L21_V001/00004948.jpg",
    ocrText: "Cuu nam thanh nien di xe may roi deo Prenn o Da Lat",
    damSummary: "A dark green rescue hat with an official emblem fills the foreground.",
    objects: [{ classEntity: "Hat", description: "Dark green rescue hat" }],
  },
  {
    videoId: "L21_V001",
    keyframeN: 251,
    frameIdx: 12156,
    ptsTimeS: 405.2,
    score: 0.899,
    imagePath: "/data/keyframe/L21_V001/00012156.jpg",
    ocrText: "Hop tac Viet Nam Campuchia",
    damSummary: "A mature green tree appears in a report about regional environmental cooperation.",
    objects: [{ classEntity: "Tree", description: "Broad green tree" }],
  },
  {
    videoId: "L21_V001",
    keyframeN: 417,
    frameIdx: 19707,
    ptsTimeS: 656.9,
    score: 0.938,
    imagePath: "/data/keyframe/L21_V001/00019707.jpg",
    ocrText: "Dak Lak giam 4 don vi hanh chinh cap xa sau khi sap nhap",
    damSummary: "A speaker wearing rectangular glasses appears during a local administration report.",
    objects: [{ classEntity: "Glasses", description: "Thin rectangular spectacles" }],
  },
  {
    videoId: "L21_V001",
    keyframeN: 601,
    frameIdx: 27681,
    ptsTimeS: 922.7,
    score: 0.912,
    imagePath: "/data/keyframe/L21_V001/00027681.jpg",
    ocrText: "Bang gia dat moi o TP HCM",
    damSummary: "The HTV logo appears over a report about updated land prices in Ho Chi Minh City.",
    objects: [{ classEntity: "HTV logo", description: "Broadcast channel logo" }],
  },
  {
    videoId: "L21_V001",
    keyframeN: 833,
    frameIdx: 37834,
    ptsTimeS: 1261.1333,
    score: 0.887,
    imagePath: "/data/keyframe/L21_V001/00037834.jpg",
    ocrText: "Trung tam tin tuc Dai Truyen hinh Thanh pho Ho Chi Minh",
    damSummary: "The HTV9 station ident closes the morning news programme.",
    objects: [{ classEntity: "HTV9 logo", description: "Closing station ident" }],
  },
];

const resultFrames = [filmstripFrames[0], filmstripFrames[4], filmstripFrames[6]];

function requireElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing demo element #${id}`);
  return element as T;
}

const elements = {
  modal: requireElement<HTMLDivElement>("inspector-modal"),
  resultsGrid: requireElement<HTMLDivElement>("results-grid"),
  resultsCount: requireElement<HTMLSpanElement>("results-count-badge"),
  statusText: requireElement<HTMLSpanElement>("server-status-text"),
  queryInput: requireElement<HTMLTextAreaElement>("input-query"),
  jsonEditor: requireElement<HTMLTextAreaElement>("json-editor"),
  runButton: requireElement<HTMLButtonElement>("btn-run-query"),
  runLabel: requireElement<HTMLSpanElement>("btn-run-label"),
  executeJsonButton: requireElement<HTMLButtonElement>("btn-execute-json"),
  reFuseButton: requireElement<HTMLButtonElement>("btn-re-fuse"),
  reRankButton: requireElement<HTMLButtonElement>("btn-re-rank"),
  timingBadge: requireElement<HTMLSpanElement>("timing-badge"),
  sessionBadge: requireElement<HTMLSpanElement>("session-badge"),
  image: requireElement<HTMLImageElement>("inspector-img"),
  canvas: requireElement<HTMLCanvasElement>("inspector-canvas"),
  placeholder: requireElement<HTMLDivElement>("inspector-img-placeholder"),
  placeholderText: requireElement<HTMLDivElement>("placeholder-text"),
  keyframeView: requireElement<HTMLDivElement>("keyframe-view"),
  videoView: requireElement<HTMLDivElement>("video-view"),
  mappingStatus: requireElement<HTMLSpanElement>("mapping-status"),
  keyframeTab: requireElement<HTMLButtonElement>("btn-view-keyframe"),
  videoTab: requireElement<HTMLButtonElement>("btn-view-video"),
  keyframeControls: requireElement<HTMLDivElement>("keyframe-visual-controls"),
  bboxCheckbox: requireElement<HTMLInputElement>("chk-show-bboxes"),
  videoId: requireElement<HTMLSpanElement>("insp-video-id"),
  keyframeN: requireElement<HTMLSpanElement>("insp-keyframe-n"),
  frameIdx: requireElement<HTMLSpanElement>("insp-frame-idx"),
  scoreRank: requireElement<HTMLSpanElement>("insp-score-rank"),
  matchedTime: requireElement<HTMLSpanElement>("insp-matched-time"),
  videoSource: requireElement<HTMLSpanElement>("insp-video-source"),
  mappingOffset: requireElement<HTMLSpanElement>("insp-mapping-offset"),
  asrText: requireElement<HTMLDivElement>("insp-asr-text"),
  damText: requireElement<HTMLDivElement>("insp-dam-text"),
  ocrText: requireElement<HTMLDivElement>("insp-ocr-text"),
  objectsList: requireElement<HTMLDivElement>("insp-objects-list"),
  filmstrip: requireElement<HTMLDivElement>("filmstrip-scroll"),
  filmstripCount: requireElement<HTMLSpanElement>("filmstrip-count"),
  filmstripPrev: requireElement<HTMLButtonElement>("btn-filmstrip-prev"),
  filmstripNext: requireElement<HTMLButtonElement>("btn-filmstrip-next"),
  closeButton: requireElement<HTMLButtonElement>("btn-close-inspector"),
  submitButton: requireElement<HTMLButtonElement>("btn-toggle-in-submission"),
  submissionInput: requireElement<HTMLInputElement>("submission-input"),
  copyButton: requireElement<HTMLButtonElement>("btn-copy-submission"),
  clearButton: requireElement<HTMLButtonElement>("btn-clear-submission"),
  toast: requireElement<HTMLDivElement>("toast"),
};

let activeFrame = resultFrames[0];
let visualMode: VisualMode = "keyframe";

function formatTime(seconds: number): string {
  const safeSeconds = Math.max(0, Math.min(VIDEO_DURATION_S, seconds));
  const minutes = Math.floor(safeSeconds / 60);
  const remainder = Math.floor(safeSeconds % 60);
  return `${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
}

function refreshIcons(): void {
  createIcons({ icons: { ExternalLink, Maximize, Pause, Play, RotateCcw, RotateCw } });
}

function setMappingStatus(label: string, state: "pending" | "ready" | "error" = "pending"): void {
  elements.mappingStatus.textContent = label;
  elements.mappingStatus.classList.toggle("ready", state === "ready");
  elements.mappingStatus.classList.toggle("error", state === "error");
}

function showToast(message: string): void {
  elements.toast.textContent = message;
  elements.toast.classList.remove("hidden");
  window.setTimeout(() => elements.toast.classList.add("hidden"), 1800);
}

function toVideoFrame(frame: DemoFrame): VideoFrameMapping {
  return { videoId: frame.videoId, ptsTimeS: frame.ptsTimeS, posterPath: frame.imagePath };
}

const videoController = createYouTubeVideoView({
  onSessionChange: (label) => {
    elements.sessionBadge.textContent = label;
  },
  onSourceChange: (label) => {
    elements.videoSource.textContent = label;
  },
  onStatusChange: setMappingStatus,
  onToast: showToast,
  refreshIcons,
});

function setVisualMode(mode: VisualMode): void {
  visualMode = mode;
  const showVideo = mode === "video";
  elements.keyframeTab.classList.toggle("active", !showVideo);
  elements.videoTab.classList.toggle("active", showVideo);
  elements.keyframeTab.setAttribute("aria-selected", String(!showVideo));
  elements.videoTab.setAttribute("aria-selected", String(showVideo));
  elements.keyframeView.classList.toggle("hidden", showVideo);
  elements.videoView.classList.toggle("hidden", !showVideo);
  elements.placeholder.classList.add("hidden");
  elements.keyframeControls.classList.toggle("hidden", showVideo);
  if (showVideo) void videoController.activate().catch(() => undefined);
  else videoController.deactivate();
}

function renderObjects(frame: DemoFrame): void {
  elements.objectsList.replaceChildren();
  if (frame.objects.length === 0) {
    const empty = document.createElement("span");
    empty.className = "demo-empty-object";
    empty.textContent = "No distinct objects in this preview";
    elements.objectsList.appendChild(empty);
    return;
  }

  frame.objects.forEach((object) => {
    const row = document.createElement("div");
    row.className = "object-item";
    const name = document.createElement("span");
    name.className = "obj-name";
    name.textContent = object.classEntity;
    const description = document.createElement("span");
    description.className = "obj-score";
    description.textContent = object.description;
    row.append(name, description);
    elements.objectsList.appendChild(row);
  });
}

function drawBBoxes(frame: DemoFrame): void {
  const canvas = elements.canvas;
  const width = elements.image.clientWidth || 960;
  const height = elements.image.clientHeight || 540;
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) return;
  context.clearRect(0, 0, width, height);
  if (!elements.bboxCheckbox.checked || visualMode !== "keyframe") return;

  frame.objects.forEach((object, index) => {
    if (!object.bbox) return;
    const [x1, y1, x2, y2] = object.bbox;
    const colors = ["#38bdf8", "#34d399", "#f59e0b"];
    const color = colors[index % colors.length];
    context.strokeStyle = color;
    context.lineWidth = 2.5;
    context.strokeRect(x1 * width, y1 * height, (x2 - x1) * width, (y2 - y1) * height);
  });
}

function renderFilmstrip(): void {
  elements.filmstrip.replaceChildren();
  elements.filmstripCount.textContent = `${filmstripFrames.length} preview frames`;
  filmstripFrames.forEach((frame) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `filmstrip-item${frame.frameIdx === activeFrame.frameIdx ? " active" : ""}`;
    button.dataset.keyframeN = String(frame.keyframeN);
    button.title = `Frame ${frame.frameIdx} at ${formatTime(frame.ptsTimeS)}`;
    button.innerHTML = `<img src="${frame.imagePath}" alt="Keyframe ${frame.keyframeN}"><span class="filmstrip-lbl">${String(frame.keyframeN).padStart(3, "0")}</span>`;
    button.addEventListener("click", () => selectFrame(frame));
    elements.filmstrip.appendChild(button);
  });
  elements.filmstrip.querySelector(".filmstrip-item.active")?.scrollIntoView({ inline: "center", block: "nearest" });
}

function populateInspector(frame: DemoFrame): void {
  activeFrame = frame;
  videoController.deactivate();
  elements.videoId.textContent = frame.videoId;
  elements.keyframeN.textContent = String(frame.keyframeN).padStart(3, "0");
  elements.frameIdx.textContent = `${frame.frameIdx} (${frame.ptsTimeS.toFixed(1)}s)`;
  elements.scoreRank.textContent = `${frame.score.toFixed(4)} • #${Math.max(1, resultFrames.indexOf(frame) + 1)}`;
  elements.matchedTime.textContent = `${formatTime(frame.ptsTimeS)}.${Math.floor((frame.ptsTimeS % 1) * 10)}`;
  elements.mappingOffset.textContent = "0.0s";
  elements.asrText.textContent = "(Not loaded in UI preview)";
  elements.damText.textContent = frame.damSummary;
  elements.ocrText.textContent = frame.ocrText;
  elements.image.src = frame.imagePath;
  videoController.setFrame(toVideoFrame(frame));
  elements.placeholderText.textContent = `${frame.videoId} / ${String(frame.frameIdx).padStart(8, "0")}.jpg`;
  elements.image.onload = () => {
    elements.placeholder.classList.add("hidden");
    drawBBoxes(frame);
  };
  elements.image.onerror = () => elements.placeholder.classList.remove("hidden");
  renderObjects(frame);
  renderFilmstrip();
  if (visualMode === "video") void videoController.activate().catch(() => undefined);
}

function selectFrame(frame: DemoFrame): void {
  populateInspector(frame);
}

function openInspector(frame: DemoFrame): void {
  setVisualMode("keyframe");
  populateInspector(frame);
  elements.modal.classList.remove("hidden");
}

function closeInspector(): void {
  videoController.deactivate();
  elements.modal.classList.add("hidden");
}

function renderResults(): void {
  elements.resultsGrid.replaceChildren();
  elements.resultsCount.textContent = `${resultFrames.length} UI preview frames`;
  resultFrames.forEach((frame, index) => {
    const card = document.createElement("article");
    card.className = "candidate-card";
    card.tabIndex = 0;
    card.innerHTML = `
      <div class="card-media">
        <img src="${frame.imagePath}" alt="${frame.videoId} frame ${frame.frameIdx}">
        <span class="card-rank-badge">#${index + 1}</span>
        <span class="card-time-badge">${formatTime(frame.ptsTimeS)}</span>
      </div>
      <div class="card-body">
        <div class="card-title-row">
          <span class="card-vid-name">${frame.videoId} : ${frame.frameIdx}</span>
          <span class="score-final">${(frame.score * 100).toFixed(1)}%</span>
        </div>
        <div class="card-speech-snippet">${frame.damSummary}</div>
        <div class="card-tags-row">
          <span class="pill-tag active">Keyframe ${String(frame.keyframeN).padStart(3, "0")}</span>
          <span class="pill-tag">Video UI</span>
        </div>
      </div>`;
    card.addEventListener("click", () => openInspector(frame));
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openInspector(frame);
      }
    });
    elements.resultsGrid.appendChild(card);
  });
  void videoController.preload(toVideoFrame(activeFrame)).catch(() => undefined);
}

function configureDemoChrome(): void {
  document.body.classList.add("video-ui-demo");
  elements.statusText.textContent = "Sample Demo · L21_V001";
  elements.queryInput.value = "Preview keyframe-to-video mapping UI";
  elements.jsonEditor.value = JSON.stringify({
    mode: "video-ui",
    video_id: "L21_V001",
    mapping_status: "youtube",
    preview_frames: resultFrames.map((frame) => frame.frameIdx),
  }, null, 2);
  elements.jsonEditor.readOnly = true;
  elements.runLabel.textContent = "Load Demo Frames";
  elements.executeJsonButton.classList.add("hidden");
  elements.reFuseButton.disabled = true;
  elements.reRankButton.disabled = true;
  elements.timingBadge.textContent = "UI demo · no backend";
  elements.sessionBadge.classList.remove("hidden");
  elements.sessionBadge.textContent = "Mapping: loading";
}

function bindEvents(): void {
  elements.runButton.addEventListener("click", () => {
    renderResults();
    showToast("Demo frames reloaded");
  });
  elements.keyframeTab.addEventListener("click", () => setVisualMode("keyframe"));
  elements.videoTab.addEventListener("click", () => setVisualMode("video"));
  elements.bboxCheckbox.addEventListener("change", () => drawBBoxes(activeFrame));
  elements.closeButton.addEventListener("click", closeInspector);
  elements.modal.addEventListener("click", (event) => {
    if (event.target === elements.modal) closeInspector();
  });
  elements.filmstripPrev.addEventListener("click", () => elements.filmstrip.scrollBy({ left: -320, behavior: "smooth" }));
  elements.filmstripNext.addEventListener("click", () => elements.filmstrip.scrollBy({ left: 320, behavior: "smooth" }));
  elements.submitButton.addEventListener("click", () => {
    elements.submissionInput.value = `${activeFrame.videoId}, ${activeFrame.frameIdx}`;
    elements.submitButton.classList.add("in-submit");
    elements.submitButton.textContent = "✓ In submission";
  });
  elements.copyButton.addEventListener("click", () => {
    void navigator.clipboard.writeText(elements.submissionInput.value);
    showToast("Submission copied");
  });
  elements.clearButton.addEventListener("click", () => {
    elements.submissionInput.value = "No keyframe selected";
    elements.submitButton.classList.remove("in-submit");
    elements.submitButton.textContent = "+ Add to submission";
  });
  document.addEventListener("keydown", (event) => {
    if (elements.modal.classList.contains("hidden")) return;
    if (event.key === "Escape") closeInspector();
    if ((event.key === "ArrowLeft" || event.key === "ArrowRight") && !(event.target instanceof HTMLInputElement)) {
      const activeIndex = filmstripFrames.findIndex((frame) => frame.frameIdx === activeFrame.frameIdx);
      const delta = event.key === "ArrowLeft" ? -1 : 1;
      const nextFrame = filmstripFrames[activeIndex + delta];
      if (nextFrame) selectFrame(nextFrame);
    }
  });
  window.addEventListener("resize", () => drawBBoxes(activeFrame));
}

configureDemoChrome();
bindEvents();
renderResults();
refreshIcons();
