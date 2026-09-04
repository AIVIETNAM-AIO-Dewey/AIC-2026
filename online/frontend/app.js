/**
 * AIC-2026 Multimodal Retrieval Studio — Frontend Logic
 * KIS retrieval workbench with independent branch diagnostics and an optional
 * final cross-branch fusion workspace.
 */

// ──────────────────────────────────────────────────────────────────────────────
// App State
// ──────────────────────────────────────────────────────────────────────────────
import { createIcons, ExternalLink, Maximize, Pause, Play, RotateCcw, RotateCw } from "lucide";
import { createYouTubeVideoView } from "./src/youtube-video-view.ts";
import {
  createSubmissionStore,
  frameIdentity,
  orderedTrakeFrames,
  serializeOfficialSubmissionRows,
  submissionFilename,
  submissionSchemaDefaults,
} from "./src/submission-workbench.js";
import {
  nearestKeyframe,
  secondsToFrameIndex,
} from "./src/time-keyframe-map.js";
import {
  branchPoolCountLabel,
  formatOcrEvidence,
  formatKisFusionEvidence,
  formatKisFusionCardEvidence,
  renderVisibleResultPool,
  resolveWinningOcrText,
} from "./src/result-gates.js";
import {
  assessKisPlanAlignment,
  canonicalKisBundleSignature,
  canonicalKisEventsSignature,
  formatKisOverallQuery,
  formatOrderedKisEvents,
  hasCompleteKisBundle,
  normalizeKisPlanText,
  parseOrderedKisEvents,
} from "./src/kis-query-plan.js";

const state = {
  taskType: "KIS",
  queryMode: "auto", // "auto", "edit", "direct"
  parserEngine: "local",
  activeWorkspace: "kis_fusion",
  capabilities: {},
  submissionContexts: {
    text: "text:empty",
    branch1: "branch1:empty",
    branch2: "branch2:empty",
    branch3_asr: "branch3_asr:empty",
    branch3_ocr: "branch3_ocr:empty",
    kis_fusion: "kis_fusion:empty",
    image: "image:empty",
    video: "video:empty",
  },
  sessionId: null,
  keyframesRoot: "/Users/macbookpro/Downloads/AIC-HCM-BATCH-1/AIC_HCM_BATCH_1/artifacts/keyframes",
  parsedQuery: null,
  modalityResults: {},
  drilldown: null,
  discoveryCascade: null,
  temporalIntersection: null,
  kisTemporalIntersection: null,
  activeModality: "all",
  searchResults: [],
  activeInspectorItem: null,
  activeVideoKeyframes: [],
  filmstripSelection: new Set(),
  filmstripSelectionAnchor: null,
  activeBBoxObjects: [],
  inspectorMediaMode: "keyframe",
  imageQueryFile: null,
  imageQueryObjectUrl: null,
  imageResults: [],
  branch1Ready: false,
  branch1Results: [],
  branch1Response: null,
  branch2Ready: false,
  branch2Results: [],
  branch2Response: null,
  branch3AsrReady: false,
  branch3AsrResults: [],
  branch3AsrResponse: null,
  branch3OcrReady: false,
  branch3OcrResults: [],
  branch3OcrResponse: null,
  kisFusionReady: false,
  kisFusionBundleValid: false,
  kisFusionBusy: false,
  kisFusionView: "results",
  kisFusionResults: [],
  kisFusionResponse: null,
  kisQueryPlan: {
    preparedSource: "",
    generatedBundleSignature: "",
    generatedEventsSignature: "",
    bundleSource: "",
    preparing: false,
    sourceOrigin: "empty",
  },
  inspectorLocalResults: [],
  exportBackfillByContext: {},
  relatedFillStatusByContext: {},
  watch: {
    videoId: "",
    fps: null,
    durationS: null,
    frameIndexBase: 0,
    maxFrameIdx: null,
    currentFrameIdx: null,
    timingMethod: "",
    keyframes: [],
    nearest: null,
    selected: null,
    searchResults: [],
  },
};

const submissionStore = createSubmissionStore();

const SEARCH_POOL_SIZE = 100;
const FILMSTRIP_WINDOW_RADIUS = 12;
let parseAbortController = null;
let searchAbortController = null;
let drilldownAbortController = null;
let cascadeAbortController = null;
let temporalAbortController = null;
let kisTemporalAbortController = null;
let inspectorAbortController = null;
let filmstripAbortController = null;
let parseRequestId = 0;
let searchRequestId = 0;
let drilldownRequestId = 0;
let cascadeRequestId = 0;
let temporalRequestId = 0;
let kisTemporalRequestId = 0;
let inspectorRequestId = 0;
let inspectorOpenRequestId = 0;
let filmstripRequestId = 0;
let resultsRenderId = 0;
let toastTimer = null;
let lastInspectorFocus = null;
let lastExportFocus = null;
let preparedExport = null;
let exportPrepareRequestId = 0;
let csvReviewDirty = false;
let csvReviewGeneration = 0;
let exportReviewAbortController = null;
let imageSearchAbortController = null;
let imageSearchRequestId = 0;
let watchLoadAbortController = null;
let watchLoadRequestId = 0;
let standaloneScopedAbortController = null;
let standaloneScopedRequestId = 0;
let inspectorScopedAbortController = null;
let inspectorScopedRequestId = 0;
let branch1AbortController = null;
let branch2AbortController = null;
let branch3AsrAbortController = null;
let branch3OcrAbortController = null;
let kisFusionAbortController = null;
let kisQueryPlanAbortController = null;
let kisQueryPlanRequestId = 0;
let relatedFillAbortController = null;
let relatedFillRequestId = 0;
const canonicalFrameCache = new Map();
const sourceFrameCache = new Map();

// ──────────────────────────────────────────────────────────────────────────────
// DOM Elements
// ──────────────────────────────────────────────────────────────────────────────
const el = {
  taskBtns: /** @type {NodeListOf<HTMLButtonElement>} */ (document.querySelectorAll(".task-btn")),
  parserEngine: /** @type {HTMLSelectElement} */ (document.getElementById("select-parser-engine")),
  externalFallback: /** @type {HTMLInputElement} */ (document.getElementById("chk-external-fallback")),
  workspaceTabs: /** @type {NodeListOf<HTMLButtonElement>} */ (document.querySelectorAll(".workspace-tab")),
  workspacePanels: /** @type {NodeListOf<HTMLElement>} */ (document.querySelectorAll("[data-workspace-panel]")),
  btnToggleSubmissionRail: document.getElementById("btn-toggle-submission-rail"),
  btnCloseSubmissionRail: document.getElementById("btn-close-submission-rail"),
  submissionRail: document.getElementById("submission-rail"),
  submissionTabCount: document.getElementById("submission-tab-count"),
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
  resultsHeading: document.getElementById("results-heading"),
  resultsGrid: document.getElementById("results-grid"),
  resultsCount: document.getElementById("results-count-badge"),
  selectTopK: /** @type {HTMLSelectElement} */ (document.getElementById("select-top-k")),
  branch1JsonEditor: /** @type {HTMLTextAreaElement} */ (document.getElementById("branch1-json-editor")),
  btnFormatBranch1Json: /** @type {HTMLButtonElement} */ (document.getElementById("btn-format-branch1-json")),
  btnRunBranch1: /** @type {HTMLButtonElement} */ (document.getElementById("btn-run-branch1")),
  branch1HealthBadge: document.getElementById("branch1-health-badge"),
  branch1Validation: document.getElementById("branch1-validation"),
  branch1WeightSiglip: /** @type {HTMLInputElement} */ (document.getElementById("branch1-weight-siglip")),
  branch1WeightMetaclip: /** @type {HTMLInputElement} */ (document.getElementById("branch1-weight-metaclip")),
  branch1WeightBeit: /** @type {HTMLInputElement} */ (document.getElementById("branch1-weight-beit")),
  branch1WeightSiglipValue: document.getElementById("branch1-weight-siglip-value"),
  branch1WeightMetaclipValue: document.getElementById("branch1-weight-metaclip-value"),
  branch1WeightBeitValue: document.getElementById("branch1-weight-beit-value"),
  branch1NormalizedWeights: document.getElementById("branch1-normalized-weights"),
  branch1ResultsCount: document.getElementById("branch1-results-count"),
  branch1Timing: document.getElementById("branch1-timing"),
  branch1ResultsGrid: document.getElementById("branch1-results-grid"),
  branch2JsonEditor: /** @type {HTMLTextAreaElement} */ (document.getElementById("branch2-json-editor")),
  btnFormatBranch2Json: document.getElementById("btn-format-branch2-json"),
  btnRunBranch2: /** @type {HTMLButtonElement} */ (document.getElementById("btn-run-branch2")),
  branch2HealthBadge: document.getElementById("branch2-health-badge"),
  branch2Validation: document.getElementById("branch2-validation"),
  branch2WeightDense: /** @type {HTMLInputElement} */ (document.getElementById("branch2-weight-dense")),
  branch2WeightSparse: /** @type {HTMLInputElement} */ (document.getElementById("branch2-weight-sparse")),
  branch2WeightBeit: /** @type {HTMLInputElement} */ (document.getElementById("branch2-weight-beit")),
  branch2WeightPrevious: /** @type {HTMLInputElement} */ (document.getElementById("branch2-weight-previous")),
  branch2WeightDenseValue: document.getElementById("branch2-weight-dense-value"),
  branch2WeightSparseValue: document.getElementById("branch2-weight-sparse-value"),
  branch2WeightBeitValue: document.getElementById("branch2-weight-beit-value"),
  branch2WeightPreviousValue: document.getElementById("branch2-weight-previous-value"),
  branch2NormalizedWeights: document.getElementById("branch2-normalized-weights"),
  branch2ResultsCount: document.getElementById("branch2-results-count"),
  branch2Timing: document.getElementById("branch2-timing"),
  branch2ResultsGrid: document.getElementById("branch2-results-grid"),
  branch3AsrJsonEditor: /** @type {HTMLTextAreaElement} */ (document.getElementById("branch3-asr-json-editor")),
  btnFormatBranch3AsrJson: document.getElementById("btn-format-branch3-asr-json"),
  btnRunBranch3Asr: /** @type {HTMLButtonElement} */ (document.getElementById("btn-run-branch3-asr")),
  branch3AsrHealthBadge: document.getElementById("branch3-asr-health-badge"),
  branch3AsrValidation: document.getElementById("branch3-asr-validation"),
  branch3AsrResultsCount: document.getElementById("branch3-asr-results-count"),
  branch3AsrTiming: document.getElementById("branch3-asr-timing"),
  branch3AsrResultsGrid: document.getElementById("branch3-asr-results-grid"),
  branch3OcrJsonEditor: /** @type {HTMLTextAreaElement} */ (document.getElementById("branch3-ocr-json-editor")),
  btnFormatBranch3OcrJson: document.getElementById("btn-format-branch3-ocr-json"),
  btnRunBranch3Ocr: /** @type {HTMLButtonElement} */ (document.getElementById("btn-run-branch3-ocr")),
  branch3OcrHealthBadge: document.getElementById("branch3-ocr-health-badge"),
  branch3OcrValidation: document.getElementById("branch3-ocr-validation"),
  branch3OcrResultsCount: document.getElementById("branch3-ocr-results-count"),
  branch3OcrTiming: document.getElementById("branch3-ocr-timing"),
  branch3OcrResultsGrid: document.getElementById("branch3-ocr-results-grid"),
  kisFusionJsonEditor: /** @type {HTMLTextAreaElement} */ (document.getElementById("kis-fusion-json-editor")),
  btnFormatKisFusionJson: document.getElementById("btn-format-kis-fusion-json"),
  btnRunKisFusion: /** @type {HTMLButtonElement} */ (document.getElementById("btn-run-kis-fusion")),
  kisFusionHealthBadge: document.getElementById("kis-fusion-health-badge"),
  kisFusionValidation: document.getElementById("kis-fusion-validation"),
  kisFusionWeightBranch1: /** @type {HTMLInputElement} */ (document.getElementById("kis-fusion-weight-branch1")),
  kisFusionWeightBranch2: /** @type {HTMLInputElement} */ (document.getElementById("kis-fusion-weight-branch2")),
  kisFusionWeightOcr: /** @type {HTMLInputElement} */ (document.getElementById("kis-fusion-weight-ocr")),
  kisFusionWeightAsr: /** @type {HTMLInputElement} */ (document.getElementById("kis-fusion-weight-asr")),
  kisFusionWeightBranch1Value: document.getElementById("kis-fusion-weight-branch1-value"),
  kisFusionWeightBranch2Value: document.getElementById("kis-fusion-weight-branch2-value"),
  kisFusionWeightOcrValue: document.getElementById("kis-fusion-weight-ocr-value"),
  kisFusionWeightAsrValue: document.getElementById("kis-fusion-weight-asr-value"),
  kisFusionNormalizedWeights: document.getElementById("kis-fusion-normalized-weights"),
  kisFusionResultsCount: document.getElementById("kis-fusion-results-count"),
  kisFusionTiming: document.getElementById("kis-fusion-timing"),
  kisFusionResultsGrid: document.getElementById("kis-fusion-results-grid"),
  kisFusionResultsHeading: document.getElementById("kis-fusion-results-heading"),
  kisFusionControlGrid: document.getElementById("kis-fusion-control-grid"),
  kisStickyQuery: document.getElementById("kis-sticky-query"),
  kisTaskBadge: document.getElementById("kis-task-badge"),
  kisPinnedQueryText: /** @type {HTMLTextAreaElement} */ (document.getElementById("kis-pinned-query-text")),
  kisQueryPlanStatus: document.getElementById("kis-query-plan-status"),
  btnPrepareKisQuery: /** @type {HTMLButtonElement} */ (document.getElementById("btn-prepare-kis-query")),
  kisTaskGuidance: document.getElementById("kis-task-guidance"),
  kisTrakeSequencePanel: /** @type {HTMLDetailsElement} */ (document.getElementById("kis-trake-sequence-panel")),
  kisTrakeSequenceEditor: /** @type {HTMLTextAreaElement} */ (document.getElementById("kis-trake-sequence-editor")),
  kisTrakeEventCount: document.getElementById("kis-trake-event-count"),
  kisTrakeGap: /** @type {HTMLSelectElement} */ (document.getElementById("kis-trake-gap")),
  kisTrakeSequences: /** @type {HTMLSelectElement} */ (document.getElementById("kis-trake-sequences")),
  btnRunKisTrake: /** @type {HTMLButtonElement} */ (document.getElementById("btn-run-kis-trake")),
  btnDiscoveryCascade: /** @type {HTMLButtonElement} */ (document.getElementById("btn-discovery-cascade")),
  btnTemporalIntersection: /** @type {HTMLButtonElement} */ (document.getElementById("btn-temporal-intersection")),
  selectTemporalGap: /** @type {HTMLSelectElement} */ (document.getElementById("select-temporal-gap")),
  selectTemporalCandidates: /** @type {HTMLSelectElement} */ (document.getElementById("select-temporal-candidates")),
  selectTemporalSequences: /** @type {HTMLSelectElement} */ (document.getElementById("select-temporal-sequences")),
  selectTemporalPathsPerVideo: /** @type {HTMLSelectElement} */ (document.getElementById("select-temporal-paths-per-video")),
  selectTemporalReservoir: /** @type {HTMLSelectElement} */ (document.getElementById("select-temporal-reservoir")),
  
  modalityTabs: /** @type {NodeListOf<HTMLButtonElement>} */ (document.querySelectorAll(".modality-tab")),
  modalityQuerySummary: document.getElementById("modality-query-summary"),
  temporalEventsEditor: /** @type {HTMLTextAreaElement} */ (document.getElementById("temporal-events-editor")),
  temporalEventCount: document.getElementById("temporal-event-count"),
  
  // Submission & Export
  submissionInput: /** @type {HTMLInputElement} */ (document.getElementById("submission-input")),
  submissionList: document.getElementById("submission-list"),
  submissionCount: document.getElementById("submission-count"),
  submissionSchemaNote: document.getElementById("submission-schema-note"),
  submissionQueryId: /** @type {HTMLInputElement} */ (document.getElementById("submission-query-id")),
  vqaAnswerPanel: document.getElementById("vqa-answer-panel"),
  vqaAnswerInput: /** @type {HTMLTextAreaElement} */ (document.getElementById("vqa-answer-input")),
  trakeEventPanel: document.getElementById("trake-event-panel"),
  trakeEventTabs: document.getElementById("trake-event-tabs"),
  btnCopySubmission: document.getElementById("btn-copy-submission"),
  btnExportCsv: document.getElementById("btn-export-csv"),
  btnClearSubmission: document.getElementById("btn-clear-submission"),
  submissionRelatedNote: document.getElementById("submission-related-note"),
  toast: document.getElementById("toast"),

  // 100-Row Export Modal
  exportModal: document.getElementById("export-modal"),
  btnCloseExportModal: document.getElementById("btn-close-export-modal"),
  btnCancelExportModal: document.getElementById("btn-cancel-export-modal"),
  btnDownloadCsvAction: /** @type {HTMLButtonElement} */ (document.getElementById("btn-download-csv-action")),
  exportQueryId: /** @type {HTMLInputElement} */ (document.getElementById("export-query-id")),
  exportRow1Preview: document.getElementById("export-row1-preview"),
  exportCsvPreview: /** @type {HTMLTextAreaElement} */ (document.getElementById("export-csv-preview")),
  exportSchemaWarning: document.getElementById("export-schema-warning"),
  btnRevalidateCsv: /** @type {HTMLButtonElement} */ (document.getElementById("btn-revalidate-csv")),

  // Image Search Workspace
  imageDropZone: document.getElementById("image-drop-zone"),
  imageQueryFile: /** @type {HTMLInputElement} */ (document.getElementById("image-query-file")),
  imageQueryPreview: /** @type {HTMLImageElement} */ (document.getElementById("image-query-preview")),
  imageDropPrompt: document.getElementById("image-drop-prompt"),
  btnChooseImage: document.getElementById("btn-choose-image"),
  btnRunImageSearch: /** @type {HTMLButtonElement} */ (document.getElementById("btn-run-image-search")),
  btnClearImage: document.getElementById("btn-clear-image"),
  selectImageTopK: /** @type {HTMLSelectElement} */ (document.getElementById("select-image-top-k")),
  imageResultsCount: document.getElementById("image-results-count"),
  imageResultsGrid: document.getElementById("image-results-grid"),

  // Standalone Video Workspace
  videoIdForm: /** @type {HTMLFormElement} */ (document.getElementById("video-id-form")),
  videoIdInput: /** @type {HTMLInputElement} */ (document.getElementById("video-id-input")),
  standaloneVideoArea: document.getElementById("standalone-video-area"),
  standaloneVideoSource: document.getElementById("standalone-video-source"),
  watchPlaybackTime: document.getElementById("watch-playback-time"),
  watchEstimatedFrame: document.getElementById("watch-estimated-frame"),
  watchKeyframeN: document.getElementById("watch-keyframe-n"),
  watchFrameIdx: document.getElementById("watch-frame-idx"),
  watchKeyframeTime: document.getElementById("watch-keyframe-time"),
  watchMappingDelta: document.getElementById("watch-mapping-delta"),
  btnWatchPrevKeyframe: /** @type {HTMLButtonElement} */ (document.getElementById("btn-watch-prev-keyframe")),
  btnWatchNextKeyframe: /** @type {HTMLButtonElement} */ (document.getElementById("btn-watch-next-keyframe")),
  btnSubmitWatchFrame: /** @type {HTMLButtonElement} */ (document.getElementById("btn-submit-watch-frame")),
  watchExactFrameInput: /** @type {HTMLInputElement} */ (document.getElementById("watch-exact-frame-input")),
  btnSelectWatchFrame: /** @type {HTMLButtonElement} */ (document.getElementById("btn-select-watch-frame")),
  watchSelectedFrameStatus: document.getElementById("watch-selected-frame-status"),
  standaloneVideoQuery: /** @type {HTMLInputElement} */ (document.getElementById("standalone-video-query")),
  btnStandaloneVideoSearch: /** @type {HTMLButtonElement} */ (document.getElementById("btn-standalone-video-search")),
  standaloneVideoSearchResults: document.getElementById("standalone-video-search-results"),
  
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
  inspAsrEvidence: document.getElementById("insp-asr-evidence"),
  inspAsrContext: document.getElementById("insp-asr-context"),
  inspDamText: document.getElementById("insp-dam-text"),
  inspOcrText: document.getElementById("insp-ocr-text"),
  inspObjectsList: document.getElementById("insp-objects-list"),
  
  btnToggleInSubmission: document.getElementById("btn-toggle-in-submission"),
  btnCloseInspector: document.getElementById("btn-close-inspector"),
  inspectorVideoQuery: /** @type {HTMLInputElement} */ (document.getElementById("inspector-video-query")),
  inspectorParserEngine: /** @type {HTMLSelectElement} */ (document.getElementById("inspector-parser-engine")),
  btnInspectorVideoSearch: /** @type {HTMLButtonElement} */ (document.getElementById("btn-inspector-video-search")),
  inspectorLocalResults: document.getElementById("inspector-local-results"),
  
  filmstripScroll: document.getElementById("filmstrip-scroll"),
  filmstripCount: document.getElementById("filmstrip-count"),
  btnFilmstripPrev: document.getElementById("btn-filmstrip-prev"),
  btnFilmstripNext: document.getElementById("btn-filmstrip-next"),
  btnClearFilmstripSelection: /** @type {HTMLButtonElement} */ (document.getElementById("btn-clear-filmstrip-selection")),
  btnAddFilmstripSelection: /** @type {HTMLButtonElement} */ (document.getElementById("btn-add-filmstrip-selection")),
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

const standaloneVideoController = createYouTubeVideoView({
  elementPrefix: "standalone",
  onSourceChange: (label) => {
    el.standaloneVideoSource.textContent = label;
  },
  onStatusChange: (label, status = "pending") => {
    el.standaloneVideoSource.textContent = label;
    el.standaloneVideoSource.classList.toggle("ready", status === "ready");
    el.standaloneVideoSource.classList.toggle("error", status === "error");
  },
  onTimeChange: updateWatchMapping,
  onToast: showToast,
  refreshIcons: refreshVideoIcons,
});

async function initApp() {
  bindEvents();
  const initialSubmission = submissionStore.getSnapshot();
  state.taskType = initialSubmission.mode;
  submissionStore.subscribe(renderSubmissionRail);
  renderSubmissionRail(initialSubmission);
  setWorkspace("kis_fusion");
  updateTaskWorkspace(initialSubmission.mode);
  if (window.matchMedia?.("(max-width: 1100px)").matches) {
    setSubmissionRailCollapsed(true);
  }
  setInspectorMediaMode("keyframe");
  refreshVideoIcons();
  await Promise.all([
    loadServerConfig(),
    loadBranch1Health(),
    loadBranch2Health(),
    loadBranch3AsrHealth(),
    loadBranch3OcrHealth(),
    loadKisFusionHealth(),
  ]);
  // Run this last so the overall status wins over the temporary
  // "Capabilities loaded" state regardless of request completion order.
  await loadOverallHealth();
  updateQueryModeUI();
  if (typeof ResizeObserver !== "undefined") {
    const resizeObserver = new ResizeObserver(() => {
      if (!el.modal.classList.contains("hidden")) drawBBoxesOnCanvas();
    });
    resizeObserver.observe(el.keyframeView);
  }
}

async function loadOverallHealth() {
  try {
    const response = await fetch("/api/health?compact=true");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const health = await response.json();
    if (health.status === "ready") {
      const accelerator = health.device === "mps" ? "MPS" : "CPU";
      setServerStatus(`Local ${accelerator} / Qdrant ready`, "ready");
    } else if (health.status === "starting") {
      setServerStatus("Server starting", "pending");
    } else {
      const components = health.components || {};
      const requiredNames = ["api_process", "qdrant", "metadata", "branch1", "branch2"];
      const requiredUnavailable = requiredNames.some(
        (name) => components[name]?.ready !== true,
      );
      setServerStatus(
        requiredUnavailable
          ? "Server degraded · required dependencies unavailable"
          : "Server degraded · optional indexes unavailable",
        "pending",
      );
    }
  } catch (error) {
    setServerStatus("Server health unavailable", "error");
    console.warn("Could not load /api/health:", error);
  }
}

async function loadServerConfig() {
  try {
    const res = await fetch("/api/config");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const cfg = await res.json();
    if (cfg.keyframes_root) state.keyframesRoot = cfg.keyframes_root;
    state.capabilities = cfg.capabilities || {};
    const asrTab = /** @type {HTMLButtonElement | null} */ (document.querySelector('[data-modality="asr"]'));
    if (asrTab && state.capabilities.asr === false) {
      asrTab.disabled = true;
      asrTab.title = "ASR artifacts are not installed in the local dataset";
    }
    const asrWorkspaceTab = /** @type {HTMLButtonElement | null} */ (document.querySelector('[data-workspace="branch3_asr"]'));
    if (asrWorkspaceTab && state.capabilities.branch3_asr === false) {
      asrWorkspaceTab.disabled = true;
      asrWorkspaceTab.title = "ASR artifacts are not ready in the local dataset";
    }
    const ocrWorkspaceTab = /** @type {HTMLButtonElement | null} */ (document.querySelector('[data-workspace="branch3_ocr"]'));
    if (ocrWorkspaceTab && state.capabilities.branch3_ocr === false) {
      ocrWorkspaceTab.disabled = true;
      ocrWorkspaceTab.title = "OCR artifacts are not ready in the local dataset";
    }
    const fusionWorkspaceTab = /** @type {HTMLButtonElement | null} */ (document.querySelector('[data-workspace="kis_fusion"]'));
    if (fusionWorkspaceTab && state.capabilities.kis_fusion === false) {
      fusionWorkspaceTab.title = "KIS fusion requires all four branch pools to be ready";
    }
    // A successful config response only means the API answered.  Readiness is
    // reported by /api/health and the per-branch health endpoints below.
    setServerStatus("Capabilities loaded", "pending");
  } catch (err) {
    console.warn("Could not load /api/config:", err);
    setServerStatus("Server configuration unavailable", "error");
  }
}

const BRANCH1_ROLES = ["original", "entity", "action", "context", "synonym", "keyword"];

async function loadBranch1Health() {
  try {
    const response = await fetch("/api/branch1/health?compact=true");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const health = await response.json();
    state.branch1Ready = health.ready === true;
    el.btnRunBranch1.disabled = !state.branch1Ready;
    el.branch1HealthBadge.textContent = state.branch1Ready ? "ALL 3 MODELS READY" : "FAIL-CLOSED · NOT READY";
    el.branch1HealthBadge.title = state.branch1Ready
      ? "Data gate, three Qdrant vectors, and three text encoders are ready"
      : "Open /api/branch1/health to inspect the failed gate";
    validateBranch1Editor();
  } catch (error) {
    state.branch1Ready = false;
    el.btnRunBranch1.disabled = true;
    el.branch1HealthBadge.textContent = "HEALTH UNAVAILABLE";
    console.warn("Could not load Branch-1 health:", error);
  }
}

async function loadBranch2Health() {
  try {
    const response = await fetch("/api/branch2/health?compact=true");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const health = await response.json();
    state.branch2Ready = health.ready === true;
    el.btnRunBranch2.disabled = !state.branch2Ready;
    el.branch2HealthBadge.textContent = state.branch2Ready ? "DAM + BM25 + BEiT-3 COCO READY" : "FAIL-CLOSED - NOT READY";
    el.branch2HealthBadge.title = state.branch2Ready ? "DAM collection, local BM25, BGE-M3, and BEiT-3 COCO cosine are ready" : "Open /api/branch2/health to inspect the failed gate";
    validateBranch2Editor();
  } catch (error) {
    state.branch2Ready = false;
    el.btnRunBranch2.disabled = true;
    el.branch2HealthBadge.textContent = "HEALTH UNAVAILABLE";
    console.warn("Could not load Branch-2 health:", error);
  }
}

async function loadBranch3AsrHealth() {
  try {
    const response = await fetch("/api/branch3/asr/health?compact=true");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const health = await response.json();
    state.branch3AsrReady = health.ready === true;
    el.btnRunBranch3Asr.disabled = !state.branch3AsrReady;
    el.branch3AsrHealthBadge.textContent = state.branch3AsrReady
      ? "ASR INDEX READY"
      : "FAIL-CLOSED · ASR NOT READY";
    el.branch3AsrHealthBadge.title = state.branch3AsrReady
      ? "55,168 ASR segments mapped to canonical frames"
      : "Prepare the ASR SQLite index and inspect /api/branch3/asr/health";
    validateBranch3AsrEditor();
  } catch (error) {
    state.branch3AsrReady = false;
    el.btnRunBranch3Asr.disabled = true;
    el.branch3AsrHealthBadge.textContent = "HEALTH UNAVAILABLE";
    console.warn("Could not load Branch-3 ASR health:", error);
  }
}

function parseBranch3AsrBundle() {
  let value;
  try { value = JSON.parse(el.branch3AsrJsonEditor.value); } catch (error) { throw new Error(`Invalid JSON: ${error.message}`); }
  if (value?.schema_version !== "branch1.query.v1") throw new Error("schema_version must be branch1.query.v1");
  if (!Array.isArray(value.queries) || value.queries.length !== 6) throw new Error("queries must contain exactly six items");
  const roles = value.queries.map((query) => query?.role);
  if (new Set(roles).size !== 6 || !BRANCH1_ROLES.every((role) => roles.includes(role))) throw new Error(`Required roles: ${BRANCH1_ROLES.join(", ")}`);
  value.queries.forEach((query) => {
    if (typeof query.vi !== "string" || !query.vi.trim()) throw new Error(`${query.role}.vi is required`);
    if (typeof query.en !== "string" || !query.en.trim()) throw new Error(`${query.role}.en is required`);
  });
  return value;
}

function validateBranch3AsrEditor() {
  try {
    parseBranch3AsrBundle();
    el.branch3AsrValidation.textContent = state.branch3AsrReady
      ? "Valid six-role bilingual bundle. Ready to search ASR."
      : "JSON is valid, but the ASR index or canonical mapping is not ready.";
    el.branch3AsrValidation.classList.toggle("ready", state.branch3AsrReady);
    el.branch3AsrValidation.classList.toggle("error", !state.branch3AsrReady);
    el.btnRunBranch3Asr.disabled = !state.branch3AsrReady;
    return true;
  } catch (error) {
    el.branch3AsrValidation.textContent = error.message;
    el.branch3AsrValidation.classList.remove("ready");
    el.branch3AsrValidation.classList.add("error");
    el.btnRunBranch3Asr.disabled = true;
    return false;
  }
}

async function runBranch3AsrSearch() {
  if (!validateBranch3AsrEditor()) return;
  const queryBundle = parseBranch3AsrBundle();
  branch3AsrAbortController?.abort();
  branch3AsrAbortController = new AbortController();
  el.btnRunBranch3Asr.disabled = true;
  el.btnRunBranch3Asr.textContent = "Running ASR...";
  const started = performance.now();
  try {
    const response = await fetch("/api/search/branch3/asr", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: branch3AsrAbortController.signal,
      body: JSON.stringify({ query_bundle: queryBundle, per_stream_top_k: 2000, final_top_k: 500 }),
    });
    if (!response.ok) throw await responseError(response, "Branch-3 ASR search failed");
    const payload = await response.json();
    // Keep the complete API pool and diagnostics in state. Rendering is
    // intentionally limited to the first 150 cards, while future fusion and
    // audit views must still be able to consume the full response.
    state.branch3AsrResponse = payload;
    state.branch3AsrResults = (payload.results || []).map((item) => ({
      ...item,
      score: item.asr_normalized_score ?? item.score ?? 0,
      score_type: "asr_bm25_ngram",
      retrieval_modality: "asr",
    }));
    setSubmissionContext("branch3_asr", JSON.stringify(queryBundle));
    const returnedCount = Number(payload.result_count || state.branch3AsrResults.length || 0);
    el.branch3AsrResultsCount.textContent = branchPoolCountLabel(payload, returnedCount, 500);
    el.branch3AsrTiming.textContent = `${Number(payload.timing?.total_ms || (performance.now() - started)).toFixed(0)} ms · ${Object.values(payload.stream_counts || {}).join(" / ")} hits`;
    el.branch3AsrValidation.textContent = "ASR ranking complete: BM25 55% · token 30% · adjacent n-gram 15%.";
    el.branch3AsrValidation.classList.remove("error");
    el.branch3AsrValidation.classList.add("ready");
    el.branch3AsrResultsGrid.replaceChildren();
    renderVisibleResultPool(
      el.branch3AsrResultsGrid,
      state.branch3AsrResults,
      (visible, container) => renderStandardCards(visible, container, "asr", ++resultsRenderId),
    );
  } catch (error) {
    if (error.name === "AbortError") return;
    el.branch3AsrResultsCount.textContent = "Search failed";
    el.branch3AsrTiming.textContent = `${Math.round(performance.now() - started)} ms`;
    el.branch3AsrResultsGrid.innerHTML = `<div class="empty-placeholder"><div class="empty-title">Branch-3 ASR failed</div><div class="empty-desc">${escapeHtml(error.message)}</div></div>`;
    showToast(error.message, "error");
  } finally {
    el.btnRunBranch3Asr.textContent = "Run ASR";
    if (state.branch3AsrReady) el.btnRunBranch3Asr.disabled = false;
  }
}

function parseBranch3OcrBundle() {
  let value;
  try { value = JSON.parse(el.branch3OcrJsonEditor.value); } catch (error) { throw new Error(`Invalid JSON: ${error.message}`); }
  if (value?.schema_version !== "branch1.query.v1") throw new Error("schema_version must be branch1.query.v1");
  if (!Array.isArray(value.queries) || value.queries.length !== 6) throw new Error("queries must contain exactly six items");
  const roles = value.queries.map((query) => query?.role);
  if (new Set(roles).size !== 6 || !BRANCH1_ROLES.every((role) => roles.includes(role))) throw new Error(`Required roles: ${BRANCH1_ROLES.join(", ")}`);
  value.queries.forEach((query) => {
    if (typeof query.vi !== "string" || !query.vi.trim()) throw new Error(`${query.role}.vi is required`);
    if (typeof query.en !== "string" || !query.en.trim()) throw new Error(`${query.role}.en is required`);
  });
  return value;
}

function validateBranch3OcrEditor() {
  try {
    parseBranch3OcrBundle();
    el.branch3OcrValidation.textContent = state.branch3OcrReady
      ? "Valid six-role bilingual bundle. Ready to search OCR."
      : "JSON is valid, but the OCR index is not ready.";
    el.branch3OcrValidation.classList.toggle("ready", state.branch3OcrReady);
    el.branch3OcrValidation.classList.toggle("error", !state.branch3OcrReady);
    el.btnRunBranch3Ocr.disabled = !state.branch3OcrReady;
    return true;
  } catch (error) {
    el.branch3OcrValidation.textContent = error.message;
    el.branch3OcrValidation.classList.remove("ready");
    el.branch3OcrValidation.classList.add("error");
    el.btnRunBranch3Ocr.disabled = true;
    return false;
  }
}

async function loadBranch3OcrHealth() {
  try {
    const response = await fetch("/api/branch3/ocr/health?compact=true");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const health = await response.json();
    state.branch3OcrReady = health.ready === true;
    el.btnRunBranch3Ocr.disabled = !state.branch3OcrReady;
    el.branch3OcrHealthBadge.textContent = state.branch3OcrReady
      ? "OCR INDEX READY"
      : "FAIL-CLOSED · OCR NOT READY";
    el.branch3OcrHealthBadge.title = state.branch3OcrReady
      ? "OCR frame index is ready for bilingual retrieval"
      : "Prepare the OCR SQLite index and inspect /api/branch3/ocr/health";
    validateBranch3OcrEditor();
  } catch (error) {
    state.branch3OcrReady = false;
    el.btnRunBranch3Ocr.disabled = true;
    el.branch3OcrHealthBadge.textContent = "HEALTH UNAVAILABLE";
    console.warn("Could not load Branch-3 OCR health:", error);
  }
}

async function loadKisFusionHealth() {
  if (!el.kisFusionHealthBadge) return;
  try {
    const response = await fetch("/api/fusion/kis/health?compact=true");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const health = await response.json();
    state.kisFusionReady = health.ready === true;
    el.btnRunKisFusion.disabled = !state.kisFusionReady || state.kisFusionBusy;
    el.kisFusionHealthBadge.textContent = state.kisFusionReady
      ? "ALL POOLS READY"
      : "FAIL-CLOSED · FUSION NOT READY";
    el.kisFusionHealthBadge.title = state.kisFusionReady
      ? "Branch 1, Branch 2, ASR and OCR are ready for weighted RRF"
      : "All four branch health gates must be ready before KIS fusion can run";
    validateKisFusionEditor();
  } catch (error) {
    state.kisFusionReady = false;
    el.btnRunKisFusion.disabled = true;
    el.kisFusionHealthBadge.textContent = "HEALTH UNAVAILABLE";
    console.warn("Could not load KIS fusion health:", error);
  }
}

function parseKisFusionBundle() {
  let value;
  try { value = JSON.parse(el.kisFusionJsonEditor.value); } catch (error) { throw new Error(`Invalid JSON: ${error.message}`); }
  if (value?.schema_version !== "branch1.query.v1") throw new Error("schema_version must be branch1.query.v1");
  if (!Array.isArray(value.queries) || value.queries.length !== 6) throw new Error("queries must contain exactly six items");
  const roles = value.queries.map((query) => query?.role);
  if (new Set(roles).size !== 6 || !BRANCH1_ROLES.every((role) => roles.includes(role))) throw new Error(`Required roles: ${BRANCH1_ROLES.join(", ")}`);
  value.queries.forEach((query) => {
    if (typeof query.vi !== "string" || !query.vi.trim()) throw new Error(`${query.role}.vi is required`);
    if (typeof query.en !== "string" || !query.en.trim()) throw new Error(`${query.role}.en is required`);
  });
  return value;
}

function validateKisFusionEditor() {
  try {
    parseKisFusionBundle();
    state.kisFusionBundleValid = true;
    el.kisFusionValidation.textContent = state.kisFusionReady
      ? "Valid six-role bilingual bundle. Ready for KIS fusion."
      : "JSON is valid, but at least one branch fusion dependency is not ready.";
    el.kisFusionValidation.classList.toggle("ready", state.kisFusionReady);
    el.kisFusionValidation.classList.toggle("error", !state.kisFusionReady);
    el.btnRunKisFusion.disabled = !state.kisFusionReady || state.kisFusionBusy;
    updateKisOrderedButtonState();
    return true;
  } catch (error) {
    el.kisFusionValidation.textContent = error.message;
    el.kisFusionValidation.classList.remove("ready");
    el.kisFusionValidation.classList.add("error");
    el.btnRunKisFusion.disabled = true;
    state.kisFusionBundleValid = false;
    updateKisOrderedButtonState();
    return false;
  }
}

function updateKisOrderedButtonState() {
  const eventCount = kisSequenceEvents().length;
  el.kisTrakeEventCount.textContent = `${eventCount} event${eventCount === 1 ? "" : "s"}`;
  el.btnRunKisTrake.disabled = !state.kisFusionReady
    || !state.kisFusionBundleValid
    || state.kisFusionBusy
    || eventCount < 2;
}

function updateKisFusionWeights() {
  const inputs = [
    el.kisFusionWeightBranch1,
    el.kisFusionWeightBranch2,
    el.kisFusionWeightOcr,
    el.kisFusionWeightAsr,
  ];
  const values = inputs.map((input) => Number(input.value));
  [
    el.kisFusionWeightBranch1Value,
    el.kisFusionWeightBranch2Value,
    el.kisFusionWeightOcrValue,
    el.kisFusionWeightAsrValue,
  ].forEach((node, index) => { node.textContent = values[index].toFixed(2); });
  const total = values.reduce((sum, value) => sum + value, 0);
  const normalized = total > 0 ? values.map((value) => Math.round(value / total * 100)) : [0, 0, 0, 0];
  el.kisFusionNormalizedWeights.textContent = `Normalized: ${normalized[0]}% / ${normalized[1]}% / ${normalized[2]}% / ${normalized[3]}%`;
}

// Card and inspector both delegate to the canonical pure formatter. The
// card intentionally retains only its leading summary segments.
function fusionEvidence(item) {
  return formatKisFusionCardEvidence(item);
}

function fusionInspectorEvidence(item) {
  return formatKisFusionEvidence(item);
}

async function runKisFusionSearch() {
  if (!validateKisFusionEditor()) return;
  const queryBundle = parseKisFusionBundle();
  kisFusionAbortController?.abort();
  kisFusionAbortController = new AbortController();
  state.kisFusionBusy = true;
  state.kisFusionView = "results";
  state.kisTemporalIntersection = null;
  kisTemporalAbortController?.abort();
  kisTemporalAbortController = null;
  kisTemporalRequestId += 1;
  el.kisFusionResultsHeading.textContent = "KIS FINAL RESULTS";
  el.btnRunKisFusion.disabled = true;
  el.btnRunKisTrake.disabled = true;
  el.btnRunKisFusion.textContent = "Running four pools\u2026";
  const started = performance.now();
  try {
    const response = await fetch("/api/search/fusion/kis", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: kisFusionAbortController.signal,
      body: JSON.stringify({
        query_bundle: queryBundle,
        branch_weights: {
          branch1: Number(el.kisFusionWeightBranch1.value),
          branch2: Number(el.kisFusionWeightBranch2.value),
          ocr: Number(el.kisFusionWeightOcr.value),
          asr: Number(el.kisFusionWeightAsr.value),
        },
      }),
    });
    if (!response.ok) throw await responseError(response, "KIS fusion search failed");
    const payload = await response.json();
    state.kisFusionResponse = payload;
    // Keep the complete final top-150 response in state.  Unlike standalone
    // branch workspaces, the final KIS pool is already the display gate.
    state.kisFusionResults = (payload.results || []).map((item) => ({
      ...item,
      score: item.final_score ?? item.score ?? item.rrf_score ?? 0,
      score_type: item.score_type || "weighted_rrf",
      retrieval_modality: "kis_fusion",
      dam_summary: fusionEvidence(item),
    }));
    setSubmissionContext("kis_fusion", JSON.stringify(queryBundle));
    const returnedCount = Number(payload.result_count ?? state.kisFusionResults.length ?? 0);
    el.kisFusionResultsCount.textContent = branchPoolCountLabel(payload, returnedCount, 150);
    el.kisFusionTiming.textContent = `${Number(payload.timing?.total_ms || (performance.now() - started)).toFixed(0)} ms \u00b7 RRF k=${payload.rrf_k} \u00b7 BEiT top ${payload.rerank_top_k}`;
    el.kisFusionValidation.textContent = "KIS fusion complete: weighted RRF 40/30/15/15 \u00b7 BEiT-3 COCO cosine 25% / RRF 75%.";
    el.kisFusionValidation.classList.remove("error");
    el.kisFusionValidation.classList.add("ready");
    el.kisFusionResultsGrid.replaceChildren();
    renderVisibleResultPool(
      el.kisFusionResultsGrid,
      state.kisFusionResults,
      (visible, container) => renderStandardCards(visible, container, "kis_fusion", ++resultsRenderId),
      150,
    );
  } catch (error) {
    if (error.name === "AbortError") return;
    el.kisFusionResultsCount.textContent = "Search failed";
    el.kisFusionTiming.textContent = `${Math.round(performance.now() - started)} ms`;
    el.kisFusionResultsGrid.innerHTML = `<div class="empty-placeholder"><div class="empty-title">KIS fusion failed</div><div class="empty-desc">${escapeHtml(error.message)}</div></div>`;
    showToast(error.message, "error");
  } finally {
    state.kisFusionBusy = false;
    el.btnRunKisFusion.textContent = "Run KIS fusion";
    el.btnRunKisFusion.disabled = !state.kisFusionReady || !state.kisFusionBundleValid;
    updateKisOrderedButtonState();
  }
}

async function runBranch3OcrSearch() {
  if (!validateBranch3OcrEditor()) return;
  const queryBundle = parseBranch3OcrBundle();
  branch3OcrAbortController?.abort();
  branch3OcrAbortController = new AbortController();
  el.btnRunBranch3Ocr.disabled = true;
  el.btnRunBranch3Ocr.textContent = "Running OCR...";
  const started = performance.now();
  try {
    const response = await fetch("/api/search/branch3/ocr", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: branch3OcrAbortController.signal,
      body: JSON.stringify({ query_bundle: queryBundle, per_stream_top_k: 2000, final_top_k: 500 }),
    });
    if (!response.ok) throw await responseError(response, "Branch-3 OCR search failed");
    const payload = await response.json();
    state.branch3OcrResponse = payload;
    state.branch3OcrResults = (payload.results || []).map((item) => ({
      ...item,
      score: item.ocr_normalized_score ?? item.score ?? 0,
      score_type: "ocr_bm25_ngram",
      retrieval_modality: "ocr",
    }));
    setSubmissionContext("branch3_ocr", JSON.stringify(queryBundle));
    const returnedCount = Number(payload.result_count || state.branch3OcrResults.length || 0);
    el.branch3OcrResultsCount.textContent = branchPoolCountLabel(payload, returnedCount, 500);
    el.branch3OcrTiming.textContent = `${Number(payload.timing?.total_ms || (performance.now() - started)).toFixed(0)} ms · ${Object.values(payload.stream_counts || {}).join(" / ")} hits`;
    el.branch3OcrValidation.textContent = "OCR ranking complete: BM25 55% · token 30% · adjacent n-gram 15%.";
    el.branch3OcrValidation.classList.remove("error");
    el.branch3OcrValidation.classList.add("ready");
    el.branch3OcrResultsGrid.replaceChildren();
    renderVisibleResultPool(
      el.branch3OcrResultsGrid,
      state.branch3OcrResults,
      (visible, container) => renderStandardCards(visible, container, "ocr", ++resultsRenderId),
    );
  } catch (error) {
    if (error.name === "AbortError") return;
    el.branch3OcrResultsCount.textContent = "Search failed";
    el.branch3OcrTiming.textContent = `${Math.round(performance.now() - started)} ms`;
    el.branch3OcrResultsGrid.innerHTML = `<div class="empty-placeholder"><div class="empty-title">Branch-3 OCR failed</div><div class="empty-desc">${escapeHtml(error.message)}</div></div>`;
    showToast(error.message, "error");
  } finally {
    el.btnRunBranch3Ocr.textContent = "Run OCR";
    if (state.branch3OcrReady) el.btnRunBranch3Ocr.disabled = false;
  }
}

function parseBranch2Bundle() {
  let value;
  try { value = JSON.parse(el.branch2JsonEditor.value); } catch (error) { throw new Error(`Invalid JSON: ${error.message}`); }
  if (value?.schema_version !== "branch1.query.v1") throw new Error("schema_version must be branch1.query.v1");
  if (!Array.isArray(value.queries) || value.queries.length !== 6) throw new Error("queries must contain exactly six items");
  const roles = value.queries.map((query) => query?.role);
  if (new Set(roles).size !== 6 || !BRANCH1_ROLES.every((role) => roles.includes(role))) throw new Error(`Required roles: ${BRANCH1_ROLES.join(", ")}`);
  value.queries.forEach((query) => {
    if (typeof query.vi !== "string" || !query.vi.trim()) throw new Error(`${query.role}.vi is required`);
    if (typeof query.en !== "string" || !query.en.trim()) throw new Error(`${query.role}.en is required`);
  });
  return value;
}

function validateBranch2Editor() {
  try {
    parseBranch2Bundle();
    el.branch2Validation.textContent = state.branch2Ready ? "Valid six-role bilingual bundle. Ready to search." : "JSON is valid, but a server-side data or encoder gate is not ready.";
    el.branch2Validation.classList.toggle("ready", state.branch2Ready);
    el.branch2Validation.classList.toggle("error", !state.branch2Ready);
    el.branch2Validation.classList.remove("warning");
    el.btnRunBranch2.disabled = !state.branch2Ready;
    return true;
  } catch (error) {
    el.branch2Validation.textContent = error.message;
    el.branch2Validation.classList.remove("ready");
    el.branch2Validation.classList.add("error");
    el.branch2Validation.classList.remove("warning");
    el.btnRunBranch2.disabled = true;
    return false;
  }
}

function updateBranch2Weights() {
  const values = [Number(el.branch2WeightDense.value), Number(el.branch2WeightSparse.value), Number(el.branch2WeightBeit.value), Number(el.branch2WeightPrevious.value)];
  [el.branch2WeightDenseValue, el.branch2WeightSparseValue, el.branch2WeightBeitValue, el.branch2WeightPreviousValue].forEach((node, index) => { node.textContent = values[index].toFixed(2); });
  const hybridTotal = values[0] + values[1];
  const rerankTotal = values[2] + values[3];
  el.branch2NormalizedWeights.textContent = `Hybrid ${hybridTotal ? Math.round(values[0] / hybridTotal * 100) : 0}% / ${hybridTotal ? Math.round(values[1] / hybridTotal * 100) : 0}% - Rerank ${rerankTotal ? Math.round(values[2] / rerankTotal * 100) : 0}% / ${rerankTotal ? Math.round(values[3] / rerankTotal * 100) : 0}%`;
  if (hybridTotal <= 0 || rerankTotal <= 0) {
    el.branch2Validation.textContent = "Both weight groups need a positive sum.";
    el.branch2Validation.classList.remove("ready", "warning");
    el.branch2Validation.classList.add("error");
    el.btnRunBranch2.disabled = true;
  } else validateBranch2Editor();
}

async function runBranch2Search() {
  if (!validateBranch2Editor()) return;
  const queryBundle = parseBranch2Bundle();
  branch2AbortController?.abort();
  branch2AbortController = new AbortController();
  el.btnRunBranch2.disabled = true;
  el.btnRunBranch2.textContent = "Running DAM + BM25...";
  const started = performance.now();
  try {
    const response = await fetch("/api/search/branch2", { method: "POST", headers: { "Content-Type": "application/json" }, signal: branch2AbortController.signal, body: JSON.stringify({ query_bundle: queryBundle, hybrid_weights: { dense: Number(el.branch2WeightDense.value), sparse: Number(el.branch2WeightSparse.value) }, rerank_weights: { beit3: Number(el.branch2WeightBeit.value), previous: Number(el.branch2WeightPrevious.value) }, per_stream_top_k: 2000, pre_rerank_top_k: 500, rerank_top_k: 100 }) });
    if (!response.ok) throw await responseError(response, "Branch-2 search failed");
    const payload = await response.json();
    state.branch2Response = payload;
    state.branch2Results = payload.results.map((item) => ({ ...item, score: item.reranked_score ?? item.hybrid_score ?? item.score, score_type: item.rerank_score_type || "dam_dense_bm25_hybrid", retrieval_modality: "branch2", dam_summary: item.dam_winner?.description_en || item.sparse_winner?.description_en || "" }));
    setSubmissionContext("branch2", JSON.stringify(queryBundle));
    const returnedCount = Number(payload.result_count || state.branch2Results.length || 0);
    el.branch2ResultsCount.textContent = `${branchPoolCountLabel(payload, returnedCount, 500)} · top ${payload.rerank_top_k} reranked`;
    el.branch2Timing.textContent = `${Number(payload.timing.total_ms).toFixed(0)} ms - dense ${Number(payload.timing.dense_ms).toFixed(0)} - sparse ${Number(payload.timing.sparse_ms).toFixed(0)}`;
    const warnings = [];
    Object.entries(payload.tokenizer_diagnostics || {}).forEach(([model, rows]) => {
      rows.forEach((row, index) => {
        if (row.truncated) warnings.push(`${model}/${row.role || BRANCH1_ROLES[index] || "stream"}${row.language ? `:${row.language}` : ""}: ${row.token_count} > ${row.max_tokens} tokens`);
      });
    });
    el.branch2Validation.textContent = warnings.length
      ? `Tokenizer truncation warning: ${warnings.join("; ")}`
      : "All Branch-2 query streams fit their tokenizer limits.";
    el.branch2Validation.classList.remove("error");
    el.branch2Validation.classList.toggle("warning", warnings.length > 0);
    // Truncation is a quality warning, not a failed search.  Keep the
    // successful/ready state while the warning class supplies yellow styling.
    el.branch2Validation.classList.add("ready");
    el.branch2ResultsGrid.replaceChildren();
    renderVisibleResultPool(
      el.branch2ResultsGrid,
      state.branch2Results,
      (visible, container) => renderStandardCards(visible, container, "branch2", ++resultsRenderId),
    );
  } catch (error) {
    if (error.name === "AbortError") return;
    el.branch2ResultsCount.textContent = "Search failed";
    el.branch2Timing.textContent = `${Math.round(performance.now() - started)} ms`;
    el.branch2ResultsGrid.innerHTML = `<div class="empty-placeholder"><div class="empty-title">Branch-2 failed</div><div class="empty-desc">${escapeHtml(error.message)}</div></div>`;
    showToast(error.message, "error");
  } finally {
    el.btnRunBranch2.textContent = "Run Branch 2";
    // Do not re-run validation here: it would overwrite a useful tokenizer
    // truncation warning rendered from the response payload.
    if (state.branch2Ready) el.btnRunBranch2.disabled = false;
  }
}

function parseBranch1Bundle() {
  let value;
  try {
    value = JSON.parse(el.branch1JsonEditor.value);
  } catch (error) {
    throw new Error(`Invalid JSON: ${error.message}`);
  }
  if (value?.schema_version !== "branch1.query.v1") {
    throw new Error("schema_version must be branch1.query.v1");
  }
  if (!Array.isArray(value.queries) || value.queries.length !== 6) {
    throw new Error("queries must contain exactly six items");
  }
  const roles = value.queries.map((query) => query?.role);
  if (new Set(roles).size !== 6 || !BRANCH1_ROLES.every((role) => roles.includes(role))) {
    throw new Error(`Required roles: ${BRANCH1_ROLES.join(", ")}`);
  }
  value.queries.forEach((query) => {
    if (typeof query.vi !== "string" || !query.vi.trim()) throw new Error(`${query.role}.vi is required`);
    if (typeof query.en !== "string" || !query.en.trim()) throw new Error(`${query.role}.en is required`);
  });
  return value;
}

function validateBranch1Editor() {
  try {
    parseBranch1Bundle();
    el.branch1Validation.textContent = state.branch1Ready
      ? "Valid six-role bilingual bundle. Ready to search."
      : "JSON is valid, but a server-side data or encoder gate is not ready.";
    el.branch1Validation.classList.toggle("ready", state.branch1Ready);
    el.branch1Validation.classList.toggle("error", !state.branch1Ready);
    el.branch1Validation.classList.remove("warning");
    el.btnRunBranch1.disabled = !state.branch1Ready;
    return true;
  } catch (error) {
    el.branch1Validation.textContent = error.message;
    el.branch1Validation.classList.remove("ready");
    el.branch1Validation.classList.add("error");
    el.branch1Validation.classList.remove("warning");
    el.btnRunBranch1.disabled = true;
    return false;
  }
}

function formatBranch1Json() {
  try {
    el.branch1JsonEditor.value = JSON.stringify(JSON.parse(el.branch1JsonEditor.value), null, 2);
    validateBranch1Editor();
  } catch (error) {
    showToast(`Cannot format Branch-1 JSON: ${error.message}`, "error");
  }
}

function updateBranch1Weights() {
  const raw = [
    Number(el.branch1WeightSiglip.value),
    Number(el.branch1WeightMetaclip.value),
    Number(el.branch1WeightBeit.value),
  ];
  el.branch1WeightSiglipValue.textContent = raw[0].toFixed(2);
  el.branch1WeightMetaclipValue.textContent = raw[1].toFixed(2);
  el.branch1WeightBeitValue.textContent = raw[2].toFixed(2);
  const total = raw.reduce((sum, value) => sum + value, 0);
  const normalized = total > 0 ? raw.map((value) => Math.round((value / total) * 100)) : [0, 0, 0];
  el.branch1NormalizedWeights.textContent = `Normalized: ${normalized[0]}% / ${normalized[1]}% / ${normalized[2]}%`;
  if (total <= 0) {
    el.branch1Validation.textContent = "At least one model weight must be greater than zero.";
    el.branch1Validation.classList.remove("ready", "warning");
    el.branch1Validation.classList.add("error");
    el.btnRunBranch1.disabled = true;
  } else {
    validateBranch1Editor();
  }
}

function branch1Audit(item) {
  return ["siglip2", "metaclip2", "beit3"].map((model) => {
    const evidence = item.model_provenance?.[model];
    if (!evidence?.observed) return `${model}: missing → 0.0000`;
    const stream = evidence.best_query_language
      ? `${evidence.best_query_role}:${evidence.best_query_language}`
      : evidence.best_query_role;
    return `${model}: raw ${Number(evidence.raw_cosine).toFixed(4)} · norm ${Number(evidence.normalized_score).toFixed(4)} · ${stream} #${evidence.best_query_rank}`;
  }).join(" | ");
}

async function runBranch1Search() {
  if (!validateBranch1Editor()) return;
  const queryBundle = parseBranch1Bundle();
  const weights = {
    siglip2: Number(el.branch1WeightSiglip.value),
    metaclip2: Number(el.branch1WeightMetaclip.value),
    beit3: Number(el.branch1WeightBeit.value),
  };
  branch1AbortController?.abort();
  branch1AbortController = new AbortController();
  el.btnRunBranch1.disabled = true;
  el.btnRunBranch1.textContent = "Running 30 searches…";
  el.branch1Timing.textContent = "Loading encoders sequentially";
  const started = performance.now();
  try {
    const response = await fetch("/api/search/branch1", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: branch1AbortController.signal,
      body: JSON.stringify({
        query_bundle: queryBundle,
        model_weights: weights,
        per_stream_top_k: 2000,
        final_top_k: 1500,
      }),
    });
    if (!response.ok) throw await responseError(response, "Branch-1 search failed");
    const payload = await response.json();
    state.branch1Response = payload;
    state.branch1Results = payload.results.map((item) => ({
      ...item,
      score: item.final_score,
      score_type: "weighted_zsigmoid_fusion",
      dam_summary: branch1Audit(item),
      retrieval_modality: "branch1",
    }));
    setSubmissionContext("branch1", JSON.stringify(queryBundle));
    const returnedCount = Number(payload.result_count || state.branch1Results.length || 0);
    el.branch1ResultsCount.textContent = branchPoolCountLabel(payload, returnedCount, 1500);
    el.branch1Timing.textContent = `${Number(payload.timing.total_ms).toFixed(0)} ms · fusion ${Number(payload.timing.fusion_ms).toFixed(1)} ms`;
    const warnings = [];
    Object.entries(payload.tokenizer_diagnostics || {}).forEach(([model, rows]) => {
      rows.forEach((row, index) => {
        if (row.truncated) warnings.push(`${model}/${row.role || BRANCH1_ROLES[index] || "stream"}${row.language ? `:${row.language}` : ""}: ${row.token_count} > ${row.max_tokens} tokens`);
      });
    });
    el.branch1Validation.textContent = warnings.length
      ? `Tokenizer truncation warning: ${warnings.join("; ")}`
      : "All 30 query streams fit their tokenizer limits.";
    el.branch1Validation.classList.remove("error");
    el.branch1Validation.classList.toggle("warning", warnings.length > 0);
    // Truncation is a quality warning, not a failed search.  Keep the
    // successful/ready state while the warning class supplies yellow styling.
    el.branch1Validation.classList.add("ready");
    el.branch1ResultsGrid.replaceChildren();
    renderVisibleResultPool(
      el.branch1ResultsGrid,
      state.branch1Results,
      (visible, container) => renderStandardCards(visible, container, "branch1", ++resultsRenderId),
    );
  } catch (error) {
    if (error.name === "AbortError") return;
    el.branch1ResultsCount.textContent = "Search failed";
    el.branch1Timing.textContent = `${Math.round(performance.now() - started)} ms`;
    el.branch1ResultsGrid.innerHTML = `<div class="empty-placeholder"><div class="empty-title">Branch-1 failed</div><div class="empty-desc">${escapeHtml(error.message)}</div></div>`;
    showToast(error.message, "error");
  } finally {
    el.btnRunBranch1.textContent = "Run Branch 1";
    // Preserve tokenizer diagnostics shown after a successful search.
    if (state.branch1Ready) el.btnRunBranch1.disabled = false;
  }
}

function setServerStatus(label, status = "ready") {
  if (el.serverStatusText) el.serverStatusText.textContent = label;
  if (!el.statusPill) return;
  el.statusPill.classList.toggle("ready", status === "ready");
  el.statusPill.classList.toggle("error", status === "error");
  el.statusPill.classList.toggle("pending", status === "pending");
}

async function responseError(response, prefix) {
  let detail = "";
  try {
    const payload = await response.json();
    if (Array.isArray(payload?.detail)) {
      detail = payload.detail.map((issue) => {
        const path = Array.isArray(issue.loc) ? issue.loc.slice(1).join(".") : "request";
        return `${path || "request"}: ${issue.msg || "invalid value"}`;
      }).join("; ");
    } else if (typeof payload?.detail === "string") {
      detail = payload.detail;
    } else if (payload?.detail?.message) {
      detail = payload.detail.message;
    }
  } catch (_) {
    // The HTTP status remains useful when a proxy returns a non-JSON error body.
  }
  return new Error(`${prefix} (HTTP ${response.status})${detail ? `: ${detail}` : ""}`);
}

// ──────────────────────────────────────────────────────────────────────────────
// Event Listeners
// ──────────────────────────────────────────────────────────────────────────────
function bindEvents() {
  // Task selector
  el.taskBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      setWorkspace("kis_fusion");
      submissionStore.setMode(btn.dataset.task);
    });
  });

  el.parserEngine.addEventListener("change", () => {
    state.parserEngine = el.parserEngine.value;
    el.externalFallback.disabled = true;
  });

  el.workspaceTabs.forEach((tab) => {
    tab.addEventListener("click", () => setWorkspace(tab.dataset.workspace));
  });
  el.btnToggleSubmissionRail.addEventListener("click", () => {
    setSubmissionRailCollapsed(!el.submissionRail.classList.contains("collapsed"));
  });
  el.btnCloseSubmissionRail.addEventListener("click", () => setSubmissionRailCollapsed(true));

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
  el.btnFormatBranch1Json.addEventListener("click", formatBranch1Json);
  el.btnRunBranch1.addEventListener("click", () => void runBranch1Search());
  el.btnFormatBranch2Json.addEventListener("click", () => { try { el.branch2JsonEditor.value = JSON.stringify(JSON.parse(el.branch2JsonEditor.value), null, 2); validateBranch2Editor(); } catch (error) { showToast(`Cannot format Branch-2 JSON: ${error.message}`, "error"); } });
  el.btnRunBranch2.addEventListener("click", () => void runBranch2Search());
  el.btnFormatBranch3AsrJson.addEventListener("click", () => { try { el.branch3AsrJsonEditor.value = JSON.stringify(JSON.parse(el.branch3AsrJsonEditor.value), null, 2); validateBranch3AsrEditor(); } catch (error) { showToast(`Cannot format Branch-3 ASR JSON: ${error.message}`, "error"); } });
  el.btnRunBranch3Asr.addEventListener("click", () => void runBranch3AsrSearch());
  el.branch3AsrJsonEditor.addEventListener("input", validateBranch3AsrEditor);
  el.branch3AsrJsonEditor.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      void runBranch3AsrSearch();
    }
  });
  el.btnFormatBranch3OcrJson.addEventListener("click", () => { try { el.branch3OcrJsonEditor.value = JSON.stringify(JSON.parse(el.branch3OcrJsonEditor.value), null, 2); validateBranch3OcrEditor(); } catch (error) { showToast(`Cannot format Branch-3 OCR JSON: ${error.message}`, "error"); } });
  el.btnRunBranch3Ocr.addEventListener("click", () => void runBranch3OcrSearch());
  el.branch3OcrJsonEditor.addEventListener("input", validateBranch3OcrEditor);
  el.branch3OcrJsonEditor.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      void runBranch3OcrSearch();
    }
  });
  el.btnFormatKisFusionJson.addEventListener("click", () => {
    try {
      el.kisFusionJsonEditor.value = JSON.stringify(JSON.parse(el.kisFusionJsonEditor.value), null, 2);
      validateKisFusionEditor();
      updateKisPinnedQuery();
    } catch (error) {
      showToast(`Cannot format KIS fusion JSON: ${error.message}`, "error");
    }
  });
  el.btnRunKisFusion.addEventListener("click", () => void runKisFusionSearch());
  el.btnPrepareKisQuery.addEventListener("click", () => void prepareKisQueryPlan());
  el.kisPinnedQueryText.addEventListener("input", () => {
    if (state.kisQueryPlan.preparing) {
      kisQueryPlanAbortController?.abort();
      kisQueryPlanAbortController = null;
      kisQueryPlanRequestId += 1;
      state.kisQueryPlan.preparing = false;
      el.btnPrepareKisQuery.textContent = "Prepare bundle & events";
    }
    state.kisQueryPlan.sourceOrigin = "user";
    updateKisQueryPlanStatus();
  });
  el.kisPinnedQueryText.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      void prepareKisQueryPlan();
    }
  });
  el.kisFusionJsonEditor.addEventListener("input", () => {
    validateKisFusionEditor();
    updateKisPinnedQuery();
  });
  el.kisFusionJsonEditor.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      void runKisFusionSearch();
    }
  });
  [
    el.kisFusionWeightBranch1,
    el.kisFusionWeightBranch2,
    el.kisFusionWeightOcr,
    el.kisFusionWeightAsr,
  ].forEach((input) => input.addEventListener("input", updateKisFusionWeights));
  el.kisTrakeSequenceEditor.addEventListener("input", () => {
    updateKisTaskUi();
    syncSubmissionEventsFromKisSequence();
  });
  el.btnRunKisTrake.addEventListener("click", () => void runTemporalIntersection());
  el.branch2JsonEditor.addEventListener("input", validateBranch2Editor);
  [el.branch2WeightDense, el.branch2WeightSparse, el.branch2WeightBeit, el.branch2WeightPrevious].forEach((input) => input.addEventListener("input", updateBranch2Weights));
  el.branch1JsonEditor.addEventListener("input", validateBranch1Editor);
  el.branch1JsonEditor.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      void runBranch1Search();
    }
  });
  [el.branch1WeightSiglip, el.branch1WeightMetaclip, el.branch1WeightBeit].forEach((input) => {
    input.addEventListener("input", updateBranch1Weights);
  });
  el.btnDiscoveryCascade.addEventListener("click", () => void runDiscoveryCascade());
  el.btnTemporalIntersection.addEventListener("click", () => void runTemporalIntersection());
  el.temporalEventsEditor.addEventListener("input", () => {
    updateTemporalButtonState();
    syncSubmissionEventsFromQuery(true);
  });

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
  el.btnRevalidateCsv.addEventListener("click", () => void revalidateEditedCsv());
  el.exportCsvPreview.addEventListener("input", () => {
    csvReviewGeneration += 1;
    exportReviewAbortController?.abort();
    exportReviewAbortController = null;
    csvReviewDirty = true;
    preparedExport = null;
    el.btnDownloadCsvAction.disabled = true;
    el.btnRevalidateCsv.disabled = false;
    el.exportSchemaWarning.textContent = "Edited rows are not validated. Revalidate before download.";
  });
  if (el.exportQueryId) el.exportQueryId.addEventListener("input", updateExportPreview);
  if (el.exportModal) {
    el.exportModal.addEventListener("click", (event) => {
      if (event.target === el.exportModal) closeExportModal();
    });
  }

  el.btnClearSubmission.addEventListener("click", () => {
    const snapshot = submissionStore.getSnapshot();
    relatedFillAbortController?.abort();
    relatedFillRequestId += 1;
    delete state.relatedFillStatusByContext[relatedFillStatusKey(snapshot.contextKey, snapshot.mode)];
    submissionStore.clear();
    showToast(`Cleared the ${state.taskType} draft.`, "success");
  });
  el.submissionQueryId.addEventListener("input", () => submissionStore.setQueryId(el.submissionQueryId.value));
  el.vqaAnswerInput.addEventListener("input", () => {
    const result = submissionStore.setAnswer(el.vqaAnswerInput.value);
    const message = result.ok ? "" : "Q&A answers cannot exceed 100 characters.";
    el.vqaAnswerInput.setCustomValidity(message);
    if (message) showToast(message, "error");
  });

  // Image-query workspace
  el.btnChooseImage.addEventListener("click", () => el.imageQueryFile.click());
  el.imageDropZone.addEventListener("click", () => el.imageQueryFile.click());
  el.imageDropZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      el.imageQueryFile.click();
    }
  });
  el.imageQueryFile.addEventListener("change", () => selectImageQueryFile(el.imageQueryFile.files?.[0] || null));
  el.imageDropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    el.imageDropZone.classList.add("dragging");
  });
  el.imageDropZone.addEventListener("dragleave", () => el.imageDropZone.classList.remove("dragging"));
  el.imageDropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    el.imageDropZone.classList.remove("dragging");
    selectImageQueryFile(event.dataTransfer?.files?.[0] || null);
  });
  el.imageDropZone.addEventListener("paste", (event) => {
    const file = [...(event.clipboardData?.files || [])].find((candidate) => candidate.type.startsWith("image/"));
    if (file) {
      event.preventDefault();
      selectImageQueryFile(file);
    }
  });
  el.btnRunImageSearch.addEventListener("click", () => void runImageSearch());
  el.btnClearImage.addEventListener("click", clearImageQuery);

  // Standalone video browser
  el.videoIdForm.addEventListener("submit", (event) => {
    event.preventDefault();
    void loadStandaloneVideo(el.videoIdInput.value);
  });
  el.btnWatchPrevKeyframe.addEventListener("click", (event) => {
    event.preventDefault();
    stepWatchSourceFrame(-1);
  });
  el.btnWatchNextKeyframe.addEventListener("click", (event) => {
    event.preventDefault();
    stepWatchSourceFrame(1);
  });
  el.btnSelectWatchFrame.addEventListener("click", () => {
    void selectWatchSourceFrame(Number(el.watchExactFrameInput.value), true);
  });
  el.watchExactFrameInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      void selectWatchSourceFrame(Number(el.watchExactFrameInput.value), true);
    }
  });
  el.btnSubmitWatchFrame.addEventListener("click", () => {
    if (state.watch.selected) {
      void addSourceFrameToSubmission(state.watch.selected, { source: "video-player-source-frame" });
    }
  });
  el.btnStandaloneVideoSearch.addEventListener("click", () => void runStandaloneVideoSearch());
  el.standaloneVideoQuery.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      void runStandaloneVideoSearch();
    }
  });

  // Inspector modal
  el.btnCloseInspector.addEventListener("click", closeInspector);
  el.btnViewKeyframe.addEventListener("click", () => setInspectorMediaMode("keyframe"));
  el.btnViewVideo.addEventListener("click", () => setInspectorMediaMode("video"));
  el.btnToggleInSubmission.addEventListener("click", toggleCurrentInSubmission);
  el.btnInspectorVideoSearch.addEventListener("click", () => void runInspectorVideoSearch());
  el.inspectorVideoQuery.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      void runInspectorVideoSearch();
    }
  });
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
  el.btnClearFilmstripSelection.addEventListener("click", clearFilmstripSelection);
  el.btnAddFilmstripSelection.addEventListener("click", () => void addFilmstripSelection());

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
  // Submission drafts intentionally survive searches and workspace switches.
  updateInspectorSubmitBtn();
}

function setSubmissionRailCollapsed(collapsed) {
  el.submissionRail.classList.toggle("collapsed", collapsed);
  el.submissionRail.closest(".workbench-shell")?.classList.toggle(
    "submission-collapsed",
    collapsed,
  );
  el.btnToggleSubmissionRail.setAttribute("aria-expanded", String(!collapsed));
  if (!collapsed) el.submissionRail.focus?.({ preventScroll: true });
}

function kisSequenceEvents() {
  return parseOrderedKisEvents(el.kisTrakeSequenceEditor.value);
}

function currentKisBundle() {
  try {
    const bundle = JSON.parse(el.kisFusionJsonEditor.value);
    return hasCompleteKisBundle(bundle) ? bundle : null;
  } catch {
    return null;
  }
}

function setKisQueryPlanStatus(message, status = "") {
  el.kisQueryPlanStatus.textContent = message;
  el.kisQueryPlanStatus.classList.toggle("ready", status === "ready");
  el.kisQueryPlanStatus.classList.toggle("warning", status === "warning");
  el.kisQueryPlanStatus.classList.toggle("error", status === "error");
}

function updateKisQueryPlanStatus() {
  const source = normalizeKisPlanText(el.kisPinnedQueryText.value);
  const bundle = currentKisBundle();
  const events = kisSequenceEvents();
  const plan = state.kisQueryPlan;
  el.btnPrepareKisQuery.disabled = plan.preparing || !source;

  if (plan.preparing) {
    setKisQueryPlanStatus("Preparing one linked bundle and event list locally…");
    return;
  }
  if (!source) {
    setKisQueryPlanStatus("Enter one overall query, then prepare its linked bundle and events.");
    return;
  }
  const stale = Boolean(plan.preparedSource)
    && normalizeKisPlanText(plan.preparedSource) !== source;
  if (stale) {
    setKisQueryPlanStatus(
      "Overall query changed. Existing JSON and events are preserved; prepare again before relying on them.",
      "warning",
    );
    return;
  }
  if (!bundle) {
    setKisQueryPlanStatus("No complete six-role bundle yet. Prepare this overall query first.", "warning");
    return;
  }

  const alignment = assessKisPlanAlignment(source, bundle, events);
  if (!alignment.sourceAligned) {
    setKisQueryPlanStatus(
      "Warning: the six-role JSON does not appear related to the overall query. Search remains available.",
      "warning",
    );
    return;
  }
  if (!alignment.eventsAligned) {
    setKisQueryPlanStatus(
      `Warning: E${alignment.mismatchedEventOrders.join(", E")} may not match the overall query. Search remains available.`,
      "warning",
    );
    return;
  }

  const bundleSignature = canonicalKisBundleSignature(bundle);
  const eventSignature = canonicalKisEventsSignature(el.kisTrakeSequenceEditor.value);
  const bundleEdited = Boolean(plan.generatedBundleSignature)
    && bundleSignature !== plan.generatedBundleSignature;
  const eventsEdited = Boolean(plan.generatedEventsSignature)
    && eventSignature !== plan.generatedEventsSignature;
  const editedLabel = bundleEdited || eventsEdited ? " · editable fields modified" : "";
  const localDraftLabel = plan.bundleSource === "local_deterministic"
    ? " · local draft; review bilingual fields"
    : "";
  if (plan.preparedSource) {
    if (state.taskType === "TRAKE" && events.length < 2) {
      setKisQueryPlanStatus(
        "TRAKE needs at least two ordered events. Add event lines or make the temporal order explicit in the overall query.",
        "warning",
      );
      return;
    }
    setKisQueryPlanStatus(
      events.length >= 2
        ? `Linked plan ready · ${events.length} related ordered events${editedLabel}${localDraftLabel}`
        : `Single-scene KIS plan ready${editedLabel}${localDraftLabel}`,
      "ready",
    );
    return;
  }
  setKisQueryPlanStatus(
    events.length >= 2
      ? `Manual bundle and ${events.length} related events are ready. Prepare to establish one-source tracking.`
      : "Manual bundle is ready. Prepare to derive ordered events from the same overall query.",
    "ready",
  );
}

function updateKisPinnedQuery() {
  const bundle = currentKisBundle();
  const canSyncFromBundle = !normalizeKisPlanText(el.kisPinnedQueryText.value)
    || state.kisQueryPlan.sourceOrigin === "empty"
    || state.kisQueryPlan.sourceOrigin === "bundle";
  if (bundle && canSyncFromBundle) {
    const nextSource = formatKisOverallQuery(bundle);
    if (nextSource) {
      el.kisPinnedQueryText.value = nextSource;
      state.kisQueryPlan.sourceOrigin = "bundle";
    }
  }
  updateKisQueryPlanStatus();
}

function hasNonemptyKisEditorText() {
  return /"(?:vi|en)"\s*:\s*"(?!\s*")[^"]+/.test(el.kisFusionJsonEditor.value);
}

async function prepareKisQueryPlan() {
  const source = normalizeKisPlanText(el.kisPinnedQueryText.value);
  if (!source) {
    showToast("Enter one overall KIS query first.", "error");
    return;
  }

  kisQueryPlanAbortController?.abort();
  kisQueryPlanAbortController = new AbortController();
  const requestId = ++kisQueryPlanRequestId;
  state.kisQueryPlan.preparing = true;
  updateKisQueryPlanStatus();
  el.btnPrepareKisQuery.textContent = "Preparing…";

  try {
    const existingBundle = currentKisBundle();
    const response = await fetch("/api/query/kis/plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: kisQueryPlanAbortController.signal,
      body: JSON.stringify({
        query: source,
        task_type: state.taskType,
        ...(existingBundle ? { query_bundle: existingBundle } : {}),
      }),
    });
    if (!response.ok) throw await responseError(response, "KIS query preparation failed");
    const plan = await response.json();
    if (requestId !== kisQueryPlanRequestId) return;
    if (!hasCompleteKisBundle(plan.query_bundle) || !Array.isArray(plan.events)) {
      throw new Error("KIS query preparation returned an invalid plan");
    }

    const nextBundleSignature = canonicalKisBundleSignature(plan.query_bundle);
    const nextEventsText = formatOrderedKisEvents(plan.events);
    const nextEventsSignature = canonicalKisEventsSignature(nextEventsText);
    const currentBundleSignature = canonicalKisBundleSignature(existingBundle);
    const currentEventsSignature = canonicalKisEventsSignature(el.kisTrakeSequenceEditor.value);
    const emptyEventsSignature = canonicalKisEventsSignature("");
    const previousBundleSnapshot = state.kisQueryPlan.generatedBundleSignature;
    const previousEventsSnapshot = state.kisQueryPlan.generatedEventsSignature;
    const protectedBundle = previousBundleSnapshot
      ? currentBundleSignature !== previousBundleSnapshot
      : hasNonemptyKisEditorText();
    const protectedEvents = previousEventsSnapshot
      ? currentEventsSignature !== previousEventsSnapshot
      : currentEventsSignature !== emptyEventsSignature;
    const replacesManualWork = (
      protectedBundle && currentBundleSignature !== nextBundleSignature
    ) || (
      protectedEvents && currentEventsSignature !== nextEventsSignature
    );
    if (replacesManualWork && !window.confirm(
      "Preparing from the overall query will replace manually edited JSON or event lines. Continue?",
    )) {
      return;
    }

    el.kisFusionJsonEditor.value = JSON.stringify(plan.query_bundle, null, 2);
    el.kisTrakeSequenceEditor.value = nextEventsText;
    state.kisQueryPlan.preparedSource = source;
    state.kisQueryPlan.generatedBundleSignature = nextBundleSignature;
    state.kisQueryPlan.generatedEventsSignature = nextEventsSignature;
    state.kisQueryPlan.bundleSource = String(plan.bundle_source || "local_deterministic");
    state.kisQueryPlan.sourceOrigin = "prepared";
    validateKisFusionEditor();
    if (plan.events.length >= 2) el.kisTrakeSequencePanel.open = true;
    syncSubmissionEventsFromKisSequence();
    updateKisTaskUi();
    showToast(
      plan.events.length >= 2
        ? `Prepared one bundle and ${plan.events.length} linked events. No search was run.`
        : "Prepared one single-scene KIS bundle. No search was run.",
      "success",
    );
  } catch (error) {
    if (error.name === "AbortError") return;
    console.error(error);
    setKisQueryPlanStatus(error.message, "error");
    showToast(error.message, "error");
  } finally {
    if (requestId === kisQueryPlanRequestId) {
      state.kisQueryPlan.preparing = false;
      kisQueryPlanAbortController = null;
      el.btnPrepareKisQuery.textContent = "Prepare bundle & events";
      updateKisQueryPlanStatus();
    }
  }
}

function updateKisTaskUi() {
  const trake = state.taskType === "TRAKE";
  el.kisTaskBadge.textContent = state.taskType === "VQA" ? "Q&A" : state.taskType;
  el.kisFusionControlGrid.classList.remove("hidden");
  if (trake) el.kisTrakeSequencePanel.open = true;
  el.kisFusionResultsHeading.textContent = state.kisFusionView === "sequence"
    ? "ORDERED KIS RESULTS"
    : "KIS FINAL RESULTS";
  if (state.taskType === "VQA") {
    el.kisTaskGuidance.textContent = "Q&A uses full KIS Fusion for evidence. Ordered KIS events are available when the question depends on multiple moments; the final answer remains human-authored.";
  } else if (trake) {
    el.kisTaskGuidance.textContent = "TRAKE runs the complete four-branch KIS pipeline for every event, then requires one video with strictly increasing source-frame indexes.";
  } else {
    el.kisTaskGuidance.textContent = "KIS uses full four-branch fusion. Open Ordered KIS Events for a multi-action query; each event receives its own complete KIS run.";
  }
  updateKisOrderedButtonState();
  updateKisPinnedQuery();
}

function updateTaskWorkspace(mode) {
  const changed = state.taskType !== mode;
  state.taskType = mode;
  updateKisTaskUi();
  if (!changed || state.activeWorkspace !== "kis_fusion") return;
  if (state.kisFusionView === "sequence" && state.kisTemporalIntersection) {
    renderTemporalIntersection(20, ++resultsRenderId);
  } else if (state.kisFusionResults.length) {
    el.kisFusionResultsCount.textContent = `${state.kisFusionResults.length} results`;
    el.kisFusionResultsGrid.replaceChildren();
    renderVisibleResultPool(
      el.kisFusionResultsGrid,
      state.kisFusionResults,
      (visible, container) => renderStandardCards(
        visible,
        container,
        "kis_fusion",
        ++resultsRenderId,
      ),
      150,
    );
  } else {
    el.kisFusionResultsCount.textContent = "Not run";
    el.kisFusionTiming.textContent = "Ready";
    el.kisFusionResultsGrid.innerHTML = `<div class="empty-placeholder"><div class="empty-icon">RRF</div><div class="empty-title">KIS Fusion ready</div><div class="empty-desc">Run one full KIS search, or open Ordered KIS Events to search multiple moments chronologically.</div></div>`;
  }
}

function syncSubmissionEventsFromKisSequence() {
  if (state.taskType !== "TRAKE") return;
  submissionStore.setTrakeEvents(kisSequenceEvents().map((event) => ({
    order: event.order,
    label: event.description,
  })));
}

function setWorkspace(workspace) {
  if (!new Set(["text", "branch1", "branch2", "branch3_asr", "branch3_ocr", "kis_fusion", "image", "video"]).has(workspace)) return;
  state.activeWorkspace = workspace;
  submissionStore.setContext(state.submissionContexts[workspace]);
  el.workspaceTabs.forEach((tab) => {
    const active = tab.dataset.workspace === workspace;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  el.workspacePanels.forEach((panel) => panel.classList.toggle("hidden", panel.dataset.workspacePanel !== workspace));
  if (workspace !== "video") standaloneVideoController.deactivate();
  else if (state.watch.nearest?.frame) void standaloneVideoController.activate().catch(() => undefined);
}

function contextHash(value) {
  let hash = 2166136261;
  const text = String(value || "");
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(36);
}

function setSubmissionContext(workspace, identity) {
  const contextKey = `${workspace}:${contextHash(identity)}`;
  state.submissionContexts[workspace] = contextKey;
  if (state.activeWorkspace === workspace) submissionStore.setContext(contextKey);
}

function currentSubmissionDraft(snapshot = submissionStore.getSnapshot()) {
  return snapshot.drafts[snapshot.mode];
}

function submissionItemCount(snapshot = submissionStore.getSnapshot()) {
  const draft = currentSubmissionDraft(snapshot);
  return snapshot.mode === "TRAKE"
    ? Object.keys(draft.eventSlots || {}).length
    : (draft.items || []).length + (draft.suggestedItems || []).length;
}

function submissionDraftItems(draft) {
  return [...(draft.items || []), ...(draft.suggestedItems || [])];
}

function relatedFillStatusKey(contextKey, mode) {
  return `${contextKey}::${mode}`;
}

function syncSubmissionEventsFromQuery(editorOnly = false) {
  const sourceEvents = editorOnly
    ? String(el.temporalEventsEditor.value || "")
      .split(/\r?\n/)
      .map((line) => line.trim().replace(/^E\d+\s*[:.)-]\s*/i, ""))
      .filter(Boolean)
      .slice(0, 6)
      .map((line, index) => ({ order: index + 1, description: line, global_scene_en: line }))
    : (Array.isArray(state.parsedQuery?.trake_events) ? state.parsedQuery.trake_events : []);
  const events = sourceEvents.map((event, index) => ({
    order: Number(event.order) || index + 1,
    label: event.description || event.scene_en || event.global_scene_en || `Event ${index + 1}`,
  }));
  submissionStore.setTrakeEvents(events);
}

function renderSubmissionRail(snapshot) {
  updateTaskWorkspace(snapshot.mode);
  const draft = currentSubmissionDraft(snapshot);
  const count = submissionItemCount(snapshot);
  el.taskBtns.forEach((button) => {
    const active = button.dataset.task === snapshot.mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  el.submissionCount.textContent = String(count);
  el.submissionTabCount.textContent = String(count);
  el.submissionSchemaNote.textContent = submissionSchemaDefaults[snapshot.mode];
  el.submissionQueryId.value = draft.queryId || "1";
  el.exportQueryId.value = draft.queryId || "1";
  el.vqaAnswerPanel.classList.toggle("hidden", snapshot.mode !== "VQA");
  el.trakeEventPanel.classList.toggle("hidden", snapshot.mode !== "TRAKE");
  if (snapshot.mode === "VQA") el.vqaAnswerInput.value = draft.answer || "";

  renderTrakeEventTabs(snapshot.drafts.TRAKE);
  el.submissionList.replaceChildren();
  if (snapshot.mode === "TRAKE") renderTrakeSubmissionItems(draft);
  else renderFrameSubmissionItems(draft);

  if (!el.submissionList.children.length) {
    const empty = document.createElement("div");
    empty.className = "submission-empty";
    empty.textContent = snapshot.mode === "TRAKE"
      ? "Choose an active event, then add a frame or an ordered sequence."
      : "Add a frame from any result, inspector, or video player.";
    el.submissionList.appendChild(empty);
  }

  const top = snapshot.mode === "TRAKE"
    ? orderedTrakeFrames(draft)[0]?.item
    : submissionDraftItems(draft)[0];
  el.submissionInput.value = top ? `${top.video_id}, ${top.frame_idx}` : "No keyframe selected";
  const relatedStatus = state.relatedFillStatusByContext[
    relatedFillStatusKey(snapshot.contextKey, snapshot.mode)
  ];
  const suggestionCount = draft.suggestedItems?.length || 0;
  el.submissionRelatedNote.classList.toggle(
    "hidden",
    snapshot.mode === "TRAKE" || (!relatedStatus && suggestionCount === 0),
  );
  if (snapshot.mode !== "TRAKE") {
    if (relatedStatus?.state === "loading") {
      el.submissionRelatedNote.textContent = "Finding visually related indexed frames…";
      el.submissionRelatedNote.classList.add("loading");
      el.submissionRelatedNote.classList.remove("error");
    } else if (relatedStatus?.state === "error") {
      el.submissionRelatedNote.textContent = relatedStatus.message;
      el.submissionRelatedNote.classList.add("error");
      el.submissionRelatedNote.classList.remove("loading");
    } else {
      el.submissionRelatedNote.textContent = suggestionCount
        ? `${suggestionCount} auto-related frame${suggestionCount === 1 ? "" : "s"} from the first verified selection.`
        : "No related frames were returned for the selected seed.";
      el.submissionRelatedNote.classList.remove("loading", "error");
    }
  }
  updateInspectorSubmitBtn();
  updateWatchSubmitButton();
}

function renderTrakeEventTabs(draft) {
  el.trakeEventTabs.replaceChildren();
  const events = draft.events?.length
    ? draft.events
    : Array.from({ length: Math.max(1, Object.keys(draft.eventSlots || {}).length) }, (_, index) => ({
      order: index + 1,
      label: `Event ${index + 1}`,
    }));
  events.forEach((event) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "trake-event-tab";
    button.classList.toggle("active", draft.activeEvent === event.order);
    button.classList.toggle("filled", Boolean(draft.eventSlots?.[String(event.order)]));
    button.textContent = `E${event.order}`;
    button.title = event.label;
    button.addEventListener("click", () => submissionStore.setActiveEvent(event.order));
    el.trakeEventTabs.appendChild(button);
  });
}

function renderFrameSubmissionItems(draft) {
  (draft.items || []).forEach((item, index) => {
    const identity = frameIdentity(item);
    const row = createSubmissionRow(item, {
      index,
      identity,
      label: `#${index + 1}`,
      onRemove: () => removeSubmissionFrame(identity),
      onMove: (direction) => submissionStore.reorderFrame(index, index + direction),
      related: false,
    });
    row.draggable = true;
    row.addEventListener("dragstart", (event) => event.dataTransfer?.setData("text/plain", String(index)));
    row.addEventListener("dragover", (event) => event.preventDefault());
    row.addEventListener("drop", (event) => {
      event.preventDefault();
      const from = Number.parseInt(event.dataTransfer?.getData("text/plain") || "", 10);
      submissionStore.reorderFrame(from, index);
    });
    el.submissionList.appendChild(row);
  });
  (draft.suggestedItems || []).forEach((item, index) => {
    const identity = frameIdentity(item);
    el.submissionList.appendChild(createSubmissionRow(item, {
      identity,
      label: `A${index + 1}`,
      related: true,
      onRemove: () => removeSubmissionFrame(identity),
      onMove: null,
    }));
  });
}

function removeSubmissionFrame(identityOrOrder) {
  const before = submissionStore.getSnapshot();
  submissionStore.removeFrame(identityOrOrder);
  const after = submissionStore.getSnapshot();
  if (after.mode !== "TRAKE" && !currentSubmissionDraft(after).relatedSeed) {
    delete state.relatedFillStatusByContext[
      relatedFillStatusKey(before.contextKey, before.mode)
    ];
    relatedFillAbortController?.abort();
    relatedFillRequestId += 1;
    renderSubmissionRail(after);
  }
}

function renderTrakeSubmissionItems(draft) {
  const eventMap = new Map((draft.events || []).map((event) => [event.order, event]));
  const orders = new Set([
    ...(draft.events || []).map((event) => event.order),
    ...Object.keys(draft.eventSlots || {}).map(Number),
  ]);
  [...orders].sort((a, b) => a - b).forEach((order) => {
    const item = draft.eventSlots?.[String(order)];
    if (!item) {
      const empty = document.createElement("button");
      empty.type = "button";
      empty.className = "submission-event-empty";
      empty.innerHTML = `<strong>E${order}</strong><span>${escapeHtml(eventMap.get(order)?.label || "Awaiting frame")}</span>`;
      empty.addEventListener("click", () => submissionStore.setActiveEvent(order));
      el.submissionList.appendChild(empty);
      return;
    }
    el.submissionList.appendChild(createSubmissionRow(item, {
      identity: String(order),
      label: `E${order}`,
      onRemove: () => removeSubmissionFrame(order),
      onMove: null,
      related: false,
    }));
  });
}

function createSubmissionRow(item, options) {
  const row = document.createElement("article");
  row.className = "submission-item";
  row.classList.toggle("auto-related", options.related === true);
  row.dataset.identity = options.identity;
  const validationLabel = item.validation === "canonical"
    ? "indexed frame verified"
    : item.validation === "source_timeline"
      ? "source frame verified"
      : "awaiting server verification";
  const frameUid = frameIdentity(item);
  const timeLabel = item.pts_time_s === null || item.pts_time_s === undefined
    ? "unknown"
    : `${Number(item.pts_time_s).toFixed(3)}s`;
  const fpsLabel = item.fps ? Number(item.fps).toFixed(3) : "unknown";
  const relatedLabel = options.related ? "Auto-related" : item.source || "manual";
  const previewFrameIdx = Number.isInteger(Number(item.preview_frame_idx))
    ? Number(item.preview_frame_idx)
    : null;
  const previewLabel = item.validation === "source_timeline" && previewFrameIdx !== null
    ? `nearest indexed preview: frame ${previewFrameIdx}`
    : "exact indexed image";
  row.innerHTML = `
    <div class="submission-item-head">
      <span class="submission-item-order">${escapeHtml(options.label)}</span>
      <span class="submission-item-source">${escapeHtml(relatedLabel)}</span>
      <button type="button" class="submission-remove" aria-label="Remove frame">×</button>
    </div>
    <div class="submission-item-identity"><strong>${escapeHtml(item.video_id)}</strong><span>frame ${item.frame_idx}</span></div>
    <div class="submission-item-meta">${item.keyframe_n == null ? "not an indexed keyframe" : `KF ${item.keyframe_n}`} · ${timeLabel} · ${escapeHtml(validationLabel)}</div>
    <details class="submission-item-details">
      <summary>Preview &amp; metadata</summary>
      <img src="${getImageUrl(item)}" alt="${escapeHtml(item.video_id)}, ${escapeHtml(previewLabel)} for source frame ${item.frame_idx}" loading="lazy" decoding="async">
      <dl>
        <div><dt>frame_uid</dt><dd>${escapeHtml(frameUid)}</dd></div>
        <div><dt>frame_idx</dt><dd>${item.frame_idx} (submission)</dd></div>
        <div><dt>keyframe_n</dt><dd>${item.keyframe_n ?? "unknown"} (navigation)</dd></div>
        <div><dt>preview</dt><dd>${escapeHtml(previewLabel)}</dd></div>
        <div><dt>timestamp</dt><dd>${timeLabel}</dd></div>
        <div><dt>FPS</dt><dd>${fpsLabel}</dd></div>
        <div><dt>source</dt><dd>${escapeHtml(relatedLabel)}</dd></div>
      </dl>
    </details>
    ${options.onMove ? `<div class="submission-reorder"><button type="button" data-move="-1" aria-label="Move up">↑</button><button type="button" data-move="1" aria-label="Move down">↓</button><span>drag to reorder</span></div>` : ""}`;
  row.querySelector(".submission-remove").addEventListener("click", options.onRemove);
  row.querySelectorAll("[data-move]").forEach((button) => button.addEventListener("click", () => options.onMove(Number(/** @type {HTMLElement} */ (button).dataset.move))));
  const preview = /** @type {HTMLImageElement} */ (row.querySelector(".submission-item-details img"));
  preview.addEventListener("error", () => {
    preview.hidden = true;
  }, { once: true });
  return row;
}

async function resolveCanonicalFrame(item, signal = undefined) {
  const identity = frameIdentity(item);
  if (!identity) throw new Error("This item has no valid video_id/frame_idx identity.");
  if (!canonicalFrameCache.has(identity)) {
    const [videoId, frameIndex] = identity.split(":");
    const response = await fetch(
      `/api/frame/${encodeURIComponent(videoId)}/${encodeURIComponent(frameIndex)}`,
      { signal },
    );
    if (!response.ok) throw await responseError(response, "Exact indexed frame verification failed");
    const payload = await response.json();
    const canonical = payload?.exact_match === true ? payload.keyframe : null;
    if (!canonical || frameIdentity(canonical) !== identity) {
      throw new Error("The server returned a different frame identity; selection was blocked.");
    }
    canonicalFrameCache.set(identity, canonical);
  }
  return { ...item, ...canonicalFrameCache.get(identity), validation: "canonical" };
}

async function resolveSourceFrame(item, signal = undefined) {
  const identity = frameIdentity(item);
  if (!identity) throw new Error("Enter a valid zero-based source-frame index.");
  if (!sourceFrameCache.has(identity)) {
    const [videoId, frameIndex] = identity.split(":");
    const response = await fetch(
      `/api/video/${encodeURIComponent(videoId)}/source-frame/${encodeURIComponent(frameIndex)}`,
      { signal },
    );
    if (!response.ok) throw await responseError(response, "Source-frame verification failed");
    const payload = await response.json();
    const verified = payload?.exact_match === true ? payload.source_frame : null;
    if (
      !verified
      || frameIdentity(verified) !== identity
      || Number(verified.frame_index_base) !== 0
      || !Number.isInteger(Number(verified.max_frame_idx))
      || Number(verified.frame_idx) > Number(verified.max_frame_idx)
      || !Number.isFinite(Number(verified.pts_time_s))
    ) {
      throw new Error("The server returned an inconsistent source-frame identity; selection was blocked.");
    }
    sourceFrameCache.set(identity, verified);
  }
  return { ...item, ...sourceFrameCache.get(identity) };
}

async function fillRelatedSubmissionFrames(seed, contextKey, mode) {
  if (mode === "TRAKE" || state.capabilities.related_frame_fill === false) return;
  relatedFillAbortController?.abort();
  relatedFillAbortController = new AbortController();
  const requestId = ++relatedFillRequestId;
  const statusKey = relatedFillStatusKey(contextKey, mode);
  Object.entries(state.relatedFillStatusByContext).forEach(([key, status]) => {
    if (status?.state === "loading") delete state.relatedFillStatusByContext[key];
  });
  state.relatedFillStatusByContext[statusKey] = { state: "loading" };
  renderSubmissionRail(submissionStore.getSnapshot());
  try {
    const response = await fetch("/api/submission/related-frames", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: relatedFillAbortController.signal,
      body: JSON.stringify({
        video_id: seed.video_id,
        frame_idx: Number(seed.frame_idx),
        limit: 99,
      }),
    });
    if (!response.ok) throw await responseError(response, "Related-frame fill failed");
    const payload = await response.json();
    const snapshot = submissionStore.getSnapshot();
    if (
      requestId !== relatedFillRequestId
      || snapshot.contextKey !== contextKey
      || snapshot.mode !== mode
    ) {
      delete state.relatedFillStatusByContext[statusKey];
      return;
    }
    const applied = submissionStore.setRelatedFrames(seed, payload.results || [], mode);
    if (!applied.ok) {
      delete state.relatedFillStatusByContext[statusKey];
      return;
    }
    state.relatedFillStatusByContext[statusKey] = { state: "ready" };
    renderSubmissionRail(submissionStore.getSnapshot());
    showToast(`Added ${applied.count} visually related frame suggestions.`, "success");
  } catch (error) {
    if (error.name === "AbortError" || requestId !== relatedFillRequestId) {
      delete state.relatedFillStatusByContext[statusKey];
      return;
    }
    state.relatedFillStatusByContext[statusKey] = {
      state: "error",
      message: `Auto-related fill unavailable: ${error.message}`,
    };
    renderSubmissionRail(submissionStore.getSnapshot());
    showToast("The verified seed was kept, but related-frame fill is unavailable.", "error");
  }
}

async function verifyAndAddSubmissionFrame(item, context, resolver) {
  const startSnapshot = submissionStore.getSnapshot();
  try {
    const verified = await resolver(item);
    const currentSnapshot = submissionStore.getSnapshot();
    if (
      currentSnapshot.contextKey !== startSnapshot.contextKey
      || currentSnapshot.mode !== startSnapshot.mode
    ) return;
    const eventOrder = context.eventOrder || item?.event_order || null;
    const result = submissionStore.addFrame(verified, {
      source: context.source || item?.retrieval_modality || "result",
      eventOrder,
      validation: verified.validation,
    });
    if (!result.ok) {
      const message = result.reason === "draft-full"
        ? "The submission draft already contains 100 frames."
        : result.reason === "invalid-trake-order"
          ? "TRAKE frames must use one video and increase in exact frame order."
          : "This result does not contain a valid source-frame reference.";
      showToast(message, "error");
      return;
    }
    const suffix = state.taskType === "TRAKE" ? ` to E${result.eventOrder}` : "";
    showToast(`Added exact source frame ${result.frame.video_id}, ${result.frame.frame_idx}${suffix}.`, "success");
    if (result.firstManual) {
      void fillRelatedSubmissionFrames(
        result.frame,
        currentSnapshot.contextKey,
        currentSnapshot.mode,
      );
    }
  } catch (error) {
    showToast(error.message, "error");
  }
}

async function addFrameToSubmission(item, context = {}) {
  return verifyAndAddSubmissionFrame(item, context, resolveCanonicalFrame);
}

async function addSourceFrameToSubmission(item, context = {}) {
  return verifyAndAddSubmissionFrame(item, context, resolveSourceFrame);
}

async function addCanonicalTrakeSequence(items, videoId) {
  const expectedEventCount = state.activeWorkspace === "kis_fusion"
    ? kisSequenceEvents().length
    : getUsableTemporalEvents().length;
  if (!Array.isArray(items) || expectedEventCount < 2 || items.length !== expectedEventCount) {
    showToast("TRAKE requires one verified frame for every ordered event.", "error");
    return;
  }
  submissionStore.setMode("TRAKE");
  const startSnapshot = submissionStore.getSnapshot();
  try {
    const frames = await Promise.all((items || []).map((item) => resolveCanonicalFrame(item)));
    const snapshot = submissionStore.getSnapshot();
    if (
      snapshot.contextKey !== startSnapshot.contextKey
      || snapshot.mode !== "TRAKE"
    ) return;
    const ordered = frames
      .map((frame, index) => ({ ...frame, event_order: Number(items[index]?.event_order) || index + 1 }))
      .sort((left, right) => left.event_order - right.event_order);
    const result = submissionStore.addSequence(ordered, {
      source: "ordered-sequence",
      validation: "canonical",
    });
    if (!result.ok) throw new Error("The ordered sequence contains an invalid frame identity.");
    showToast(`Added ${ordered.length} exact ordered events from ${videoId}.`, "success");
  } catch (error) {
    showToast(error.message, "error");
  }
}

function setSearchBusy(isBusy) {
  el.btnRunQuery.disabled = isBusy;
  el.btnExecuteJson.disabled = isBusy;
  el.selectTopK.disabled = isBusy;
  el.modalityTabs.forEach((tab) => {
    tab.disabled = isBusy
      || Boolean(state.drilldown)
      || Boolean(state.discoveryCascade)
      || Boolean(state.temporalIntersection);
  });
  updateDiscoveryButtonState(isBusy);
  updateTemporalButtonState(isBusy);
  el.btnRunQuery.setAttribute("aria-busy", String(isBusy));
}

function updateDiscoveryButtonState(isBusy = false) {
  const hasVisualQuery = Boolean(String(state.parsedQuery?.global_scene_en || "").trim());
  const hasObjectQuery = Array.isArray(state.parsedQuery?.objects_en)
    && state.parsedQuery.objects_en.some((query) => String(query || "").trim());
  const hasRawPools = Object.keys(state.modalityResults).length > 0;
  el.btnDiscoveryCascade.disabled = isBusy
    || !hasVisualQuery
    || !hasObjectQuery
    || !hasRawPools
    || Boolean(state.drilldown)
    || Boolean(state.discoveryCascade)
    || Boolean(state.temporalIntersection);
}

function getUsableTemporalEvents() {
  const sourceEditor = state.activeWorkspace === "kis_fusion" && state.taskType === "TRAKE"
    ? el.kisTrakeSequenceEditor
    : el.temporalEventsEditor;
  const editorLines = String(sourceEditor?.value || "")
    .split(/\r?\n/)
    .map((line) => line.trim().replace(/^E\d+\s*[:.)-]\s*/i, ""))
    .filter(Boolean)
    .slice(0, 6);
  if (editorLines.length > 0) {
    return editorLines.map((query, index) => ({
      order: index + 1,
      description: query,
      global_scene_en: query,
    }));
  }
  if (!Array.isArray(state.parsedQuery?.trake_events)) return [];
  return state.parsedQuery.trake_events
    .map((event, index) => ({
      order: Number(event?.order) || index + 1,
      description: String(event?.description || `Event ${Number(event?.order) || index + 1}`).trim(),
      global_scene_en: String(event?.scene_en || "").trim(),
    }))
    .filter((event) => event.global_scene_en)
    .sort((left, right) => left.order - right.order);
}

function syncTemporalEventsEditorFromParsedQuery() {
  const events = Array.isArray(state.parsedQuery?.trake_events)
    ? state.parsedQuery.trake_events
      .map((event) => String(event?.scene_en || "").trim())
      .filter(Boolean)
      .slice(0, 6)
    : [];
  el.temporalEventsEditor.value = events.join("\n");
  updateTemporalButtonState();
  syncSubmissionEventsFromQuery();
}

function updateTemporalButtonState(isBusy = false) {
  const events = getUsableTemporalEvents();
  const hasRawPools = Object.keys(state.modalityResults).length > 0;
  const blockedByDerivedView = Boolean(
    state.drilldown || state.discoveryCascade || state.temporalIntersection,
  );
  el.btnTemporalIntersection.disabled = isBusy
    || events.length < 2
    || !hasRawPools
    || blockedByDerivedView;
  el.selectTemporalGap.disabled = isBusy || events.length < 2 || blockedByDerivedView;
  el.temporalEventsEditor.disabled = isBusy || blockedByDerivedView;
  el.temporalEventCount.textContent = `${events.length} usable event${events.length === 1 ? "" : "s"}`;
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
  if (state.drilldown || state.discoveryCascade || state.temporalIntersection) return;
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

function formatPoolDiagnostics(pool) {
  const diagnostics = pool?.query_diagnostics;
  if (!diagnostics) return "";
  const tokenSummary = `${diagnostics.token_count}/${diagnostics.max_tokens} SigLIP tokens`;
  if (!diagnostics.truncated) return tokenSummary;
  return `⚠ ${tokenSummary}; effective query: ${diagnostics.effective_query || ""}`;
}

function updateModalityQuerySummary() {
  if (state.temporalIntersection) {
    const data = state.temporalIntersection.data;
    const eventQueries = (data.event_pools || [])
      .map((event) => `E${event.order}: <code>${escapeHtml(event.query || "")}</code>`)
      .join("<br>");
    el.modalityQuerySummary.innerHTML = `<strong>Ordered SigLIP intersection</strong> · each event is searched independently, video IDs are intersected, and timestamps must increase.<br>Cross-modal fusion: <strong>OFF</strong> · sequence score: <strong>(shared-scene anchor cosine + weakest event cosine) / 2</strong>; event mean is the next tie-break.<br>${eventQueries}`;
    return;
  }
  if (state.discoveryCascade) {
    const discovery = state.discoveryCascade.data;
    el.modalityQuerySummary.innerHTML = `<strong>Explicit DAM → SigLIP discovery cascade</strong> · DAM searches each <code>objects_en</code> entry independently and only selects video scope.<br>Final frame rank is raw <code>global_scene_en</code> SigLIP cosine. DAM score is not added; cross-modal gating is applied and displayed.`;
    if (discovery.siglip_query_diagnostics?.truncated) {
      el.modalityQuerySummary.innerHTML += `<br>⚠ Effective SigLIP query: ${escapeHtml(discovery.siglip_query_diagnostics.effective_query || "")}`;
    }
    return;
  }
  if (state.drilldown) {
    const pool = state.drilldown.pool;
    const diagnostics = formatPoolDiagnostics(pool);
    el.modalityQuerySummary.innerHTML = `<strong>${escapeHtml(pool.display_name)}</strong> · manual video scope · query source <code>global_scene_en</code> · query <code>${escapeHtml(formatPoolQuery(pool))}</code><br>${escapeHtml(pool.score_description)}${diagnostics ? `<br>${escapeHtml(diagnostics)}` : ""}`;
    return;
  }
  if (!Object.keys(state.modalityResults).length) {
    el.modalityQuerySummary.textContent = "Each modality uses only its own parsed subquery and raw score.";
    return;
  }
  if (state.activeModality === "all") {
    el.modalityQuerySummary.innerHTML = MODALITY_ORDER.map((modality) => {
      const pool = state.modalityResults[modality];
      if (!pool) return "";
      const query = pool.status === "ok" ? formatPoolQuery(pool) : pool.reason;
      const diagnostics = formatPoolDiagnostics(pool);
      return `<strong>${escapeHtml(pool.display_name)}</strong>: <code>${escapeHtml(query)}</code> · ${escapeHtml(pool.score_type)}${diagnostics ? ` · ${escapeHtml(diagnostics)}` : ""}`;
    }).filter(Boolean).join("<br>");
    return;
  }
  const pool = state.modalityResults[state.activeModality];
  if (!pool) return;
  const query = pool.status === "ok" ? formatPoolQuery(pool) : pool.reason;
  const diagnostics = formatPoolDiagnostics(pool);
  el.modalityQuerySummary.innerHTML = `<strong>${escapeHtml(pool.display_name)}</strong> · query source <code>${escapeHtml(pool.query_source)}</code> · query <code>${escapeHtml(query)}</code><br>${escapeHtml(pool.score_description)}${diagnostics ? `<br>${escapeHtml(diagnostics)}` : ""}`;
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
    data.task_type = "KIS";
    el.jsonEditor.value = JSON.stringify(data, null, 2);
  } catch (_) {}
}

function clearVideoDrilldown({ restoreControls = true } = {}) {
  const previous = state.drilldown;
  drilldownAbortController?.abort();
  drilldownAbortController = null;
  drilldownRequestId += 1;
  state.drilldown = null;

  if (restoreControls && previous) {
    el.selectTopK.value = previous.previousTopK;
    state.activeModality = previous.previousActiveModality;
  }
  el.modalityTabs.forEach((tab) => {
    const active = tab.dataset.modality === state.activeModality;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
    tab.disabled = false;
  });
  updateDiscoveryButtonState();
  updateTemporalButtonState();
}

function exitVideoDrilldown() {
  const previousTimingText = state.drilldown?.previousTimingText;
  clearVideoDrilldown();
  if (previousTimingText) el.timingBadge.textContent = previousTimingText;
  setServerStatus("No fusion / no reranking", "ready");
  renderModalityResults();
}

function clearDiscoveryCascade({ restoreControls = true } = {}) {
  const previous = state.discoveryCascade;
  cascadeAbortController?.abort();
  cascadeAbortController = null;
  cascadeRequestId += 1;
  state.discoveryCascade = null;

  if (restoreControls && previous) {
    el.selectTopK.value = previous.previousTopK;
    state.activeModality = previous.previousActiveModality;
  }
  el.modalityTabs.forEach((tab) => {
    const active = tab.dataset.modality === state.activeModality;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
    tab.disabled = false;
  });
  updateDiscoveryButtonState();
  updateTemporalButtonState();
}

function exitDiscoveryCascade() {
  const previousTimingText = state.discoveryCascade?.previousTimingText;
  clearDiscoveryCascade();
  if (previousTimingText) el.timingBadge.textContent = previousTimingText;
  setServerStatus("No fusion / no reranking", "ready");
  renderModalityResults();
}

function clearTemporalIntersection({ restoreControls = true } = {}) {
  const previous = state.temporalIntersection;
  temporalAbortController?.abort();
  temporalAbortController = null;
  temporalRequestId += 1;
  state.temporalIntersection = null;

  if (restoreControls && previous) {
    el.selectTopK.value = previous.previousTopK;
    state.activeModality = previous.previousActiveModality;
  }
  el.modalityTabs.forEach((tab) => {
    const active = tab.dataset.modality === state.activeModality;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
    tab.disabled = false;
  });
  updateDiscoveryButtonState();
  updateTemporalButtonState();
}

function exitTemporalIntersection() {
  const previousTimingText = state.temporalIntersection?.previousTimingText;
  clearTemporalIntersection();
  if (previousTimingText) el.timingBadge.textContent = previousTimingText;
  setServerStatus("No fusion / no reranking", "ready");
  renderModalityResults();
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
  setSubmissionContext("text", query);

  parseAbortController?.abort();
  searchAbortController?.abort();
  clearVideoDrilldown();
  clearDiscoveryCascade();
  clearTemporalIntersection();
  searchRequestId += 1;
  parseAbortController = new AbortController();
  const requestId = ++parseRequestId;
  setSearchBusy(true);
  clearSubmissionSelection();
  state.parsedQuery = null;
  el.temporalEventsEditor.value = "";
  updateTemporalButtonState(true);
  state.modalityResults = {};
  state.searchResults = [];
  el.resultsCount.textContent = "Parsing…";
  el.resultsGrid.innerHTML = `<div class="empty-placeholder" aria-live="polite"><div class="empty-icon">⏳</div><div class="empty-title">Parsing query</div><div class="empty-desc">Previous results were cleared to prevent stale selections.</div></div>`;
  el.timingBadge.textContent = "Parsing query...";
  setServerStatus("Parsing query…", "pending");

  try {
    if (state.parserEngine === "direct") {
      state.parsedQuery = {
        task_type: "KIS",
        language: "en",
        original_query: query,
        global_scene_en: query,
        objects_en: [query],
        speech_vi: "",
        ocr_keywords: [],
        is_temporal_trake: false,
        trake_events: [],
        vqa_question: "",
      };
      el.jsonEditor.value = JSON.stringify(state.parsedQuery, null, 2);
      syncTemporalEventsEditorFromParsedQuery();
      if (state.queryMode === "auto") await handleExecuteJsonClick();
      else {
        el.timingBadge.textContent = "Direct contract ready (no parser model used)";
        setServerStatus("Direct query ready", "ready");
      }
      return;
    }
    const parseRes = await fetch("/api/parse", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: query,
        task_type: "KIS",
        engine: state.parserEngine,
      }),
      signal: parseAbortController.signal,
    });

    if (!parseRes.ok) throw await responseError(parseRes, "Parse failed");
    const parseData = await parseRes.json();
    if (requestId !== parseRequestId) return;
    state.parsedQuery = parseData.parsed_query;
    syncTemporalEventsEditorFromParsedQuery();

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

  if (!parsedJson || typeof parsedJson !== "object" || Array.isArray(parsedJson)) {
    showToast("Parsed query must be one JSON object.", "error");
    return;
  }

  if (typeof parsedJson.original_query !== "string" || !parsedJson.original_query.trim()) {
    const fallbackOriginal = el.inputQuery.value.trim()
      || (typeof parsedJson.global_scene_en === "string" ? parsedJson.global_scene_en.trim() : "");
    if (!fallbackOriginal) {
      showToast("Missing required field: original_query", "error");
      return;
    }
    parsedJson.original_query = fallbackOriginal;
  }

  delete parsedJson.weights;
  parsedJson.task_type = "KIS";
  setSubmissionContext("text", parsedJson.original_query || JSON.stringify(parsedJson));
  state.parsedQuery = parsedJson;
  el.jsonEditor.value = JSON.stringify(parsedJson, null, 2);
  syncTemporalEventsEditorFromParsedQuery();

  searchAbortController?.abort();
  clearVideoDrilldown();
  clearDiscoveryCascade();
  clearTemporalIntersection();
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

    if (!res.ok) throw await responseError(res, "Search failed");
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

async function runTemporalIntersection() {
  const isKisOrdered = state.activeWorkspace === "kis_fusion";
  const events = isKisOrdered ? kisSequenceEvents() : getUsableTemporalEvents();
  if (events.length < 2) {
    showToast(
      isKisOrdered
        ? "Ordered KIS search needs at least two non-empty event lines."
        : "Ordered search needs at least two trake_events with non-empty scene_en values.",
      "error",
    );
    return;
  }
  let queryBundle = null;
  if (isKisOrdered) {
    if (!validateKisFusionEditor()) return;
    queryBundle = parseKisFusionBundle();
  }

  const requestController = new AbortController();
  let requestId;
  if (isKisOrdered) {
    kisTemporalAbortController?.abort();
    kisTemporalAbortController = requestController;
    requestId = ++kisTemporalRequestId;
  } else {
    temporalAbortController?.abort();
    temporalAbortController = requestController;
    requestId = ++temporalRequestId;
  }
  const isCurrentRequest = () => isKisOrdered
    ? requestId === kisTemporalRequestId
    : requestId === temporalRequestId;
  if (isKisOrdered) state.kisTemporalIntersection = null;
  else state.temporalIntersection = null;
  const previousTopK = el.selectTopK.value;
  const previousActiveModality = state.activeModality;
  const previousTimingText = el.timingBadge.textContent;
  const maxGapSeconds = Number(
    isKisOrdered ? el.kisTrakeGap.value : el.selectTemporalGap.value,
  ) || 30;
  // Ordered KIS focuses the complete six-role bundle independently for every
  // event. Never let a stale diagnostic shared-scene anchor alter that path.
  const anchorQuery = isKisOrdered
    ? ""
    : String(state.parsedQuery?.global_scene_en || "").trim();

  setSearchBusy(true);
  if (isKisOrdered) {
    state.kisFusionBusy = true;
    el.btnRunKisFusion.disabled = true;
    el.btnRunKisTrake.disabled = true;
    el.btnRunKisTrake.textContent = "Running full KIS per event…";
    setSubmissionContext(
      "kis_fusion",
      `ordered-kis:${JSON.stringify({ query_bundle: queryBundle, events })}`,
    );
    if (state.taskType === "TRAKE") {
      submissionStore.setTrakeEvents(events.map((event) => ({
        order: event.order,
        label: event.description,
      })));
    }
  }
  clearSubmissionSelection();
  if (!el.modal.classList.contains("hidden")) closeInspector();
  state.searchResults = [];
  const resultsGrid = isKisOrdered ? el.kisFusionResultsGrid : el.resultsGrid;
  const resultsCount = isKisOrdered ? el.kisFusionResultsCount : el.resultsCount;
  const timingBadge = isKisOrdered ? el.kisFusionTiming : el.timingBadge;
  if (isKisOrdered) el.kisFusionResultsHeading.textContent = "ORDERED KIS RESULTS";
  else el.resultsHeading.textContent = "⛓ ORDERED SIGLIP INTERSECTION";
  resultsCount.textContent = `Searching ${events.length} ordered events…`;
  resultsGrid.innerHTML = `<div class="empty-placeholder" aria-live="polite"><div class="empty-icon">⛓</div><div class="empty-title">${isKisOrdered ? "Running full KIS for every event" : "Intersecting ordered visual events"}</div><div class="empty-desc">${isKisOrdered ? "Each event uses Branch 1, Branch 2, OCR, ASR, weighted RRF and final BEiT-3 reranking before same-video source-frame ordering." : "Every prompt runs as an independent sequence search. Candidate video IDs are intersected before timestamp order is enforced."}</div></div>`;
  timingBadge.textContent = `Searching ${events.length} event pools...`;
  setServerStatus(
    isKisOrdered ? "Ordered full KIS fusion…" : "Ordered SigLIP intersection…",
    "pending",
  );

  try {
    const endpoint = isKisOrdered
      ? "/api/search/fusion/kis/temporal"
      : "/api/search/temporal-intersection";
    const requestBody = isKisOrdered
      ? {
        task_type: state.taskType,
        query_bundle: queryBundle,
        events,
        branch_weights: {
          branch1: Number(el.kisFusionWeightBranch1.value),
          branch2: Number(el.kisFusionWeightBranch2.value),
          ocr: Number(el.kisFusionWeightOcr.value),
          asr: Number(el.kisFusionWeightAsr.value),
        },
        top_k_sequences: Number(el.kisTrakeSequences.value) || 100,
        max_gap_seconds: maxGapSeconds,
      }
      : {
        events,
        anchor_query: anchorQuery || null,
        top_k_per_event: Number(el.selectTemporalCandidates.value) || 300,
        top_k_sequences: Number(el.selectTemporalSequences.value) || 100,
        paths_per_video: Number(el.selectTemporalPathsPerVideo.value) || 1,
        sequence_reservoir_size: el.selectTemporalReservoir.value === "same"
          ? null
          : Number(el.selectTemporalReservoir.value),
        max_gap_seconds: maxGapSeconds,
      };
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody),
      signal: requestController.signal,
    });
    if (!response.ok) throw await responseError(response, "Ordered search failed");
    const data = await response.json();
    if (!isCurrentRequest()) return;

    const intersectionState = {
      data,
      previousTopK,
      previousActiveModality,
      previousTimingText,
    };
    if (isKisOrdered) {
      state.kisTemporalIntersection = intersectionState;
      state.kisFusionView = "sequence";
    } else {
      state.temporalIntersection = intersectionState;
    }
    el.selectTopK.value = "20";
    el.modalityTabs.forEach((tab) => {
      tab.classList.remove("active");
      tab.setAttribute("aria-selected", "false");
    });
    timingBadge.textContent = `${data.event_count} event pools completed in ${data.execution_time_ms}ms`;
    setServerStatus(
      isKisOrdered
        ? "Ordered KIS · full fusion per event"
        : "Ordered SigLIP · cross-modal fusion off",
      "ready",
    );
    if (isKisOrdered) renderTemporalIntersection(20, ++resultsRenderId);
    else renderModalityResults();
    const count = Number(data.ordered_sequence_count ?? data.sequences?.length ?? 0);
    showToast(
      count > 0 ? `Found ${count} ordered video sequence${count === 1 ? "" : "s"}.` : "No video satisfied every event in order.",
      count > 0 ? "success" : "error",
    );
  } catch (error) {
    if (error.name === "AbortError") return;
    console.error(error);
    if (isKisOrdered) {
      el.kisFusionResultsCount.textContent = "Sequence search failed";
      el.kisFusionTiming.textContent = "Ready";
      el.kisFusionResultsGrid.innerHTML = `<div class="empty-placeholder"><div class="empty-title">Ordered KIS failed</div><div class="empty-desc">${escapeHtml(error.message)}</div></div>`;
    } else {
      el.resultsHeading.textContent = "🎯 RAW MODALITY RESULTS";
      el.timingBadge.textContent = previousTimingText;
      renderModalityResults();
    }
    setServerStatus(isKisOrdered ? "Ordered KIS failed" : "No fusion / no reranking", isKisOrdered ? "error" : "ready");
    showToast(error.message, "error");
  } finally {
    if (isCurrentRequest()) setSearchBusy(false);
    if (isKisOrdered && isCurrentRequest()) {
      state.kisFusionBusy = false;
      kisTemporalAbortController = null;
      el.btnRunKisTrake.textContent = "Run ordered KIS fusion";
      el.btnRunKisFusion.disabled = !state.kisFusionReady || !state.kisFusionBundleValid;
      updateKisTaskUi();
    } else if (!isKisOrdered && isCurrentRequest()) {
      temporalAbortController = null;
    }
  }
}

async function runDiscoveryCascade() {
  const visualQuery = String(state.parsedQuery?.global_scene_en || "").trim();
  const objectQueries = Array.isArray(state.parsedQuery?.objects_en)
    ? state.parsedQuery.objects_en.filter((query) => String(query || "").trim())
    : [];
  if (!visualQuery || objectQueries.length === 0) {
    showToast("Discovery needs both global_scene_en and at least one objects_en query.", "error");
    return;
  }

  cascadeAbortController?.abort();
  cascadeAbortController = new AbortController();
  const requestId = ++cascadeRequestId;
  const previousTopK = el.selectTopK.value;
  const previousActiveModality = state.activeModality;
  const previousTimingText = el.timingBadge.textContent;

  setSearchBusy(true);
  clearSubmissionSelection();
  if (!el.modal.classList.contains("hidden")) closeInspector();
  state.searchResults = [];
  el.resultsHeading.textContent = "🧭 EXPLICIT DISCOVERY CASCADE";
  el.resultsCount.textContent = "Discovering candidate videos…";
  el.resultsGrid.innerHTML = `<div class="empty-placeholder" aria-live="polite"><div class="empty-icon">🧭</div><div class="empty-title">Discovering videos without score fusion</div><div class="empty-desc">Each DAM object independently selects its Top 20 raw frames. SigLIP then ranks Top 10 frames inside every resulting video.</div></div>`;
  el.timingBadge.textContent = "Running explicit DAM → SigLIP discovery cascade...";
  setServerStatus("DAM gating → raw SigLIP ranking…", "pending");

  try {
    const response = await fetch("/api/discover/dam-to-siglip", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        parsed_query: state.parsedQuery,
        dam_top_frames_per_object: 20,
        siglip_top_frames_per_video: 10,
      }),
      signal: cascadeAbortController.signal,
    });
    if (!response.ok) throw await responseError(response, "Discovery cascade failed");
    const data = await response.json();
    if (requestId !== cascadeRequestId) return;

    state.discoveryCascade = {
      data,
      previousTopK,
      previousActiveModality,
      previousTimingText,
    };
    el.selectTopK.value = "50";
    el.modalityTabs.forEach((tab) => {
      tab.classList.remove("active");
      tab.setAttribute("aria-selected", "false");
    });
    el.timingBadge.textContent = `Cascade searched ${data.unique_candidate_video_count} unique videos in ${data.execution_time_ms}ms`;
    setServerStatus("Explicit DAM → SigLIP cascade · no score fusion", "ready");
    renderModalityResults();
    showToast(`Discovered ${data.unique_candidate_video_count} candidate videos.`, "success");
  } catch (error) {
    if (error.name === "AbortError") return;
    console.error(error);
    el.resultsHeading.textContent = "🎯 RAW MODALITY RESULTS";
    el.timingBadge.textContent = previousTimingText;
    setServerStatus("No fusion / no reranking", "ready");
    renderModalityResults();
    showToast(error.message, "error");
  } finally {
    if (requestId === cascadeRequestId) setSearchBusy(false);
  }
}

async function searchInsideVideo(item, sourceModality) {
  const visualQuery = String(state.parsedQuery?.global_scene_en || "").trim();
  if (!visualQuery) {
    showToast("This action needs a non-empty global_scene_en SigLIP query.", "error");
    return;
  }

  const videoId = String(item.video_id || "").toUpperCase().replace(/-/g, "_");
  if (!videoId) {
    showToast("The selected result has no video ID.", "error");
    return;
  }

  drilldownAbortController?.abort();
  drilldownAbortController = new AbortController();
  const requestId = ++drilldownRequestId;
  const previousTopK = el.selectTopK.value;
  const previousActiveModality = state.activeModality;
  const previousTimingText = el.timingBadge.textContent;

  setSearchBusy(true);
  clearSubmissionSelection();
  if (!el.modal.classList.contains("hidden")) closeInspector();
  state.searchResults = [];
  el.resultsCount.textContent = `Searching ${videoId}…`;
  el.resultsGrid.innerHTML = `<div class="empty-placeholder" aria-live="polite"><div class="empty-icon">⏳</div><div class="empty-title">Searching inside ${escapeHtml(videoId)}</div><div class="empty-desc">Running the same SigLIP cosine only over frames from this video.</div></div>`;
  el.timingBadge.textContent = `Running manual SigLIP drill-down in ${videoId}...`;
  setServerStatus("Video-scoped SigLIP search…", "pending");

  try {
    const response = await fetch(`/api/video/${encodeURIComponent(videoId)}/search/siglip`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        parsed_query: state.parsedQuery,
        top_k: SEARCH_POOL_SIZE,
      }),
      signal: drilldownAbortController.signal,
    });
    if (!response.ok) throw await responseError(response, "Video drill-down failed");
    const data = await response.json();
    if (requestId !== drilldownRequestId) return;

    state.drilldown = {
      videoId: data.video_id,
      pool: data.modality_result,
      evaluatedFrames: data.evaluated_frames,
      sourceModality,
      sourceRank: item.rank,
      previousTopK,
      previousActiveModality,
      previousTimingText,
    };
    state.activeModality = "siglip";
    el.selectTopK.value = "50";
    el.modalityTabs.forEach((tab) => {
      const active = tab.dataset.modality === "siglip";
      tab.classList.toggle("active", active);
      tab.setAttribute("aria-selected", String(active));
    });

    el.timingBadge.textContent = `SigLIP searched ${data.evaluated_frames} frames in ${data.execution_time_ms}ms`;
    setServerStatus("Manual video scope · no fusion", "ready");
    renderModalityResults();
    showToast(`Showing SigLIP cosine inside ${data.video_id}.`, "success");
  } catch (error) {
    if (error.name === "AbortError") return;
    console.error(error);
    el.timingBadge.textContent = previousTimingText;
    setServerStatus("No fusion / no reranking", "ready");
    renderModalityResults();
    showToast(error.message, "error");
  } finally {
    if (requestId === drilldownRequestId) setSearchBusy(false);
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Additive Image Search and Video Browser Workspaces
// ──────────────────────────────────────────────────────────────────────────────
function selectImageQueryFile(file) {
  if (!file) return;
  if (!new Set(["image/jpeg", "image/png", "image/webp"]).has(file.type)) {
    showToast("Choose a JPEG, PNG, or WebP image.", "error");
    return;
  }
  imageSearchAbortController?.abort();
  imageSearchAbortController = null;
  imageSearchRequestId += 1;
  if (state.imageQueryObjectUrl) URL.revokeObjectURL(state.imageQueryObjectUrl);
  state.imageQueryFile = file;
  state.imageResults = [];
  setSubmissionContext("image", `${file.name}:${file.size}:${file.lastModified}:${file.type}`);
  state.imageQueryObjectUrl = URL.createObjectURL(file);
  el.imageQueryPreview.src = state.imageQueryObjectUrl;
  el.imageQueryPreview.classList.remove("hidden");
  el.imageDropPrompt.classList.add("hidden");
  el.btnRunImageSearch.disabled = false;
  el.imageResultsCount.textContent = `${file.name || "pasted image"} ready`;
  el.imageResultsGrid.innerHTML = `<div class="empty-placeholder"><div class="empty-icon">🖼</div><div class="empty-title">New image ready</div><div class="empty-desc">Run Direct SigLIP image search to retrieve matching frames.</div></div>`;
}

function clearImageQuery() {
  imageSearchAbortController?.abort();
  imageSearchRequestId += 1;
  if (state.imageQueryObjectUrl) URL.revokeObjectURL(state.imageQueryObjectUrl);
  state.imageQueryFile = null;
  state.imageQueryObjectUrl = null;
  state.imageResults = [];
  state.submissionContexts.image = "image:empty";
  if (state.activeWorkspace === "image") submissionStore.setContext("image:empty");
  el.imageQueryFile.value = "";
  el.imageQueryPreview.removeAttribute("src");
  el.imageQueryPreview.classList.add("hidden");
  el.imageDropPrompt.classList.remove("hidden");
  el.btnRunImageSearch.disabled = true;
  el.imageResultsCount.textContent = "No image selected";
  el.imageResultsGrid.innerHTML = `<div class="empty-placeholder"><div class="empty-icon">🖼</div><div class="empty-title">Image Search Ready</div><div class="empty-desc">Choose an unknown visual example to find similar indexed frames.</div></div>`;
}

async function runImageSearch() {
  if (!state.imageQueryFile) return;
  imageSearchAbortController?.abort();
  imageSearchAbortController = new AbortController();
  const requestId = ++imageSearchRequestId;
  el.btnRunImageSearch.disabled = true;
  el.imageResultsCount.textContent = "Embedding and searching…";
  el.imageResultsGrid.innerHTML = `<div class="empty-placeholder"><div class="empty-icon">⏳</div><div class="empty-title">Running SigLIP image search</div><div class="empty-desc">No text parser or cross-modal fusion is used.</div></div>`;
  try {
    const body = new FormData();
    body.append("file", state.imageQueryFile, state.imageQueryFile.name || "query-image.png");
    body.append("top_k", el.selectImageTopK.value);
    const response = await fetch("/api/search/image", {
      method: "POST",
      body,
      signal: imageSearchAbortController.signal,
    });
    if (!response.ok) throw await responseError(response, "Image search failed");
    const data = await response.json();
    if (requestId !== imageSearchRequestId) return;
    state.imageResults = data.modality_result?.results || [];
    renderImageResults(data);
  } catch (error) {
    if (error.name === "AbortError") return;
    console.error(error);
    el.imageResultsCount.textContent = "Search failed";
    el.imageResultsGrid.innerHTML = `<div class="empty-placeholder"><div class="empty-icon">⚠</div><div class="empty-title">Image search failed</div><div class="empty-desc">${escapeHtml(error.message)}</div></div>`;
    showToast(error.message, "error");
  } finally {
    if (requestId === imageSearchRequestId) el.btnRunImageSearch.disabled = !state.imageQueryFile;
  }
}

function renderImageResults(data) {
  const renderId = ++resultsRenderId;
  el.imageResultsGrid.replaceChildren();
  el.imageResultsCount.textContent = `${state.imageResults.length} raw SigLIP cosine results · ${Number(data.execution_time_ms || 0).toFixed(1)}ms`;
  if (!state.imageResults.length) {
    el.imageResultsGrid.innerHTML = `<div class="empty-placeholder"><div class="empty-icon">∅</div><div class="empty-title">No image matches</div></div>`;
    return;
  }
  renderStandardCards(state.imageResults, el.imageResultsGrid, "image", renderId);
}

function normalizeRequestedVideoId(value) {
  const videoId = String(value || "").trim().toUpperCase().replaceAll("-", "_");
  return /^[A-Z0-9]+_V\d+$/.test(videoId) ? videoId : "";
}

function deriveTimelineFps(keyframes) {
  const samples = (keyframes || []).flatMap((frame) => {
    const index = Number(frame.frame_idx);
    const seconds = Number(frame.pts_time_s);
    return Number.isFinite(index) && Number.isFinite(seconds) && seconds > 0 ? [index / seconds] : [];
  }).sort((left, right) => left - right);
  if (!samples.length) return null;
  return Number(samples[Math.floor(samples.length / 2)].toFixed(4));
}

async function fetchVideoTimeline(videoId, signal = undefined) {
  const canonicalId = normalizeRequestedVideoId(videoId);
  if (!canonicalId) throw new Error("Use a dataset video ID such as L25_V060.");
  const endpoints = [
    `/api/video/${encodeURIComponent(canonicalId)}/timeline`,
    `/api/video/${encodeURIComponent(canonicalId)}/keyframes`,
  ];
  let lastError = null;
  for (const endpoint of endpoints) {
    try {
      const response = await fetch(endpoint, { signal });
      if (!response.ok) {
        lastError = await responseError(response, "Timeline unavailable");
        if (response.status !== 404) throw lastError;
        continue;
      }
      const data = await response.json();
      const responseVideoId = normalizeRequestedVideoId(data.video_id || canonicalId);
      if (responseVideoId !== canonicalId) {
        throw new Error("Timeline response returned a different video identity.");
      }
      const keyframes = (data.keyframes || [])
        .map((frame) => ({ ...frame, video_id: canonicalId, validation: "canonical" }))
        .sort((left, right) => Number(left.frame_idx) - Number(right.frame_idx));
      if (!keyframes.length) throw new Error(`No indexed keyframes were found for ${canonicalId}.`);
      const fps = Number(data.fps) || deriveTimelineFps(keyframes);
      if (!Number.isFinite(fps) || fps <= 0) throw new Error(`FPS metadata is invalid for ${canonicalId}.`);
      const validAnchors = keyframes.every((frame, index) => (
        Number.isInteger(Number(frame.frame_idx))
        && Number(frame.frame_idx) >= 0
        && Number.isFinite(Number(frame.pts_time_s))
        && Number(frame.pts_time_s) >= 0
        && (
          index === 0
          || (
            Number(frame.frame_idx) > Number(keyframes[index - 1].frame_idx)
            && Number(frame.pts_time_s) > Number(keyframes[index - 1].pts_time_s)
          )
        )
      ));
      if (!validAnchors) throw new Error(`Indexed-frame anchors are inconsistent for ${canonicalId}.`);
      const frameIndexBase = Number(data.frame_index_base ?? 0);
      const fallbackMax = Number(keyframes[keyframes.length - 1].frame_idx);
      const maxFrameIdx = Number(data.max_frame_idx ?? fallbackMax);
      if (frameIndexBase !== 0 || !Number.isInteger(maxFrameIdx) || maxFrameIdx < fallbackMax) {
        throw new Error(`Source-frame bounds are inconsistent for ${canonicalId}.`);
      }
      return {
        videoId: responseVideoId,
        fps,
        durationS: Number(data.duration_s) || Number(keyframes[keyframes.length - 1].pts_time_s),
        frameIndexBase,
        maxFrameIdx,
        timingMethod: String(data.timing_method || "exact-anchor-piecewise-linear-v1"),
        keyframes,
      };
    } catch (error) {
      if (error.name === "AbortError") throw error;
      lastError = error;
      if (!String(error.message || "").includes("404")) break;
    }
  }
  throw lastError || new Error(`Timeline unavailable for ${canonicalId}.`);
}

async function loadStandaloneVideo(requestedVideoId) {
  const videoId = normalizeRequestedVideoId(requestedVideoId);
  if (!videoId) {
    showToast("Use a dataset video ID such as L25_V060.", "error");
    return;
  }
  el.videoIdInput.value = videoId;
  setSubmissionContext("video", videoId);
  standaloneScopedAbortController?.abort();
  standaloneScopedAbortController = null;
  standaloneScopedRequestId += 1;
  watchLoadAbortController?.abort();
  watchLoadAbortController = new AbortController();
  const requestId = ++watchLoadRequestId;
  state.watch.videoId = "";
  state.watch.fps = null;
  state.watch.durationS = null;
  state.watch.frameIndexBase = 0;
  state.watch.maxFrameIdx = null;
  state.watch.currentFrameIdx = null;
  state.watch.timingMethod = "";
  state.watch.keyframes = [];
  state.watch.nearest = null;
  state.watch.selected = null;
  state.watch.searchResults = [];
  el.watchExactFrameInput.value = "";
  el.watchSelectedFrameStatus.textContent = "No source frame selected";
  updateWatchMapping(0);
  el.standaloneVideoSearchResults.replaceChildren();
  el.standaloneVideoSearchResults.classList.add("hidden");
  el.btnStandaloneVideoSearch.disabled = true;
  el.standaloneVideoSource.textContent = "Loading verified source-frame timeline…";
  el.standaloneVideoSource.classList.remove("ready", "error");
  el.standaloneVideoArea.classList.remove("hidden");
  el.btnSubmitWatchFrame.disabled = true;
  standaloneVideoController.deactivate();
  try {
    const timeline = await fetchVideoTimeline(videoId, watchLoadAbortController.signal);
    if (requestId !== watchLoadRequestId) return;
    state.watch.videoId = timeline.videoId;
    state.watch.fps = timeline.fps;
    state.watch.durationS = timeline.durationS;
    state.watch.frameIndexBase = timeline.frameIndexBase;
    state.watch.maxFrameIdx = timeline.maxFrameIdx;
    state.watch.timingMethod = timeline.timingMethod;
    state.watch.keyframes = timeline.keyframes;
    state.watch.nearest = nearestKeyframe(timeline.keyframes, 0);
    const initialSourceFrame = await resolveSourceFrame(
      { video_id: timeline.videoId, frame_idx: timeline.frameIndexBase },
      watchLoadAbortController.signal,
    );
    if (requestId !== watchLoadRequestId) return;
    el.watchExactFrameInput.min = String(timeline.frameIndexBase);
    el.watchExactFrameInput.max = String(timeline.maxFrameIdx);
    el.watchExactFrameInput.value = String(timeline.frameIndexBase);
    standaloneVideoController.setFrame(toVideoFrame(initialSourceFrame));
    updateWatchMapping(Number(initialSourceFrame.pts_time_s) || 0);
    await standaloneVideoController.preload(toVideoFrame(initialSourceFrame));
    if (requestId !== watchLoadRequestId) return;
    await standaloneVideoController.activate();
    if (requestId !== watchLoadRequestId) return;
    el.standaloneVideoSource.textContent = `${timeline.videoId} · source frames ${timeline.frameIndexBase}–${timeline.maxFrameIdx} · ${timeline.keyframes.length} indexed previews`;
    el.btnStandaloneVideoSearch.disabled = false;
    updateWatchSubmitButton();
    showToast(`Loaded ${timeline.videoId}.`, "success");
  } catch (error) {
    if (error.name === "AbortError" || requestId !== watchLoadRequestId) return;
    console.error(error);
    el.standaloneVideoSource.textContent = error.message;
    el.standaloneVideoSource.classList.add("error");
    showToast(error.message, "error");
  }
}

function updateWatchMapping(playbackSeconds) {
  const seconds = Number(playbackSeconds);
  el.watchPlaybackTime.textContent = Number.isFinite(seconds) ? `${seconds.toFixed(1)}s` : "—";
  const currentFrameIdx = secondsToFrameIndex(
    seconds,
    state.watch.keyframes,
    state.watch.fps,
    state.watch.frameIndexBase,
    state.watch.maxFrameIdx,
  );
  state.watch.currentFrameIdx = currentFrameIdx;
  el.watchEstimatedFrame.textContent = currentFrameIdx === null ? "—" : String(currentFrameIdx);
  const match = nearestKeyframe(state.watch.keyframes, seconds);
  state.watch.nearest = match;
  if (!match) {
    el.watchKeyframeN.textContent = "—";
    el.watchFrameIdx.textContent = "—";
    el.watchKeyframeTime.textContent = "—";
    el.watchMappingDelta.textContent = "—";
    updateWatchSubmitButton();
    return;
  }
  el.watchKeyframeN.textContent = `#${match.frame.keyframe_n}`;
  el.watchFrameIdx.textContent = String(match.frame.frame_idx);
  el.watchKeyframeTime.textContent = `${Number(match.frame.pts_time_s).toFixed(3)}s`;
  const sign = match.deltaSeconds > 0 ? "+" : "";
  el.watchMappingDelta.textContent = `${sign}${match.deltaSeconds.toFixed(3)}s`;
  const poster = document.getElementById("standalone-video-poster");
  if (poster && poster.getAttribute("src") !== getImageUrl(match.frame)) poster.setAttribute("src", getImageUrl(match.frame));
  updateWatchSubmitButton();
}

function updateWatchSubmitButton() {
  if (!el.btnSubmitWatchFrame) return;
  const frame = state.watch.selected;
  el.btnSubmitWatchFrame.disabled = !frame;
  const selected = frame ? submissionStore.hasFrame(frame) : false;
  el.btnSubmitWatchFrame.classList.toggle("in-submit", selected);
  el.btnSubmitWatchFrame.textContent = selected
    ? "✓ Selected source frame is in submission"
    : "+ Add selected source frame";
  const activeFrameIdx = frame?.frame_idx ?? state.watch.currentFrameIdx;
  el.btnWatchPrevKeyframe.disabled = !Number.isInteger(Number(activeFrameIdx))
    || Number(activeFrameIdx) <= state.watch.frameIndexBase;
  el.btnWatchNextKeyframe.disabled = !Number.isInteger(Number(activeFrameIdx))
    || !Number.isInteger(Number(state.watch.maxFrameIdx))
    || Number(activeFrameIdx) >= Number(state.watch.maxFrameIdx);
}

function stepWatchSourceFrame(direction) {
  const typed = Number(el.watchExactFrameInput.value);
  const current = Number.isInteger(typed)
    ? typed
    : state.watch.selected?.frame_idx ?? state.watch.currentFrameIdx;
  if (!Number.isInteger(Number(current))) return;
  const next = Math.max(
    state.watch.frameIndexBase,
    Math.min(Number(state.watch.maxFrameIdx), Number(current) + direction),
  );
  if (next === Number(current)) return;
  void selectWatchSourceFrame(next, true);
}

async function selectWatchSourceFrame(frameIndex, seek = false) {
  if (
    !state.watch.videoId
    || !Number.isInteger(frameIndex)
    || frameIndex < state.watch.frameIndexBase
    || frameIndex > state.watch.maxFrameIdx
  ) {
    showToast(
      `Enter a whole frame index from ${state.watch.frameIndexBase} to ${state.watch.maxFrameIdx}.`,
      "error",
    );
    return;
  }
  el.btnSelectWatchFrame.disabled = true;
  try {
    const sourceFrame = await resolveSourceFrame({
      video_id: state.watch.videoId,
      frame_idx: frameIndex,
    });
    if (state.watch.videoId !== sourceFrame.video_id) return;
    state.watch.selected = sourceFrame;
    el.watchExactFrameInput.value = String(sourceFrame.frame_idx);
    el.watchSelectedFrameStatus.textContent = sourceFrame.indexed_keyframe
      ? `Selected source frame ${sourceFrame.frame_idx} · indexed KF ${sourceFrame.keyframe_n} · ${Number(sourceFrame.pts_time_s).toFixed(3)}s`
      : `Selected source frame ${sourceFrame.frame_idx} · ${Number(sourceFrame.pts_time_s).toFixed(3)}s · nearest indexed preview is frame ${sourceFrame.preview_frame_idx}`;
    el.watchSelectedFrameStatus.classList.remove("error");
    if (seek) {
      standaloneVideoController.seekTo(Number(sourceFrame.pts_time_s));
      updateWatchMapping(Number(sourceFrame.pts_time_s));
    }
    const poster = document.getElementById("standalone-video-poster");
    if (poster) poster.setAttribute("src", getImageUrl(sourceFrame));
    updateWatchSubmitButton();
  } catch (error) {
    state.watch.selected = null;
    el.watchSelectedFrameStatus.textContent = error.message;
    el.watchSelectedFrameStatus.classList.add("error");
    updateWatchSubmitButton();
    showToast(error.message, "error");
  } finally {
    el.btnSelectWatchFrame.disabled = false;
  }
}

function directParsedQuery(query) {
  return {
    task_type: "KIS",
    language: "en",
    original_query: query,
    global_scene_en: query,
    objects_en: [query],
    speech_vi: "",
    ocr_keywords: [],
    is_temporal_trake: false,
    trake_events: [],
    vqa_question: "",
  };
}

function currentVisualQuery() {
  if (state.activeWorkspace === "kis_fusion") {
    try {
      const bundle = JSON.parse(el.kisFusionJsonEditor.value);
      const original = (bundle.queries || []).find((query) => query.role === "original");
      const text = String(original?.en || original?.vi || "").trim();
      if (text) return text;
    } catch {
      // The scoped-search input stays editable when the main bundle is invalid.
    }
  }
  return String(state.parsedQuery?.global_scene_en || "").trim();
}

async function parseScopedQuery(query, engine, signal = undefined) {
  if (engine === "direct") return directParsedQuery(query);
  const response = await fetch("/api/parse", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      task_type: "KIS",
      engine,
    }),
    signal,
  });
  if (!response.ok) throw await responseError(response, "Scoped query parse failed");
  return (await response.json()).parsed_query;
}

async function searchVideoWithText(videoId, query, engine = "direct", topK = 50, signal = undefined) {
  const canonicalId = normalizeRequestedVideoId(videoId);
  if (!canonicalId) throw new Error("The active frame has no valid dataset video ID.");
  if (!String(query || "").trim()) throw new Error("Enter a visual description to search inside this video.");
  const parsedQuery = await parseScopedQuery(String(query).trim(), engine, signal);
  const visualQuery = String(parsedQuery.global_scene_en || query).trim();
  let contextBundle;
  if (state.activeWorkspace === "kis_fusion" && state.kisFusionBundleValid) {
    try {
      contextBundle = parseKisFusionBundle();
    } catch {
      contextBundle = undefined;
    }
  }
  const response = await fetch(`/api/video/${encodeURIComponent(canonicalId)}/search/visual-fusion`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: visualQuery,
      query_bundle: contextBundle,
      top_k: topK,
    }),
    signal,
  });
  if (!response.ok) throw await responseError(response, "Inside-video search failed");
  return response.json();
}

async function runStandaloneVideoSearch() {
  const query = el.standaloneVideoQuery.value.trim();
  if (!state.watch.videoId || !query) {
    showToast("Load a video and enter a visual description first.", "error");
    return;
  }
  el.btnStandaloneVideoSearch.disabled = true;
  standaloneScopedAbortController?.abort();
  standaloneScopedAbortController = new AbortController();
  const requestId = ++standaloneScopedRequestId;
  el.standaloneVideoSearchResults.classList.remove("hidden");
  el.standaloneVideoSearchResults.innerHTML = `<div class="empty-placeholder"><div class="empty-icon">⏳</div><div class="empty-title">Searching ${escapeHtml(state.watch.videoId)}</div></div>`;
  try {
    const data = await searchVideoWithText(
      state.watch.videoId,
      query,
      "direct",
      100,
      standaloneScopedAbortController.signal,
    );
    if (requestId !== standaloneScopedRequestId) return;
    state.watch.searchResults = data.modality_result?.results || [];
    el.standaloneVideoSearchResults.replaceChildren();
    renderStandardCards(
      state.watch.searchResults,
      el.standaloneVideoSearchResults,
      "visual_fusion",
      ++resultsRenderId,
    );
  } catch (error) {
    if (error.name === "AbortError") return;
    el.standaloneVideoSearchResults.innerHTML = `<div class="empty-placeholder"><div class="empty-title">Search failed</div><div class="empty-desc">${escapeHtml(error.message)}</div></div>`;
    showToast(error.message, "error");
  } finally {
    if (requestId === standaloneScopedRequestId) el.btnStandaloneVideoSearch.disabled = false;
  }
}

async function runInspectorVideoSearch() {
  const item = state.activeInspectorItem;
  if (!item) return;
  const query = el.inspectorVideoQuery.value.trim();
  el.btnInspectorVideoSearch.disabled = true;
  inspectorScopedAbortController?.abort();
  inspectorScopedAbortController = new AbortController();
  const requestId = ++inspectorScopedRequestId;
  el.inspectorLocalResults.classList.remove("hidden");
  el.inspectorLocalResults.innerHTML = `<div class="inspector-local-status">Searching ${escapeHtml(item.video_id)}…</div>`;
  try {
    const data = await searchVideoWithText(
      item.video_id,
      query,
      el.inspectorParserEngine.value,
      30,
      inspectorScopedAbortController.signal,
    );
    if (requestId !== inspectorScopedRequestId || state.activeInspectorItem?.video_id !== item.video_id) return;
    state.inspectorLocalResults = data.modality_result?.results || [];
    renderInspectorLocalResults(state.inspectorLocalResults);
  } catch (error) {
    if (error.name === "AbortError") return;
    el.inspectorLocalResults.innerHTML = `<div class="inspector-local-status error">${escapeHtml(error.message)}</div>`;
    showToast(error.message, "error");
  } finally {
    if (requestId === inspectorScopedRequestId) el.btnInspectorVideoSearch.disabled = false;
  }
}

function renderInspectorLocalResults(results) {
  el.inspectorLocalResults.replaceChildren();
  results.slice(0, 20).forEach((item, index) => {
    const row = document.createElement("div");
    row.className = "inspector-local-result";
    row.innerHTML = `
      <button type="button" class="inspector-local-open"><strong>#${item.rank || index + 1}</strong><span>frame ${item.frame_idx} · ${Number(item.pts_time_s || 0).toFixed(1)}s</span><em>${Number(item.score || 0).toFixed(4)}</em></button>
      <button type="button" class="inspector-local-add" aria-label="Add frame ${item.frame_idx} to submission">+</button>`;
    row.querySelector(".inspector-local-open").addEventListener("click", () => void openStandardInspector(item, true));
    row.querySelector(".inspector-local-add").addEventListener("click", () => void addFrameToSubmission(item, { source: "video-scoped-visual-fusion" }));
    el.inspectorLocalResults.appendChild(row);
  });
  if (!results.length) el.inspectorLocalResults.innerHTML = `<div class="inspector-local-status">No frames returned.</div>`;
}

// ──────────────────────────────────────────────────────────────────────────────
// Raw Modality Result Rendering
// ──────────────────────────────────────────────────────────────────────────────
function getImageUrl(item) {
  const vid = item.video_id;
  const relpath = String(item.image_relpath || item.preview_image_relpath || "").replace(/^\/+/, "");
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

  if (state.temporalIntersection) {
    renderTemporalIntersection(limit, renderId);
    return;
  }

  if (state.discoveryCascade) {
    renderDiscoveryCascade(limit, renderId);
    return;
  }

  if (state.drilldown) {
    renderVideoDrilldown(limit, renderId);
    return;
  }

  el.resultsHeading.textContent = "🎯 RAW MODALITY RESULTS";

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

function getAllCascadeCandidates(cascades) {
  const pools = cascades.map((cascade) => cascade.results || []);
  const maxLength = pools.reduce((maximum, results) => Math.max(maximum, results.length), 0);
  const seen = new Set();
  const candidates = [];
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

function mergeUniqueCandidates(...candidateLists) {
  const seen = new Set();
  const merged = [];
  candidateLists.forEach((list) => {
    (list || []).forEach((item) => {
      const key = `${item.video_id}:${item.frame_idx}`;
      if (seen.has(key)) return;
      seen.add(key);
      merged.push(item);
    });
  });
  return merged;
}

function renderTemporalIntersection(limit, renderId) {
  const kisTarget = state.activeWorkspace === "kis_fusion"
    && state.kisFusionView === "sequence";
  const intersection = kisTarget
    ? state.kisTemporalIntersection
    : state.temporalIntersection;
  if (!intersection) return;
  const data = intersection.data;
  const fullKis = data.event_fusion_applied === true;
  const targetGrid = kisTarget ? el.kisFusionResultsGrid : el.resultsGrid;
  const targetCount = kisTarget ? el.kisFusionResultsCount : el.resultsCount;
  const sequences = (data.sequences || []).slice(0, limit);
  const allSequenceFrames = (data.sequences || [])
    .flatMap((sequence) => sequence.matched_events || []);
  state.searchResults = mergeUniqueCandidates(allSequenceFrames, getAllSearchCandidates());
  if (kisTarget) el.kisFusionResultsHeading.textContent = "ORDERED KIS RESULTS";
  else el.resultsHeading.textContent = "⛓ ORDERED SIGLIP INTERSECTION";
  targetCount.textContent = `${sequences.length} shown of ${data.ordered_sequence_count || 0} ordered video sequences`;
  targetGrid.replaceChildren();

  const overview = document.createElement("section");
  overview.className = "temporal-overview-section";
  const eventTrace = (data.event_pools || [])
    .map((event) => `E${event.order}: ${escapeHtml(event.query || "")}`)
    .join(" → ");
  overview.innerHTML = `
    <div class="video-drilldown-header">
      <div>
        <div class="video-drilldown-eyebrow">Same-video intersection · strictly increasing timestamps</div>
        <div class="video-drilldown-title">${fullKis ? "Ordered sequence search using full KIS Fusion per event" : "Ordered sequence search using SigLIP only"}</div>
        <div class="video-drilldown-query">${eventTrace}</div>
      </div>
      ${kisTarget ? "" : '<button type="button" class="btn-back-pools">← Back to four raw pools</button>'}
    </div>
    <div class="video-drilldown-audit">
      <span>${data.event_count} ${fullKis ? "full KIS event runs" : "independent event pools"}</span>
      <span>Top ${data.top_k_per_event} frames/event</span>
      <span>${data.paths_per_video || 1} path${Number(data.paths_per_video || 1) === 1 ? "" : "s"}/video</span>
      <span>${data.sequence_reservoir_count ?? data.ordered_sequence_count ?? 0}/${data.sequence_reservoir_size ?? data.top_k_sequences} sequence reservoir</span>
      <span>${data.intersection_video_count || 0} common videos</span>
      ${fullKis ? "" : `<span>shared-scene anchor ${data.anchor_query_applied ? "ON" : "OFF"}</span>`}
      <span>max consecutive gap ${data.max_gap_seconds}s</span>
      ${fullKis ? "<span>0-based canonical source-frame order</span>" : ""}
      <span>${fullKis ? "Branch 1 + Branch 2 + OCR + ASR" : "cross-modal fusion OFF"}</span>
      <span>${fullKis ? "score = weakest KIS final score" : "score = (anchor + weakest event) / 2"}</span>
      <span>${fullKis ? "BEiT-3 final rerank ON" : "reranking OFF"}</span>
    </div>`;
  overview.querySelector(".btn-back-pools")?.addEventListener("click", exitTemporalIntersection);
  targetGrid.appendChild(overview);

  if (sequences.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-placeholder temporal-empty-state";
    empty.innerHTML = `<div class="empty-icon">∅</div><div class="empty-title">No ordered intersection</div><div class="empty-desc">The event searches returned individual frames, but no common video contained every event in the requested timestamp order. Edit the event prompts or increase the maximum gap.</div>`;
    targetGrid.appendChild(empty);
    return;
  }

  sequences.forEach((sequence) => {
    const section = document.createElement("section");
    section.className = "temporal-sequence-section";
    const gaps = (sequence.gaps_seconds || []).map((gap) => `${Number(gap).toFixed(1)}s`).join(" · ");
    const sequenceIsComplete = Array.isArray(sequence.matched_events)
      && sequence.matched_events.length === Number(data.event_count);
    section.innerHTML = `
      <div class="temporal-sequence-header">
        <div>
          <div class="video-drilldown-eyebrow">Sequence #${sequence.rank}</div>
          <div class="temporal-sequence-title">${escapeHtml(sequence.video_id)}</div>
        </div>
        <div class="temporal-sequence-metrics">
          <span title="${fullKis ? "Primary sequence score: weakest event-level KIS final score" : "Primary sequence score: arithmetic mean of shared-scene anchor cosine and weakest event cosine"}">sequence ${Number(sequence.sequence_score || 0).toFixed(4)}</span>
          ${data.anchor_query_applied ? `<span title="Primary discovery rank: raw SigLIP cosine for the shared global scene">anchor cosine ${Number(sequence.context_anchor_score || 0).toFixed(4)}</span>` : ""}
          <span title="Primary sequence rank: the weakest event-level ${fullKis ? "KIS final score" : "raw SigLIP cosine"}">min ${fullKis ? "KIS" : "cosine"} ${Number(sequence.minimum_event_score || 0).toFixed(4)}</span>
          <span title="Secondary sequence rank: arithmetic mean of event-level ${fullKis ? "KIS final scores" : "raw SigLIP cosines"}">mean ${fullKis ? "KIS" : "cosine"} ${Number(sequence.mean_event_score || 0).toFixed(4)}</span>
          <span>span ${Number(sequence.span_seconds || 0).toFixed(1)}s</span>
          <span>gaps ${escapeHtml(gaps || "-")}</span>
          <span>global rank sum ${sequence.global_rank_sum}</span>
        </div>
        ${state.taskType === "TRAKE" && sequenceIsComplete ? '<button type="button" class="btn-add-sequence">+ Add whole sequence</button>' : ""}
      </div>`;
    section.querySelector(".btn-add-sequence")?.addEventListener("click", () => {
      void addCanonicalTrakeSequence(sequence.matched_events || [], sequence.video_id);
    });

    const eventGrid = document.createElement("div");
    eventGrid.className = "temporal-event-grid";
    (sequence.matched_events || []).forEach((match, matchIndex) => {
      const event = document.createElement("article");
      event.className = "temporal-event-step";
      event.innerHTML = `
        <div class="temporal-event-header">
          <span class="temporal-event-number">E${match.event_order}</span>
          <div>
            <div class="temporal-event-description">${escapeHtml(match.event_description || `Event ${match.event_order}`)}</div>
            <div class="temporal-event-query">${escapeHtml(match.event_query || "")}</div>
          </div>
        </div>`;
      const cardGrid = document.createElement("div");
      cardGrid.className = "temporal-event-card";
      renderStandardCards([match], cardGrid, fullKis ? "kis_fusion" : "siglip", renderId);
      event.appendChild(cardGrid);
      eventGrid.appendChild(event);

      if (matchIndex < sequence.matched_events.length - 1) {
        const arrow = document.createElement("div");
        arrow.className = "temporal-event-arrow";
        arrow.setAttribute("aria-hidden", "true");
        arrow.textContent = "→";
        eventGrid.appendChild(arrow);
      }
    });
    section.appendChild(eventGrid);
    targetGrid.appendChild(section);
  });
}

function renderDiscoveryCascade(limit, renderId) {
  const data = state.discoveryCascade.data;
  const cascades = data.cascades || [];
  const totalShown = cascades.reduce(
    (total, cascade) => total + Math.min(cascade.results?.length || 0, limit),
    0,
  );
  state.searchResults = getAllCascadeCandidates(cascades);
  el.resultsHeading.textContent = "🧭 EXPLICIT DISCOVERY CASCADE";
  el.resultsCount.textContent = `${totalShown} shown across ${cascades.length} independent object cascades`;

  const overview = document.createElement("section");
  overview.className = "cascade-overview-section";
  overview.innerHTML = `
    <div class="video-drilldown-header">
      <div>
        <div class="video-drilldown-eyebrow">Explicit cross-modal gating · scores never added</div>
        <div class="video-drilldown-title">DAM discovers videos; raw SigLIP cosine ranks frames</div>
        <div class="video-drilldown-query">Each objects_en entry is searched independently. Top ${data.dam_top_frames_per_object} raw DAM frames are deduplicated into video scopes; Top ${data.siglip_top_frames_per_video} SigLIP frames from each video are merged by unchanged cosine.</div>
      </div>
      <button type="button" class="btn-back-pools">← Back to four raw pools</button>
    </div>
    <div class="video-drilldown-audit">
      <span>${data.object_query_count} object cascades</span>
      <span>${data.unique_candidate_video_count} unique videos</span>
      <span>${data.unique_evaluated_frames} video frames evaluated</span>
      <span>DAM score excluded from final rank</span>
      <span>no score fusion</span>
      <span>no learned reranker</span>
    </div>`;
  overview.querySelector(".btn-back-pools").addEventListener("click", exitDiscoveryCascade);
  el.resultsGrid.appendChild(overview);

  cascades.forEach((cascade, cascadeIndex) => {
    const section = document.createElement("section");
    section.className = "modality-pool-section cascade-pool-section";
    const list = (cascade.results || []).slice(0, limit);
    const videoTrace = (cascade.candidate_videos || [])
      .map((candidate) => `${candidate.video_id} DAM#${candidate.dam_raw_frame_rank}`)
      .join(" · ");
    section.innerHTML = `
      <div class="modality-pool-header">
        <div>
          <div class="modality-pool-title">Object ${cascadeIndex + 1}: ${escapeHtml(cascade.object_query)}</div>
          <div class="modality-pool-meta">${cascade.dam_frames_considered} DAM frames → ${cascade.candidate_video_count} videos → ${cascade.siglip_frames_per_video} scoped SigLIP frames/video</div>
        </div>
        <span class="count-badge">${list.length} shown of ${cascade.result_count} · final rank: raw cosine</span>
      </div>
      <div class="cascade-video-trace" title="Candidate videos and their best raw DAM frame ranks">${escapeHtml(videoTrace)}</div>`;
    const grid = document.createElement("div");
    grid.className = "modality-pool-grid";
    if (list.length === 0) {
      grid.innerHTML = `<div class="pool-status-message">No scoped SigLIP frames returned for this object.</div>`;
    } else {
      renderStandardCards(list, grid, "cascade", renderId);
    }
    section.appendChild(grid);
    el.resultsGrid.appendChild(section);
  });
}

function renderVideoDrilldown(limit, renderId) {
  const drilldown = state.drilldown;
  const pool = drilldown.pool;
  const list = (pool.results || []).slice(0, limit);
  const sourceRank = Number.isFinite(Number(drilldown.sourceRank))
    ? `raw rank #${drilldown.sourceRank}`
    : "selected result";
  state.searchResults = pool.results || [];
  el.resultsHeading.textContent = "🔎 VIDEO-SCOPED SIGLIP";
  el.resultsCount.textContent = `${list.length} of ${pool.result_count} SigLIP results inside ${drilldown.videoId}`;

  const section = document.createElement("section");
  section.className = "video-drilldown-section";
  section.innerHTML = `
    <div class="video-drilldown-header">
      <div>
        <div class="video-drilldown-eyebrow">Manual drill-down from ${escapeHtml(String(drilldown.sourceModality).toUpperCase())} ${escapeHtml(sourceRank)}</div>
        <div class="video-drilldown-title">SigLIP cosine restricted to video ${escapeHtml(drilldown.videoId)}</div>
        <div class="video-drilldown-query">${escapeHtml(formatPoolQuery(pool))}</div>
      </div>
      <button type="button" class="btn-back-pools">← Back to four original pools</button>
    </div>
    <div class="video-drilldown-audit">
      <span>${drilldown.evaluatedFrames} frames evaluated</span>
      <span>raw cosine only</span>
      <span>no fusion</span>
      <span>no reranking</span>
      <span>source score not reused</span>
    </div>`;
  section.querySelector(".btn-back-pools").addEventListener("click", exitVideoDrilldown);

  const grid = document.createElement("div");
  grid.className = "modality-pool-grid";
  if (pool.status !== "ok" || list.length === 0) {
    grid.innerHTML = `<div class="pool-status-message">${escapeHtml(pool.reason || "No frames returned in this video.")}</div>`;
  } else {
    renderStandardCards(list, grid, "siglip", renderId);
  }
  section.appendChild(grid);
  el.resultsGrid.appendChild(section);
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
  if (modality === "kis_fusion") {
    return fusionEvidence(item);
  }
  if (modality === "visual_fusion") {
    const modelLabel = { siglip2: "SigLIP2", metaclip2: "MetaCLIP2", beit3: "BEiT-3" };
    const evidence = Object.entries(item.model_provenance || {}).map(([model, value]) => {
      if (!value?.observed) return `${modelLabel[model] || model} missing→0.000`;
      return `${modelLabel[model] || model} ${Number(value.raw_cosine || 0).toFixed(3)}→${Number(value.normalized_score || 0).toFixed(3)}`;
    });
    return evidence.join(" · ") || "SigLIP2 + MetaCLIP2 + BEIT3 visual fusion";
  }
  if (modality === "branch2") {
    const evidence = (raw, normalized, observed, digits = 3) => {
      if (observed === false || raw == null) return `missing→${Number(normalized || 0).toFixed(3)}`;
      return `${Number(raw).toFixed(digits)}→${Number(normalized || 0).toFixed(3)}`;
    };
    const dense = evidence(item.dense_raw, item.dense_normalized, item.dense_observed, 3);
    const sparse = evidence(item.sparse_raw, item.sparse_normalized, item.sparse_observed, 4);
    const beit = item.beit3_normalized == null ? "-" : `${Number(item.beit3_raw_cosine).toFixed(3)}→${Number(item.beit3_normalized).toFixed(3)}`;
    const delta = item.rank_delta == null ? "-" : (Number(item.rank_delta) > 0 ? `+${item.rank_delta}` : item.rank_delta);
    return `Dense ${dense} · Sparse ${sparse} · BEiT-3 ${beit} · rank Δ ${delta}`;
  }
  if (modality === "cascade") {
    return `Video via DAM raw rank #${item.dam_discovery_rank} at frame ${item.dam_discovery_frame_idx} · SigLIP rank #${item.video_scope_rank} inside ${item.video_id}`;
  }
  if (modality === "asr") {
    const transcript = item.transcript || item.asr_transcript || "No speech text";
    const role = item.asr_best_query_role ? ` · ${item.asr_best_query_role}${item.asr_best_query_language ? `:${item.asr_best_query_language}` : ""}` : "";
    const score = item.asr_normalized_score == null ? "" : ` · norm ${Number(item.asr_normalized_score).toFixed(3)}`;
    const bm25 = item.bm25_relevance == null ? "" : ` · BM25 ${Number(item.bm25_relevance).toFixed(3)}`;
    const token = item.token_coverage == null ? "" : ` · token ${Number(item.token_coverage).toFixed(3)}`;
    const ngram = item.ngram_coverage == null ? "" : ` · bigram ${Number(item.ngram_coverage).toFixed(3)}`;
    const span = item.asr_start_s == null ? "" : ` · ${Number(item.asr_start_s).toFixed(2)}-${Number(item.asr_end_s ?? item.asr_start_s).toFixed(2)}s`;
    return `${transcript}${role}${score}${bm25}${token}${ngram}${span}`;
  }
  if (modality === "ocr") {
    return formatOcrEvidence(item);
  }
  if (modality === "dam") {
    const subjects = (item.subject_scores || [])
      .map((entry) => {
        const status = entry.matched === false ? "unmatched" : "matched";
        const region = entry.best_region ? ` → ${entry.best_region}` : "";
        return `${entry.subject}: ${Number(entry.cosine).toFixed(4)} ${status}${region}`;
      })
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
    card.setAttribute("role", "group");
    card.setAttribute("aria-label", `${item.video_id}, frame ${item.frame_idx}, rank ${rank}`);

    const canDrillDown = !state.drilldown && modality !== "visual_fusion";
    const modalityLabel = modality === "visual_fusion" ? "VISUAL TRIO" : modality.toUpperCase();

    card.innerHTML = `
      <div class="card-media">
        <img src="${imgUrl}" alt="Keyframe from ${escapeHtml(item.video_id)}" loading="lazy" decoding="async">
        <span class="card-rank-badge">#${rank}</span>
        <span class="card-modality-badge">${escapeHtml(modalityLabel)}</span>
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
          ${modality === "cascade" ? `<span class="pill-tag">DAM gate #${item.dam_discovery_rank}</span><span class="pill-tag">video SigLIP #${item.video_scope_rank}</span>` : ""}
        </div>
        <button type="button" class="btn-add-submission-card">+ Add to submission</button>
        ${canDrillDown ? `<button type="button" class="btn-video-drilldown">Search inside this video</button>` : ""}
      </div>`;

    const image = card.querySelector("img");
    image.addEventListener("error", () => {
      image.hidden = true;
      image.parentElement.classList.add("img-fallback");
    }, { once: true });
    const openCard = () => void openStandardInspector(item);
    const addButton = card.querySelector(".btn-add-submission-card");
    addButton.addEventListener("click", (event) => {
      event.stopPropagation();
      void addFrameToSubmission(item, { source: modality, eventOrder: item.event_order });
    });
    addButton.addEventListener("keydown", (event) => event.stopPropagation());
    const drilldownButton = card.querySelector(".btn-video-drilldown");
    if (drilldownButton) {
      drilldownButton.addEventListener("click", (event) => {
        event.stopPropagation();
        void openStandardInspector(item).then(() => {
          el.inspectorVideoQuery.value = String(item.event_query || currentVisualQuery());
          el.inspectorVideoQuery.focus();
        });
      });
      drilldownButton.addEventListener("keydown", (event) => event.stopPropagation());
    }
    card.addEventListener("click", openCard);
    card.addEventListener("keydown", (event) => {
      if (isInteractiveTarget(event.target)) return;
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
  const openRequestId = ++inspectorOpenRequestId;
  let canonicalItem;
  try {
    canonicalItem = await resolveCanonicalFrame(item);
  } catch (error) {
    if (openRequestId === inspectorOpenRequestId) showToast(error.message, "error");
    return;
  }
  if (openRequestId !== inspectorOpenRequestId) return;
  item = canonicalItem;
  if (el.modal.classList.contains("hidden")) lastInspectorFocus = document.activeElement;
  const videoChanged = state.activeInspectorItem?.video_id !== item.video_id;
  state.activeInspectorItem = item;
  if (videoChanged) {
    state.inspectorLocalResults = [];
    el.inspectorLocalResults.replaceChildren();
    el.inspectorLocalResults.classList.add("hidden");
    el.inspectorVideoQuery.value = "";
  }
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

  // KIS keeps ASR evidence under branch_provenance when a visual voter was
  // encountered first. Prefer that winning segment over macro context.
  const kisBranchEvidence = item.retrieval_modality === "kis_fusion"
    ? (item.branch_provenance || {})
    : {};
  const asrEvidence = item.retrieval_modality === "kis_fusion"
    ? (kisBranchEvidence.asr || {})
    : item;
  const ocrEvidence = item.retrieval_modality === "kis_fusion"
    ? (kisBranchEvidence.ocr || {})
    : item;
  if (item.retrieval_modality === "kis_fusion") {
    // The remaining inspector code predates nested branch evidence.  Merge a
    // shallow, local view so its existing score fields resolve to the winning
    // ASR/OCR evidence without mutating the final response held in state.
    item = { ...item, ...asrEvidence, ...ocrEvidence };
  }
  el.inspAsrText.textContent = asrEvidence.asr_transcript
    || asrEvidence.transcript
    || "(No speech / silent frame)";
  if (el.inspAsrEvidence) {
    if (asrEvidence.asr_best_query_role || asrEvidence.bm25_raw != null) {
      const roles = asrEvidence.asr_stream_provenance || {};
      const observed = Object.entries(roles)
        .map(([role, evidence]) => `${role}: ${evidence ? Number(evidence.combined_score ?? 0).toFixed(3) : "not observed"}`)
        .join(" · ");
      el.inspAsrEvidence.textContent = [
        `winner=${asrEvidence.asr_best_query_role || "-"}`,
        `combined=${Number(asrEvidence.asr_raw_score ?? 0).toFixed(4)}`,
        `normalized=${Number(asrEvidence.asr_normalized_score ?? asrEvidence.score ?? 0).toFixed(4)}`,
        // KIS keeps ASR and OCR lexical evidence in separate nested branch
        // objects.  Do not let the local inspector merge make OCR's BM25
        // fields overwrite the winning ASR evidence.
        `BM25=${Number(asrEvidence.bm25_raw ?? 0).toFixed(4)} → ${Number(asrEvidence.bm25_relevance ?? 0).toFixed(4)}`,
        `token=${Number(asrEvidence.token_coverage ?? 0).toFixed(4)}`,
        `bigram=${Number(asrEvidence.ngram_coverage ?? asrEvidence.adjacent_bigram_coverage ?? 0).toFixed(4)}`,
        asrEvidence.asr_segment_id ? `segment=${asrEvidence.asr_segment_id}` : "",
        asrEvidence.asr_start_s != null
          ? `span=${Number(asrEvidence.asr_start_s).toFixed(2)}-${Number(asrEvidence.asr_end_s ?? asrEvidence.asr_start_s).toFixed(2)}s`
          : "",
        observed ? `streams: ${observed}` : "",
      ].filter(Boolean).join(" · ");
    } else {
      el.inspAsrEvidence.textContent = "(No ASR ranking evidence)";
    }
  }
  if (el.inspAsrContext) el.inspAsrContext.textContent = "(Loading macro audio context...)";
  el.inspDamText.textContent = item.retrieval_modality === "kis_fusion"
    ? fusionInspectorEvidence(item)
    : item.retrieval_modality === "visual_fusion"
      ? resultEvidence(item, "visual_fusion")
      : item.dam_summary || "(No visual description available)";
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
        if (frameIdentity(active) !== frameIdentity(item)) return;
        state.activeBBoxObjects = data.dam_objects || [];
        if (el.inspAsrContext) {
          el.inspAsrContext.textContent = data.macro_audio_transcript || "(No macro audio context)";
        }
        // Preserve OCR evidence returned by the winning result.  The detail
        // endpoint is allowed to fill an empty card field, but must never
        // replace the text that actually contributed to ranking.
        const nestedOcrEvidence = active.retrieval_modality === "kis_fusion"
          && active.branch_provenance?.ocr
          && (
            active.branch_provenance.ocr.ocr_best_query_role != null
            || active.branch_provenance.ocr.ocr_stream_provenance != null
            || active.branch_provenance.ocr.ocr_raw_score != null
            || Object.prototype.hasOwnProperty.call(active.branch_provenance.ocr, "ocr_text")
          );
        const winningOcrEvidence = active.retrieval_modality === "ocr"
          || active.ocr_best_query_role != null
          || active.ocr_stream_provenance != null
          || active.ocr_raw_score != null
          || Boolean(nestedOcrEvidence);
        if (el.inspOcrText && !winningOcrEvidence && !active.ocr_text && data.keyframe) {
          const detailOcrText = resolveWinningOcrText(
            active,
            data.keyframe,
            winningOcrEvidence,
          );
          if (detailOcrText) el.inspOcrText.textContent = detailOcrText;
        }
        if (data.keyframe?.dam_summary_en && item.retrieval_modality !== "kis_fusion") {
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
    state.filmstripSelection.clear();
    state.filmstripSelectionAnchor = null;
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
  updateFilmstripSelectionUi();
  if (!keyframes.length) return;

  const activeIndex = Math.max(0, keyframes.findIndex((kf) => kf.keyframe_n === currentKeyframeN));
  const start = Math.max(0, activeIndex - FILMSTRIP_WINDOW_RADIUS);
  const end = Math.min(keyframes.length, activeIndex + FILMSTRIP_WINDOW_RADIUS + 1);
  const fragment = document.createDocumentFragment();

  if (start > 0) fragment.appendChild(createFilmstripJump(keyframes, start, -1));
  keyframes.slice(start, end).forEach((kf, offset) => {
    fragment.appendChild(createFilmstripItem(kf, currentKeyframeN, start + offset));
  });
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

function createFilmstripItem(kf, currentKeyframeN, absoluteIndex) {
  const isActive = kf.keyframe_n === currentKeyframeN;
  const identity = frameIdentity(kf);
  const isSelected = state.filmstripSelection.has(identity);
  const item = document.createElement("div");
  item.className = "filmstrip-item" + (isActive ? " active" : "") + (isSelected ? " selected" : "");
  item.dataset.keyframeN = kf.keyframe_n;
  item.dataset.frameUid = identity;
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
  label.innerHTML = `<strong>F${kf.frame_idx}</strong><small>KF ${kf.keyframe_n}</small>`;
  const addButton = document.createElement("button");
  addButton.type = "button";
  addButton.className = "filmstrip-add";
  addButton.textContent = isSelected ? "✓" : "+";
  addButton.title = "Select this frame; Shift-click selects the range";
  addButton.setAttribute("aria-label", `Select exact frame ${kf.frame_idx}`);
  addButton.setAttribute("aria-pressed", String(isSelected));
  addButton.addEventListener("click", (event) => {
    event.stopPropagation();
    toggleFilmstripSelection(kf, absoluteIndex, event.shiftKey);
  });
  item.append(image, label, addButton);

  item.addEventListener("click", () => selectFilmstripKeyframe(kf));
  item.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      selectFilmstripKeyframe(kf);
    }
  });
  return item;
}

function updateFilmstripSelectionUi() {
  const count = state.filmstripSelection.size;
  el.filmstripCount.textContent = `${state.activeVideoKeyframes.length} keyframes · ${count} selected`;
  el.btnClearFilmstripSelection.disabled = count === 0;
  el.btnAddFilmstripSelection.disabled = count === 0;
  el.btnAddFilmstripSelection.textContent = `Add selected (${count})`;
  el.filmstripScroll.querySelectorAll(".filmstrip-item").forEach((item) => {
    const selected = state.filmstripSelection.has(
      /** @type {HTMLElement} */ (item).dataset.frameUid || "",
    );
    item.classList.toggle("selected", selected);
    const button = item.querySelector(".filmstrip-add");
    if (button) {
      button.textContent = selected ? "✓" : "+";
      button.setAttribute("aria-pressed", String(selected));
    }
  });
}

function toggleFilmstripSelection(frame, absoluteIndex, selectRange = false) {
  const identity = frameIdentity(frame);
  if (!identity) return;
  if (selectRange && Number.isInteger(state.filmstripSelectionAnchor)) {
    const start = Math.min(state.filmstripSelectionAnchor, absoluteIndex);
    const end = Math.max(state.filmstripSelectionAnchor, absoluteIndex);
    for (let index = start; index <= end && state.filmstripSelection.size < 100; index += 1) {
      const candidateIdentity = frameIdentity(state.activeVideoKeyframes[index]);
      if (candidateIdentity) state.filmstripSelection.add(candidateIdentity);
    }
  } else if (state.filmstripSelection.has(identity)) {
    state.filmstripSelection.delete(identity);
  } else if (state.filmstripSelection.size < 100) {
    state.filmstripSelection.add(identity);
  } else {
    showToast("A submission can contain at most 100 selected frames.", "error");
  }
  state.filmstripSelectionAnchor = absoluteIndex;
  updateFilmstripSelectionUi();
}

function clearFilmstripSelection() {
  state.filmstripSelection.clear();
  state.filmstripSelectionAnchor = null;
  updateFilmstripSelectionUi();
}

async function addFilmstripSelection() {
  const selectedFrames = state.activeVideoKeyframes.filter((frame) => (
    state.filmstripSelection.has(frameIdentity(frame))
  ));
  if (!selectedFrames.length) return;
  const startSnapshot = submissionStore.getSnapshot();
  el.btnAddFilmstripSelection.disabled = true;
  el.btnAddFilmstripSelection.textContent = "Verifying exact frames…";
  try {
    const canonicalFrames = await Promise.all(
      selectedFrames.map((frame) => resolveCanonicalFrame(frame)),
    );
    const snapshot = submissionStore.getSnapshot();
    if (
      snapshot.contextKey !== startSnapshot.contextKey
      || snapshot.mode !== startSnapshot.mode
    ) return;
    if (snapshot.mode === "TRAKE") {
      const draft = currentSubmissionDraft(snapshot);
      const eventCount = draft.events?.length || 0;
      if (eventCount < 2 || canonicalFrames.length !== eventCount) {
        throw new Error(
          `TRAKE needs exactly ${eventCount || "the configured number of"} selected frames, one for each sequencing prompt.`,
        );
      }
      const sequence = [...canonicalFrames]
        .sort((left, right) => Number(left.pts_time_s) - Number(right.pts_time_s))
        .map((frame, index) => ({ ...frame, event_order: index + 1 }));
      const result = submissionStore.addSequence(sequence, {
        source: "filmstrip-range",
        validation: "canonical",
      });
      if (!result.ok) {
        throw new Error("The selected frames must belong to one video and increase in exact frame order.");
      }
      showToast(`Added ${sequence.length} exact frames as one ordered TRAKE sequence.`, "success");
    } else {
      const result = submissionStore.addFrames(canonicalFrames, {
        source: "filmstrip-range",
        validation: "canonical",
      });
      if (!result.ok) throw new Error("No new exact frame could be added to the draft.");
      showToast(`Added ${result.added} exact frame${result.added === 1 ? "" : "s"}.`, "success");
      if (result.firstManual) {
        void fillRelatedSubmissionFrames(
          canonicalFrames[0],
          snapshot.contextKey,
          snapshot.mode,
        );
      }
    }
    clearFilmstripSelection();
  } catch (error) {
    showToast(error.message, "error");
    updateFilmstripSelectionUi();
  }
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
  inspectorOpenRequestId += 1;
  inspectorScopedAbortController?.abort();
  inspectorScopedRequestId += 1;
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
  state.filmstripSelection.clear();
  state.filmstripSelectionAnchor = null;
  updateFilmstripSelectionUi();
  state.activeBBoxObjects = [];
  state.activeInspectorItem = null;
  drawBBoxesOnCanvas();
  if (lastInspectorFocus instanceof HTMLElement) lastInspectorFocus.focus();
  lastInspectorFocus = null;
}

function toggleCurrentInSubmission() {
  if (!state.activeInspectorItem) return;
  const item = state.activeInspectorItem;
  const snapshot = submissionStore.getSnapshot();
  const draft = currentSubmissionDraft(snapshot);
  const eventOrder = snapshot.mode === "TRAKE" ? draft.activeEvent : null;
  if (submissionStore.hasFrame(item, snapshot.mode, eventOrder)) {
    removeSubmissionFrame(snapshot.mode === "TRAKE" ? eventOrder : frameIdentity(item));
    showToast(`Removed ${item.video_id}, ${item.frame_idx} from the draft.`);
  } else {
    void addFrameToSubmission(item, { source: item.retrieval_modality || "inspector", eventOrder });
  }
  updateInspectorSubmitBtn();
}

function updateInspectorSubmitBtn() {
  if (!state.activeInspectorItem) return;
  const item = state.activeInspectorItem;
  const snapshot = submissionStore.getSnapshot();
  const eventOrder = snapshot.mode === "TRAKE" ? currentSubmissionDraft(snapshot).activeEvent : null;
  const isSelected = submissionStore.hasFrame(item, snapshot.mode, eventOrder);

  el.btnToggleInSubmission.classList.toggle("in-submit", isSelected);
  el.btnToggleInSubmission.innerHTML = isSelected
    ? `<span>✓ In ${snapshot.mode === "TRAKE" ? `E${eventOrder}` : "submission"}</span>`
    : `<span>+ Add to ${snapshot.mode === "TRAKE" ? `E${eventOrder}` : "submission"}</span>`;
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
// Official BTC headerless submission CSV exporter (maximum 100 rows)
// ──────────────────────────────────────────────────────────────────────────────
function openExportModal() {
  if (submissionItemCount() === 0) {
    showToast("Add at least one human-selected frame before preparing a submission.", "error");
    return;
  }
  lastExportFocus = document.activeElement;
  const title = document.getElementById("export-modal-title");
  if (title) title.textContent = state.taskType === "TRAKE"
    ? "📥 Review TRAKE 100-Row Sequence CSV"
    : `📥 Review ${state.taskType === "VQA" ? "Q&A" : state.taskType} 100-Row Submission CSV`;
  el.exportModal.classList.remove("hidden");
  setBackgroundInert(el.exportModal, true);
  el.exportQueryId.value = currentSubmissionDraft().queryId || "1";
  void updateExportPreview();
  requestAnimationFrame(() => el.exportQueryId.focus({ preventScroll: true }));
}

function closeExportModal() {
  el.exportModal.classList.add("hidden");
  setBackgroundInert(el.exportModal, false);
  if (lastExportFocus instanceof HTMLElement) lastExportFocus.focus();
  lastExportFocus = null;
}

async function updateExportPreview() {
  exportReviewAbortController?.abort();
  exportReviewAbortController = null;
  const qId = el.exportQueryId ? el.exportQueryId.value.trim() || "1" : "1";
  submissionStore.setQueryId(qId);
  if (el.exportQueryId) {
    el.exportQueryId.setCustomValidity(isValidQueryId(qId)
      ? ""
      : "Use only letters, numbers, underscores, and hyphens.");
  }
  const requestId = ++exportPrepareRequestId;
  preparedExport = null;
  el.exportRow1Preview.textContent = "Preparing canonical rows…";
  el.exportCsvPreview.value = "";
  el.exportSchemaWarning.textContent = "";
  el.btnDownloadCsvAction.disabled = true;
  if (!isValidQueryId(qId)) return;
  let prepared;
  try {
    prepared = await prepareSubmission(qId);
  } catch (error) {
    if (requestId !== exportPrepareRequestId) return;
    preparedExport = null;
    el.exportRow1Preview.textContent = "Submission preparation failed";
    el.exportSchemaWarning.textContent = error.message;
    el.btnDownloadCsvAction.disabled = true;
    el.btnRevalidateCsv.disabled = true;
    showToast(error.message, "error");
    return;
  }
  if (requestId !== exportPrepareRequestId) return;
  preparedExport = prepared;
  const csvContent = previewCsvContent(prepared);
  const rowCount = preparedCsvRowCount(prepared, csvContent);
  el.exportCsvPreview.value = csvContent;
  csvReviewDirty = false;
  el.btnRevalidateCsv.disabled = true;
  const verification = prepared.server_verified ? "server-validated" : "client fallback · not server-verified";
  const expected = state.taskType === "TRAKE" ? "100 complete ordered sequences" : "100 rows";
  el.exportRow1Preview.textContent = `${rowCount} row${rowCount === 1 ? "" : "s"} · ${verification} · target: ${expected} (1–100 accepted)`;
  const messages = [...(prepared.warnings || []), ...(prepared.errors || [])];
  messages.unshift(submissionSchemaDefaults[state.taskType]);
  el.exportSchemaWarning.textContent = messages.join(" · ");
  if (el.btnDownloadCsvAction) {
    const downloadable = canDownloadPreparedSubmission(prepared);
    el.btnDownloadCsvAction.disabled = !downloadable;
    const label = el.btnDownloadCsvAction.querySelector("span");
    if (label) label.textContent = downloadable
      ? `⚡ Download CSV (${rowCount} Rows)`
      : `Waiting for a server-validated ${state.taskType === "TRAKE" ? "sequence " : ""}CSV`;
  }
}

function serverOfficialCsvContent(prepared) {
  return prepared?.server_verified === true && typeof prepared?.official_csv?.content === "string"
    ? prepared.official_csv.content
    : "";
}

function previewCsvContent(prepared) {
  const officialContent = serverOfficialCsvContent(prepared);
  if (officialContent) return officialContent;
  const lines = serializeOfficialSubmissionRows(prepared?.rows || [], state.taskType);
  return lines.length ? `${lines.join("\n")}\n` : "";
}

function preparedCsvRowCount(prepared, content = "") {
  const declared = Number(prepared?.official_csv?.row_count ?? prepared?.row_count);
  if (Number.isInteger(declared) && declared >= 0) return declared;
  try {
    return parseCsvRecords(content).length;
  } catch {
    return 0;
  }
}

function canDownloadPreparedSubmission(prepared) {
  const content = serverOfficialCsvContent(prepared);
  const rowCount = preparedCsvRowCount(prepared, content);
  return Boolean(
    content
    && rowCount >= 1
    && rowCount <= 100
    && !(prepared?.errors || []).length
    && prepared?.official_csv?.valid === true
    && prepared?.valid_for_download === true,
  );
}

function parseCsvRecords(text) {
  const records = [];
  let record = [];
  let field = "";
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (quoted) {
      if (character === '"' && text[index + 1] === '"') {
        field += '"';
        index += 1;
      } else if (character === '"') quoted = false;
      else field += character;
      continue;
    }
    if (character === '"' && field === "") quoted = true;
    else if (character === ",") {
      record.push(field.trim());
      field = "";
    } else if (character === "\n" || character === "\r") {
      if (character === "\r" && text[index + 1] === "\n") index += 1;
      record.push(field.trim());
      field = "";
      if (record.some((value) => value !== "")) records.push(record);
      record = [];
    } else field += character;
  }
  if (field || record.length) {
    record.push(field.trim());
    if (record.some((value) => value !== "")) records.push(record);
  }
  if (quoted) throw new Error("CSV contains an unclosed quoted field.");
  return records;
}

function editedCsvPrepareRequest(records, queryId) {
  const mode = state.taskType;
  if (records.length < 1 || records.length > 100) {
    throw new Error(`${mode === "VQA" ? "Q&A" : mode} CSV must contain between 1 and 100 data rows.`);
  }
  if (mode === "TRAKE") {
    const configuredEventCount = currentSubmissionDraft().events?.length || 0;
    const eventCount = configuredEventCount || Math.max(0, (records[0]?.length || 0) - 1);
    if (eventCount < 2) throw new Error("TRAKE needs at least two event frame columns.");
    const manualSequences = records.map((record, rowIndex) => {
      if (record.length !== eventCount + 1) {
        throw new Error(`TRAKE row ${rowIndex + 1} must contain video_id plus exactly ${eventCount} event frame indexes.`);
      }
      const videoId = normalizeRequestedVideoId(record[0]);
      if (!videoId) throw new Error(`TRAKE row ${rowIndex + 1} has an invalid video ID.`);
      const events = record.slice(1).map((value, index) => {
        const frameIdx = Number(value);
        if (!Number.isInteger(frameIdx) || frameIdx < 0) throw new Error(`TRAKE row ${rowIndex + 1}, E${index + 1} has an invalid frame index.`);
        return { event_order: index + 1, video_id: videoId, frame_idx: frameIdx };
      });
      return { video_id: videoId, events };
    });
    return {
      task_type: "TRAKE",
      mode: "review",
      query_id: queryId,
      target_rows: records.length,
      event_count: eventCount,
      manual_selections: [],
      candidate_reservoir: [],
      manual_sequences: manualSequences,
      candidate_sequences: [],
    };
  }
  const manualSelections = records.map((record, rowIndex) => {
    const requiredColumns = mode === "VQA" ? 3 : 2;
    if (record.length !== requiredColumns) throw new Error(`Row ${rowIndex + 1} must have ${requiredColumns} columns for ${mode}.`);
    const videoId = normalizeRequestedVideoId(record[0]);
    const frameIdx = Number(record[1]);
    if (!videoId || !Number.isInteger(frameIdx) || frameIdx < 0) throw new Error(`Row ${rowIndex + 1} has an invalid video or frame index.`);
    return { video_id: videoId, frame_idx: frameIdx };
  });
  const answer = mode === "VQA" ? records[0][2] : undefined;
  if (mode === "VQA" && (!answer || records.some((record) => record[2] !== answer))) {
    throw new Error("Q&A requires one non-empty human answer shared by every row.");
  }
  if (mode === "VQA" && Array.from(answer).length > 100) throw new Error("Q&A answer cannot exceed 100 characters.");
  return {
    task_type: mode,
    mode: "review",
    query_id: queryId,
    target_rows: records.length,
    manual_selections: manualSelections,
    candidate_reservoir: [],
    ...(mode === "VQA" ? { vqa_answer: answer } : {}),
  };
}

async function revalidateEditedCsv() {
  const queryId = el.exportQueryId.value.trim() || "1";
  const requestId = ++exportPrepareRequestId;
  const reviewGeneration = csvReviewGeneration;
  const reviewMode = state.taskType;
  const reviewContext = submissionStore.getSnapshot().contextKey;
  exportReviewAbortController?.abort();
  exportReviewAbortController = new AbortController();
  const reviewController = exportReviewAbortController;
  const isCurrentReview = () => requestId === exportPrepareRequestId
    && reviewGeneration === csvReviewGeneration
    && reviewMode === state.taskType
    && reviewContext === submissionStore.getSnapshot().contextKey
    && queryId === (el.exportQueryId.value.trim() || "1")
    && csvReviewDirty;
  el.btnRevalidateCsv.disabled = true;
  el.exportSchemaWarning.textContent = "Revalidating every edited row against the canonical index…";
  try {
    const records = parseCsvRecords(el.exportCsvPreview.value);
    const request = editedCsvPrepareRequest(records, queryId);
    const response = await fetch("/api/submission/prepare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
      signal: reviewController.signal,
    });
    if (!response.ok) throw await responseError(response, "CSV revalidation failed");
    const prepared = await response.json();
    if (!isCurrentReview()) return;
    preparedExport = { ...prepared, server_verified: true };
    if (!canDownloadPreparedSubmission(preparedExport)) {
      throw new Error((prepared.errors || []).join(" · ") || "The server did not return a valid official CSV payload.");
    }
    const csvContent = serverOfficialCsvContent(preparedExport);
    const rowCount = preparedCsvRowCount(preparedExport, csvContent);
    el.exportCsvPreview.value = csvContent;
    csvReviewDirty = false;
    el.btnDownloadCsvAction.disabled = false;
    el.exportRow1Preview.textContent = `${rowCount} row${rowCount === 1 ? "" : "s"} · server-validated after manual editing`;
    el.exportSchemaWarning.textContent = (prepared.warnings || []).join(" · ");
    showToast("Edited CSV rows passed canonical server validation.", "success");
  } catch (error) {
    if (error.name === "AbortError" || !isCurrentReview()) return;
    preparedExport = null;
    csvReviewDirty = true;
    el.btnDownloadCsvAction.disabled = true;
    el.btnRevalidateCsv.disabled = false;
    el.exportSchemaWarning.textContent = error.message;
    showToast(error.message, "error");
  } finally {
    if (exportReviewAbortController === reviewController) exportReviewAbortController = null;
  }
}

function isValidQueryId(queryId) {
  return /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(queryId);
}

function activeCandidateReservoir() {
  let candidates = [];
  if (state.activeWorkspace === "image") {
    candidates = state.imageResults;
  } else if (state.activeWorkspace === "branch1") {
    candidates = state.branch1Results;
  } else if (state.activeWorkspace === "branch2") {
    candidates = state.branch2Results;
  } else if (state.activeWorkspace === "branch3_asr") {
    candidates = state.branch3AsrResults;
  } else if (state.activeWorkspace === "branch3_ocr") {
    candidates = state.branch3OcrResults;
  } else if (state.activeWorkspace === "kis_fusion") {
    candidates = state.kisFusionView === "sequence" && state.kisTemporalIntersection
      ? [
        ...(state.kisTemporalIntersection?.data?.sequences || []),
        ...(state.kisTemporalIntersection?.data?.reserve_sequences || []),
      ].flatMap((sequence) => sequence.matched_events || [])
      : state.kisFusionResults;
  } else if (state.activeWorkspace === "video") {
    candidates = state.watch.searchResults;
  } else if (state.temporalIntersection) {
    candidates = [
      ...(state.temporalIntersection.data.sequences || []),
      ...(state.temporalIntersection.data.reserve_sequences || []),
    ].flatMap((sequence) => sequence.matched_events || []);
  } else if (state.discoveryCascade) {
    candidates = getAllCascadeCandidates(state.discoveryCascade.data.cascades || []);
  } else if (state.drilldown) {
    candidates = state.drilldown.pool?.results || [];
  } else if (state.activeModality === "all") {
    // Explicit round-robin across visible raw pools; scores remain incomparable and unfused.
    candidates = state.searchResults;
  } else {
    candidates = state.modalityResults[state.activeModality]?.results || [];
  }
  const contextKey = submissionStore.getSnapshot().contextKey;
  const scopedBackfill = state.exportBackfillByContext[contextKey] || [];
  return mergeUniqueCandidates(candidates, scopedBackfill);
}

function compactFrame(item) {
  return {
    video_id: item.video_id,
    frame_idx: Number(item.frame_idx),
    keyframe_n: Number(item.keyframe_n) || null,
    pts_time_s: Number.isFinite(Number(item.pts_time_s)) ? Number(item.pts_time_s) : null,
    image_relpath: item.image_relpath || "",
    preview_image_relpath: item.preview_image_relpath || "",
    indexed_keyframe: item.indexed_keyframe === true,
    validation: item.validation || "unverified",
    frame_index_base: item.frame_index_base ?? null,
    max_frame_idx: item.max_frame_idx ?? null,
    timing_method: item.timing_method || "",
    preview_frame_idx: item.preview_frame_idx ?? null,
    preview_keyframe_n: item.preview_keyframe_n ?? null,
    preview_pts_time_s: item.preview_pts_time_s ?? null,
    related_seed_frame_idx: item.related_seed_frame_idx ?? null,
    source: item.source || item.retrieval_modality || "candidate",
  };
}

function normalizeTrakeSequence(sequence, eventCount) {
  const sourceEvents = sequence?.events || sequence?.matched_events || [];
  if (!Number.isInteger(eventCount) || eventCount < 1 || sourceEvents.length !== eventCount) return null;
  const events = sourceEvents
    .map((item, index) => ({ event_order: Number(item.event_order) || index + 1, ...compactFrame(item) }))
    .sort((left, right) => left.event_order - right.event_order);
  if (events.some((event, index) => (
    event.event_order !== index + 1
    || !Number.isInteger(event.frame_idx)
    || event.frame_idx < 0
  ))) return null;
  const videoId = normalizeRequestedVideoId(sequence.video_id || events[0]?.video_id);
  if (!videoId || events.some((event) => normalizeRequestedVideoId(event.video_id || videoId) !== videoId)) return null;
  if (events.some((event, index) => index > 0 && event.frame_idx <= events[index - 1].frame_idx)) return null;
  return {
    video_id: videoId,
    events: events.map((event) => ({ ...event, video_id: videoId })),
  };
}

function trakeSequenceIdentity(sequence) {
  return `${sequence.video_id}:${sequence.events.map((event) => event.frame_idx).join(":")}`;
}

function orderedCandidateSequences(eventCount, excluded = new Set()) {
  const intersection = state.activeWorkspace === "kis_fusion"
    && state.kisFusionView === "sequence"
    ? state.kisTemporalIntersection
    : state.temporalIntersection;
  const rawSequences = [
    ...(intersection?.data?.sequences || []),
    ...(intersection?.data?.reserve_sequences || []),
  ];
  const seen = new Set(excluded);
  return rawSequences.flatMap((sequence) => {
    const normalized = normalizeTrakeSequence(sequence, eventCount);
    if (!normalized) return [];
    const identity = trakeSequenceIdentity(normalized);
    if (seen.has(identity)) return [];
    seen.add(identity);
    return [normalized];
  }).slice(0, 500);
}

function buildSubmissionPrepareRequest(queryId) {
  const snapshot = submissionStore.getSnapshot();
  const draft = currentSubmissionDraft(snapshot);
  const request = {
    task_type: snapshot.mode,
    mode: "exact_100",
    query_id: queryId,
    target_rows: 100,
    manual_selections: snapshot.mode === "TRAKE" ? [] : (draft.items || []).map(compactFrame),
    candidate_reservoir: mergeUniqueCandidates(
      draft.suggestedItems || [],
      activeCandidateReservoir(),
    ).map(compactFrame),
  };
  if (snapshot.mode === "VQA") request.vqa_answer = draft.answer || "";
  if (snapshot.mode === "TRAKE") {
    const ordered = orderedTrakeFrames(draft);
    const eventCount = draft.events?.length || ordered.length;
    request.event_count = eventCount;
    const manualSequence = normalizeTrakeSequence({
      video_id: ordered[0]?.item.video_id || "",
      events: ordered.map(({ order, item }) => ({ event_order: order, ...compactFrame(item) })),
    }, eventCount);
    if (!manualSequence) {
      throw new Error(`Complete all ${eventCount || "requested"} TRAKE events in one video with strictly increasing frame indexes before preparing the CSV.`);
    }
    request.manual_sequences = [manualSequence];
    request.candidate_sequences = orderedCandidateSequences(
      eventCount,
      new Set([trakeSequenceIdentity(manualSequence)]),
    );
  }
  return request;
}

async function prepareSubmission(queryId) {
  const contextKey = submissionStore.getSnapshot().contextKey;
  const request = buildSubmissionPrepareRequest(queryId);
  let response;
  try {
    response = await fetch("/api/submission/prepare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
  } catch (error) {
    console.warn("Server submission preparation is unreachable; preview only:", error);
    if (request.task_type !== "TRAKE") {
      await ensureClientBackfill(request.manual_selections, request.candidate_reservoir, 100, contextKey);
    }
    return prepareSubmissionClientFallback(queryId);
  }
  if (response.ok) {
    const data = await response.json();
    return { ...data, server_verified: true };
  }
  if (![404, 501].includes(response.status)) {
    throw await responseError(response, "Submission preparation failed");
  }
  if (request.task_type !== "TRAKE") {
    await ensureClientBackfill(request.manual_selections, request.candidate_reservoir, 100, contextKey);
  }
  return prepareSubmissionClientFallback(queryId);
}

async function ensureClientBackfill(manualSelections, candidates, target, contextKey) {
  const seeds = mergeUniqueCandidates(manualSelections, candidates).slice(0, 8);
  const seenVideos = new Set();
  const backfill = [];
  for (const seed of seeds) {
    if (backfill.length >= target) break;
    if (!seed?.video_id || seenVideos.has(seed.video_id)) continue;
    seenVideos.add(seed.video_id);
    try {
      const timeline = await fetchVideoTimeline(seed.video_id);
      const seedTime = Number(seed.pts_time_s);
      const ordered = [...timeline.keyframes].sort((left, right) => {
        if (Number.isFinite(seedTime)) {
          const delta = Math.abs(Number(left.pts_time_s) - seedTime) - Math.abs(Number(right.pts_time_s) - seedTime);
          if (delta) return delta;
        }
        return Number(left.keyframe_n) - Number(right.keyframe_n);
      });
      backfill.push(...ordered.map((frame) => ({ ...frame, source: "canonical-neighbour", validation: "canonical" })));
    } catch {
      // Other shortlisted videos can still provide a safe client-side fallback.
    }
  }
  state.exportBackfillByContext[contextKey] = mergeUniqueCandidates(
    state.exportBackfillByContext[contextKey] || [],
    backfill,
  );
}

function prepareSubmissionClientFallback(queryId) {
  const snapshot = submissionStore.getSnapshot();
  const draft = currentSubmissionDraft(snapshot);
  if (snapshot.mode === "TRAKE") return prepareTrakeClientFallback(queryId, draft);
  const rows = [];
  const seen = new Set();
  const addCandidate = (candidate) => {
    if (!candidate?.video_id || !Number.isInteger(Number(candidate.frame_idx)) || rows.length >= 100) return;
    const frameIndex = Number(candidate.frame_idx);
    if (frameIndex < 0) return;
    const key = `${candidate.video_id}:${frameIndex}`;
    if (seen.has(key)) return;
    seen.add(key);
    rows.push({
      video_id: candidate.video_id,
      frame_idx: frameIndex,
      answer: snapshot.mode === "VQA" ? draft.answer || "" : undefined,
    });
  };
  submissionDraftItems(draft).forEach(addCandidate);
  activeCandidateReservoir().forEach(addCandidate);
  const answerMissing = snapshot.mode === "VQA" && !String(draft.answer || "").trim();
  return {
    ok: false,
    task_type: snapshot.mode,
    query_id: queryId,
    row_count: rows.length,
    complete: false,
    missing_rows: Math.max(0, 100 - rows.length),
    rows,
    warnings: ["Server preparation endpoint unavailable; rows are canonical client candidates but are not server-verified."],
    errors: [
      "Download is disabled until /api/submission/prepare validates every row.",
      ...(answerMissing ? ["A human Q&A answer is required."] : []),
    ],
    server_verified: false,
  };
}

function prepareTrakeClientFallback(queryId, draft) {
  const ordered = orderedTrakeFrames(draft);
  const expected = draft.events?.length || ordered.length;
  const manualSequence = normalizeTrakeSequence({
    video_id: ordered[0]?.item.video_id || "",
    events: ordered.map(({ order, item }) => ({ event_order: order, ...compactFrame(item) })),
  }, expected);
  const sequences = [];
  const seen = new Set();
  if (manualSequence) {
    sequences.push(manualSequence);
    seen.add(trakeSequenceIdentity(manualSequence));
  }
  sequences.push(...orderedCandidateSequences(expected, seen));
  const rows = sequences.slice(0, 100);
  return {
    ok: false,
    task_type: "TRAKE",
    query_id: queryId,
    row_count: rows.length,
    complete: false,
    missing_rows: Math.max(0, 100 - rows.length),
    rows,
    warnings: ["Preview contains only complete ordered sequences returned by the current search; no rows were fabricated."],
    errors: [
      "Download is disabled until /api/submission/prepare validates the complete ordered sequence rows.",
      ...(!manualSequence ? [`Select all ${expected} event frames from one video in strictly increasing frame order.`] : []),
    ],
    server_verified: false,
  };
}

function executeDownload100Csv() {
  const qId = el.exportQueryId ? el.exportQueryId.value.trim() || "1" : "1";
  if (!isValidQueryId(qId)) {
    showToast("Query ID may only contain letters, numbers, underscores, and hyphens.", "error");
    el.exportQueryId.focus();
    return;
  }
  if (!canDownloadPreparedSubmission(preparedExport)) {
    showToast("Prepare a server-validated official CSV before downloading.", "error");
    return;
  }
  const csvContent = serverOfficialCsvContent(preparedExport);
  const rowCount = preparedCsvRowCount(preparedExport, csvContent);
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  
  const link = document.createElement("a");
  link.setAttribute("href", url);
  link.setAttribute("download", submissionFilename(qId, state.taskType));
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);

  closeExportModal();
  showToast(`📥 Exported ${rowCount} official submission rows for ${qId}.`, "success");
}

// ──────────────────────────────────────────────────────────────────────────────
// Run App
// ──────────────────────────────────────────────────────────────────────────────
if (document.readyState === "loading") {
  window.addEventListener("DOMContentLoaded", initApp, { once: true });
} else {
  void initApp();
}
