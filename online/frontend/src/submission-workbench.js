const STORAGE_VERSION = 3;
const STORAGE_KEY = "aic2026.submission-workbench.v3";
const TASK_MODES = ["KIS", "VQA", "TRAKE"];
export const MAX_VQA_ANSWER_CHARACTERS = 100;

export function normalizeVqaAnswer(value) {
  return Array.from(String(value || "")).slice(0, MAX_VQA_ANSWER_CHARACTERS).join("");
}

function emptyDraft(mode) {
  if (mode === "TRAKE") {
    return {
      activeEvent: 1,
      events: [],
      eventSlots: {},
      queryId: "1",
    };
  }
  return {
    answer: "",
    items: [],
    queryId: "1",
  };
}

function emptyState() {
  return {
    version: STORAGE_VERSION,
    mode: "KIS",
    contextKey: "default",
    contexts: {
      default: emptyContext(),
    },
  };
}

function emptyContext() {
  return {
    KIS: emptyDraft("KIS"),
    VQA: emptyDraft("VQA"),
    TRAKE: emptyDraft("TRAKE"),
  };
}

function finiteNumber(value, fallback = null) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

export function normalizeVideoId(value) {
  return String(value || "").trim().toUpperCase().replaceAll("-", "_");
}

export function frameIdentity(item) {
  const videoId = normalizeVideoId(item?.video_id);
  const frameIdx = finiteNumber(item?.frame_idx);
  if (!videoId || !Number.isInteger(frameIdx) || frameIdx < 0) return "";
  return `${videoId}:${frameIdx}`;
}

export function normalizeSubmissionFrame(item, context = {}) {
  const identity = frameIdentity(item);
  if (!identity) return null;
  const videoId = normalizeVideoId(item.video_id);
  const frameIdx = Number(item.frame_idx);
  const keyframeN = finiteNumber(item.keyframe_n);
  const ptsTimeS = finiteNumber(item.pts_time_s);
  const score = finiteNumber(item.score);
  return {
    video_id: videoId,
    frame_idx: frameIdx,
    keyframe_n: Number.isInteger(keyframeN) && keyframeN > 0 ? keyframeN : null,
    pts_time_s: ptsTimeS !== null && ptsTimeS >= 0 ? ptsTimeS : null,
    image_relpath: String(item.image_relpath || ""),
    score,
    rank: finiteNumber(item.rank),
    source: String(context.source || item.retrieval_modality || item.source || "manual"),
    manual: context.manual !== false,
    validation: String(context.validation || item.validation || "unverified"),
    added_at: String(context.addedAt || new Date().toISOString()),
  };
}

function sanitizeStoredFrame(item) {
  const normalized = normalizeSubmissionFrame(item, {
    source: item?.source,
    manual: item?.manual !== false,
    validation: item?.validation,
    addedAt: item?.added_at,
  });
  return normalized;
}

function sanitizeDraft(mode, raw) {
  const draft = emptyDraft(mode);
  draft.queryId = String(raw?.queryId || "1").slice(0, 64);
  if (mode === "TRAKE") {
    draft.activeEvent = Math.max(1, Number.parseInt(raw?.activeEvent, 10) || 1);
    draft.events = Array.isArray(raw?.events)
      ? raw.events.map((event, index) => ({
        order: Math.max(1, Number.parseInt(event?.order, 10) || index + 1),
        label: String(event?.label || event?.description || `Event ${index + 1}`).slice(0, 300),
      }))
      : [];
    Object.entries(raw?.eventSlots || {}).forEach(([order, item]) => {
      const normalized = sanitizeStoredFrame(item);
      if (normalized) draft.eventSlots[String(Math.max(1, Number.parseInt(order, 10) || 1))] = normalized;
    });
    return draft;
  }
  draft.answer = normalizeVqaAnswer(raw?.answer);
  const seen = new Set();
  draft.items = (Array.isArray(raw?.items) ? raw.items : []).flatMap((item) => {
    const normalized = sanitizeStoredFrame(item);
    const identity = frameIdentity(normalized);
    if (!normalized || !identity || seen.has(identity)) return [];
    seen.add(identity);
    return [normalized];
  });
  return draft;
}

function loadState(storage, key) {
  const fallback = emptyState();
  if (!storage) return fallback;
  try {
    const raw = JSON.parse(storage.getItem(key) || "null");
    if (!raw || raw.version !== STORAGE_VERSION) return fallback;
    const mode = TASK_MODES.includes(raw.mode) ? raw.mode : "KIS";
    const contexts = {};
    Object.entries(raw.contexts || {}).slice(0, 100).forEach(([contextKey, drafts]) => {
      contexts[String(contextKey).slice(0, 200)] = {
        KIS: sanitizeDraft("KIS", drafts?.KIS),
        VQA: sanitizeDraft("VQA", drafts?.VQA),
        TRAKE: sanitizeDraft("TRAKE", drafts?.TRAKE),
      };
    });
    const contextKey = String(raw.contextKey || "default").slice(0, 200);
    if (!contexts[contextKey]) contexts[contextKey] = emptyContext();
    return {
      version: STORAGE_VERSION,
      mode,
      contextKey,
      contexts,
    };
  } catch {
    return fallback;
  }
}

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

export function createSubmissionStore(options = {}) {
  const storage = options.storage === undefined
    ? (typeof window !== "undefined" ? window.localStorage : null)
    : options.storage;
  const storageKey = options.storageKey || STORAGE_KEY;
  let data = loadState(storage, storageKey);
  const listeners = new Set();

  function persist() {
    try {
      storage?.setItem(storageKey, JSON.stringify(data));
    } catch {
      // A full/disabled localStorage must not block retrieval or submission editing.
    }
  }

  function emit() {
    persist();
    const snapshot = getSnapshot();
    listeners.forEach((listener) => listener(snapshot));
  }

  function contextDrafts() {
    if (!data.contexts[data.contextKey]) data.contexts[data.contextKey] = emptyContext();
    return data.contexts[data.contextKey];
  }

  function getSnapshot() {
    return clone({
      version: data.version,
      mode: data.mode,
      contextKey: data.contextKey,
      drafts: contextDrafts(),
    });
  }

  function currentDraft() {
    return contextDrafts()[data.mode];
  }

  function setContext(value) {
    const contextKey = String(value || "default").trim().slice(0, 200) || "default";
    if (contextKey === data.contextKey) return;
    data.contextKey = contextKey;
    if (!data.contexts[contextKey]) data.contexts[contextKey] = emptyContext();
    emit();
  }

  function setMode(mode) {
    if (!TASK_MODES.includes(mode) || data.mode === mode) return;
    data.mode = mode;
    emit();
  }

  function setQueryId(value) {
    currentDraft().queryId = String(value || "").slice(0, 64);
    emit();
  }

  function setAnswer(value) {
    if (data.mode !== "VQA") return { ok: false, reason: "wrong-mode" };
    const answer = String(value || "");
    if (Array.from(answer).length > MAX_VQA_ANSWER_CHARACTERS) {
      return { ok: false, reason: "answer-too-long", maxCharacters: MAX_VQA_ANSWER_CHARACTERS };
    }
    currentDraft().answer = answer;
    emit();
    return { ok: true };
  }

  function setTrakeEvents(events) {
    const draft = contextDrafts().TRAKE;
    draft.events = (Array.isArray(events) ? events : []).map((event, index) => ({
      order: Math.max(1, Number.parseInt(event?.order, 10) || index + 1),
      label: String(event?.label || event?.description || `Event ${index + 1}`).slice(0, 300),
    }));
    const allowedOrders = new Set(draft.events.map((event) => String(event.order)));
    Object.keys(draft.eventSlots).forEach((order) => {
      if (!allowedOrders.has(order)) delete draft.eventSlots[order];
    });
    if (draft.events.length && !draft.events.some((event) => event.order === draft.activeEvent)) {
      draft.activeEvent = draft.events[0].order;
    }
    emit();
  }

  function setActiveEvent(order) {
    const next = Math.max(1, Number.parseInt(order, 10) || 1);
    contextDrafts().TRAKE.activeEvent = next;
    emit();
  }

  function addFrame(item, context = {}) {
    const frame = normalizeSubmissionFrame(item, context);
    if (!frame) return { ok: false, reason: "invalid-frame" };
    const mode = context.mode && TASK_MODES.includes(context.mode) ? context.mode : data.mode;
    const draft = contextDrafts()[mode];
    if (mode === "TRAKE") {
      const order = Math.max(1, Number.parseInt(context.eventOrder, 10) || draft.activeEvent || 1);
      draft.eventSlots[String(order)] = frame;
      draft.activeEvent = order;
      emit();
      return { ok: true, mode, eventOrder: order, frame: clone(frame) };
    }
    const identity = frameIdentity(frame);
    const existingIndex = draft.items.findIndex((candidate) => frameIdentity(candidate) === identity);
    if (existingIndex >= 0) {
      draft.items[existingIndex] = { ...draft.items[existingIndex], ...frame, manual: true };
      emit();
      return { ok: true, mode, existing: true, frame: clone(frame) };
    }
    draft.items.push(frame);
    emit();
    return { ok: true, mode, existing: false, frame: clone(frame) };
  }

  function addSequence(items, context = {}) {
    if (!Array.isArray(items) || items.length === 0) return { ok: false, reason: "empty-sequence" };
    const results = items.map((item, index) => addFrame(item, {
      ...context,
      mode: "TRAKE",
      eventOrder: Number.parseInt(item?.event_order, 10) || index + 1,
    }));
    return { ok: results.every((result) => result.ok), results };
  }

  function removeFrame(identityOrOrder) {
    const draft = currentDraft();
    if (data.mode === "TRAKE") {
      delete draft.eventSlots[String(identityOrOrder)];
      emit();
      return;
    }
    draft.items = draft.items.filter((item) => frameIdentity(item) !== identityOrOrder);
    emit();
  }

  function reorderFrame(fromIndex, toIndex) {
    const draft = currentDraft();
    if (data.mode === "TRAKE") return;
    const from = Number.parseInt(fromIndex, 10);
    const to = Number.parseInt(toIndex, 10);
    if (!Number.isInteger(from) || !Number.isInteger(to)
      || from < 0 || to < 0 || from >= draft.items.length || to >= draft.items.length
      || from === to) return;
    const [item] = draft.items.splice(from, 1);
    draft.items.splice(to, 0, item);
    emit();
  }

  function updateFrame(identityOrOrder, patch) {
    const draft = currentDraft();
    const applyPatch = (item) => sanitizeStoredFrame({ ...item, ...patch, manual: true });
    if (data.mode === "TRAKE") {
      const order = String(identityOrOrder);
      const next = applyPatch(draft.eventSlots[order]);
      if (next) draft.eventSlots[order] = next;
      emit();
      return Boolean(next);
    }
    const index = draft.items.findIndex((item) => frameIdentity(item) === identityOrOrder);
    if (index < 0) return false;
    const next = applyPatch(draft.items[index]);
    if (!next) return false;
    const duplicate = draft.items.some((item, candidateIndex) => (
      candidateIndex !== index && frameIdentity(item) === frameIdentity(next)
    ));
    if (duplicate) return false;
    draft.items[index] = next;
    emit();
    return true;
  }

  function clear(mode = data.mode) {
    if (!TASK_MODES.includes(mode)) return;
    const drafts = contextDrafts();
    const queryId = drafts[mode]?.queryId || "1";
    drafts[mode] = emptyDraft(mode);
    drafts[mode].queryId = queryId;
    emit();
  }

  function hasFrame(item, mode = data.mode, eventOrder = null) {
    const identity = frameIdentity(item);
    if (!identity) return false;
    const draft = contextDrafts()[mode];
    if (mode === "TRAKE") {
      if (eventOrder !== null) return frameIdentity(draft.eventSlots[String(eventOrder)]) === identity;
      return Object.values(draft.eventSlots).some((candidate) => frameIdentity(candidate) === identity);
    }
    return draft.items.some((candidate) => frameIdentity(candidate) === identity);
  }

  return {
    addFrame,
    addSequence,
    clear,
    getSnapshot,
    hasFrame,
    removeFrame,
    reorderFrame,
    setActiveEvent,
    setAnswer,
    setContext,
    setMode,
    setQueryId,
    setTrakeEvents,
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    updateFrame,
  };
}

export function orderedTrakeFrames(draft) {
  return Object.entries(draft?.eventSlots || {})
    .map(([order, item]) => ({ order: Number(order), item }))
    .filter(({ order, item }) => Number.isInteger(order) && frameIdentity(item))
    .sort((left, right) => left.order - right.order);
}

export const submissionSchemaDefaults = Object.freeze({
  KIS: "video_id, frame_idx · no header · up to 100 rows",
  VQA: "video_id, frame_idx, answer · answer max 100 characters · no header",
  TRAKE: "video_id, frame_E1, …, frame_EN · one complete ordered sequence per row · up to 100 rows",
});

export function csvCell(value) {
  const text = String(value ?? "");
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

export function serializeOfficialSubmissionRows(rows, mode) {
  return (Array.isArray(rows) ? rows : []).slice(0, 100).map((row) => {
    if (Array.isArray(row)) return row.map(csvCell).join(",");
    if (!row || typeof row !== "object") throw new Error("Submission rows must contain structured frame data.");
    if (mode === "VQA") {
      if (Array.from(String(row.answer || "")).length > MAX_VQA_ANSWER_CHARACTERS) {
        throw new Error("Q&A answer cannot exceed 100 characters.");
      }
      return [row.video_id, row.frame_idx, row.answer || ""].map(csvCell).join(",");
    }
    if (mode === "TRAKE") {
      const events = row.events || row.matched_events || [];
      if (events.length) return [row.video_id, ...events.map((event) => event.frame_idx)].map(csvCell).join(",");
      if (Array.isArray(row.frame_indices)) return [row.video_id, ...row.frame_indices].map(csvCell).join(",");
      throw new Error("A TRAKE row must contain one frame index for every event.");
    }
    return [row.video_id, row.frame_idx].map(csvCell).join(",");
  });
}

export function submissionFilename(queryFileId, mode) {
  const suffix = mode === "VQA" ? "qa" : String(mode || "kis").toLowerCase();
  const requested = String(queryFileId || "")
    .trim()
    .replace(/\.(?:txt|csv)$/i, "")
    .replace(/[\u0000-\u001f<>:"/\\|?*]+/g, "_")
    .replace(/\s+/g, "_")
    .replace(/^\.+|\.+$/g, "")
    .slice(0, 120);
  const stem = requested || "submission";
  const suffixedStem = new RegExp(`(?:^|[-_])${suffix}$`, "i").test(stem) ? stem : `${stem}-${suffix}`;
  return `${suffixedStem}.csv`;
}
