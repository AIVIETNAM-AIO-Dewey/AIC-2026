const STORAGE_VERSION = 4;
const STORAGE_KEY = "aic2026.submission-workbench.v4";
const LEGACY_STORAGE_KEY = "aic2026.submission-workbench.v3";
const TASK_MODES = ["KIS", "VQA", "TRAKE"];
const MAX_SUBMISSION_FRAMES = 100;
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
    suggestedItems: [],
    relatedSeed: null,
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
  if (value === null || value === undefined || value === "") return fallback;
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
  const pointId = finiteNumber(item.point_id ?? item.global_idx);
  const fps = finiteNumber(item.fps);
  const maxFrameIdx = finiteNumber(item.max_frame_idx);
  const previewFrameIdx = finiteNumber(item.preview_frame_idx);
  const previewKeyframeN = finiteNumber(item.preview_keyframe_n);
  const previewPtsTimeS = finiteNumber(item.preview_pts_time_s);
  const relatedSeedFrameIdx = finiteNumber(item.related_seed_frame_idx);
  return {
    video_id: videoId,
    frame_idx: frameIdx,
    frame_uid: identity,
    keyframe_n: Number.isInteger(keyframeN) && keyframeN >= 0 ? keyframeN : null,
    pts_time_s: ptsTimeS !== null && ptsTimeS >= 0 ? ptsTimeS : null,
    fps: fps !== null && fps > 0 ? fps : null,
    point_id: Number.isInteger(pointId) && pointId > 0 ? pointId : null,
    image_relpath: String(item.image_relpath || ""),
    preview_image_relpath: String(item.preview_image_relpath || ""),
    indexed_keyframe: item.indexed_keyframe === true
      || item.validation === "canonical"
      || context.validation === "canonical",
    frame_index_base: Number(item.frame_index_base) === 0 ? 0 : null,
    max_frame_idx: Number.isInteger(maxFrameIdx) && maxFrameIdx >= frameIdx
      ? maxFrameIdx
      : null,
    duration_s: finiteNumber(item.duration_s),
    timing_method: String(item.timing_method || ""),
    preview_frame_idx: Number.isInteger(previewFrameIdx) && previewFrameIdx >= 0
      ? previewFrameIdx
      : null,
    preview_keyframe_n: Number.isInteger(previewKeyframeN) && previewKeyframeN >= 0
      ? previewKeyframeN
      : null,
    preview_pts_time_s: previewPtsTimeS !== null && previewPtsTimeS >= 0
      ? previewPtsTimeS
      : null,
    related_seed_frame_idx: Number.isInteger(relatedSeedFrameIdx) && relatedSeedFrameIdx >= 0
      ? relatedSeedFrameIdx
      : null,
    score,
    related_rank: finiteNumber(item.related_rank),
    related_score: finiteNumber(item.related_score),
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
  draft.suggestedItems = (Array.isArray(raw?.suggestedItems) ? raw.suggestedItems : [])
    .flatMap((item) => {
      const normalized = normalizeSubmissionFrame(item, {
        source: item?.source || "auto-related",
        manual: false,
        validation: item?.validation,
        addedAt: item?.added_at,
      });
      const identity = frameIdentity(normalized);
      if (!normalized || !identity || seen.has(identity)) return [];
      seen.add(identity);
      return [normalized];
    })
    .slice(0, Math.max(0, MAX_SUBMISSION_FRAMES - draft.items.length));
  const relatedSeed = frameIdentity(raw?.relatedSeed);
  draft.relatedSeed = relatedSeed ? sanitizeStoredFrame(raw.relatedSeed) : null;
  return draft;
}

function loadState(storage, key) {
  const fallback = emptyState();
  if (!storage) return fallback;
  try {
    const serialized = storage.getItem(key)
      || (key === STORAGE_KEY ? storage.getItem(LEGACY_STORAGE_KEY) : null);
    const raw = JSON.parse(serialized || "null");
    if (!raw || ![3, STORAGE_VERSION].includes(raw.version)) return fallback;
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
      const conflicts = Object.entries(draft.eventSlots).some(([candidateOrder, candidate]) => {
        if (Number(candidateOrder) === order) return false;
        if (frameIdentity(candidate).split(":")[0] !== frame.video_id) return true;
        return Number(candidateOrder) < order
          ? Number(candidate.frame_idx) >= frame.frame_idx
          : Number(candidate.frame_idx) <= frame.frame_idx;
      });
      if (conflicts) return { ok: false, reason: "invalid-trake-order" };
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
    const suggestedIndex = draft.suggestedItems.findIndex(
      (candidate) => frameIdentity(candidate) === identity,
    );
    if (suggestedIndex >= 0) draft.suggestedItems.splice(suggestedIndex, 1);
    if (
      draft.items.length + draft.suggestedItems.length >= MAX_SUBMISSION_FRAMES
      && draft.suggestedItems.length
    ) draft.suggestedItems.pop();
    if (draft.items.length + draft.suggestedItems.length >= MAX_SUBMISSION_FRAMES) {
      return { ok: false, reason: "draft-full", maxFrames: MAX_SUBMISSION_FRAMES };
    }
    const firstManual = draft.items.length === 0;
    draft.items.push(frame);
    emit();
    return { ok: true, mode, existing: false, firstManual, frame: clone(frame) };
  }

  function addFrames(items, context = {}) {
    if (!Array.isArray(items) || items.length === 0) return { ok: false, reason: "empty-frames" };
    const mode = context.mode && TASK_MODES.includes(context.mode) ? context.mode : data.mode;
    if (mode === "TRAKE") return { ok: false, reason: "use-trake-sequence" };
    const draft = contextDrafts()[mode];
    let added = 0;
    let existing = 0;
    let invalid = 0;
    for (const item of items) {
      const frame = normalizeSubmissionFrame(item, context);
      const identity = frameIdentity(frame);
      if (!frame || !identity) {
        invalid += 1;
        continue;
      }
      const manualIndex = draft.items.findIndex(
        (candidate) => frameIdentity(candidate) === identity,
      );
      if (manualIndex >= 0) {
        draft.items[manualIndex] = { ...draft.items[manualIndex], ...frame, manual: true };
        existing += 1;
        continue;
      }
      draft.suggestedItems = draft.suggestedItems.filter(
        (candidate) => frameIdentity(candidate) !== identity,
      );
      if (
        draft.items.length + draft.suggestedItems.length >= MAX_SUBMISSION_FRAMES
        && draft.suggestedItems.length
      ) draft.suggestedItems.pop();
      if (draft.items.length + draft.suggestedItems.length >= MAX_SUBMISSION_FRAMES) break;
      draft.items.push(frame);
      added += 1;
    }
    if (added || existing) emit();
    return {
      ok: added > 0 || existing > 0,
      mode,
      added,
      existing,
      invalid,
      firstManual: added > 0 && draft.items.length === added,
    };
  }

  function setRelatedFrames(seed, items, mode = data.mode) {
    if (!TASK_MODES.includes(mode) || mode === "TRAKE") {
      return { ok: false, reason: "unsupported-mode" };
    }
    const draft = contextDrafts()[mode];
    const normalizedSeed = normalizeSubmissionFrame(seed, {
      source: "related-seed",
      manual: true,
      validation: seed?.validation === "source_timeline" ? "source_timeline" : "canonical",
    });
    const seedIdentity = frameIdentity(normalizedSeed);
    if (!seedIdentity || frameIdentity(draft.items[0]) !== seedIdentity) {
      return { ok: false, reason: "stale-seed" };
    }
    const manualIdentities = new Set(draft.items.map(frameIdentity));
    const seen = new Set(manualIdentities);
    const capacity = Math.max(0, MAX_SUBMISSION_FRAMES - draft.items.length);
    draft.suggestedItems = (Array.isArray(items) ? items : []).flatMap((item) => {
      const normalized = normalizeSubmissionFrame(item, {
        source: "auto-related",
        manual: false,
        validation: "canonical",
      });
      const identity = frameIdentity(normalized);
      if (!normalized || !identity || seen.has(identity)) return [];
      seen.add(identity);
      return [normalized];
    }).slice(0, capacity);
    draft.relatedSeed = normalizedSeed;
    emit();
    return { ok: true, count: draft.suggestedItems.length };
  }

  function clearRelatedFrames(mode = data.mode) {
    if (!TASK_MODES.includes(mode) || mode === "TRAKE") return;
    const draft = contextDrafts()[mode];
    if (!draft.suggestedItems.length && !draft.relatedSeed) return;
    draft.suggestedItems = [];
    draft.relatedSeed = null;
    emit();
  }

  function addSequence(items, context = {}) {
    if (!Array.isArray(items) || items.length === 0) return { ok: false, reason: "empty-sequence" };
    const draft = contextDrafts().TRAKE;
    const normalized = items.map((item, index) => ({
      order: Number.parseInt(item?.event_order, 10) || index + 1,
      frame: normalizeSubmissionFrame(item, context),
    })).sort((left, right) => left.order - right.order);
    if (normalized.some(({ frame }) => !frame)) return { ok: false, reason: "invalid-frame" };
    if (draft.events.length && normalized.length !== draft.events.length) {
      return { ok: false, reason: "event-count-mismatch" };
    }
    const videoIds = new Set(normalized.map(({ frame }) => frame.video_id));
    const ordered = normalized.every(({ order, frame }, index) => (
      order === index + 1
      && (index === 0 || frame.frame_idx > normalized[index - 1].frame.frame_idx)
    ));
    if (videoIds.size !== 1 || !ordered) {
      return { ok: false, reason: "invalid-trake-order" };
    }
    normalized.forEach(({ order, frame }) => {
      draft.eventSlots[String(Math.max(1, order))] = frame;
    });
    draft.activeEvent = normalized[normalized.length - 1].order;
    emit();
    return { ok: true, count: normalized.length };
  }

  function removeFrame(identityOrOrder) {
    const draft = currentDraft();
    if (data.mode === "TRAKE") {
      delete draft.eventSlots[String(identityOrOrder)];
      emit();
      return;
    }
    const removedSeed = frameIdentity(draft.relatedSeed) === identityOrOrder;
    draft.items = draft.items.filter((item) => frameIdentity(item) !== identityOrOrder);
    draft.suggestedItems = draft.suggestedItems.filter(
      (item) => frameIdentity(item) !== identityOrOrder,
    );
    if (removedSeed) {
      draft.relatedSeed = null;
      draft.suggestedItems = [];
    }
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
    return [...draft.items, ...draft.suggestedItems]
      .some((candidate) => frameIdentity(candidate) === identity);
  }

  return {
    addFrame,
    addFrames,
    addSequence,
    clear,
    clearRelatedFrames,
    getSnapshot,
    hasFrame,
    removeFrame,
    reorderFrame,
    setActiveEvent,
    setAnswer,
    setContext,
    setMode,
    setQueryId,
    setRelatedFrames,
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
