import assert from "node:assert/strict";
import test from "node:test";

import {
  createSubmissionStore,
  frameIdentity,
  MAX_VQA_ANSWER_CHARACTERS,
  orderedTrakeFrames,
  serializeOfficialSubmissionRows,
  submissionFilename,
} from "../src/submission-workbench.js";

function memoryStorage() {
  const values = new Map();
  return {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
  };
}

const frame = (video, index, keyframe = index) => ({
  video_id: video,
  frame_idx: index,
  keyframe_n: keyframe,
  pts_time_s: index / 25,
});

test("submission store keeps manual order, deduplicates, edits, and persists", () => {
  const storage = memoryStorage();
  const store = createSubmissionStore({ storage, storageKey: "test" });
  assert.equal(store.addFrame(frame("l25-v060", 100), { source: "siglip" }).ok, true);
  store.addFrame(frame("L25_V060", 200), { source: "video-player" });
  store.addFrame(frame("L25_V060", 100), { source: "ocr" });
  assert.deepEqual(store.getSnapshot().drafts.KIS.items.map(frameIdentity), [
    "L25_V060:100",
    "L25_V060:200",
  ]);
  store.reorderFrame(1, 0);
  assert.deepEqual(store.getSnapshot().drafts.KIS.items.map(frameIdentity), [
    "L25_V060:200",
    "L25_V060:100",
  ]);
  assert.equal(store.updateFrame("L25_V060:200", { frame_idx: 220 }), true);

  const restored = createSubmissionStore({ storage, storageKey: "test" });
  assert.deepEqual(restored.getSnapshot().drafts.KIS.items.map(frameIdentity), [
    "L25_V060:220",
    "L25_V060:100",
  ]);
});

test("verified arbitrary source-frame metadata remains distinct from its indexed preview", () => {
  const storage = memoryStorage();
  const store = createSubmissionStore({ storage, storageKey: "source-frame" });
  const sourceFrame = {
    video_id: "L25_V001",
    frame_idx: 34,
    keyframe_n: null,
    pts_time_s: 1.36,
    fps: 25,
    image_relpath: "",
    indexed_keyframe: false,
    validation: "source_timeline",
    frame_index_base: 0,
    max_frame_idx: 249,
    preview_frame_idx: 17,
    preview_keyframe_n: 1,
    preview_pts_time_s: 0.68,
    preview_image_relpath: "keyframes/L25_V001/00000017.jpg",
    related_seed_frame_idx: 17,
  };
  store.addFrame(sourceFrame, { validation: "source_timeline" });
  const restored = createSubmissionStore({ storage, storageKey: "source-frame" });
  const item = restored.getSnapshot().drafts.KIS.items[0];
  assert.equal(item.frame_uid, "L25_V001:34");
  assert.equal(item.frame_idx, 34);
  assert.equal(item.keyframe_n, null);
  assert.equal(item.validation, "source_timeline");
  assert.equal(item.preview_frame_idx, 17);
  assert.equal(item.preview_image_relpath, "keyframes/L25_V001/00000017.jpg");
  assert.equal(item.related_seed_frame_idx, 17);
});

test("VQA has a human answer and TRAKE fills explicit event slots", () => {
  const store = createSubmissionStore({ storage: memoryStorage(), storageKey: "tasks" });
  store.setMode("VQA");
  store.setAnswer("câu trả lời của người dùng");
  store.addFrame(frame("L01_V001", 10));
  assert.equal(store.getSnapshot().drafts.VQA.answer, "câu trả lời của người dùng");

  store.setMode("TRAKE");
  store.setTrakeEvents([{ order: 1, label: "first" }, { order: 2, label: "second" }]);
  store.addFrame(frame("L01_V001", 10), { eventOrder: 1 });
  store.addFrame(frame("L01_V001", 20), { eventOrder: 2 });
  assert.deepEqual(
    orderedTrakeFrames(store.getSnapshot().drafts.TRAKE).map(({ order, item }) => [order, item.frame_idx]),
    [[1, 10], [2, 20]],
  );

  store.setTrakeEvents([{ order: 1, label: "first only" }]);
  assert.deepEqual(Object.keys(store.getSnapshot().drafts.TRAKE.eventSlots), ["1"]);
});

test("TRAKE rejects cross-video and non-increasing frame selections before export", () => {
  const store = createSubmissionStore({ storage: memoryStorage(), storageKey: "trake-order" });
  store.setMode("TRAKE");
  store.setTrakeEvents([{ order: 1, label: "first" }, { order: 2, label: "second" }]);
  assert.equal(store.addFrame(frame("L01_V001", 20), { eventOrder: 2 }).ok, true);
  assert.equal(
    store.addFrame(frame("L02_V001", 10), { eventOrder: 1 }).reason,
    "invalid-trake-order",
  );
  assert.equal(
    store.addFrame(frame("L01_V001", 30), { eventOrder: 1 }).reason,
    "invalid-trake-order",
  );
  assert.equal(store.addFrame(frame("L01_V001", 10), { eventOrder: 1 }).ok, true);

  assert.equal(store.addSequence([
    { ...frame("L01_V001", 10), event_order: 1 },
    { ...frame("L02_V001", 20), event_order: 2 },
  ]).reason, "invalid-trake-order");
  assert.equal(store.addSequence([
    { ...frame("L01_V001", 30), event_order: 1 },
    { ...frame("L01_V001", 20), event_order: 2 },
  ]).reason, "invalid-trake-order");
});

test("Q&A answer blocks values beyond the official 100-character limit", () => {
  const store = createSubmissionStore({ storage: memoryStorage(), storageKey: "answer-limit" });
  store.setMode("VQA");
  assert.equal(store.setAnswer("🐟".repeat(MAX_VQA_ANSWER_CHARACTERS)).ok, true);
  const rejected = store.setAnswer("🐟".repeat(MAX_VQA_ANSWER_CHARACTERS + 1));
  assert.equal(rejected.ok, false);
  assert.equal(rejected.reason, "answer-too-long");
  assert.equal(Array.from(store.getSnapshot().drafts.VQA.answer).length, MAX_VQA_ANSWER_CHARACTERS);
});

test("official CSV rows are headerless, omit query ID, and quote special answers", () => {
  assert.deepEqual(serializeOfficialSubmissionRows([
    { query_id: "ignored", video_id: "L00_V000", frame_idx: 1234 },
  ], "KIS"), ["L00_V000,1234"]);
  assert.deepEqual(serializeOfficialSubmissionRows([
    { query_id: "ignored", video_id: "L01_V028", frame_idx: 3450, answer: 'Có 3 người, anh ấy nói "Xin chào"' },
  ], "VQA"), ['L01_V028,3450,"Có 3 người, anh ấy nói ""Xin chào"""']);
  assert.deepEqual(serializeOfficialSubmissionRows([
    {
      query_id: "ignored",
      video_id: "L10_V001",
      events: [{ frame_idx: 1200 }, { frame_idx: 1850 }, { frame_idx: 2100 }, { frame_idx: 2450 }],
    },
  ], "TRAKE"), ["L10_V001,1200,1850,2100,2450"]);
});

test("query identifier is filename metadata and uses the official qa suffix fallback", () => {
  assert.equal(submissionFilename("query-4-trake.txt", "TRAKE"), "query-4-trake.csv");
  assert.equal(submissionFilename("", "VQA"), "submission-qa.csv");
  assert.equal(submissionFilename("folder/query 3 qa.csv", "VQA"), "folder_query_3_qa.csv");
  assert.equal(submissionFilename("1", "KIS"), "1-kis.csv");
});

test("drafts are isolated and restored per query/workspace context", () => {
  const store = createSubmissionStore({ storage: memoryStorage(), storageKey: "contexts" });
  store.setContext("text:query-a");
  store.addFrame(frame("L01_V001", 10));
  store.setContext("text:query-b");
  assert.deepEqual(store.getSnapshot().drafts.KIS.items, []);
  store.addFrame(frame("L02_V001", 20));
  store.setContext("text:query-a");
  assert.deepEqual(store.getSnapshot().drafts.KIS.items.map(frameIdentity), ["L01_V001:10"]);
  store.setContext("text:query-b");
  assert.deepEqual(store.getSnapshot().drafts.KIS.items.map(frameIdentity), ["L02_V001:20"]);
});

test("first manual frame can fill visible related suggestions without changing manual order", () => {
  const store = createSubmissionStore({ storage: memoryStorage(), storageKey: "related" });
  const seed = frame("L01_V001", 10, 1);
  const added = store.addFrame(seed, { validation: "canonical" });
  assert.equal(added.firstManual, true);
  const result = store.setRelatedFrames(seed, [
    frame("L01_V001", 10, 1),
    { ...frame("L02_V001", 20, 2), related_rank: 1, related_score: 0.02 },
    { ...frame("L03_V001", 30, 3), related_rank: 2, related_score: 0.01 },
    frame("L02_V001", 20, 2),
  ]);
  assert.deepEqual(result, { ok: true, count: 2 });
  let draft = store.getSnapshot().drafts.KIS;
  assert.deepEqual(draft.items.map(frameIdentity), ["L01_V001:10"]);
  assert.deepEqual(draft.suggestedItems.map(frameIdentity), ["L02_V001:20", "L03_V001:30"]);
  assert(draft.suggestedItems.every((item) => item.manual === false));

  store.addFrame(frame("L02_V001", 20, 2), { source: "filmstrip" });
  draft = store.getSnapshot().drafts.KIS;
  assert.deepEqual(draft.items.map(frameIdentity), ["L01_V001:10", "L02_V001:20"]);
  assert.deepEqual(draft.suggestedItems.map(frameIdentity), ["L03_V001:30"]);
});

test("bulk add is atomic, canonical-identity deduplicated, and bounded to 100 rows", () => {
  const storage = memoryStorage();
  const store = createSubmissionStore({ storage, storageKey: "bulk" });
  let emissions = 0;
  store.subscribe(() => { emissions += 1; });
  const frames = Array.from({ length: 105 }, (_, index) => frame("L01_V001", index * 10, index + 1));
  const result = store.addFrames([...frames, frames[0]], { validation: "canonical" });
  assert.equal(result.added, 100);
  assert.equal(result.firstManual, true);
  assert.equal(emissions, 1);
  assert.equal(store.getSnapshot().drafts.KIS.items.length, 100);
});

test("a human selection displaces the last automatic suggestion at the 100-row cap", () => {
  const store = createSubmissionStore({ storage: memoryStorage(), storageKey: "manual-priority" });
  const seed = frame("L01_V001", 10, 1);
  store.addFrame(seed);
  store.setRelatedFrames(
    seed,
    Array.from({ length: 99 }, (_, index) => frame("L02_V001", index + 100, index + 1)),
  );
  const result = store.addFrame(frame("L03_V001", 900, 1));
  const draft = store.getSnapshot().drafts.KIS;
  assert.equal(result.ok, true);
  assert.equal(draft.items.length, 2);
  assert.equal(draft.suggestedItems.length, 98);
  assert.equal(draft.items.at(-1).frame_uid, "L03_V001:900");
});

test("version 3 drafts migrate without losing manual selections", () => {
  const storage = memoryStorage();
  storage.setItem("migration", JSON.stringify({
    version: 3,
    mode: "KIS",
    contextKey: "default",
    contexts: {
      default: {
        KIS: { queryId: "9", answer: "", items: [frame("L04_V001", 40, 4)] },
        VQA: { queryId: "1", answer: "", items: [] },
        TRAKE: { queryId: "1", activeEvent: 1, events: [], eventSlots: {} },
      },
    },
  }));
  const restored = createSubmissionStore({ storage, storageKey: "migration" });
  assert.equal(restored.getSnapshot().version, 4);
  assert.deepEqual(restored.getSnapshot().drafts.KIS.items.map(frameIdentity), ["L04_V001:40"]);
  assert.deepEqual(restored.getSnapshot().drafts.KIS.suggestedItems, []);
});
