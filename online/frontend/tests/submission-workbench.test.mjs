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
