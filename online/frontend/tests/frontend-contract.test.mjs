import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  extractHtmlIds,
  extractRequiredIds,
  validateFrontendContract,
} from "../../../scripts/check_frontend.mjs";

test("contract parser extracts HTML and JavaScript ids", () => {
  assert.deepEqual(extractHtmlIds('<div id="alpha"></div><button id="beta"></button>'), [
    "alpha",
    "beta",
  ]);
  assert.deepEqual(
    extractRequiredIds(
      'document.getElementById("alpha"); requireElement<HTMLButtonElement>("beta");',
    ),
    ["alpha", "beta"],
  );
});

test("frontend source and HTML keep a valid DOM contract", () => {
  assert.deepEqual(validateFrontendContract(), []);
});

test("video controller reports time during polling, seeking, and pause sampling", () => {
  const source = readFileSync(new URL("../src/youtube-video-view.ts", import.meta.url), "utf8");
  assert.match(source, /options\.onTimeChange\?\.\(currentSeconds\)/);
  assert.match(source, /function samplePlayerTime\(\)/);
  assert.match(source, /if \(!playing\) samplePlayerTime\(\)/);
  assert.match(source, /seekTo\(seconds\)[\s\S]*?updatePlaybackUi\(\)/);
});

test("async workspaces and edited CSV review guard against stale responses", () => {
  const source = readFileSync(new URL("../app.js", import.meta.url), "utf8");
  assert.match(source, /function selectImageQueryFile[\s\S]*?imageSearchRequestId \+= 1/);
  assert.match(source, /async function loadStandaloneVideo[\s\S]*?standaloneScopedRequestId \+= 1/);
  assert.match(source, /currentSnapshot\.contextKey !== editContext \|\| currentSnapshot\.mode !== editMode/);
  assert.match(source, /const isCurrentReview = \(\) =>[\s\S]*?reviewGeneration === csvReviewGeneration[\s\S]*?reviewContext === submissionStore\.getSnapshot\(\)\.contextKey/);
  assert.match(source, /ensureClientBackfill\(request\.manual_selections, request\.candidate_reservoir, 100, contextKey\)/);
});
