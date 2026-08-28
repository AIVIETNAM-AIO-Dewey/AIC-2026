import assert from "node:assert/strict";
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
