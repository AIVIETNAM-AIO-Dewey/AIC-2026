import assert from "node:assert/strict";
import test from "node:test";

import { estimateRawFrame, nearestKeyframe } from "../src/time-keyframe-map.js";

const keyframes = [
  { keyframe_n: 1, frame_idx: 0, pts_time_s: 0 },
  { keyframe_n: 2, frame_idx: 250, pts_time_s: 10 },
  { keyframe_n: 3, frame_idx: 500, pts_time_s: 20 },
];

test("nearest canonical keyframe uses binary search and earlier midpoint tie", () => {
  assert.equal(nearestKeyframe(keyframes, 9).frame.keyframe_n, 2);
  assert.equal(nearestKeyframe(keyframes, 15).frame.keyframe_n, 2);
  assert.equal(nearestKeyframe(keyframes, 19).frame.keyframe_n, 3);
  assert.equal(nearestKeyframe(keyframes, -5).frame.keyframe_n, 1);
  assert.equal(nearestKeyframe([], 1), null);
});

test("estimated raw frame is informational and validates fps", () => {
  assert.equal(estimateRawFrame(10.4, 25), 260);
  assert.equal(estimateRawFrame(1, 0), null);
  assert.equal(estimateRawFrame(-1, 25), null);
});
