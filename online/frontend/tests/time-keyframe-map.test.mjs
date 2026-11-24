import assert from "node:assert/strict";
import test from "node:test";

import {
  estimateRawFrame,
  frameIndexToSeconds,
  nearestKeyframe,
  secondsToFrameIndex,
} from "../src/time-keyframe-map.js";

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

test("source-frame conversion preserves exact organizer anchors", () => {
  for (const frame of keyframes) {
    assert.equal(frameIndexToSeconds(frame.frame_idx, keyframes, 25), frame.pts_time_s);
    assert.equal(
      secondsToFrameIndex(frame.pts_time_s, keyframes, 25, 0, 500),
      frame.frame_idx,
    );
  }
});

test("source-frame conversion round trips arbitrary indices at common dataset FPS", () => {
  for (const fps of [25, 29.97, 30, 26.44]) {
    const anchors = [
      { frame_idx: 0, pts_time_s: 0 },
      { frame_idx: 173, pts_time_s: 173 / fps },
      { frame_idx: 509, pts_time_s: 509 / fps },
    ];
    for (const frameIndex of [0, 1, 86, 173, 174, 333, 509, 777]) {
      const seconds = frameIndexToSeconds(frameIndex, anchors, fps);
      assert.equal(secondsToFrameIndex(seconds, anchors, fps, 0, 999), frameIndex);
    }
  }
});

test("source-frame conversion honors anchor drift, bounds, and invalid input", () => {
  const drifted = [
    { frame_idx: 17, pts_time_s: 0.68 },
    { frame_idx: 51, pts_time_s: 2.04 },
    { frame_idx: 87, pts_time_s: 3.48 },
  ];
  assert.equal(frameIndexToSeconds(34, drifted, 25), 1.36);
  assert.equal(secondsToFrameIndex(1.36, drifted, 25, 0, 249), 34);
  assert.equal(secondsToFrameIndex(999, drifted, 25, 0, 249), 249);
  assert.equal(secondsToFrameIndex(0, drifted, 25, 0, 249), 0);
  assert.equal(frameIndexToSeconds(-1, drifted, 25), null);
  assert.equal(frameIndexToSeconds(1.5, drifted, 25), null);
  assert.equal(secondsToFrameIndex(-1, drifted, 25), null);
});
