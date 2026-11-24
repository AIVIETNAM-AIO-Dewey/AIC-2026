function secondsOf(frame) {
  const seconds = Number(frame?.pts_time_s);
  return Number.isFinite(seconds) ? seconds : null;
}

function frameIndexOf(frame) {
  const frameIndex = Number(frame?.frame_idx);
  return Number.isInteger(frameIndex) && frameIndex >= 0 ? frameIndex : null;
}

function validFps(value) {
  const fps = Number(value);
  return Number.isFinite(fps) && fps > 0 ? fps : null;
}

function clampFrameIndex(frameIndex, minimum, maximum) {
  const lower = Number.isInteger(minimum) && minimum >= 0 ? minimum : 0;
  const upper = Number.isInteger(maximum) && maximum >= lower ? maximum : null;
  return Math.max(lower, upper === null ? frameIndex : Math.min(upper, frameIndex));
}

function frameInsertionPoint(keyframes, targetFrameIndex) {
  let low = 0;
  let high = keyframes.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    const frameIndex = frameIndexOf(keyframes[middle]);
    if (frameIndex === null) return null;
    if (frameIndex < targetFrameIndex) low = middle + 1;
    else high = middle;
  }
  return low;
}

function timeInsertionPoint(keyframes, targetSeconds) {
  let low = 0;
  let high = keyframes.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    const seconds = secondsOf(keyframes[middle]);
    if (seconds === null) return null;
    if (seconds < targetSeconds) low = middle + 1;
    else high = middle;
  }
  return low;
}

/**
 * Return the nearest canonical keyframe to a playback position.
 * Ties deliberately resolve to the earlier frame so scrubbing at an exact
 * midpoint cannot jump forward unexpectedly.
 */
export function nearestKeyframe(keyframes, playbackSeconds) {
  if (!Array.isArray(keyframes) || keyframes.length === 0) return null;
  const target = Number(playbackSeconds);
  if (!Number.isFinite(target)) return null;
  let low = 0;
  let high = keyframes.length - 1;
  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    const seconds = secondsOf(keyframes[middle]);
    if (seconds === null) return nearestKeyframeLinear(keyframes, target);
    if (seconds < target) low = middle + 1;
    else high = middle - 1;
  }
  const before = keyframes[Math.max(0, high)];
  const after = keyframes[Math.min(keyframes.length - 1, low)];
  const beforeDelta = Math.abs(target - Number(before.pts_time_s));
  const afterDelta = Math.abs(Number(after.pts_time_s) - target);
  const frame = beforeDelta <= afterDelta ? before : after;
  return {
    frame,
    index: frame === before ? Math.max(0, high) : Math.min(keyframes.length - 1, low),
    deltaSeconds: Number((Number(frame.pts_time_s) - target).toFixed(4)),
  };
}

function nearestKeyframeLinear(keyframes, target) {
  let best = null;
  keyframes.forEach((frame, index) => {
    const seconds = secondsOf(frame);
    if (seconds === null) return;
    const delta = seconds - target;
    if (!best || Math.abs(delta) < Math.abs(best.deltaSeconds)
      || (Math.abs(delta) === Math.abs(best.deltaSeconds) && delta < best.deltaSeconds)) {
      best = { frame, index, deltaSeconds: Number(delta.toFixed(4)) };
    }
  });
  return best;
}

export function estimateRawFrame(playbackSeconds, fps) {
  const seconds = Number(playbackSeconds);
  const framesPerSecond = Number(fps);
  if (!Number.isFinite(seconds) || seconds < 0 || !Number.isFinite(framesPerSecond) || framesPerSecond <= 0) {
    return null;
  }
  return Math.max(0, Math.round(seconds * framesPerSecond));
}

/**
 * Convert an exact zero-based source-frame index to playback seconds.
 * Organizer timestamps are immutable anchors. Non-indexed frames interpolate
 * between the surrounding anchors, preserving every known frame exactly.
 */
export function frameIndexToSeconds(frameIndex, keyframes, fps) {
  const target = Number(frameIndex);
  const framesPerSecond = validFps(fps);
  if (!Number.isInteger(target) || target < 0 || framesPerSecond === null) return null;
  if (!Array.isArray(keyframes) || keyframes.length === 0) {
    return Number((target / framesPerSecond).toFixed(6));
  }
  const position = frameInsertionPoint(keyframes, target);
  if (position === null) return null;
  const exact = keyframes[position];
  if (exact && frameIndexOf(exact) === target) return secondsOf(exact);

  let seconds;
  if (position === 0) {
    const firstFrame = frameIndexOf(keyframes[0]);
    const firstTime = secondsOf(keyframes[0]);
    if (firstFrame === null || firstTime === null) return null;
    seconds = firstFrame > 0 && firstTime > 0
      ? firstTime * (target / firstFrame)
      : target / framesPerSecond;
  } else if (position < keyframes.length) {
    const leftFrame = frameIndexOf(keyframes[position - 1]);
    const rightFrame = frameIndexOf(keyframes[position]);
    const leftTime = secondsOf(keyframes[position - 1]);
    const rightTime = secondsOf(keyframes[position]);
    if (
      leftFrame === null || rightFrame === null || leftTime === null || rightTime === null
      || rightFrame <= leftFrame || rightTime <= leftTime
    ) return null;
    const fraction = (target - leftFrame) / (rightFrame - leftFrame);
    seconds = leftTime + fraction * (rightTime - leftTime);
  } else {
    const last = keyframes[keyframes.length - 1];
    const lastFrame = frameIndexOf(last);
    const lastTime = secondsOf(last);
    if (lastFrame === null || lastTime === null) return null;
    seconds = lastTime + (target - lastFrame) / framesPerSecond;
  }
  return Number(Math.max(0, seconds).toFixed(6));
}

/**
 * Convert playback seconds to the closest zero-based source-frame index using
 * the exact inverse anchor segments. The returned value is always bounded.
 */
export function secondsToFrameIndex(
  playbackSeconds,
  keyframes,
  fps,
  minimumFrameIndex = 0,
  maximumFrameIndex = null,
) {
  const target = Number(playbackSeconds);
  const framesPerSecond = validFps(fps);
  if (!Number.isFinite(target) || target < 0 || framesPerSecond === null) return null;
  if (!Array.isArray(keyframes) || keyframes.length === 0) {
    return clampFrameIndex(
      Math.round(target * framesPerSecond),
      minimumFrameIndex,
      maximumFrameIndex,
    );
  }
  const position = timeInsertionPoint(keyframes, target);
  if (position === null) return null;
  const exact = keyframes[position];
  if (exact && secondsOf(exact) === target) {
    return clampFrameIndex(frameIndexOf(exact), minimumFrameIndex, maximumFrameIndex);
  }

  let frameIndex;
  if (position === 0) {
    const firstFrame = frameIndexOf(keyframes[0]);
    const firstTime = secondsOf(keyframes[0]);
    if (firstFrame === null || firstTime === null) return null;
    frameIndex = firstFrame > 0 && firstTime > 0
      ? target * (firstFrame / firstTime)
      : target * framesPerSecond;
  } else if (position < keyframes.length) {
    const leftFrame = frameIndexOf(keyframes[position - 1]);
    const rightFrame = frameIndexOf(keyframes[position]);
    const leftTime = secondsOf(keyframes[position - 1]);
    const rightTime = secondsOf(keyframes[position]);
    if (
      leftFrame === null || rightFrame === null || leftTime === null || rightTime === null
      || rightFrame <= leftFrame || rightTime <= leftTime
    ) return null;
    const fraction = (target - leftTime) / (rightTime - leftTime);
    frameIndex = leftFrame + fraction * (rightFrame - leftFrame);
  } else {
    const last = keyframes[keyframes.length - 1];
    const lastFrame = frameIndexOf(last);
    const lastTime = secondsOf(last);
    if (lastFrame === null || lastTime === null) return null;
    frameIndex = lastFrame + (target - lastTime) * framesPerSecond;
  }
  return clampFrameIndex(
    Math.round(frameIndex),
    minimumFrameIndex,
    maximumFrameIndex,
  );
}
