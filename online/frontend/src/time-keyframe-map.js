function secondsOf(frame) {
  const seconds = Number(frame?.pts_time_s);
  return Number.isFinite(seconds) ? seconds : null;
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
