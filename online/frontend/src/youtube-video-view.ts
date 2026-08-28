export type MappingStatusState = "pending" | "ready" | "error";

export interface VideoFrameMapping {
  videoId: string;
  ptsTimeS: number;
  posterPath: string;
}

interface MediaInfo {
  author: string;
  length: number;
  title: string;
  watch_url: string;
}

interface YouTubePlayer {
  cueVideoById(options: { videoId: string; startSeconds: number }): void;
  getCurrentTime(): number;
  loadVideoById(options: { videoId: string; startSeconds: number }): void;
  mute(): void;
  pauseVideo(): void;
  playVideo(): void;
  seekTo(seconds: number, allowSeekAhead: boolean): void;
  unMute(): void;
}

interface YouTubePlayerEvent {
  data: number;
  target: YouTubePlayer;
}

interface YouTubeNamespace {
  Player: new (elementId: string, options: {
    events: {
      onError: (event: YouTubePlayerEvent) => void;
      onReady: (event: YouTubePlayerEvent) => void;
      onStateChange: (event: YouTubePlayerEvent) => void;
    };
  }) => YouTubePlayer;
}

declare global {
  interface Window {
    YT?: YouTubeNamespace;
    onYouTubeIframeAPIReady?: () => void;
  }
}

export interface YouTubeVideoViewOptions {
  /** @deprecated Mapping state is reported through onStatusChange. */
  onSessionChange?(label: string): void;
  onSourceChange(label: string): void;
  onStatusChange(label: string, state?: MappingStatusState): void;
  onToast(message: string): void;
  refreshIcons(): void;
}

export interface YouTubeVideoViewController {
  activate(): Promise<void>;
  deactivate(): void;
  openOnYouTube(): void;
  preload(frame: VideoFrameMapping): Promise<void>;
  requestFullscreen(): Promise<void>;
  seekBy(deltaSeconds: number): void;
  seekTo(seconds: number): void;
  setFrame(frame: VideoFrameMapping): void;
  togglePlayback(): Promise<void>;
}

function requireElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing video view element #${id}`);
  return element as T;
}

function parseYouTubeId(watchUrl: string): string | null {
  try {
    const url = new URL(watchUrl);
    if (url.hostname === "youtu.be") return url.pathname.split("/").filter(Boolean)[0] ?? null;
    return url.searchParams.get("v") ?? url.pathname.match(/\/embed\/([^/?]+)/)?.[1] ?? null;
  } catch {
    return null;
  }
}

function formatTime(seconds: number): string {
  const safeSeconds = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const minutes = Math.floor(safeSeconds / 60);
  const remainder = Math.floor(safeSeconds % 60);
  return `${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
}

function describeYouTubeError(code: number): string {
  if (code === 100) return "Video unavailable or private (100)";
  if (code === 101 || code === 150) return `Embedding disabled (${code})`;
  if (code === 153) return "Missing player referrer (153)";
  if (code === 2) return "Invalid YouTube video ID (2)";
  if (code === 5) return "HTML5 player error (5)";
  return `YouTube player error (${code})`;
}

export function createYouTubeVideoView(options: YouTubeVideoViewOptions): YouTubeVideoViewController {
  const elements = {
    backButton: requireElement<HTMLButtonElement>("btn-video-back"),
    centerPlay: requireElement<HTMLButtonElement>("btn-video-center-play"),
    forwardButton: requireElement<HTMLButtonElement>("btn-video-forward"),
    fullscreenButton: requireElement<HTMLButtonElement>("btn-video-fullscreen"),
    mock: requireElement<HTMLDivElement>("video-mock"),
    openButton: requireElement<HTMLButtonElement>("btn-open-youtube"),
    playButton: requireElement<HTMLButtonElement>("btn-video-play"),
    poster: requireElement<HTMLImageElement>("video-poster"),
    progress: requireElement<HTMLInputElement>("video-progress"),
    time: requireElement<HTMLSpanElement>("video-time"),
    youtubeFrame: requireElement<HTMLIFrameElement>("youtube-player"),
  };

  let active = false;
  let apiPromise: Promise<YouTubeNamespace> | null = null;
  let currentEmbedKey: string | null = null;
  let currentSeconds = 0;
  let currentFrame: VideoFrameMapping | null = null;
  let isPlaying = false;
  let mediaInfo: MediaInfo | null = null;
  let mediaInfoPromise: Promise<MediaInfo> | null = null;
  let mediaInfoPromiseVideoId: string | null = null;
  let mediaInfoVideoId: string | null = null;
  let pendingPreview: { requestId: number; targetSeconds: number } | null = null;
  let playbackTimer: number | null = null;
  let player: YouTubePlayer | null = null;
  let playerAttemptId = 0;
  let playerPromise: Promise<void> | null = null;
  let playerReady = false;
  let preloadPromise: Promise<void> | null = null;
  let preloadPromiseVideoId: string | null = null;
  let previewPauseTimer: number | null = null;
  let previewRequestId = 0;
  let youtubeVideoId: string | null = null;

  function duration(): number | null {
    const seconds = Number(mediaInfo?.length);
    return Number.isFinite(seconds) && seconds > 0 ? seconds : null;
  }

  function clampSeconds(seconds: number): number {
    const safeSeconds = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
    const knownDuration = duration();
    return knownDuration === null ? safeSeconds : Math.min(knownDuration, safeSeconds);
  }

  function updatePlaybackUi(): void {
    const knownDuration = duration();
    elements.progress.disabled = knownDuration === null;
    elements.progress.max = String(knownDuration ?? Math.max(1, Math.ceil(currentSeconds)));
    elements.progress.value = clampSeconds(currentSeconds).toFixed(1);
    elements.time.textContent = `${formatTime(currentSeconds)} / ${knownDuration === null ? "--:--" : formatTime(knownDuration)}`;
  }

  function setPlaying(playing: boolean): void {
    isPlaying = playing;
    if (playbackTimer !== null) {
      window.clearInterval(playbackTimer);
      playbackTimer = null;
    }
    elements.centerPlay.classList.toggle("hidden", playing);
    elements.playButton.innerHTML = `<i data-lucide="${playing ? "pause" : "play"}"></i>`;
    elements.playButton.title = playing ? "Pause" : "Play";
    elements.playButton.setAttribute("aria-label", playing ? "Pause" : "Play");
    options.refreshIcons();
    if (playing) {
      playbackTimer = window.setInterval(() => {
        if (!player) return;
        currentSeconds = player.getCurrentTime();
        updatePlaybackUi();
      }, 250);
    }
  }

  function clearPreviewPauseTimer(): void {
    if (previewPauseTimer !== null) {
      window.clearTimeout(previewPauseTimer);
      previewPauseTimer = null;
    }
  }

  function cancelPreview(showPoster: boolean): void {
    previewRequestId += 1;
    pendingPreview = null;
    clearPreviewPauseTimer();
    try {
      player?.pauseVideo();
      player?.unMute();
    } catch {
      // A failed/removed iframe can reject commands while its state is reset.
    }
    elements.mock.classList.toggle("preview-ready", !showPoster);
  }

  async function loadMediaInfo(videoId: string): Promise<{ info: MediaInfo; parsedVideoId: string }> {
    const encodedVideoId = encodeURIComponent(videoId);
    const apiUrl = `/api/video/${encodedVideoId}/media-info`;
    const staticUrl = `/data/media-info/${encodedVideoId}.json`;
    const isDemo = new URLSearchParams(window.location.search).get("demo") === "video-ui";
    const urls = isDemo ? [staticUrl, apiUrl] : [apiUrl, staticUrl];
    let response: Response | null = null;
    for (const url of urls) {
      try {
        const candidate = await fetch(url);
        if (candidate.ok) {
          response = candidate;
          break;
        }
      } catch {
        // The static demo fallback remains available when FastAPI is offline.
      }
    }
    if (!response) throw new Error("Media info is unavailable for this video");
    const info = await response.json() as MediaInfo;
    const parsedVideoId = parseYouTubeId(info.watch_url);
    if (!parsedVideoId) throw new Error("The media info does not contain a valid YouTube URL");
    return { info, parsedVideoId };
  }

  function getMediaInfo(): Promise<MediaInfo> {
    if (!currentFrame) return Promise.reject(new Error("No video frame selected"));
    const videoId = currentFrame.videoId;
    if (mediaInfo && mediaInfoVideoId === videoId) return Promise.resolve(mediaInfo);
    if (mediaInfoPromise && mediaInfoPromiseVideoId === videoId) return mediaInfoPromise;

    const pending = loadMediaInfo(videoId).then(({ info, parsedVideoId }) => {
      if (currentFrame?.videoId !== videoId) throw new Error("Video selection changed while loading media info");
      mediaInfo = info;
      mediaInfoVideoId = videoId;
      youtubeVideoId = parsedVideoId;
      currentSeconds = clampSeconds(currentSeconds);
      elements.openButton.disabled = false;
      options.onSourceChange(`${info.author} - YouTube`);
      updatePlaybackUi();
      return info;
    });
    const retryable = pending.catch((error: unknown) => {
      if (mediaInfoPromise === retryable) {
        mediaInfoPromise = null;
        mediaInfoPromiseVideoId = null;
      }
      throw error;
    });
    mediaInfoPromise = retryable;
    mediaInfoPromiseVideoId = videoId;
    return retryable;
  }

  function loadYouTubeApi(): Promise<YouTubeNamespace> {
    if (window.YT?.Player) return Promise.resolve(window.YT);
    if (apiPromise) return apiPromise;

    const pending = new Promise<YouTubeNamespace>((resolve, reject) => {
      const previousReady = window.onYouTubeIframeAPIReady;
      let settled = false;
      let script = document.querySelector<HTMLScriptElement>('script[src="https://www.youtube.com/iframe_api"]');
      const finish = (callback: () => void) => {
        if (settled) return;
        settled = true;
        window.clearTimeout(timeout);
        if (window.onYouTubeIframeAPIReady === handleReady) window.onYouTubeIframeAPIReady = previousReady;
        callback();
      };
      const handleReady = () => {
        previousReady?.();
        if (window.YT?.Player) finish(() => resolve(window.YT as YouTubeNamespace));
        else finish(() => reject(new Error("YouTube player API is unavailable")));
      };
      const timeout = window.setTimeout(
        () => finish(() => reject(new Error("YouTube player API timed out"))),
        15000,
      );
      window.onYouTubeIframeAPIReady = handleReady;
      if (!script) {
        script = document.createElement("script");
        script.src = "https://www.youtube.com/iframe_api";
        script.async = true;
        script.onerror = () => finish(() => reject(new Error("Could not load YouTube player API")));
        document.head.appendChild(script);
      }
    });
    const retryable = pending.catch((error: unknown) => {
      if (apiPromise === retryable) apiPromise = null;
      if (!window.YT?.Player) {
        document.querySelector<HTMLScriptElement>('script[src="https://www.youtube.com/iframe_api"]')?.remove();
      }
      throw error;
    });
    apiPromise = retryable;
    return retryable;
  }

  function cueFrame(): void {
    if (!currentFrame || !youtubeVideoId) return;
    const targetSeconds = clampSeconds(currentFrame.ptsTimeS);
    if (playerReady && player) {
      player.cueVideoById({ videoId: youtubeVideoId, startSeconds: targetSeconds });
    } else {
      const embedKey = `${youtubeVideoId}:${targetSeconds}`;
      if (currentEmbedKey !== embedKey) {
        const params = new URLSearchParams({
          controls: "1",
          enablejsapi: "1",
          origin: window.location.origin,
          playsinline: "1",
          rel: "0",
          start: String(Math.floor(targetSeconds)),
        });
        elements.youtubeFrame.src = `https://www.youtube.com/embed/${youtubeVideoId}?${params}`;
        currentEmbedKey = embedKey;
      }
      elements.mock.classList.add("player-visible");
    }
    currentSeconds = targetSeconds;
    updatePlaybackUi();
  }

  function finalizePreview(requestId: number): void {
    const preview = pendingPreview;
    if (!preview || preview.requestId !== requestId || requestId !== previewRequestId || !player) return;
    pendingPreview = null;
    previewPauseTimer = null;
    const playerSeconds = player.getCurrentTime();
    currentSeconds = clampSeconds(Number.isFinite(playerSeconds) ? playerSeconds : preview.targetSeconds);
    player.pauseVideo();
    player.unMute();
    elements.mock.classList.add("preview-ready");
    setPlaying(false);
    updatePlaybackUi();
    options.onStatusChange(`Ready at ${formatTime(currentSeconds)}`, "ready");
  }

  function handlePlayerStateChange(state: number): void {
    if (state === 1 && pendingPreview) {
      const requestId = pendingPreview.requestId;
      setPlaying(false);
      clearPreviewPauseTimer();
      previewPauseTimer = window.setTimeout(() => finalizePreview(requestId), 140);
      return;
    }
    setPlaying(state === 1);
  }

  function failPlayerAttempt(attemptId: number, error: unknown): void {
    if (attemptId !== playerAttemptId) return;
    cancelPreview(true);
    playerAttemptId += 1;
    playerReady = false;
    player = null;
    playerPromise = null;
    currentEmbedKey = null;
    elements.youtubeFrame.removeAttribute("src");
    elements.mock.classList.remove("player-ready", "player-visible");
    const message = error instanceof Error ? error.message : "YouTube player unavailable";
    options.onStatusChange(message, "error");
    options.onToast(message);
  }

  function ensurePlayer(): Promise<void> {
    if (playerReady) return Promise.resolve();
    if (playerPromise) return playerPromise;
    const attemptId = ++playerAttemptId;
    let ready = false;
    const pending = (async () => {
      options.onStatusChange("Loading YouTube");
      await getMediaInfo();
      if (attemptId !== playerAttemptId) throw new Error("Player request superseded");
      cueFrame();
      const yt = await loadYouTubeApi();
      if (attemptId !== playerAttemptId) throw new Error("Player request superseded");
      await new Promise<void>((resolve, reject) => {
        player = new yt.Player("youtube-player", {
          events: {
            onReady: (event) => {
              if (attemptId !== playerAttemptId) {
                try {
                  event.target.pauseVideo();
                } catch {
                  // The superseded iframe may already have been detached.
                }
                return;
              }
              player = event.target;
              playerReady = true;
              ready = true;
              elements.mock.classList.add("player-ready");
              options.onStatusChange("Mapping ready", "ready");
              cueFrame();
              resolve();
            },
            onStateChange: (event) => {
              if (attemptId === playerAttemptId) handlePlayerStateChange(event.data);
            },
            onError: (event) => {
              if (attemptId !== playerAttemptId) return;
              const error = new Error(describeYouTubeError(event.data));
              if (ready) failPlayerAttempt(attemptId, error);
              else reject(error);
            },
          },
        });
      });
    })();
    const retryable = pending.catch((error: unknown) => {
      failPlayerAttempt(attemptId, error);
      throw error;
    });
    playerPromise = retryable;
    return retryable;
  }

  async function preparePreview(): Promise<void> {
    if (!currentFrame) return;
    if (!youtubeVideoId) await getMediaInfo();
    if (!youtubeVideoId) return;
    const requestId = ++previewRequestId;
    const targetSeconds = clampSeconds(currentFrame.ptsTimeS);
    pendingPreview = { requestId, targetSeconds };
    clearPreviewPauseTimer();
    currentSeconds = targetSeconds;
    elements.mock.classList.remove("preview-ready");
    setPlaying(false);
    updatePlaybackUi();
    options.onStatusChange(`Seeking ${formatTime(targetSeconds)}`);
    await ensurePlayer();
    if (!active || requestId !== previewRequestId || !player || !youtubeVideoId) return;
    player.mute();
    player.loadVideoById({ videoId: youtubeVideoId, startSeconds: targetSeconds });
  }

  async function playPrepared(): Promise<void> {
    await ensurePlayer();
    if (!player || !currentFrame) return;
    const needsSeek = pendingPreview !== null || !elements.mock.classList.contains("preview-ready");
    previewRequestId += 1;
    pendingPreview = null;
    clearPreviewPauseTimer();
    player.unMute();
    if (needsSeek) player.seekTo(clampSeconds(currentFrame.ptsTimeS), true);
    elements.mock.classList.add("preview-ready");
    player.playVideo();
  }

  const controller: YouTubeVideoViewController = {
    async activate() {
      active = true;
      await preparePreview();
    },
    deactivate() {
      active = false;
      cancelPreview(true);
      setPlaying(false);
    },
    openOnYouTube() {
      if (!mediaInfo) return;
      const separator = mediaInfo.watch_url.includes("?") ? "&" : "?";
      window.open(`${mediaInfo.watch_url}${separator}t=${Math.floor(currentSeconds)}s`, "_blank", "noopener,noreferrer");
    },
    preload(frame) {
      controller.setFrame(frame);
      if (preloadPromise && preloadPromiseVideoId === frame.videoId) return preloadPromise;
      const videoId = frame.videoId;
      const pending = (async () => {
        options.onStatusChange("Loading video metadata");
        await getMediaInfo();
        if (currentFrame?.videoId === videoId) options.onStatusChange("Video metadata ready", "ready");
      })();
      const retryable = pending.catch((error: unknown) => {
        if (preloadPromise === retryable) {
          preloadPromise = null;
          preloadPromiseVideoId = null;
        }
        const message = error instanceof Error ? error.message : "Preload unavailable";
        if (currentFrame?.videoId === videoId) options.onStatusChange(message, "error");
        throw error;
      });
      preloadPromise = retryable;
      preloadPromiseVideoId = videoId;
      return retryable;
    },
    requestFullscreen() {
      return elements.mock.requestFullscreen?.() ?? Promise.resolve();
    },
    seekBy(deltaSeconds) {
      controller.seekTo(currentSeconds + deltaSeconds);
    },
    seekTo(seconds) {
      currentSeconds = clampSeconds(seconds);
      player?.seekTo(currentSeconds, true);
      if (!isPlaying) player?.pauseVideo();
      updatePlaybackUi();
    },
    setFrame(frame) {
      if (currentFrame && currentFrame.videoId !== frame.videoId) {
        active = false;
        cancelPreview(true);
        if (playerPromise && !playerReady && !player) {
          playerAttemptId += 1;
          playerPromise = null;
        }
        currentEmbedKey = null;
        elements.openButton.disabled = true;
        mediaInfo = null;
        mediaInfoPromise = null;
        mediaInfoPromiseVideoId = null;
        mediaInfoVideoId = null;
        preloadPromise = null;
        preloadPromiseVideoId = null;
        youtubeVideoId = null;
        options.onSourceChange("Loading media info");
      }
      currentFrame = frame;
      currentSeconds = frame.ptsTimeS;
      elements.poster.src = frame.posterPath;
      elements.mock.classList.remove("preview-ready");
      updatePlaybackUi();
    },
    async togglePlayback() {
      if (isPlaying) {
        player?.pauseVideo?.();
        return;
      }
      await playPrepared();
    },
  };

  elements.centerPlay.addEventListener("click", () => void controller.togglePlayback().catch(() => undefined));
  elements.playButton.addEventListener("click", () => void controller.togglePlayback().catch(() => undefined));
  elements.backButton.addEventListener("click", () => controller.seekBy(-5));
  elements.forwardButton.addEventListener("click", () => controller.seekBy(5));
  elements.progress.addEventListener("input", () => controller.seekTo(Number(elements.progress.value)));
  elements.openButton.addEventListener("click", () => controller.openOnYouTube());
  elements.fullscreenButton.addEventListener("click", () => void controller.requestFullscreen());

  updatePlaybackUi();
  return controller;
}
