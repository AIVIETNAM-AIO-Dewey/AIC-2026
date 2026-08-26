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
  onSessionChange(label: string): void;
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

function formatTime(seconds: number, duration: number): string {
  const safeSeconds = Math.max(0, Math.min(duration, seconds));
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
  let pendingPreview: { requestId: number; targetSeconds: number } | null = null;
  let playbackTimer: number | null = null;
  let player: YouTubePlayer | null = null;
  let playerPromise: Promise<void> | null = null;
  let playerReady = false;
  let preloadPromise: Promise<void> | null = null;
  let previewPauseTimer: number | null = null;
  let previewRequestId = 0;
  let youtubeVideoId: string | null = null;

  function duration(): number {
    return mediaInfo?.length || 1;
  }

  function updatePlaybackUi(): void {
    elements.progress.max = String(duration());
    elements.progress.value = currentSeconds.toFixed(1);
    elements.time.textContent = `${formatTime(currentSeconds, duration())} / ${formatTime(duration(), duration())}`;
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
    player?.pauseVideo?.();
    player?.unMute?.();
    elements.mock.classList.toggle("preview-ready", !showPoster);
  }

  async function loadMediaInfo(videoId: string): Promise<MediaInfo> {
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
    mediaInfo = info;
    youtubeVideoId = parsedVideoId;
    elements.openButton.disabled = false;
    options.onSourceChange(`${info.author} - YouTube`);
    options.onSessionChange("Mapping: YouTube");
    updatePlaybackUi();
    return info;
  }

  function getMediaInfo(): Promise<MediaInfo> {
    if (mediaInfo) return Promise.resolve(mediaInfo);
    if (!currentFrame) return Promise.reject(new Error("No video frame selected"));
    mediaInfoPromise ??= loadMediaInfo(currentFrame.videoId);
    return mediaInfoPromise;
  }

  function loadYouTubeApi(): Promise<YouTubeNamespace> {
    if (window.YT?.Player) return Promise.resolve(window.YT);
    if (apiPromise) return apiPromise;
    apiPromise = new Promise((resolve, reject) => {
      const previousReady = window.onYouTubeIframeAPIReady;
      const timeout = window.setTimeout(() => reject(new Error("YouTube player API timed out")), 15000);
      window.onYouTubeIframeAPIReady = () => {
        previousReady?.();
        window.clearTimeout(timeout);
        if (window.YT?.Player) resolve(window.YT);
        else reject(new Error("YouTube player API is unavailable"));
      };
      if (document.querySelector('script[src="https://www.youtube.com/iframe_api"]')) return;
      const script = document.createElement("script");
      script.src = "https://www.youtube.com/iframe_api";
      script.async = true;
      script.onerror = () => {
        window.clearTimeout(timeout);
        reject(new Error("Could not load YouTube player API"));
      };
      document.head.appendChild(script);
    });
    return apiPromise;
  }

  function cueFrame(): void {
    if (!currentFrame || !youtubeVideoId) return;
    if (playerReady && player) {
      player.cueVideoById({ videoId: youtubeVideoId, startSeconds: currentFrame.ptsTimeS });
    } else {
      const embedKey = `${youtubeVideoId}:${currentFrame.ptsTimeS}`;
      if (currentEmbedKey !== embedKey) {
        const params = new URLSearchParams({
          controls: "1",
          enablejsapi: "1",
          origin: window.location.origin,
          playsinline: "1",
          rel: "0",
          start: String(Math.floor(currentFrame.ptsTimeS)),
        });
        elements.youtubeFrame.src = `https://www.youtube.com/embed/${youtubeVideoId}?${params}`;
        currentEmbedKey = embedKey;
      }
      elements.mock.classList.add("player-visible");
    }
    currentSeconds = currentFrame.ptsTimeS;
    updatePlaybackUi();
  }

  function finalizePreview(requestId: number): void {
    const preview = pendingPreview;
    if (!preview || preview.requestId !== requestId || requestId !== previewRequestId || !player) return;
    pendingPreview = null;
    previewPauseTimer = null;
    const playerSeconds = player.getCurrentTime();
    currentSeconds = Number.isFinite(playerSeconds) ? playerSeconds : preview.targetSeconds;
    player.pauseVideo();
    player.unMute();
    elements.mock.classList.add("preview-ready");
    setPlaying(false);
    updatePlaybackUi();
    options.onStatusChange(`Ready at ${formatTime(currentSeconds, duration())}`, "ready");
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

  async function ensurePlayer(): Promise<void> {
    if (playerReady) return;
    if (playerPromise) return playerPromise;
    playerPromise = (async () => {
      try {
        options.onStatusChange("Loading YouTube");
        const info = await getMediaInfo();
        cueFrame();
        const yt = await loadYouTubeApi();
        await new Promise<void>((resolve, reject) => {
          player = new yt.Player("youtube-player", {
            events: {
              onReady: (event) => {
                player = event.target;
                playerReady = true;
                elements.mock.classList.add("player-ready");
                options.onSourceChange(`${info.author} - YouTube`);
                options.onStatusChange("Mapping ready", "ready");
                cueFrame();
                resolve();
              },
              onStateChange: (event) => handlePlayerStateChange(event.data),
              onError: (event) => {
                const error = new Error(describeYouTubeError(event.data));
                playerReady = false;
                elements.mock.classList.remove("player-ready");
                options.onStatusChange(error.message, "error");
                cancelPreview(true);
                reject(error);
              },
            },
          });
        });
      } catch (error) {
        playerPromise = null;
        elements.mock.classList.remove("player-ready");
        cancelPreview(true);
        const message = error instanceof Error ? error.message : "YouTube player unavailable";
        options.onStatusChange(message, "error");
        options.onToast(message);
        throw error;
      }
    })();
    return playerPromise;
  }

  async function preparePreview(): Promise<void> {
    if (!currentFrame) return;
    if (!youtubeVideoId) await getMediaInfo();
    if (!youtubeVideoId) return;
    const requestId = ++previewRequestId;
    const targetSeconds = currentFrame.ptsTimeS;
    pendingPreview = { requestId, targetSeconds };
    clearPreviewPauseTimer();
    currentSeconds = targetSeconds;
    elements.mock.classList.remove("preview-ready");
    setPlaying(false);
    updatePlaybackUi();
    options.onStatusChange(`Seeking ${formatTime(targetSeconds, duration())}`);
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
    if (needsSeek) player.seekTo(currentFrame.ptsTimeS, true);
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
      if (preloadPromise) return preloadPromise;
      preloadPromise = (async () => {
        options.onStatusChange("Preloading video");
        await getMediaInfo();
        cueFrame();
        await loadYouTubeApi();
        await ensurePlayer();
        options.onStatusChange("Video preloaded", "ready");
        options.onSessionChange("Mapping: preloaded");
      })().catch((error: unknown) => {
        preloadPromise = null;
        const message = error instanceof Error ? error.message : "Preload unavailable";
        options.onStatusChange(message, "error");
        options.onSessionChange("Mapping: fallback");
        throw error;
      });
      return preloadPromise;
    },
    requestFullscreen() {
      return elements.mock.requestFullscreen?.() ?? Promise.resolve();
    },
    seekBy(deltaSeconds) {
      controller.seekTo(currentSeconds + deltaSeconds);
    },
    seekTo(seconds) {
      currentSeconds = Math.max(0, Math.min(duration(), seconds));
      player?.seekTo?.(currentSeconds, true);
      if (!isPlaying) player?.pauseVideo?.();
      updatePlaybackUi();
    },
    setFrame(frame) {
      if (currentFrame && currentFrame.videoId !== frame.videoId) {
        active = false;
        cancelPreview(true);
        currentEmbedKey = null;
        elements.openButton.disabled = true;
        mediaInfo = null;
        mediaInfoPromise = null;
        preloadPromise = null;
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
