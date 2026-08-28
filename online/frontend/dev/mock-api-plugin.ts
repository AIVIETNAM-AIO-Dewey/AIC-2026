import type { IncomingMessage, ServerResponse } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import type { Plugin } from "vite";

const MOCK_VIDEO_ID = "L21_V001";
const MOCK_FRAME_COUNT = 128;
const MOCK_POOL_SIZE = 100;

const frames = Array.from({ length: MOCK_FRAME_COUNT }, (_, index) => {
  const keyframeN = index + 1;
  const frameIdx = 4 + index * 47;
  const filename = `${String(frameIdx).padStart(8, "0")}.jpg`;
  return {
    keyframe_n: keyframeN,
    frame_idx: frameIdx,
    pts_time_s: Number((frameIdx / 30).toFixed(4)),
    filename,
    score: Number((0.99 - index * 0.003).toFixed(4)),
    video_id: MOCK_VIDEO_ID,
    image_relpath: `keyframes/${MOCK_VIDEO_ID}/${filename}`,
    score_type: "cosine",
    submission_string: `${MOCK_VIDEO_ID}, ${frameIdx}`,
    asr_transcript: `Deterministic sample transcript for keyframe ${keyframeN}.`,
    dam_summary: `Synthetic ${MOCK_VIDEO_ID} keyframe ${keyframeN} returned by the lightweight mock API.`,
    ocr_text: `HTV TIN TUC ${keyframeN}`,
  };
});

function rotatedFrames(offset: number): typeof frames {
  return frames
    .map((_, index) => frames[(index + offset) % frames.length])
    .slice(0, MOCK_POOL_SIZE);
}

function sendJson(response: ServerResponse, value: unknown, status = 200): void {
  response.statusCode = status;
  response.setHeader("Content-Type", "application/json; charset=utf-8");
  response.end(JSON.stringify(value));
}

function pathnameOf(request: IncomingMessage): string {
  return new URL(request.url ?? "/", "http://localhost").pathname;
}

function escapeXml(value: string): string {
  return value.replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&apos;",
  })[character] ?? character);
}

function sendPlaceholderImage(response: ServerResponse, videoId: string, filename: string): void {
  const frameNumber = Number.parseInt(path.parse(filename).name, 10);
  const label = escapeXml(
    `${videoId} · frame ${Number.isFinite(frameNumber) ? frameNumber : "preview"}`,
  );
  const svg = [
    '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360" viewBox="0 0 640 360">',
    '<rect width="640" height="360" fill="#111827"/>',
    '<rect x="24" y="24" width="592" height="312" rx="18" fill="#1f2937" stroke="#38bdf8" stroke-width="2"/>',
    '<text x="320" y="166" text-anchor="middle" fill="#e5e7eb" font-family="system-ui,sans-serif" font-size="24">AIC mock keyframe</text>',
    `<text x="320" y="204" text-anchor="middle" fill="#7dd3fc" font-family="ui-monospace,monospace" font-size="18">${label}</text>`,
    "</svg>",
  ].join("");
  response.statusCode = 200;
  response.setHeader("Content-Type", "image/svg+xml; charset=utf-8");
  response.setHeader("Cache-Control", "public, max-age=3600");
  response.end(svg);
}

export function mockApiPlugin(workspaceRoot: string): Plugin {
  return {
    name: "aic-lightweight-mock-api",
    configureServer(server) {
      server.middlewares.use(async (request, response, next) => {
        const pathname = pathnameOf(request);

        if (pathname === "/api/config") {
          sendJson(response, {
            keyframes_root: path.join(workspaceRoot, "data", "keyframe"),
            experiment_mode: "nofusion",
            task_types: ["KIS"],
            modalities: ["siglip", "dam", "ocr", "asr"],
            fusion_enabled: false,
            reranking_enabled: false,
          });
          return;
        }

        if (pathname === "/api/parse") {
          sendJson(response, {
            execution_time_ms: 1.2,
            parsed_query: {
              task_type: "KIS",
              language: "vi",
              original_query: "Local sample query",
              global_scene_en: "Vietnamese morning news broadcast",
              objects_en: ["news presenter", "city"],
              speech_vi: "tin tuc buoi sang",
              ocr_keywords: ["HTV"],
              is_temporal_trake: false,
              trake_events: [],
              vqa_question: "",
            },
          });
          return;
        }

        if (pathname === "/api/search") {
          const makePool = (
            modality: string,
            displayName: string,
            query: string | string[],
            scoreType: string,
            resultFrames: typeof frames,
          ) => ({
            modality,
            display_name: displayName,
            status: "ok",
            reason: "",
            query,
            query_source: modality === "dam" ? "objects_en" : modality === "ocr" ? "ocr_keywords" : modality === "asr" ? "speech_vi" : "global_scene_en",
            score_type: scoreType,
            score_description: `Raw ${scoreType} score for the ${displayName} pool`,
            result_count: resultFrames.length,
            execution_time_ms: 1.1,
            results: resultFrames.map((frame, index) => ({
              ...frame,
              rank: index + 1,
              score_type: scoreType,
              transcript: frame.asr_transcript,
              matched_keywords: modality === "ocr" ? ["HTV"] : undefined,
              subject_scores: modality === "dam" ? [{ subject: "news presenter", cosine: frame.score }] : undefined,
            })),
          });
          sendJson(response, {
            session_id: "local-mock-session",
            execution_time_ms: 4.8,
            experiment_mode: "nofusion",
            fusion_applied: false,
            reranking_applied: false,
            modality_results: {
              siglip: makePool("siglip", "SigLIP visual scene", "Vietnamese morning news broadcast", "cosine", rotatedFrames(0)),
              dam: makePool("dam", "DAM detected objects", ["news presenter", "city"], "mean_best_region_cosine", rotatedFrames(7)),
              ocr: makePool("ocr", "OCR on-screen text", ["HTV"], "keyword_match_ratio", rotatedFrames(14)),
              asr: makePool("asr", "ASR spoken speech", "tin tuc buoi sang", "cosine", rotatedFrames(21)),
            },
          });
          return;
        }

        const mediaMatch = pathname.match(/^\/api\/video\/([^/]+)\/media-info$/);
        if (mediaMatch) {
          const videoId = decodeURIComponent(mediaMatch[1]);
          try {
            const content = await readFile(path.join(workspaceRoot, "data", "media-info", `${videoId}.json`), "utf8");
            sendJson(response, JSON.parse(content));
          } catch {
            sendJson(response, { detail: "Media info not found" }, 404);
          }
          return;
        }

        const keyframesMatch = pathname.match(/^\/api\/video\/([^/]+)\/keyframes$/);
        if (keyframesMatch) {
          sendJson(response, { video_id: decodeURIComponent(keyframesMatch[1]), total_keyframes: frames.length, keyframes: frames });
          return;
        }

        const detailMatch = pathname.match(/^\/api\/keyframe\/([^/]+)\/(\d+)$/);
        if (detailMatch) {
          const keyframeN = Number(detailMatch[2]);
          const frame = frames.find((item) => item.keyframe_n === keyframeN);
          if (!frame) {
            sendJson(response, { detail: "Keyframe not found" }, 404);
            return;
          }
          sendJson(response, {
            keyframe: frame,
            macro_audio_transcript: frame.asr_transcript,
            dam_objects: [{ class_entity: "Broadcast frame", description_en: frame.dam_summary, bbox: [0.04, 0.05, 0.96, 0.94] }],
          });
          return;
        }

        const imageMatch = pathname.match(/^\/(?:keyframes|data\/keyframe)\/([^/]+)\/(\d+\.jpg)$/i);
        if (imageMatch) {
          sendPlaceholderImage(
            response,
            decodeURIComponent(imageMatch[1]),
            decodeURIComponent(imageMatch[2]),
          );
          return;
        }

        next();
      });
    },
  };
}
