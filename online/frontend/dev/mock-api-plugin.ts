import type { IncomingMessage, ServerResponse } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import type { Plugin } from "vite";

const frames = [
  { keyframe_n: 1, frame_idx: 4, pts_time_s: 0.1333, filename: "00000004.jpg", score: 0.947 },
  { keyframe_n: 2, frame_idx: 31, pts_time_s: 1.0333, filename: "00000031.jpg", score: 0.921 },
  { keyframe_n: 101, frame_idx: 4948, pts_time_s: 164.9333, filename: "00004948.jpg", score: 0.905 },
  { keyframe_n: 251, frame_idx: 12156, pts_time_s: 405.2, filename: "00012156.jpg", score: 0.899 },
  { keyframe_n: 417, frame_idx: 19707, pts_time_s: 656.9, filename: "00019707.jpg", score: 0.938 },
  { keyframe_n: 601, frame_idx: 27681, pts_time_s: 922.7, filename: "00027681.jpg", score: 0.912 },
  { keyframe_n: 833, frame_idx: 37834, pts_time_s: 1261.1333, filename: "00037834.jpg", score: 0.887 },
].map((frame) => ({
  ...frame,
  video_id: "L21_V001",
  final_score: frame.score,
  stage1_score: frame.score,
  submission_string: `L21_V001, ${frame.frame_idx}`,
  asr_transcript: "Sample transcript for local UI testing.",
  dam_summary: "Sample L21_V001 keyframe returned by the lightweight development server.",
  ocr_text: "HTV Tin Tuc",
}));

function sendJson(response: ServerResponse, value: unknown, status = 200): void {
  response.statusCode = status;
  response.setHeader("Content-Type", "application/json; charset=utf-8");
  response.end(JSON.stringify(value));
}

function pathnameOf(request: IncomingMessage): string {
  return new URL(request.url ?? "/", "http://localhost").pathname;
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
            default_weights: { vis: 0.35, dam: 0.3, asr: 0.35, ocr: 0 },
            mode: "lightweight-mock",
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
              weights: { vis: 0.35, dam: 0.3, asr: 0.35, ocr: 0 },
            },
          });
          return;
        }

        if (pathname === "/api/search" || pathname === "/api/search/cached") {
          sendJson(response, {
            session_id: "local-mock-session",
            execution_time_ms: 4.8,
            total_candidates_evaluated: frames.length,
            results: [frames[4], frames[0], frames[6]],
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

        const imageMatch = pathname.match(/^\/keyframes\/L21_V001\/(\d+)\.jpg$/i);
        if (imageMatch) {
          const keyframeN = Number(imageMatch[1]);
          const frame = frames.find((item) => item.keyframe_n === keyframeN);
          if (frame) {
            response.statusCode = 302;
            response.setHeader("Location", `/data/keyframe/L21_V001/${frame.filename}`);
            response.end();
            return;
          }
        }

        next();
      });
    },
  };
}
