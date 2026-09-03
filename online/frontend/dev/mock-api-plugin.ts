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
    fps: 30,
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

/**
 * Keep mock pools at the same boundaries as the real API.  The generic
 * ``frames`` fixture stays small for timeline/media routes, while these
 * deterministic canonical-looking rows exercise the full fusion gates:
 * 1,500 / 500 / 500 / 500.  All branch fixtures derive the same first 500
 * identities so the RRF mock also exercises cross-branch agreement.
 */
const gateFrames = Array.from({ length: 1500 }, (_, index) => {
  const keyframeN = index + 1;
  const frameIdx = 4 + index * 47;
  const filename = `${String(frameIdx).padStart(8, "0")}.jpg`;
  const template = frames[index % frames.length];
  return {
    ...template,
    keyframe_n: keyframeN,
    frame_idx: frameIdx,
    pts_time_s: Number((frameIdx / 30).toFixed(4)),
    filename,
    video_id: MOCK_VIDEO_ID,
    image_relpath: `keyframes/${MOCK_VIDEO_ID}/${filename}`,
    submission_string: `${MOCK_VIDEO_ID}, ${frameIdx}`,
    asr_transcript: `Deterministic sample transcript for keyframe ${keyframeN}.`,
    dam_summary: `Synthetic ${MOCK_VIDEO_ID} keyframe ${keyframeN} returned by the lightweight mock API.`,
    ocr_text: `HTV TIN TUC ${keyframeN}`,
  };
});

const branch1MockResults = gateFrames.map((frame, index) => {
  const finalScore = Number((0.98 - index * 0.006).toFixed(6));
  const provenance = Object.fromEntries(
    [
      ["siglip2", 0.45],
      ["metaclip2", 0.30],
      ["beit3", 0.25],
    ].map(([model, weight], modelIndex) => [model, {
      observed: true,
      raw_cosine: Number((0.91 - index * 0.004 - modelIndex * 0.02).toFixed(6)),
      normalized_score: Number((0.86 - index * 0.005).toFixed(6)),
      best_query_role: ["original", "entity", "action", "context", "synonym", "keyword"][index % 6],
      best_query_language: model === "beit3" ? "en" : (index % 2 === 0 ? "vi" : "en"),
      best_query_rank: (index % 6) + 1,
      query_scores: {},
      normalization_mean: 0.6,
      normalization_std: 0.1,
      weight,
    }]),
  );
  return {
    ...frame,
    frame_uid: `${frame.video_id}:${frame.frame_idx}`,
    point_id: index + 1,
    global_idx: index + 1,
    rank: index + 1,
    final_score: finalScore,
    score: finalScore,
    score_type: "weighted_zsigmoid_fusion",
    best_model: "siglip2",
    best_stream_rank: index + 1,
    model_provenance: provenance,
  };
});

const branch2MockResults = gateFrames.slice(0, 500).map((frame, index) => {
  const hybridScore = Number((0.94 - index * 0.005).toFixed(6));
  const rerankedScore = Number((0.96 - ((index * 7) % 100) * 0.004).toFixed(6));
  return {
    ...frame,
    frame_uid: `${frame.video_id}:${frame.frame_idx}`,
    point_id: index + 1,
    global_idx: index + 1,
    rank: index + 1,
    hybrid_rank: index + 1,
    pre_rerank_rank: index + 1,
    hybrid_score: hybridScore,
    dense_raw: hybridScore + 0.02,
    dense_normalized: 0.8,
    sparse_raw: 1 / (61 + index),
    sparse_normalized: 0.6,
    beit3_raw_cosine: 0.9 - index * 0.003,
    beit3_normalized: 0.82,
    previous_normalized: 0.78,
    reranked_score: rerankedScore,
    score: rerankedScore,
    score_type: "beit3_coco_cosine_blend",
    rerank_score_type: "beit3_coco_cosine_blend",
    rank_delta: 0,
    beit3_best_query_role: "original",
    dense_best_query_language: "en",
    sparse_best_query_language: "en",
    beit3_best_query_language: "en",
    dam_winner: {
      point_id: index + 1,
      region_id: `${frame.video_id}:${frame.frame_idx}:d000`,
      bbox: [0.05, 0.05, 0.75, 0.9],
      description_en: "A broadcast frame with a visible presenter and city background.",
      best_query_role: "entity",
    },
    sparse_winner: {
      region_id: `${frame.video_id}:${frame.frame_idx}:d000`,
      description_en: "A broadcast frame with a visible presenter and city background.",
      query_role: "keyword",
      rank: index + 1,
      bm25_raw: -1.0,
    },
  };
});

const branch3AsrMockResults = gateFrames.slice(0, 500).map((frame, index) => ({
  ...frame,
  frame_uid: `${frame.video_id}:${frame.frame_idx}`,
  point_id: index + 1,
  global_idx: index + 1,
  rank: index + 1,
  score: Number((0.93 - index * 0.004).toFixed(6)),
  asr_raw_score: Number((0.88 - index * 0.003).toFixed(6)),
  asr_normalized_score: Number((0.93 - index * 0.004).toFixed(6)),
  asr_best_query_role: ["original", "entity", "action", "context", "synonym", "keyword"][index % 6],
  asr_best_query_language: index % 2 === 0 ? "vi" : "en",
  asr_segment_id: `${frame.video_id}:seg_${index}`,
  asr_start_s: Number((frame.pts_time_s - 0.5).toFixed(3)),
  asr_end_s: Number((frame.pts_time_s + 0.5).toFixed(3)),
  asr_transcript: frame.asr_transcript,
  bm25_raw: Number((-1.0 + index * 0.003).toFixed(6)),
  bm25_relevance: Number((0.88 - index * 0.003).toFixed(6)),
  token_coverage: Number((0.80 - (index % 5) * 0.03).toFixed(6)),
  ngram_coverage: Number((0.76 - (index % 7) * 0.025).toFixed(6)),
  asr_stream_provenance: Object.fromEntries(
    ["original", "entity", "action", "context", "synonym", "keyword"].flatMap((role) => ["vi", "en"].map((language) => [`${role}:${language}`, {
      role,
      language,
      stream: `${role}:${language}`,
      rank: index + 1,
      combined_score: Number((0.88 - index * 0.003).toFixed(6)),
    }])),
  ),
  score_type: "asr_bm25_ngram",
}));

const branch3OcrMockResults = gateFrames.slice(0, 500).map((frame, index) => ({
  ...frame,
  frame_uid: `${frame.video_id}:${frame.frame_idx}`,
  point_id: index + 1,
  rank: index + 1,
  score: Number((0.91 - index * 0.004).toFixed(6)),
  ocr_raw_score: Number((0.86 - index * 0.003).toFixed(6)),
  ocr_combined_score: Number((0.86 - index * 0.003).toFixed(6)),
  ocr_normalized_score: Number((0.91 - index * 0.004).toFixed(6)),
  ocr_best_query_role: ["original", "entity", "action", "context", "synonym", "keyword"][index % 6],
  ocr_best_query_language: index % 2 === 0 ? "vi" : "en",
  ocr_text: frame.ocr_text,
  full_text: frame.ocr_text,
  bm25_raw: Number((-1.0 + index * 0.003).toFixed(6)),
  bm25_relevance: Number((0.86 - index * 0.003).toFixed(6)),
  ocr_bm25_raw: Number((-1.0 + index * 0.003).toFixed(6)),
  ocr_bm25_relevance: Number((0.86 - index * 0.003).toFixed(6)),
  token_coverage: Number((0.80 - (index % 5) * 0.03).toFixed(6)),
  ocr_token_coverage: Number((0.80 - (index % 5) * 0.03).toFixed(6)),
  ngram_coverage: Number((0.76 - (index % 7) * 0.025).toFixed(6)),
  adjacent_bigram_coverage: Number((0.76 - (index % 7) * 0.025).toFixed(6)),
  matched_keywords: ["htv"],
  matched_terms: ["htv"],
  ocr_stream_provenance: Object.fromEntries(
    ["original", "entity", "action", "context", "synonym", "keyword"].flatMap((role) => ["vi", "en"].map((language) => [`${role}:${language}`, {
      role,
      language,
      stream: `${role}:${language}`,
      rank: index + 1,
      combined_score: Number((0.86 - index * 0.003).toFixed(6)),
      bm25_raw: Number((-1.0 + index * 0.003).toFixed(6)),
      bm25_relevance: Number((0.86 - index * 0.003).toFixed(6)),
      token_coverage: Number((0.80 - (index % 5) * 0.03).toFixed(6)),
      ngram_coverage: Number((0.76 - (index % 7) * 0.025).toFixed(6)),
    }])),
  ),
  score_type: "ocr_bm25_ngram",
}));

// The lightweight mock also exposes the final KIS contract so the workspace
// can be exercised without a Qdrant container.  Keep identity fields
// canonical and make the synthetic evidence obey the same rank-only formula
// as the production service.
const kisFusionMockResults = gateFrames.slice(0, 150).map((frame, index) => {
  const rank = index + 1;
  const frameUid = `${frame.video_id}:${frame.frame_idx}`;
  const isReranked = index < 100;
  // Model a deterministic adjacent-pair swap in the final BEiT window.
  // The RRF fields retain their pre-rerank positions while the array order is
  // the final rank, so rank_delta is mathematically meaningful in mock UI.
  const preRerankRank = isReranked
    ? (rank % 2 === 0 ? rank - 1 : rank + 1)
    : rank;
  const rrfContributions = {
    branch1: 0.40 / (60 + preRerankRank),
    branch2: 0.30 / (60 + preRerankRank),
    ocr: 0.15 / (60 + preRerankRank),
    asr: 0.15 / (60 + preRerankRank),
  };
  const rrfScore = Object.values(rrfContributions).reduce((sum, value) => sum + value, 0);
  // The BEiT margin is intentionally larger than the paired RRF difference,
  // which makes the final list ordered by the displayed final score.
  const beit3Normalized = Number((0.996 - (rank - 1) * 0.004).toFixed(6));
  const rrfNormalized = Number((0.75 - (preRerankRank - 1) * 0.0005).toFixed(6));
  const finalScore = Number((0.25 * beit3Normalized + 0.75 * rrfNormalized).toFixed(6));
  const publicScore = isReranked ? finalScore : Number(rrfScore.toFixed(8));
  const beit3QueryScores = Object.fromEntries(
    ["original", "entity", "action", "context", "synonym", "keyword"].map((role, roleIndex) => [
      role,
      { cosine: Number((0.78 - index * 0.001 - roleIndex * 0.004).toFixed(6)), rank: roleIndex + 1, role, language: "en" },
    ]),
  );
  const canonical = {
    frame_uid: frameUid,
    point_id: rank,
    global_idx: rank,
    video_id: frame.video_id,
    frame_idx: frame.frame_idx,
    keyframe_n: frame.keyframe_n,
    pts_time_s: frame.pts_time_s,
    fps: frame.fps,
    image_relpath: frame.image_relpath,
    submission_string: frame.submission_string,
  };
  const rerankFields = isReranked
    ? {
      beit3_raw_cosine: Number((0.88 - index * 0.002).toFixed(6)),
      beit3_normalized: beit3Normalized,
      rrf_normalized: rrfNormalized,
      beit3_best_query_role: "original",
      beit3_best_query_language: "en",
      beit3_query_scores: beit3QueryScores,
      rank_delta: preRerankRank - rank,
      rerank_formula: {
        beit3_weight: 0.25,
        previous_weight: 0.75,
        previous_score_field: "rrf_score",
        expression: "beit3_weight * normalized_beit3 + previous_weight * normalized_rrf",
      },
    }
    : {};
  return {
    ...canonical,
    rank,
    pre_rerank_rank: preRerankRank,
    rrf_score: Number(rrfScore.toFixed(8)),
    final_score: publicScore,
    score: publicScore,
    score_type: isReranked ? "beit3_coco_cosine_blend" : "weighted_rrf",
    branch_agreement_count: 4,
    observed_branches: ["branch1", "branch2", "ocr", "asr"],
    branch_ranks: { branch1: preRerankRank, branch2: preRerankRank, ocr: preRerankRank, asr: preRerankRank },
    rrf_contributions: rrfContributions,
    branch_normalized_scores: { branch1: 0.85, branch2: 0.8, ocr: 0.7, asr: 0.7 },
    branch_provenance: {
      branch1: {
        final_score: 0.85,
        best_model: "siglip2",
        best_query_role: "original",
        best_query_language: "vi",
        model_provenance: {
          siglip2: { observed: true, raw_cosine: 0.86, normalized_score: 0.85, best_query_role: "original", best_query_language: "vi", best_query_rank: 1 },
          metaclip2: { observed: true, raw_cosine: 0.81, normalized_score: 0.78, best_query_role: "entity", best_query_language: "en", best_query_rank: 2 },
          beit3: { observed: true, raw_cosine: 0.79, normalized_score: 0.76, best_query_role: "context", best_query_language: "en", best_query_rank: 3 },
        },
      },
      branch2: {
        dense_observed: true,
        dense_raw: 0.79,
        dense_normalized: 0.8,
        dense_rank: preRerankRank,
        dense_best_query_role: "entity",
        dense_best_query_language: "en",
        sparse_observed: true,
        sparse_raw: 0.04,
        sparse_normalized: 0.65,
        sparse_rank: preRerankRank,
        sparse_best_query_role: "keyword",
        sparse_best_query_language: "en",
        hybrid_score: 0.8,
        hybrid_rank: preRerankRank,
        pre_rerank_rank: preRerankRank,
        reranked_score: 0.82,
        beit3_raw_cosine: 0.78,
        beit3_normalized: 0.76,
        beit3_best_query_role: "original",
        beit3_best_query_language: "en",
        dam_winner: { description_en: "mock DAM region", bbox: [0.1, 0.1, 0.4, 0.4] },
      },
      ocr: { ocr_text: `HTV TIN TUC ${frame.keyframe_n}`, ocr_best_query_role: "keyword", ocr_best_query_language: "vi", bm25_relevance: 0.7, token_coverage: 0.8, adjacent_bigram_coverage: 0.6, matched_terms: ["htv", "tin", "tuc"] },
      asr: { asr_transcript: `Deterministic sample transcript for keyframe ${frame.keyframe_n}.`, asr_best_query_role: "original", asr_best_query_language: "vi", bm25_relevance: 0.7, token_coverage: 0.8, adjacent_bigram_coverage: 0.6, asr_segment_id: `${frame.video_id}:s${frame.keyframe_n}`, asr_start_s: frame.pts_time_s, asr_end_s: frame.pts_time_s + 1.5 },
    },
    ...rerankFields,
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

function csvCell(value: unknown): string {
  const text = String(value ?? "");
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function officialCsvPayload(taskType: string, rows: Array<Record<string, unknown>>): Record<string, unknown> {
  const lines = rows.slice(0, 100).map((row) => {
    if (taskType === "VQA") return [row.video_id, row.frame_idx, row.answer].map(csvCell).join(",");
    if (taskType === "TRAKE") {
      const events = Array.isArray(row.events) ? row.events as Array<{ frame_idx?: number }> : [];
      return [row.video_id, ...events.map((event) => event.frame_idx)].map(csvCell).join(",");
    }
    return [row.video_id, row.frame_idx].map(csvCell).join(",");
  });
  return {
    encoding: "UTF-8",
    delimiter: ",",
    has_header: false,
    line_ending: "CRLF",
    row_count: lines.length,
    max_rows: 100,
    valid: lines.length >= 1 && lines.length <= 100,
    content: lines.join("\r\n"),
  };
}

function pathnameOf(request: IncomingMessage): string {
  return new URL(request.url ?? "/", "http://localhost").pathname;
}

async function readJsonBody(request: IncomingMessage): Promise<Record<string, unknown>> {
  const chunks: Buffer[] = [];
  for await (const chunk of request) chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  if (!chunks.length) return {};
  try {
    return JSON.parse(Buffer.concat(chunks).toString("utf8")) as Record<string, unknown>;
  } catch {
    return {};
  }
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
            experiment_mode: "cpu_qdrant_workbench",
            task_types: ["KIS"],
            modalities: ["siglip", "dam", "ocr", "asr"],
            fusion_enabled: true,
            reranking_enabled: true,
            capabilities: {
              image_search: true,
              video_timeline: true,
              submission_prepare: true,
              parser_modes: ["local", "direct"],
              qwen_fallback_control: false,
              branch1_three_model: true,
              branch2_dam_hybrid: true,
              branch3_asr: true,
              branch3_ocr: true,
              kis_fusion: true,
            },
            qwen_enabled: false,
            gemini_enabled: false,
          });
          return;
        }

        if (pathname === "/api/health") {
          sendJson(response, {
            status: "ready",
            device: "cpu",
            production_ready: false,
            qwen_enabled: false,
            gemini_enabled: false,
            optional_degraded_components: [],
            components: {
              api_process: { ready: true },
              qdrant: { ready: true },
              metadata: { ready: true },
              branch1: { ready: true },
              branch2: { ready: true },
              branch3_asr: { ready: true, required: false },
              branch3_ocr: { ready: true, required: false },
              kis_fusion: { ready: true, required: false, production_ready: false },
            },
          });
          return;
        }

        if (pathname === "/api/branch1/health") {
          sendJson(response, {
            status: "ready",
            ready: true,
            // The mock deliberately mirrors the real provenance policy: a
            // green operational gate does not prove immutable offline
            // checkpoint identity, so it must not advertise production
            // readiness.
            production_ready: false,
            fail_closed: true,
            models: {
              siglip2: { data: { ready: true }, collection: { ready: true }, text_encoder: { ready: true } },
              metaclip2: { data: { ready: true }, collection: { ready: true }, text_encoder: { ready: true } },
              beit3: { data: { ready: true }, collection: { ready: true }, text_encoder: { ready: true } },
            },
          });
          return;
        }

        if (pathname === "/api/branch2/health") {
          sendJson(response, {
            status: "ready",
            ready: true,
            production_ready: false,
            fail_closed: true,
            components: {
              dam_data: { ready: true },
              dam_collection: { ready: true },
              bm25: { ready: true },
              bge_text_encoder: { ready: true },
              beit3_collection: { ready: true },
              beit3_text_encoder: { ready: true },
              frame_mapping: { ready: true },
            },
          });
          return;
        }

        if (pathname === "/api/branch3/asr/health") {
          sendJson(response, {
            status: "ready",
            ready: true,
            production_ready: false,
            required: false,
            segments: 55168,
            mapped_segments: 55168,
            schema_version: "branch3.asr-index.v2",
            revision_verified: false,
            warnings: ["ASR offline model revision is not cryptographically verified"],
          });
          return;
        }

        if (pathname === "/api/branch3/ocr/health") {
          sendJson(response, {
            status: "ready",
            ready: true,
            production_ready: false,
            required: false,
            fail_closed: false,
            schema_version: "branch3.ocr-index.v3",
            sqlite_user_version: 3,
            frames: 247956,
            fts_frames: 247956,
            fts_unmapped_rows: 0,
            mapped_frames: 247956,
            videos: 873,
            distinct_point_ids: 247956,
            min_point_id: 1,
            max_point_id: 247956,
            manifest_build_id: "mock-ocr-build",
            internal_build_id: "mock-ocr-build",
            opened_build_id: "mock-ocr-build",
            fts_content_verified: true,
            fts_content_fingerprint: "mock-ocr-fts-fingerprint",
            lexical_contract_version: "branch3.ocr-lexical.v3",
            source_audit_performed: true,
            source_stale: false,
            source_drift_is_blocking: false,
            source_audit_age_s: 0,
            database_matches_manifest: true,
            source_matches_manifest: true,
            source_stat_matches_manifest: true,
            canonical_metadata_matches_manifest: true,
            canonical_metadata_stat_matches_manifest: true,
            canonical_fingerprint_matches: true,
            source_fingerprint_matches_manifest: true,
            source_file_count_matches_manifest: true,
            source_directory_file_count: 873,
            source_directory_matches_manifest: true,
            database_stat_matches_manifest: true,
            internal_metadata_matches: true,
            build_id_matches: true,
            connection_generation: 1,
             connection_reopened: false,
             connection_error: null,
             source_validation_age_s: 0,
             artifact_summary: {
               source_total: 873,
               source_verified: 873,
               source_failed: 0,
               hash_recomputed: 0,
             },
             artifact_diagnostics: [],
            offline_identity: {
              model_id: null,
              engine: "OCR source export",
              revision_verified: false,
            },
            revision_verified: false,
            warnings: ["OCR offline model revision is not cryptographically verified"],
          });
          return;
        }

        if (pathname === "/api/fusion/kis/health") {
          sendJson(response, {
            schema_version: "kis.fusion.health.v1",
            branch: "final_fusion",
            task_type: "KIS",
            status: "ready",
            ready: true,
            required: false,
            production_ready: false,
            fail_closed: true,
            rrf_k: 60,
            final_top_k: 150,
            rerank_top_k: 100,
            beit3_weight: 0.25,
            rrf_weight: 0.75,
            branch_weights: { branch1: 0.4, branch2: 0.3, ocr: 0.15, asr: 0.15 },
            components: {
              branch1: { ready: true, production_ready: false },
              branch2: { ready: true, production_ready: false },
              asr: { ready: true, required: false, production_ready: false },
              ocr: { ready: true, required: false, production_ready: false },
              canonical_frame_mapping: { ready: true },
              beit3: { ready: true },
              query_cache: { ready: true },
              execution_contract: {
                ready: true,
                branches: { branch1: true, branch2: true, asr: true, ocr: true },
              },
              resource_qualification: { ready: false, production_ready: false },
            },
            api_rss_bytes: 0,
            });
          return;
        }

        if (pathname === "/api/search/branch1") {
          await readJsonBody(request);
          sendJson(response, {
            schema_version: "branch1.result.v1",
            fusion_applied: true,
            reranking_applied: false,
            weights: { siglip2: 0.45, metaclip2: 0.30, beit3: 0.25 },
            per_stream_top_k: 2000,
            final_top_k: 1500,
            candidate_union_count: branch1MockResults.length,
            result_count: branch1MockResults.length,
            stream_count: 30,
            tokenizer_diagnostics: {
              siglip2: Array.from({ length: 12 }, (_, index) => ({ role: ["original", "entity", "action", "context", "synonym", "keyword"][Math.floor(index / 2)], language: index % 2 === 0 ? "vi" : "en", token_count: 8, max_tokens: 64, truncated: false })),
              metaclip2: Array.from({ length: 12 }, (_, index) => ({ role: ["original", "entity", "action", "context", "synonym", "keyword"][Math.floor(index / 2)], language: index % 2 === 0 ? "vi" : "en", token_count: 8, max_tokens: 77, truncated: false })),
              beit3: Array.from({ length: 6 }, (_, index) => ({ role: ["original", "entity", "action", "context", "synonym", "keyword"][index], language: "en", token_count: 8, max_tokens: 64, truncated: false })),
            },
            query_streams: {
              siglip2: Array.from({ length: 12 }, (_, index) => ({ role: ["original", "entity", "action", "context", "synonym", "keyword"][Math.floor(index / 2)], language: index % 2 === 0 ? "vi" : "en", stream: `${["original", "entity", "action", "context", "synonym", "keyword"][Math.floor(index / 2)]}:${index % 2 === 0 ? "vi" : "en"}` })),
              metaclip2: Array.from({ length: 12 }, (_, index) => ({ role: ["original", "entity", "action", "context", "synonym", "keyword"][Math.floor(index / 2)], language: index % 2 === 0 ? "vi" : "en", stream: `${["original", "entity", "action", "context", "synonym", "keyword"][Math.floor(index / 2)]}:${index % 2 === 0 ? "vi" : "en"}` })),
              beit3: Array.from({ length: 6 }, (_, index) => ({ role: ["original", "entity", "action", "context", "synonym", "keyword"][index], language: "en", stream: `${["original", "entity", "action", "context", "synonym", "keyword"][index]}:en` })),
            },
            candidate_count_before_gate: branch1MockResults.length,
            gate_top_k: 1500,
            future_fusion_eligible: true,
            timing: { models: {}, fusion_ms: 0.2, total_ms: 3.4 },
            results: branch1MockResults,
          });
          return;
        }

        if (pathname === "/api/search/fusion/kis") {
          await readJsonBody(request);
          sendJson(response, {
            schema_version: "kis.fusion.result.v1",
            task_type: "KIS",
            fusion_applied: true,
            fusion_method: "weighted_rrf",
            reranking_applied: true,
            rrf_k: 60,
            branch_weights: { branch1: 0.4, branch2: 0.3, ocr: 0.15, asr: 0.15 },
            branch_pool_counts: { branch1: 1500, branch2: 500, ocr: 500, asr: 500 },
            branch_pool_limits: { branch1: 1500, branch2: 500, ocr: 500, asr: 500 },
            final_top_k: 150,
            candidate_count_before_gate: 1500,
            pre_rerank_top_k: 150,
            rerank_top_k: 100,
            beit3_weight: 0.25,
            rrf_weight: 0.75,
            result_count: kisFusionMockResults.length,
            timing: { branches: {}, rrf_ms: 0.2, beit3_rerank_ms: 0.3, total_ms: 1.5 },
            rerank: { candidate_count: 150, rerank_count: 100, weights: { beit3: 0.25, previous: 0.75 }, scoring: "cosine", checkpoint_task: "BEiT-3 COCO Retrieval" },
            results: kisFusionMockResults,
          });
          return;
        }

        if (pathname === "/api/search/branch2") {
          await readJsonBody(request);
          sendJson(response, {
            schema_version: "branch2.result.v1",
            fusion_applied: true,
            reranking_applied: true,
            hybrid_weights: { dense: 0.70, sparse: 0.30 },
            rerank_weights: { beit3: 0.40, previous: 0.60 },
            per_stream_top_k: 2000,
            pre_rerank_top_k: 500,
            rerank_top_k: 100,
            result_count: branch2MockResults.length,
            candidate_count_before_gate: branch2MockResults.length,
            gate_top_k: 500,
            future_fusion_eligible: true,
            dense_candidate_count: 80,
            sparse_candidate_count: 80,
            stream_count: 18,
            query_streams: {
              dam_dense: ["original", "entity", "action", "context", "synonym", "keyword"].map((role) => ({ role, language: "en", stream: `${role}:en` })),
              bm25_sparse: ["original", "entity", "action", "context", "synonym", "keyword"].map((role) => ({ role, language: "en", stream: `${role}:en` })),
              beit3_rerank: ["original", "entity", "action", "context", "synonym", "keyword"].map((role) => ({ role, language: "en", stream: `${role}:en` })),
            },
            timing: { bge_cache_hit: false, beit_cache_hit: false, dense_ms: 1.1, sparse_ms: 0.8, total_ms: 4.7 },
            tokenizer_diagnostics: {
              bge_m3: Array.from({ length: 6 }, () => ({ token_count: 8, max_tokens: 512, truncated: false })),
              beit3: Array.from({ length: 6 }, () => ({ token_count: 8, max_tokens: 64, truncated: false })),
            },
            rerank: {
              candidate_count: branch2MockResults.length,
              rerank_count: 100,
              weights: { beit3: 0.40, previous: 0.60 },
              text_encoder_output: "language_head",
              checkpoint_task: "BEiT-3 COCO Retrieval",
              scoring: "cosine",
            },
            results: branch2MockResults,
          });
          return;
        }

        if (pathname === "/api/search/branch3/asr") {
          await readJsonBody(request);
          sendJson(response, {
            schema_version: "branch3.asr.result.v1",
            fusion_applied: false,
            reranking_applied: false,
            query_roles: ["original", "entity", "action", "context", "synonym", "keyword"],
            per_stream_top_k: 2000,
            final_top_k: 500,
            candidate_segment_count: 100,
            candidate_frame_count: 100,
            query_languages: ["vi", "en"],
            query_streams: ["original", "entity", "action", "context", "synonym", "keyword"].flatMap((role) => ["vi", "en"].map((language) => ({ role, language, stream: `${role}:${language}` }))),
            stream_count: 12,
            stream_counts: Object.fromEntries(["original", "entity", "action", "context", "synonym", "keyword"].flatMap((role) => ["vi", "en"].map((language) => [`${role}:${language}`, 100]))),
            result_count: branch3AsrMockResults.length,
            candidate_count_before_gate: branch3AsrMockResults.length,
            gate_top_k: 500,
            future_fusion_eligible: true,
            timing: { asr_ms: 2.1, total_ms: 2.1 },
            results: branch3AsrMockResults,
          });
          return;
        }

        if (pathname === "/api/search/branch3/ocr") {
          await readJsonBody(request);
          sendJson(response, {
            schema_version: "branch3.ocr.result.v1",
            fusion_applied: false,
            reranking_applied: false,
            query_roles: ["original", "entity", "action", "context", "synonym", "keyword"],
            query_languages: ["vi", "en"],
            query_streams: ["original", "entity", "action", "context", "synonym", "keyword"].flatMap((role) => ["vi", "en"].map((language) => ({ role, language, stream: `${role}:${language}` }))),
            stream_count: 12,
            per_stream_top_k: 2000,
            final_top_k: 500,
            candidate_frame_count: 100,
            stream_counts: Object.fromEntries(["original", "entity", "action", "context", "synonym", "keyword"].flatMap((role) => ["vi", "en"].map((language) => [`${role}:${language}`, 100]))),
            result_count: branch3OcrMockResults.length,
            candidate_count_before_gate: branch3OcrMockResults.length,
            gate_top_k: 500,
            future_fusion_eligible: true,
            timing: { ocr_ms: 2.1, total_ms: 2.1 },
            results: branch3OcrMockResults,
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
              is_temporal_trake: true,
              trake_events: [
                {
                  order: 1,
                  description: "A yellow cyclist crosses the finish line",
                  scene_en: "A low finish-line shot of a cyclist in a yellow jersey and black shorts",
                  objects_en: ["yellow cyclist"],
                  speech_vi: "",
                  ocr_keywords: [],
                },
                {
                  order: 2,
                  description: "A blue cyclist in black shorts follows",
                  scene_en: "A low finish-line shot of a cyclist in a blue jersey and black shorts",
                  objects_en: ["blue cyclist in black shorts"],
                  speech_vi: "",
                  ocr_keywords: [],
                },
                {
                  order: 3,
                  description: "A blue cyclist in red shorts arrives next",
                  scene_en: "A low finish-line shot of a cyclist in a blue jersey and red shorts",
                  objects_en: ["blue cyclist in red shorts"],
                  speech_vi: "",
                  ocr_keywords: [],
                },
              ],
              vqa_question: "",
            },
          });
          return;
        }

        if (pathname === "/api/search/temporal-intersection") {
          const eventDefinitions = [
            {
              order: 1,
              description: "A yellow cyclist crosses the finish line",
              query: "A low finish-line shot of a cyclist in a yellow jersey and black shorts",
              frame: frames[3],
              rank: 12,
            },
            {
              order: 2,
              description: "A blue cyclist in black shorts follows",
              query: "A low finish-line shot of a cyclist in a blue jersey and black shorts",
              frame: frames[8],
              rank: 27,
            },
            {
              order: 3,
              description: "A blue cyclist in red shorts arrives next",
              query: "A low finish-line shot of a cyclist in a blue jersey and red shorts",
              frame: frames[13],
              rank: 39,
            },
          ];
          const matchedEvents = eventDefinitions.map((event) => ({
            ...event.frame,
            rank: event.rank,
            score: Number((0.82 - event.order * 0.03).toFixed(4)),
            score_type: "cosine",
            event_order: event.order,
            event_description: event.description,
            event_query: event.query,
          }));
          const gapsSeconds = matchedEvents.slice(1).map((event, index) =>
            Number((event.pts_time_s - matchedEvents[index].pts_time_s).toFixed(4))
          );
          const meanEventScore = matchedEvents.reduce((total, event) => total + event.score, 0)
            / matchedEvents.length;
          const minimumEventScore = Math.min(...matchedEvents.map((event) => event.score));
          sendJson(response, {
            operation: "ordered_siglip_intersection",
            experiment_mode: "nofusion_temporal_intersection",
            event_count: eventDefinitions.length,
            top_k_per_event: 300,
            top_k_sequences: 100,
            paths_per_video: 1,
            sequence_reservoir_size: 100,
            sequence_reservoir_count: 1,
            max_gap_seconds: 30,
            anchor_query: "A low ground-level bicycle race finish line",
            anchor_query_applied: true,
            anchor_pool: {
              query: "A low ground-level bicycle race finish line",
              score_type: "cosine",
              result_count: 100,
              candidate_video_count: 7,
            },
            intersection_video_count: 1,
            ordered_sequence_count: 1,
            same_modality_event_aggregation_applied: true,
            same_modality_event_aggregation: "mean_context_anchor_and_minimum_event_then_event_mean",
            cross_modal_fusion_applied: false,
            fusion_applied: false,
            reranking_applied: false,
            ranking_rule: "minimum event cosine desc, mean cosine desc, global rank sum asc, span asc, video_id asc",
            event_pools: eventDefinitions.map((event) => ({
              order: event.order,
              description: event.description,
              query: event.query,
              query_source: "events[].global_scene_en",
              score_type: "cosine",
              result_count: 100,
              candidate_video_count: 7,
            })),
            sequences: [{
              rank: 1,
              video_id: MOCK_VIDEO_ID,
              context_anchor_rank: 4,
              context_anchor_score: 0.86,
              context_anchor_frame: frames[2],
              minimum_event_score: Number(minimumEventScore.toFixed(6)),
              mean_event_score: Number(meanEventScore.toFixed(6)),
              sequence_score: Number(((0.86 + minimumEventScore) / 2).toFixed(6)),
              score_type: "mean_context_anchor_and_minimum_event_raw_siglip_cosine",
              ranking_values: {
                sequence_score: Number(((0.86 + minimumEventScore) / 2).toFixed(6)),
                context_anchor_score: 0.86,
                minimum_event_score: Number(minimumEventScore.toFixed(6)),
                mean_event_score: Number(meanEventScore.toFixed(6)),
              },
              global_rank_sum: matchedEvents.reduce((total, event) => total + event.rank, 0),
              span_seconds: Number((matchedEvents.at(-1)!.pts_time_s - matchedEvents[0].pts_time_s).toFixed(4)),
              gaps_seconds: gapsSeconds,
              matched_events: matchedEvents,
            }],
            reserve_sequences: [],
            execution_time_ms: 3.4,
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

        if (pathname === "/api/search/image") {
          const resultFrames = rotatedFrames(11).slice(0, 50).map((frame, index) => ({
            ...frame,
            rank: index + 1,
            score: Number((0.94 - index * 0.004).toFixed(4)),
            score_type: "cosine",
            retrieval_modality: "image",
          }));
          sendJson(response, {
            task_type: "KIS",
            experiment_mode: "nofusion",
            operation: "image_query",
            query_modality: "image",
            scope: "global",
            video_id: null,
            evaluated_frames: frames.length,
            fusion_applied: false,
            reranking_applied: false,
            modality_result: {
              modality: "siglip",
              display_name: "SigLIP image similarity",
              status: "ok",
              reason: "",
              query: "<uploaded image>",
              query_source: "uploaded_image",
              score_type: "cosine",
              score_description: "Raw cosine between uploaded-image SigLIP vector and full-frame image",
              result_count: resultFrames.length,
              execution_time_ms: 1.2,
              results: resultFrames,
            },
            execution_time_ms: 1.5,
          });
          return;
        }

        if (pathname === "/api/submission/prepare") {
          const body = await readJsonBody(request);
          const taskType = String(body.task_type || "KIS").toUpperCase();
          const queryId = String(body.query_id || "1");
          const canonicalFrameKeys = new Set(frames.map((frame) => `${frame.video_id}:${frame.frame_idx}`));
          if (taskType === "TRAKE") {
            type SubmittedSequence = { video_id?: string; events?: Array<{ event_order?: number; frame_idx?: number; pts_time_s?: number }> };
            const manualSequences = (Array.isArray(body.manual_sequences) ? body.manual_sequences : []) as SubmittedSequence[];
            const candidateSequences = (Array.isArray(body.candidate_sequences) ? body.candidate_sequences : []) as SubmittedSequence[];
            const expected = Number(body.event_count) || manualSequences[0]?.events?.length || candidateSequences[0]?.events?.length || 0;
            const targetRows = Math.min(100, Math.max(1, Number(body.target_rows) || 100));
            const normalizeSequence = (sequence: SubmittedSequence): Record<string, unknown> | null => {
              const sequenceVideoId = String(sequence?.video_id || "").toUpperCase().replaceAll("-", "_");
              const events = Array.isArray(sequence?.events) ? [...sequence.events] : [];
              if (!sequenceVideoId || events.length !== expected || expected < 1) return null;
              events.sort((left, right) => Number(left.event_order) - Number(right.event_order));
              const canonicalEvents = events.map((event, index) => {
                const canonical = frames.find((frame) => frame.video_id === sequenceVideoId && frame.frame_idx === Number(event.frame_idx));
                if (!canonical || Number(event.event_order || index + 1) !== index + 1) return null;
                return { ...event, event_order: index + 1, pts_time_s: event.pts_time_s ?? canonical.pts_time_s };
              });
              if (canonicalEvents.some((event) => event === null)) return null;
              const typedEvents = canonicalEvents as Array<{ event_order: number; frame_idx?: number; pts_time_s?: number }>;
              const increasing = typedEvents.every((event, index) => index === 0
                || Number(event.frame_idx) > Number(typedEvents[index - 1].frame_idx));
              return increasing ? { video_id: sequenceVideoId, events: typedEvents } : null;
            };
            const invalidManual = manualSequences.find((sequence) => normalizeSequence(sequence) === null);
            if (invalidManual) {
              sendJson(response, {
                ok: false,
                task_type: "TRAKE",
                query_id: queryId,
                row_count: 0,
                complete: false,
                valid_for_download: false,
                missing_rows: targetRows,
                rows: [],
                official_csv: officialCsvPayload("TRAKE", []),
                warnings: [],
                errors: ["A manual TRAKE frame is not present in the canonical mock timeline."],
              }, 400);
              return;
            }
            const seen = new Set<string>();
            const rows: Array<Record<string, unknown>> = [];
            [...manualSequences, ...candidateSequences].forEach((sequence) => {
              if (rows.length >= targetRows) return;
              const normalized = normalizeSequence(sequence);
              if (!normalized) return;
              const events = normalized.events as Array<{ frame_idx?: number }>;
              const identity = `${normalized.video_id}:${events.map((event) => event.frame_idx).join(":")}`;
              if (seen.has(identity)) return;
              seen.add(identity);
              rows.push(normalized);
            });
            const complete = rows.length === targetRows;
            const validForDownload = rows.length >= 1 && rows.length <= 100;
            sendJson(response, {
              ok: validForDownload,
              task_type: "TRAKE",
              query_id: queryId,
              row_count: rows.length,
              complete,
              valid_for_download: validForDownload,
              missing_rows: Math.max(0, targetRows - rows.length),
              rows,
              official_csv: officialCsvPayload("TRAKE", rows),
              warnings: complete ? [] : [`Only ${rows.length} complete canonical sequences were available; BTC accepts up to 100 rows.`],
              errors: validForDownload ? [] : ["No complete ordered TRAKE sequence is available."],
            });
            return;
          }
          const answer = String(body.vqa_answer || "");
          const manual = Array.isArray(body.manual_selections) ? body.manual_selections : [];
          const candidates = Array.isArray(body.candidate_reservoir) ? body.candidate_reservoir : [];
          const invalidManual = manual.find((value) => {
            if (!value || typeof value !== "object") return true;
            const item = value as { video_id?: string; frame_idx?: number };
            const videoId = String(item.video_id || "").toUpperCase().replaceAll("-", "_");
            return !canonicalFrameKeys.has(`${videoId}:${Number(item.frame_idx)}`);
          });
          if (invalidManual) {
            sendJson(response, {
              ok: false,
              task_type: taskType,
              query_id: queryId,
              row_count: 0,
              complete: false,
              missing_rows: 100,
              rows: [],
              warnings: [],
              errors: ["A manual frame is not present in the canonical mock timeline."],
            }, 400);
            return;
          }
          const seen = new Set<string>();
          const chosen: Array<{ video_id: string; frame_idx: number }> = [];
          [...manual, ...candidates, ...frames].forEach((value) => {
            if (chosen.length >= 100 || !value || typeof value !== "object") return;
            const item = value as { video_id?: string; frame_idx?: number };
            const videoId = String(item.video_id || MOCK_VIDEO_ID).toUpperCase().replaceAll("-", "_");
            const frameIdx = Number(item.frame_idx);
            const key = `${videoId}:${frameIdx}`;
            if (!Number.isInteger(frameIdx) || frameIdx < 0 || !canonicalFrameKeys.has(key) || seen.has(key)) return;
            seen.add(key);
            chosen.push({ video_id: videoId, frame_idx: frameIdx });
          });
          const answerMissing = taskType === "VQA" && !answer.trim();
          const targetRows = Math.min(100, Math.max(1, Number(body.target_rows) || 100));
          const limited = chosen.slice(0, targetRows);
          const answerTooLong = Array.from(answer).length > 100;
          const rows = limited.map((item) => ({ ...item, ...(taskType === "VQA" ? { answer } : {}) }));
          const validForDownload = rows.length >= 1 && rows.length <= 100 && !answerMissing && !answerTooLong;
          sendJson(response, {
            ok: validForDownload,
            task_type: taskType,
            query_id: queryId,
            row_count: rows.length,
            complete: rows.length === targetRows && !answerMissing && !answerTooLong,
            valid_for_download: validForDownload,
            missing_rows: Math.max(0, targetRows - rows.length),
            rows,
            official_csv: officialCsvPayload(taskType, rows),
            warnings: [],
            errors: [
              ...(answerMissing ? ["A human Q&A answer is required."] : []),
              ...(answerTooLong ? ["Q&A answer cannot exceed 100 characters."] : []),
            ],
          });
          return;
        }

        const drilldownMatch = pathname.match(/^\/api\/video\/([^/]+)\/search\/siglip$/);
        if (drilldownMatch) {
          const videoId = decodeURIComponent(drilldownMatch[1]).toUpperCase().replaceAll("-", "_");
          const resultFrames = rotatedFrames(0).map((frame, index) => ({
            ...frame,
            video_id: videoId,
            image_relpath: `keyframes/${videoId}/${frame.filename}`,
            submission_string: `${videoId}, ${frame.frame_idx}`,
            rank: index + 1,
            score_type: "cosine",
            scope: "video",
            scope_video_id: videoId,
          }));
          sendJson(response, {
            experiment_mode: "nofusion",
            operation: "manual_video_drilldown",
            video_id: videoId,
            scope_selected_by_user: true,
            evaluated_frames: frames.length,
            fusion_applied: false,
            reranking_applied: false,
            execution_time_ms: 1.6,
            modality_result: {
              modality: "siglip",
              display_name: `SigLIP inside ${videoId}`,
              status: "ok",
              reason: "",
              query: "Vietnamese morning news broadcast",
              query_source: "global_scene_en",
              score_type: "cosine",
              score_description: `Raw full-frame image/text cosine restricted to ${videoId}; no modality scores are combined`,
              result_count: resultFrames.length,
              evaluated_frames: frames.length,
              execution_time_ms: 1.4,
              scope: "video",
              video_id: videoId,
              fusion_applied: false,
              reranking_applied: false,
              results: resultFrames,
            },
          });
          return;
        }

        if (pathname === "/api/discover/dam-to-siglip") {
          const makeCascade = (objectIndex: number, objectQuery: string, offset: number) => {
            const resultFrames = rotatedFrames(offset).slice(0, 20).map((frame, index) => ({
              ...frame,
              rank: index + 1,
              score_type: "cosine",
              scope: "dam_to_siglip_cascade",
              video_scope_rank: (index % 10) + 1,
              discovery_object_index: objectIndex,
              discovery_object_query: objectQuery,
              candidate_video_order: 1,
              dam_discovery_rank: objectIndex + 2,
              dam_discovery_frame_idx: frames[offset].frame_idx,
              dam_discovery_keyframe_n: frames[offset].keyframe_n,
              dam_discovery_score: 0.72 - objectIndex * 0.01,
              dam_discovery_score_type: "best_region_cosine",
            }));
            return {
              cascade_id: `dam_object_${objectIndex}`,
              display_name: `DAM object ${objectIndex} → scoped SigLIP`,
              object_query: objectQuery,
              object_query_source: `objects_en[${objectIndex - 1}]`,
              dam_score_type: "best_region_cosine",
              dam_frames_considered: 20,
              candidate_video_count: 1,
              candidate_videos: [{
                candidate_video_order: 1,
                video_id: MOCK_VIDEO_ID,
                dam_raw_frame_rank: objectIndex + 2,
                dam_frame_idx: frames[offset].frame_idx,
                dam_keyframe_n: frames[offset].keyframe_n,
                dam_score: 0.72 - objectIndex * 0.01,
                dam_score_type: "best_region_cosine",
                evaluated_frames: frames.length,
              }],
              siglip_query: "Vietnamese morning news broadcast",
              siglip_query_source: "global_scene_en",
              siglip_score_type: "cosine",
              siglip_frames_per_video: 10,
              evaluated_video_frames: frames.length,
              result_count: resultFrames.length,
              results: resultFrames,
              final_ranking: "Raw SigLIP cosine over scoped frames",
              dam_score_used_in_final_rank: false,
              fusion_applied: false,
              cross_modal_gating_applied: true,
              learned_reranker_applied: false,
            };
          };
          const cascades = [
            makeCascade(1, "news presenter", 0),
            makeCascade(2, "city", 10),
          ];
          sendJson(response, {
            operation: "dam_to_siglip_discovery_cascade",
            experiment_mode: "nofusion_with_explicit_cascade",
            cascade_applied: true,
            cross_modal_gating_applied: true,
            fusion_applied: false,
            dam_score_used_in_final_rank: false,
            learned_reranker_applied: false,
            dam_top_frames_per_object: 20,
            siglip_top_frames_per_video: 10,
            object_query_count: 2,
            unique_candidate_video_count: 1,
            unique_evaluated_frames: frames.length,
            result_count: cascades.reduce((total, cascade) => total + cascade.result_count, 0),
            siglip_query_diagnostics: { token_count: 6, max_tokens: 64, truncated: false },
            cascades,
            execution_time_ms: 2.8,
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
            sendJson(response, {
              author: "AIC local mock",
              length: 240,
              title: `${videoId} mock video`,
              watch_url: "https://www.youtube.com/watch?v=aqz-KE-bpKQ",
            });
          }
          return;
        }

        const keyframesMatch = pathname.match(/^\/api\/video\/([^/]+)\/keyframes$/);
        if (keyframesMatch) {
          sendJson(response, { video_id: decodeURIComponent(keyframesMatch[1]), total_keyframes: frames.length, keyframes: frames });
          return;
        }

        const timelineMatch = pathname.match(/^\/api\/video\/([^/]+)\/timeline$/);
        if (timelineMatch) {
          const videoId = decodeURIComponent(timelineMatch[1]).toUpperCase().replaceAll("-", "_");
          sendJson(response, {
            video_id: videoId,
            fps: 30,
            keyframe_count: frames.length,
            keyframes: frames.map((frame) => ({ ...frame, video_id: videoId })),
          });
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
