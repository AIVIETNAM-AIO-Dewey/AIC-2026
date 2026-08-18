export type TaskType = "kis" | "qa" | "trake" | "ocr";
export type SearchTaskType = Exclude<TaskType, "ocr">;

export type OcrLine = {
  line_id: string;
  raw_text: string;
  normalized_text: string;
  confidence: number | null;
  accepted: boolean;
  polygon_xy: [number, number][] | null;
  polygon_clamped: boolean;
  reading_order: number;
};

export type StructuredOcr = {
  terminal_status: "success" | "empty" | "error";
  full_text: string;
  width: number;
  height: number;
  run_id: string;
  model_revisions: string[];
  source_image_sha256: string | null;
  lines: OcrLine[];
};

export type OcrMatch = {
  query: string;
  normalized_query: string;
  matched_text: string;
  lexical_score: number;
  fuzzy_similarity: number | null;
  final_score: number;
  match_type: "exact" | "accent_folded" | "fuzzy" | "trigram_candidate";
  fuzzy_enabled: boolean;
};

export type FrameHit = {
  rank: number;
  score: number;
  video_id: string;
  frame_idx: number;
  keyframe_n?: number;
  pts_time_s: number;
  image_url: string;
  modality_scores: Record<string, number>;
  evidence: { modality: string; text?: string; score: number }[];
  ocr?: StructuredOcr | null;
  ocr_match?: OcrMatch | null;
};

export type Sequence = {
  rank: number;
  video_id: string;
  score: number;
  events: { event_index: number; frame: FrameHit }[];
};

export type SearchResponse = {
  request_id: string;
  task_type: TaskType;
  degraded: boolean;
  results: FrameHit[];
  sequences: Sequence[];
  answer?: string;
  confidence?: number;
  evidence_frame_uids: string[];
  ocr_search?: {
    query: string;
    normalized_query: string;
    fuzzy_enabled: boolean;
    strategies: string[];
    latency_ms: number;
  };
};

export type TaskCapability = { ready: boolean; missing: string[] };

export type Capabilities = {
  qdrant_ready: boolean;
  openai_configured: boolean;
  image_answers_enabled: boolean;
  search_ready: boolean;
  tasks: Record<TaskType, TaskCapability>;
  collections: Record<string, boolean>;
  models: Record<string, boolean>;
};

export type OcrDatasetStatus = {
  manifest_id: string;
  status: "not_started" | "running" | "interrupted" | "failed" | "completed";
  total_frames: number;
  processed_frames: number;
  remaining_frames: number;
  counters: Record<string, number>;
  output_exists: boolean;
};

export type OcrJobs = {
  enabled: boolean;
  model_id: string;
  active_manifest_id: string | null;
  started_at: string | null;
  last_exit_code: number | null;
  datasets: OcrDatasetStatus[];
};

const base = import.meta.env.VITE_API_BASE ?? "";

export async function search(
  task_type: SearchTaskType,
  raw_query_vi: string,
): Promise<SearchResponse> {
  const response = await fetch(`${base}/api/v1/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      task_type,
      raw_query_vi,
      top_k: 100,
      use_images_for_answer: true,
    }),
  });
  if (!response.ok) throw new Error((await response.json()).detail ?? "Tìm kiếm thất bại");
  return response.json();
}

export async function searchOcr(
  query: string,
  fuzzy: boolean,
): Promise<SearchResponse> {
  const response = await fetch(`${base}/api/v1/ocr/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, fuzzy, top_k: 100 }),
  });
  if (!response.ok) throw new Error((await response.json()).detail ?? "Tìm OCR thất bại");
  const payload = await response.json();
  return {
    request_id: payload.request_id,
    task_type: "ocr",
    degraded: false,
    results: payload.results,
    sequences: [],
    evidence_frame_uids: [],
    ocr_search: {
      query: payload.query,
      normalized_query: payload.normalized_query,
      fuzzy_enabled: payload.fuzzy_enabled,
      strategies: payload.strategies,
      latency_ms: payload.latency_ms,
    },
  };
}

async function ocrJobRequest(path: string, manifest_id?: string): Promise<OcrJobs> {
  const response = await fetch(`${base}${path}`, {
    method: manifest_id ? "POST" : "GET",
    headers: manifest_id ? { "Content-Type": "application/json" } : undefined,
    body: manifest_id ? JSON.stringify({ manifest_id }) : undefined,
  });
  if (!response.ok) throw new Error((await response.json()).detail ?? "OCR job thất bại");
  return response.json();
}

export const getOcrJobs = () => ocrJobRequest("/api/v1/ocr/jobs");
export const runOcrJob = (manifestId: string) =>
  ocrJobRequest("/api/v1/ocr/jobs/run", manifestId);

export async function indexOcrJob(manifestId: string): Promise<Record<string, unknown>> {
  const response = await fetch(`${base}/api/v1/ocr/jobs/index`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ manifest_id: manifestId }),
  });
  if (!response.ok) throw new Error((await response.json()).detail ?? "Index OCR thất bại");
  return response.json();
}

export async function getCapabilities(): Promise<Capabilities> {
  const response = await fetch(`${base}/api/v1/capabilities`);
  if (!response.ok) throw new Error("Backend chưa sẵn sàng");
  return response.json();
}
