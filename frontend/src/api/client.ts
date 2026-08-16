export type TaskType = "kis" | "qa" | "trake";
export type FrameHit = { rank: number; score: number; video_id: string; frame_idx: number; keyframe_n?: number; pts_time_s: number; image_url: string; modality_scores: Record<string, number>; evidence: { modality: string; text?: string; score: number }[] };
export type Sequence = { rank: number; video_id: string; score: number; events: { event_index: number; frame: FrameHit }[] };
export type SearchResponse = { request_id: string; task_type: TaskType; degraded: boolean; results: FrameHit[]; sequences: Sequence[]; answer?: string; confidence?: number; evidence_frame_uids: string[] };

const base = import.meta.env.VITE_API_BASE ?? "";
export async function search(task_type: TaskType, raw_query_vi: string): Promise<SearchResponse> {
  const response = await fetch(`${base}/api/v1/search`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ task_type, raw_query_vi, top_k: 100, use_images_for_answer: true }) });
  if (!response.ok) throw new Error((await response.json()).detail ?? "Search failed");
  return response.json();
}
