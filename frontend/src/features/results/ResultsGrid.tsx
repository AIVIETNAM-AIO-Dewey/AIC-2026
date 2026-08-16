import type { FrameHit } from "../../api/client";
export function ResultsGrid({ hits, basket, toggle }: { hits: FrameHit[]; basket: FrameHit[]; toggle: (hit: FrameHit) => void }) {
  return <section className="grid">{hits.map(hit => <article key={`${hit.video_id}:${hit.frame_idx}`}><img src={hit.image_url} alt={`${hit.video_id} frame ${hit.frame_idx}`} /><strong>#{hit.rank} · {hit.video_id}</strong><small>frame_idx {hit.frame_idx} · {hit.pts_time_s.toFixed(2)}s</small><small>{Object.entries(hit.modality_scores).map(([name, score]) => `${name}: ${score.toFixed(3)}`).join(" · ")}</small><button onClick={() => toggle(hit)}>{basket.some(item => item.video_id === hit.video_id && item.frame_idx === hit.frame_idx) ? "Bỏ chọn" : "Chọn"}</button></article>)}</section>;
}
