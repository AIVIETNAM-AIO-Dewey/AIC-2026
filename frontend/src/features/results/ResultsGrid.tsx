import type { FrameHit } from "../../api/client";
import { OcrOverlay } from "./OcrOverlay";

export function ResultsGrid({
  hits,
  basket,
  toggle,
}: {
  hits: FrameHit[];
  basket: FrameHit[];
  toggle: (hit: FrameHit) => void;
}) {
  return (
    <section className="grid" aria-label="Danh sách keyframe">
      {hits.map((hit) => {
        const selected = basket.some(
          (item) => item.video_id === hit.video_id && item.frame_idx === hit.frame_idx,
        );
        return (
          <article key={`${hit.video_id}:${hit.frame_idx}`}>
            <div className="frame-preview">
              <img src={hit.image_url} alt={`${hit.video_id} frame ${hit.frame_idx}`} />
              {hit.ocr ? <OcrOverlay ocr={hit.ocr} /> : null}
            </div>
            <strong>
              #{hit.rank} · {hit.video_id}
            </strong>
            <small>
              frame_idx {hit.frame_idx} · {hit.pts_time_s.toFixed(2)}s
            </small>
            {hit.ocr ? (
              <details className="ocr-details">
                <summary>
                  OCR {hit.ocr.terminal_status} · {hit.ocr.lines.length} dòng
                </summary>
                <small>{hit.ocr.full_text || "Không có text"}</small>
                <small>Model: {hit.ocr.model_revisions.join(", ")}</small>
              </details>
            ) : null}
            <small>
              {Object.entries(hit.modality_scores)
                .map(([name, score]) => `${name}: ${score.toFixed(3)}`)
                .join(" · ")}
            </small>
            <button type="button" onClick={() => toggle(hit)}>
              {selected ? "Bỏ chọn" : "Chọn frame"}
            </button>
          </article>
        );
      })}
    </section>
  );
}
