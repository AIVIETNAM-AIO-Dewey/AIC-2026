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
            {hit.ocr_match ? (
              <div className="ocr-match" data-match={hit.ocr_match.match_type}>
                <strong>{hit.ocr_match.match_type.replace("_", " ")}</strong>
                <small>Lexical: {hit.ocr_match.lexical_score.toFixed(3)}</small>
                <small>
                  Levenshtein: {hit.ocr_match.fuzzy_similarity === null ? "tắt" : hit.ocr_match.fuzzy_similarity.toFixed(3)}
                </small>
                <small>Final: {hit.ocr_match.final_score.toFixed(3)}</small>
                <small>
                  Lý do: {hit.ocr_match.match_type === "exact"
                    ? "cụm chữ khớp trực tiếp"
                    : hit.ocr_match.match_type === "accent_folded"
                      ? "khớp sau khi bỏ dấu"
                      : hit.ocr_match.match_type === "fuzzy"
                        ? "lỗi OCR gần truy vấn theo edit distance"
                        : "ứng viên gần theo character trigram"}
                </small>
              </div>
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
