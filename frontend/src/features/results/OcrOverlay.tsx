import type { StructuredOcr } from "../../api/client";

export function OcrOverlay({ ocr }: { ocr: StructuredOcr }) {
  const lines = ocr.lines.filter((line) => line.polygon_xy && line.polygon_xy.length >= 3);
  return (
    <svg
      className="ocr-overlay"
      viewBox={`0 0 ${ocr.width} ${ocr.height}`}
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label={`${lines.length} vùng OCR`}
    >
      {lines.map((line) => (
        <polygon
          key={line.line_id}
          points={line.polygon_xy!.map(([x, y]) => `${x},${y}`).join(" ")}
          className={line.accepted ? "ocr-polygon accepted" : "ocr-polygon rejected"}
          vectorEffect="non-scaling-stroke"
        >
          <title>
            {line.raw_text} · confidence {line.confidence === null ? "N/A" : line.confidence.toFixed(3)}
          </title>
        </polygon>
      ))}
    </svg>
  );
}
