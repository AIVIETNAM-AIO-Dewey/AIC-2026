import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";

import { OcrOverlay } from "./OcrOverlay";

afterEach(cleanup);

it("renders native-coordinate polygons with the source frame viewBox", () => {
  const { container } = render(
    <OcrOverlay
      ocr={{
        terminal_status: "success",
        full_text: "non sông liền một dải",
        width: 1280,
        height: 720,
        run_id: "ocr-run",
        model_revisions: ["PP-OCRv6-small@fixture"],
        source_image_sha256: "a".repeat(64),
        lines: [
          {
            line_id: "line-0001",
            raw_text: "NON SÔNG LIỀN MỘT DẢI",
            normalized_text: "non sông liền một dải",
            confidence: 0.856,
            accepted: true,
            polygon_xy: [[100, 200], [600, 200], [600, 260], [100, 260]],
            polygon_clamped: false,
            reading_order: 0,
          },
        ],
      }}
    />,
  );

  const overlay = screen.getByRole("img", { name: "1 vùng OCR" });
  expect(overlay.getAttribute("viewBox")).toBe("0 0 1280 720");
  expect(overlay.getAttribute("preserveAspectRatio")).toBe("xMidYMid meet");
  expect(container.querySelector("polygon")?.getAttribute("points")).toBe(
    "100,200 600,200 600,260 100,260",
  );
  expect(screen.getByText(/confidence 0.856/)).toBeTruthy();
});
