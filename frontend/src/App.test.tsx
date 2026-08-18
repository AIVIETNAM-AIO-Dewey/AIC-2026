import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import App from "./App";

const capability = (kis: boolean, qa = kis, trake = kis) => ({
  qdrant_ready: kis,
  openai_configured: qa,
  image_answers_enabled: true,
  search_ready: kis,
  collections: {},
  models: {},
  tasks: {
    kis: { ready: kis, missing: kis ? [] : ["frames_sparse_current"] },
    qa: { ready: qa, missing: qa ? [] : ["gpt4o"] },
    trake: { ready: trake, missing: trake ? [] : ["frames_dense_current"] },
    ocr: { ready: kis, missing: kis ? [] : ["ocr_current"] },
  },
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("readiness", () => {
  it("shows a disabled search form until KIS artifacts are ready", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: true, json: async () => capability(false) }),
    );
    render(<App />);

    expect(await screen.findByText("Pipeline tìm kiếm chưa sẵn sàng")).toBeTruthy();
    expect(screen.getByLabelText("Vietnamese query")).toHaveProperty("disabled", true);
    expect(screen.getByRole("button", { name: "Tìm kiếm" })).toHaveProperty(
      "disabled",
      true,
    );
    expect(screen.getByText("Kết quả keyframe")).toBeTruthy();
  });

  it("enables KIS search and keeps unavailable modes disabled after ingest", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: true, json: async () => capability(true, false, false) }),
    );
    render(<App />);

    expect(await screen.findByLabelText("Vietnamese query")).toHaveProperty(
      "disabled",
      false,
    );
    expect(screen.getByRole("button", { name: "Tìm kiếm" })).toHaveProperty(
      "disabled",
      false,
    );
    expect(screen.getByRole("tab", { name: "QA" })).toHaveProperty("disabled", true);
    expect(screen.getByRole("tab", { name: "TRAKE" })).toHaveProperty("disabled", true);
    fireEvent.click(screen.getByRole("tab", { name: "KIS" }));
    await waitFor(() =>
      expect(screen.getByRole("tab", { name: "KIS" }).getAttribute("aria-selected")).toBe(
        "true",
      ),
    );
  });

  it("runs OCR-only search with a fuzzy toggle and renders score reasons", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = input.toString();
      if (url.endsWith("/api/v1/capabilities")) {
        return { ok: true, json: async () => capability(true, false, false) };
      }
      if (url.endsWith("/api/v1/ocr/jobs")) {
        return {
          ok: true,
          json: async () => ({
            enabled: false,
            model_id: "ppocrv6-small",
            active_manifest_id: null,
            started_at: null,
            last_exit_code: null,
            datasets: [],
          }),
        };
      }
      if (url.endsWith("/api/v1/ocr/search")) {
        const request = JSON.parse(init?.body as string);
        expect(request.fuzzy).toBe(false);
        return {
          ok: true,
          json: async () => ({
            request_id: "request-1",
            task_type: "ocr",
            query: request.query,
            normalized_query: request.query,
            fuzzy_enabled: false,
            strategies: ["exact_tokens", "accent_folded_tokens", "character_trigrams"],
            latency_ms: 3.2,
            results: [
              {
                rank: 1,
                score: 9,
                video_id: "L23_V001",
                frame_idx: 7,
                pts_time_s: 7,
                image_url: "/frame.jpg",
                modality_scores: { ocr: 9 },
                evidence: [],
                ocr_match: {
                  query: request.query,
                  normalized_query: request.query,
                  matched_text: "NON SÔNG LIỀN MỘT DẢI",
                  lexical_score: 9,
                  fuzzy_similarity: null,
                  final_score: 9,
                  match_type: "accent_folded",
                  fuzzy_enabled: false,
                },
              },
            ],
          }),
        };
      }
      throw new Error(`Unexpected fetch ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);

    const ocrTab = await screen.findByRole("tab", { name: "OCR" });
    await waitFor(() => expect(ocrTab).toHaveProperty("disabled", false));
    fireEvent.click(ocrTab);
    fireEvent.click(screen.getByRole("checkbox", { name: /Fuzzy OCR/ }));
    fireEvent.change(screen.getByLabelText("Vietnamese query"), {
      target: { value: "non song lien mot dai" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Tìm kiếm" }));

    expect(await screen.findByText("accent folded")).toBeTruthy();
    expect(screen.getByText("Lexical: 9.000")).toBeTruthy();
    expect(screen.getByText("Levenshtein: tắt")).toBeTruthy();
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/ocr/search",
      expect.objectContaining({ method: "POST" }),
    );
  });
});
