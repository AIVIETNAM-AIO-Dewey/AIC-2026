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
  },
});

const emptySearchResponse = {
  request_id: "request-1",
  task_type: "kis",
  degraded: false,
  results: [],
  sequences: [],
  evidence_frame_uids: [],
};

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("readiness", () => {
  it("shows a disabled structured-query form until KIS artifacts are ready", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: true, json: async () => capability(false) }),
    );
    render(<App />);

    expect(await screen.findByText("Pipeline tìm kiếm chưa sẵn sàng")).toBeTruthy();
    expect(screen.getByLabelText("Structured query JSON")).toHaveProperty(
      "disabled",
      true,
    );
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

    expect(await screen.findByLabelText("Structured query JSON")).toHaveProperty(
      "disabled",
      false,
    );
    expect(screen.getByRole("button", { name: "Tìm kiếm" })).toHaveProperty(
      "disabled",
      false,
    );
    expect(screen.getByRole("tab", { name: "QA" })).toHaveProperty("disabled", true);
    expect(screen.getByRole("tab", { name: "TRAKE" })).toHaveProperty("disabled", true);
  });

  it("sends the pasted aic26.query.v1 object without translating it", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({ ok: true, json: async () => capability(true, false, false) })
      .mockResolvedValueOnce({ ok: true, json: async () => emptySearchResponse });
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);

    const input = await screen.findByLabelText("Structured query JSON");
    fireEvent.change(input, {
      target: {
        value: JSON.stringify({
          schema_version: "aic26.query.v1",
          task_type: "kis",
          raw_query_vi: "người mặc áo đỏ",
          scene_en: "a person wearing a red shirt",
        }),
      },
    });
    fireEvent.click(screen.getByRole("button", { name: "Tìm kiếm" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    const request = fetchMock.mock.calls[1][1] as RequestInit;
    const body = JSON.parse(request.body as string);
    expect(body.query.scene_en).toBe("a person wearing a red shirt");
    expect(body.query.raw_query_vi).toBe("người mặc áo đỏ");
    expect(body.raw_query_vi).toBeUndefined();
  });
});
