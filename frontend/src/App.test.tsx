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
});
