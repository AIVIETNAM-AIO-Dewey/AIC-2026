import { fireEvent, render, screen, waitFor } from "@testing-library/react";
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

afterEach(() => vi.unstubAllGlobals());

describe("readiness", () => {
  it("hides search until KIS artifacts are ready", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: true, json: async () => capability(false) }),
    );
    render(<App />);
    expect(await screen.findByText("Chưa thể tìm kiếm")).toBeTruthy();
    expect(screen.queryByLabelText("Vietnamese query")).toBeNull();
  });

  it("shows search and disables unavailable modes after ingest", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: true, json: async () => capability(true, false, false) }),
    );
    render(<App />);
    expect(await screen.findByLabelText("Vietnamese query")).toBeTruthy();
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
