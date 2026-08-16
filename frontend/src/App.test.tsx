import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import App from "./App";

describe("task switch", () => {
  it("switches from KIS to Q&A and TRAKE", () => {
    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "QA" }));
    expect(screen.getByRole("tab", { name: "QA" }).getAttribute("aria-selected")).toBe("true");
    fireEvent.click(screen.getByRole("tab", { name: "TRAKE" }));
    expect(screen.getByRole("tab", { name: "TRAKE" }).getAttribute("aria-selected")).toBe("true");
  });
});
