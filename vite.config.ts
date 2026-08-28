import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import { mockApiPlugin } from "./online/frontend/dev/mock-api-plugin.ts";

const workspaceRoot = fileURLToPath(new URL(".", import.meta.url));
const frontendRoot = fileURLToPath(new URL("./online/frontend", import.meta.url));

export default defineConfig(({ mode }) => {
  const useMockApi = mode === "mock";
  return {
    root: frontendRoot,
    plugins: useMockApi ? [mockApiPlugin(workspaceRoot)] : [],
    server: {
      proxy: useMockApi ? undefined : {
        "/api": "http://127.0.0.1:8890",
        "/keyframes": "http://127.0.0.1:8890",
      },
    },
    build: {
      outDir: "dist",
      emptyOutDir: true,
    },
  };
});
