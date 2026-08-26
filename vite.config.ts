import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import { mockApiPlugin } from "./online/frontend/dev/mock-api-plugin.ts";

export default defineConfig(({ mode }) => {
  const useMockApi = mode === "mock";
  return {
    root: ".",
    plugins: useMockApi ? [mockApiPlugin(fileURLToPath(new URL(".", import.meta.url)))] : [],
    server: {
      proxy: useMockApi ? undefined : {
        "/api": "http://127.0.0.1:8890",
        "/keyframes": "http://127.0.0.1:8890",
      },
    },
    build: {
      outDir: "online/frontend/dist",
      emptyOutDir: true,
      rollupOptions: {
        input: fileURLToPath(new URL("./online/frontend/index.html", import.meta.url)),
      },
    },
  };
});
