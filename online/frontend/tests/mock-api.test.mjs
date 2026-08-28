import assert from "node:assert/strict";
import { dirname, join, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { createServer } from "vite";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");

test("mock API serves enough deterministic candidates and valid placeholder images", async (context) => {
  const server = await createServer({
    configFile: join(root, "vite.config.ts"),
    mode: "mock",
    server: {
      host: "127.0.0.1",
      port: 0,
      strictPort: true,
      hmr: false,
    },
  });
  await server.listen();
  context.after(() => server.close());

  const address = server.httpServer?.address();
  assert(address && typeof address === "object");
  const base = `http://127.0.0.1:${address.port}`;

  const searchResponse = await fetch(`${base}/api/search`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: "{}",
  });
  assert.equal(searchResponse.status, 200);
  const search = await searchResponse.json();
  const pools = Object.values(search.modality_results);
  assert.deepEqual(
    pools.map((pool) => pool.results.length),
    [100, 100, 100, 100],
  );
  const uniqueCandidates = new Set(
    pools.flatMap((pool) => pool.results.map((item) => `${item.video_id}:${item.frame_idx}`)),
  );
  assert(uniqueCandidates.size >= 120);

  const filmstrip = await (await fetch(`${base}/api/video/L21_V001/keyframes`)).json();
  assert.equal(filmstrip.total_keyframes, 128);

  for (const path of [
    "/keyframes/L21_V001/00000004.jpg",
    "/data/keyframe/L21_V001/00004948.jpg",
  ]) {
    const image = await fetch(`${base}${path}`);
    assert.equal(image.status, 200);
    assert.match(image.headers.get("content-type") ?? "", /^image\//);
    assert((await image.arrayBuffer()).byteLength > 100);
  }
});
