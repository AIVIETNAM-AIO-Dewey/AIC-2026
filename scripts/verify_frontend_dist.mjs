import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const dist = join(root, "online", "frontend", "dist");
const indexPath = join(dist, "index.html");

function fail(message) {
  console.error(`frontend-dist: ${message}`);
  process.exitCode = 1;
}

if (!existsSync(indexPath)) {
  fail("online/frontend/dist/index.html is missing; run npm run build first");
} else {
  const html = readFileSync(indexPath, "utf8");
  const localReferences = [...html.matchAll(/(?:src|href)=["']([^"']+)["']/g)]
    .map((match) => match[1])
    .filter((value) => !/^(?:[a-z]+:|\/\/|#)/i.test(value));

  for (const reference of localReferences) {
    const assetPath = join(dist, reference.replace(/^\.?\//, ""));
    if (!existsSync(assetPath) || !statSync(assetPath).isFile()) {
      fail(`index.html references a missing asset: ${reference}`);
    }
  }

  if (/\.tsx?(?:[?"'])/.test(html)) fail("compiled index.html still references TypeScript source");

  const assetsDir = join(dist, "assets");
  const assets = existsSync(assetsDir) ? readdirSync(assetsDir) : [];
  if (!assets.some((name) => extname(name) === ".js")) fail("compiled JavaScript asset is missing");
  if (!assets.some((name) => extname(name) === ".css")) fail("compiled CSS asset is missing");

  if (!process.exitCode) console.log(`Verified compiled frontend: ${assets.length} assets.`);
}
