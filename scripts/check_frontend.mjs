import { existsSync, readFileSync, readdirSync } from "node:fs";
import { dirname, extname, join, relative, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const DEFAULT_ROOT = resolve(SCRIPT_DIR, "..");

function collectFiles(directory, extension) {
  const files = [];
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) files.push(...collectFiles(path, extension));
    else if (extname(entry.name) === extension) files.push(path);
  }
  return files;
}

export function extractHtmlIds(html) {
  return [...html.matchAll(/\bid\s*=\s*["']([^"']+)["']/g)].map((match) => match[1]);
}

export function extractRequiredIds(source) {
  const ids = [];
  const patterns = [
    /getElementById\(\s*["'`]([^"'`]+)["'`]\s*\)/g,
    /requireElement(?:<[^>]+>)?\(\s*["'`]([^"'`]+)["'`]\s*\)/g,
  ];
  for (const pattern of patterns) {
    for (const match of source.matchAll(pattern)) ids.push(match[1]);
  }
  return ids;
}

export function validateFrontendContract(root = DEFAULT_ROOT) {
  const frontend = join(root, "online", "frontend");
  const htmlPath = join(frontend, "index.html");
  const html = readFileSync(htmlPath, "utf8");
  const htmlIds = extractHtmlIds(html);
  const availableIds = new Set(htmlIds);
  const errors = [];

  const duplicates = [...new Set(htmlIds.filter((id, index) => htmlIds.indexOf(id) !== index))];
  if (duplicates.length) errors.push(`Duplicate HTML ids: ${duplicates.join(", ")}`);

  const sources = [join(frontend, "app.js"), ...collectFiles(join(frontend, "src"), ".ts")];
  for (const sourcePath of sources) {
    const requiredIds = new Set(extractRequiredIds(readFileSync(sourcePath, "utf8")));
    const missing = [...requiredIds].filter((id) => !availableIds.has(id));
    if (missing.length) {
      errors.push(`${relative(root, sourcePath)} references missing ids: ${missing.join(", ")}`);
    }
  }

  const moduleSource = html.match(/<script[^>]+type=["']module["'][^>]+src=["']([^"']+)["']/)?.[1];
  if (!moduleSource) {
    errors.push("index.html has no module entry script");
  } else {
    const modulePath = resolve(frontend, moduleSource);
    if (!existsSync(modulePath)) errors.push(`Module entry does not exist: ${moduleSource}`);
  }

  return errors;
}

export function main(root = DEFAULT_ROOT) {
  const errors = validateFrontendContract(root);
  if (errors.length) {
    for (const error of errors) console.error(`frontend-contract: ${error}`);
    return 1;
  }
  console.log("Frontend syntax and DOM contract checks passed.");
  return 0;
}

const invokedPath = process.argv[1] ? pathToFileURL(resolve(process.argv[1])).href : "";
if (import.meta.url === invokedPath) process.exitCode = main();
