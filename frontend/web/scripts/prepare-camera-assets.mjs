import { createHash } from "node:crypto";
import { cp, mkdir, readFile, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";

const root = new URL("../public/mediapipe/", import.meta.url);
const model = new URL("face_landmarker.task", root);
const source = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const sha256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff";
const valid = (bytes) => bytes && createHash("sha256").update(bytes).digest("hex") === sha256;
const readIfPresent = async (path) => {
  try {
    return await readFile(path);
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
};

await mkdir(root, { recursive: true });
if (!valid(await readIfPresent(model))) {
  const cache = process.env.HAIRSTYLE_MODEL_DIR || join(homedir(), ".cache", "hairstyle-advisor", "models");
  let bytes = await readIfPresent(join(cache, "face_landmarker.task"));
  if (!valid(bytes)) {
    const response = await fetch(source, { signal: AbortSignal.timeout(60_000) });
    if (!response.ok) throw new Error(`Face Landmarker download failed (HTTP ${response.status})`);
    bytes = Buffer.from(await response.arrayBuffer());
  }
  if (!valid(bytes)) throw new Error("Face Landmarker SHA-256 mismatch");
  await writeFile(model, bytes);
}

await cp(new URL("../node_modules/@mediapipe/tasks-vision/wasm/", import.meta.url), new URL("wasm/", root), { recursive: true });
