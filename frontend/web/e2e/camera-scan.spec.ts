import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";
import path from "node:path";
import { createCameraDetector } from "../src/lib/camera-detector";
import { assessCameraFrame, estimateFaceShape, measureClippedFraction, measureLightAndSharpness } from "../src/lib/camera-scan";

const SAMPLE = path.resolve(__dirname, "../../../ml/datasets/samples/mae_jemison.jpg");
const SHOTS = process.env.SCREENSHOT_DIR;

test("uses the CPU detector if GPU creation fails", async () => {
  const delegates: string[] = [];
  const result = { faceLandmarks: [], faceBlendshapes: [], facialTransformationMatrixes: [] };
  const detector = await createCameraDetector(async (delegate) => {
    delegates.push(delegate);
    if (delegate === "GPU") throw new Error("GPU initialization failed");
    return { detectForVideo: () => result, close() {} };
  });
  await expect(detector.detectForVideo({} as HTMLCanvasElement, 1)).resolves.toBe(result);
  expect(delegates).toEqual(["GPU", "CPU"]);
  detector.close();
});

test("retries a failed GPU frame on the CPU and propagates a subsequent CPU error", async () => {
  const calls: string[] = [];
  const frame = {} as HTMLCanvasElement;
  const result = { faceLandmarks: [], faceBlendshapes: [], facialTransformationMatrixes: [] };
  const detector = await createCameraDetector(async (delegate) => {
    calls.push(delegate);
    return {
      detectForVideo(input, timestamp) {
        expect(input).toBe(frame);
        calls.push(`${delegate} ${timestamp}`);
        if (delegate === "GPU" || timestamp === 3) throw new Error(`${delegate} inference failed`);
        return result;
      },
      close() {
        calls.push(`close ${delegate}`);
        if (delegate === "GPU") throw new Error("GPU context lost");
      },
    };
  });
  await expect(detector.detectForVideo(frame, 1)).resolves.toBe(result);
  await expect(detector.detectForVideo(frame, 2)).resolves.toBe(result);
  await expect(detector.detectForVideo(frame, 3)).rejects.toThrow("CPU inference failed");
  detector.close();
  expect(calls).toEqual(["GPU", "GPU 1", "close GPU", "CPU", "CPU 1", "CPU 2", "CPU 3", "close CPU"]);
});

test("camera guidance rejects poor frames and ranks usable frames", () => {
  const points = Array.from({ length: 478 }, () => ({ x: .5, y: .5 }));
  points[234].x = .35;
  points[454].x = .65;
  const frontal = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const turned = [.87, 0, -.5, 0, 0, 1, 0, 0, .5, 0, .87, 0, 0, 0, 0, 1];
  const good = assessCameraFrame(points, 640, 480, frontal, { brightness: 128, sharpness: 60, clippedFraction: 0 });
  const fair = assessCameraFrame(points, 640, 480, frontal, { brightness: 90, sharpness: 50, clippedFraction: 0 });
  expect(good.score).toBeGreaterThan(fair.score!);
  expect(assessCameraFrame(points, 640, 480, turned, { brightness: 128, sharpness: 60, clippedFraction: 0 })).toMatchObject({ score: null, frontal: false });
  expect(assessCameraFrame(points, 640, 480, frontal, { brightness: 20, sharpness: 60, clippedFraction: 0 }).hint).toContain("lebih terang");
  const flat = measureLightAndSharpness({ width: 5, height: 5, data: new Uint8ClampedArray(100).fill(128), colorSpace: "srgb" });
  expect(flat).toEqual({ brightness: 128, sharpness: 0 });
  expect(assessCameraFrame(points, 640, 480, frontal, { ...flat, clippedFraction: 0 }).hint).toContain("buram");
  expect(estimateFaceShape(points, 640, 480)).toBeNull();
});

test("counts clipped RGB highlights inside the face hull and rejects more than 15 percent", () => {
  const image: ImageData = { width: 7, height: 6, data: new Uint8ClampedArray(7 * 6 * 4).fill(128), colorSpace: "srgb" };
  const channel = (x: number, y: number, c: number) => (y * image.width + x) * 4 + c;
  image.data[channel(5, 4, 0)] = 255;
  expect(measureClippedFraction(image, [{ x: 1, y: 1 }, { x: 5, y: 1 }, { x: 1, y: 4 }])).toBe(0);
  image.data[channel(5, 4, 0)] = 128;
  image.data[channel(0, 1, 0)] = 255;
  image.data[channel(1, 1, 0)] = 250;
  image.data[channel(2, 2, 1)] = 250;
  image.data[channel(3, 3, 2)] = 250;
  image.data[channel(4, 3, 0)] = 249;
  image.data[channel(5, 4, 3)] = 255;
  const hull = [{ x: 1, y: 1 }, { x: 3, y: 2 }, { x: 1, y: 4 }, { x: 5, y: 1 }, { x: 5, y: 4 }];
  const points = Array.from({ length: 478 }, () => ({ x: .5, y: .5 }));
  points[234].x = .35;
  points[454].x = .65;
  const frontal = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const assess = () => assessCameraFrame(points, 640, 480, frontal, {
    brightness: 128, sharpness: 60, clippedFraction: measureClippedFraction(image, hull),
  });
  expect(measureClippedFraction(image, hull)).toBe(.15);
  expect(assess().score).not.toBeNull();
  image.data[channel(4, 3, 0)] = 250;
  expect(measureClippedFraction(image, hull)).toBe(.2);
  expect(assess()).toMatchObject({ score: null, hint: "Kurangi cahaya langsung ke wajah." });
});

test("camera denial falls back to manual upload in Indonesian", async ({ page }) => {
  await page.addInitScript(() => {
    navigator.mediaDevices.getUserMedia = async () => { throw new DOMException("", "NotAllowedError"); };
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Pindai wajah dengan kamera" }).click();
  await expect(page.getByRole("alert").filter({ hasText: "Izin kamera ditolak." })).toHaveText(
    "Izin kamera ditolak. Gunakan pilihan unggah foto.",
  );
  await page.getByRole("button", { name: "Kembali ke unggah foto" }).click();
  await page.getByLabel("Pilih foto").setInputFiles(SAMPLE);
  await expect(page.getByRole("img", { name: "Pratinjau foto" })).toBeVisible();
});

test("live scan shows a shape and only submits the selected photo after confirmation", async ({ page }) => {
  await page.addInitScript((source) => {
    navigator.mediaDevices.getUserMedia = async () => {
      const image = new Image();
      image.src = source;
      await image.decode();
      const canvas = document.createElement("canvas");
      canvas.width = image.width;
      canvas.height = image.height;
      const ctx = canvas.getContext("2d")!;
      const stream = canvas.captureStream(15);
      const track = stream.getVideoTracks()[0];
      const timer = window.setInterval(() => {
        if (track.readyState === "ended") {
          clearInterval(timer);
          document.documentElement.dataset.testCameraStopped = "true";
        } else ctx.drawImage(image, 0, 0);
      }, 66);
      return stream;
    };
  }, `data:image/jpeg;base64,${readFileSync(SAMPLE).toString("base64")}`);

  let posts = 0;
  const assetRequests: URL[] = [];
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (/(?:face_landmarker\.task|vision_wasm_.*\.(?:wasm|js))$/.test(url.pathname)) assetRequests.push(url);
    if (request.method() === "POST" && url.pathname === "/api/analyses") {
      posts++;
    }
  });
  await page.goto("/");
  await page.getByRole("checkbox").first().check();
  await page.getByRole("button", { name: "Pindai wajah dengan kamera" }).click();
  await expect(page.getByText(/Perkiraan bentuk saat memindai: (Oval|Bulat|Persegi|Lonjong|Hati|Wajik)/)).toBeVisible({ timeout: 30_000 });
  const capture = page.getByRole("button", { name: "Gunakan foto terbaik" });
  await expect(capture).toBeEnabled({ timeout: 15_000 });
  expect(assetRequests.map(({ pathname }) => pathname)).toEqual(expect.arrayContaining([
    "/mediapipe/face_landmarker.task",
    expect.stringMatching(/^\/mediapipe\/wasm\/vision_wasm_.*\.wasm$/),
    expect.stringMatching(/^\/mediapipe\/wasm\/vision_wasm_.*\.js$/),
  ]));
  expect(new Set(assetRequests.map(({ origin }) => origin))).toEqual(new Set([new URL(page.url()).origin]));
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/pindai-wajah-langsung.png`, fullPage: true });
  await capture.click();
  await expect(page.getByRole("img", { name: "Pratinjau foto" })).toBeVisible();
  await expect(page.locator("html")).toHaveAttribute("data-test-camera-stopped", "true");
  await expect(page.getByRole("checkbox").first()).not.toBeChecked();
  expect(posts).toBe(0);

  await page.getByRole("button", { name: "Analisis wajah" }).click();
  await expect(page.getByRole("heading", { name: "2. Hasil analisis" })).toBeVisible();
  await expect(page.getByText("Foto tidak disimpan.")).toBeVisible();
  await expect(page.getByText("Perkiraan bentuk wajah")).toBeVisible();
  expect(posts).toBe(1);
  await page.getByRole("button", { name: "Hapus analisis", exact: true }).click();
  await expect(page.getByText("Analisis dan rekomendasi Anda sudah dihapus dari server.").first()).toBeVisible();
});
