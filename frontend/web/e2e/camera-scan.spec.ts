import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";
import path from "node:path";
import { assessCameraFrame, estimateFaceShape, measureLightAndSharpness } from "../src/lib/camera-scan";

const SAMPLE = path.resolve(__dirname, "../../../ml/datasets/samples/mae_jemison.jpg");
const SHOTS = process.env.SCREENSHOT_DIR;

test("camera guidance rejects poor frames and ranks usable frames", () => {
  const points = Array.from({ length: 478 }, () => ({ x: .5, y: .5 }));
  points[234].x = .35;
  points[454].x = .65;
  const frontal = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const turned = [.87, 0, -.5, 0, 0, 1, 0, 0, .5, 0, .87, 0, 0, 0, 0, 1];
  const good = assessCameraFrame(points, 640, 480, frontal, { brightness: 128, sharpness: 60 });
  const fair = assessCameraFrame(points, 640, 480, frontal, { brightness: 90, sharpness: 50 });
  expect(good.score).toBeGreaterThan(fair.score!);
  expect(assessCameraFrame(points, 640, 480, turned, { brightness: 128, sharpness: 60 })).toMatchObject({ score: null, frontal: false });
  expect(assessCameraFrame(points, 640, 480, frontal, { brightness: 20, sharpness: 60 }).hint).toContain("lebih terang");
  const flat = measureLightAndSharpness({ width: 5, height: 5, data: new Uint8ClampedArray(100).fill(128), colorSpace: "srgb" });
  expect(flat).toEqual({ brightness: 128, sharpness: 0 });
  expect(assessCameraFrame(points, 640, 480, frontal, flat).hint).toContain("buram");
  expect(estimateFaceShape(points, 640, 480)).toBeNull();
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
  page.on("request", (request) => {
    if (request.method() === "POST" && new URL(request.url()).pathname === "/api/analyses") {
      posts++;
    }
  });
  await page.goto("/");
  await page.getByRole("checkbox").first().check();
  await page.getByRole("button", { name: "Pindai wajah dengan kamera" }).click();
  await expect(page.getByText(/Perkiraan bentuk saat memindai: (Oval|Bulat|Persegi|Lonjong|Hati|Wajik)/)).toBeVisible({ timeout: 30_000 });
  const capture = page.getByRole("button", { name: "Gunakan foto terbaik" });
  await expect(capture).toBeEnabled({ timeout: 15_000 });
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
