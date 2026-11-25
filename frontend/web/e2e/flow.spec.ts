import { expect, test } from "@playwright/test";
import path from "path";

const SAMPLE = path.resolve(__dirname, "../../../ml/datasets/samples/mae_jemison.jpg");
const SHOTS = process.env.SCREENSHOT_DIR;

test("upload, analyze, recommend, delete", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "AI Hairstyle Advisor" })).toBeVisible();
  await page.getByLabel("Pilih foto").setInputFiles(SAMPLE);
  await page.getByRole("checkbox").first().check();
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/1-unggah.png`, fullPage: true });

  await page.getByRole("button", { name: "Analisis wajah" }).click();
  await expect(page.getByText("Perkiraan bentuk wajah")).toBeVisible();
  await expect(page.locator("strong", { hasText: "tidak terkalibrasi" })).toBeVisible();
  await expect(page.getByText("Foto disimpan atas persetujuan Anda.")).toBeVisible();
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/2-hasil-analisis.png`, fullPage: true });

  await page.getByLabel("Jenis rambut").selectOption("wavy");
  await page.getByLabel("Kategori gaya").selectOption("wanita");
  await page.getByRole("button", { name: "Tampilkan rekomendasi" }).click();
  await expect(page.getByText(/^4\. Rekomendasi/)).toBeVisible();
  await expect(page.locator("ol > li")).toHaveCount(5);
  await page.locator("ol > li details").first().evaluate((d) => ((d as HTMLDetailsElement).open = true));
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/3-rekomendasi.png`, fullPage: true });

  await page.getByRole("button", { name: "Hapus analisis & foto" }).click();
  await expect(page.getByText("sudah dihapus dari server")).toBeVisible();
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/4-dihapus.png` });
});

test("bad file shows Indonesian error", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Pilih foto").setInputFiles({ name: "x.jpg", mimeType: "image/jpeg", buffer: Buffer.from("bukan foto") });
  await page.getByRole("button", { name: "Analisis wajah" }).click();
  await expect(page.getByText("Format foto harus JPEG, PNG, atau WebP.")).toBeVisible();
});
