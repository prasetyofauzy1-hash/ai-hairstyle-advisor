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

  await expect(page.getByText("Data Anda di server")).toBeVisible();
  await page.getByRole("button", { name: "Hapus analisis & foto" }).click();
  await expect(page.getByText("Data Anda di server")).toHaveCount(0);
  await expect(page.getByText("Analisis, foto, dan rekomendasi Anda sudah dihapus dari server.")).toBeVisible();
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/4-dihapus.png` });
});

test("new photo resets consent; earlier analysis stays deletable", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Pilih foto").setInputFiles(SAMPLE);
  await page.getByRole("checkbox").first().check();
  await page.getByRole("button", { name: "Analisis wajah" }).click();
  await expect(page.getByText("Perkiraan bentuk wajah")).toBeVisible();
  await page.getByLabel("Pilih foto").setInputFiles(SAMPLE);
  await expect(page.getByRole("checkbox").first()).not.toBeChecked();
  await page.getByRole("button", { name: "Analisis wajah" }).click();
  await expect(page.getByText("Foto tidak disimpan.")).toBeVisible();
  await expect(page.getByRole("button", { name: "Hapus", exact: true })).toHaveCount(2);
  await page.reload();
  await expect(page.getByRole("button", { name: "Hapus", exact: true })).toHaveCount(2);
  await page.getByRole("button", { name: "Hapus", exact: true }).first().click();
  await expect(page.getByRole("button", { name: "Hapus", exact: true })).toHaveCount(1);
  await page.getByRole("button", { name: "Hapus", exact: true }).first().click();
  await expect(page.getByText("Data Anda di server")).toHaveCount(0);
});

test("bad file shows Indonesian error", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Pilih foto").setInputFiles({ name: "x.jpg", mimeType: "image/jpeg", buffer: Buffer.from("bukan foto") });
  await expect(page.getByText("Kami tidak bisa membuka foto ini.", { exact: false })).toBeVisible();
  await expect(page.getByRole("button", { name: "Analisis wajah" })).toBeDisabled();
  await page.getByLabel("Pilih foto").setInputFiles({ name: "x.txt", mimeType: "text/plain", buffer: Buffer.from("teks") });
  await expect(page.getByText("Format foto harus JPEG, PNG, atau WebP.")).toBeVisible();
});
