import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "e2e",
  timeout: 90_000,
  use: { baseURL: process.env.WEB_URL || "http://localhost:3000", viewport: { width: 1280, height: 900 } },
});
