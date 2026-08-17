import { expect, test } from "@playwright/test";
import { mockAnalyzeApi, tinyVideoFile } from "./fixtures";

test.describe("upload to report", () => {
  test("happy path shows processing then a relative result", async ({ page }) => {
    await mockAnalyzeApi(page);
    await page.goto("/");
    await expect(page.getByRole("heading", { name: /how does this cover present/i })).toBeVisible();
    await page.locator('input[type="file"]').setInputFiles(tinyVideoFile());
    await page.getByRole("button", { name: /analyze video/i }).click();
    await expect(page.getByRole("heading", { name: /analyzing/i })).toBeVisible();
    await expect(page.getByText(/scoring against the creator baseline/i)).toBeVisible();
    await expect(page.getByRole("heading", { name: /relative to a creator/i })).toBeVisible({
      timeout: 15_000,
    });
    await expect(page.getByText("72%", { exact: true })).toBeVisible();
    await expect(page.getByText(/framing is stronger/i)).toBeVisible();
  });

  test("unsupported file type stays on upload with an error", async ({ page }) => {
    await page.goto("/");
    await page.locator('input[type="file"]').setInputFiles({
      name: "notes.txt",
      mimeType: "text/plain",
      buffer: Buffer.from("not a video"),
    });
    await expect(page.getByRole("alert")).toBeVisible();
    await expect(page.getByRole("button", { name: /analyze video/i })).toBeDisabled();
  });

  test("backend failure is shown on the upload page", async ({ page }) => {
    await mockAnalyzeApi(page, { failUpload: true });
    await page.goto("/");
    await page.locator('input[type="file"]').setInputFiles(tinyVideoFile());
    await page.getByRole("button", { name: /analyze video/i }).click();
    await expect(page.getByRole("alert")).toContainText(/backend failed|could not upload/i);
  });

  test("reload during processing can resume from the same job link", async ({ page }) => {
    await mockAnalyzeApi(page);
    await page.goto("/");
    await page.locator('input[type="file"]').setInputFiles(tinyVideoFile());
    await page.getByRole("button", { name: /analyze video/i }).click();
    await expect(page.getByRole("heading", { name: /analyzing/i })).toBeVisible();
    await page.reload();
    await expect(page.getByRole("heading", { name: /analyzing|relative to a creator/i })).toBeVisible();
  });
});

test.describe("mobile", () => {
  test("upload and report remain readable", async ({ page }) => {
    await mockAnalyzeApi(page);
    await page.goto("/");
    await expect(page.getByRole("heading", { name: /how does this cover present/i })).toBeVisible();
    await page.locator('input[type="file"]').setInputFiles(tinyVideoFile());
    await page.getByRole("button", { name: /analyze video/i }).click();
    await expect(page.getByRole("heading", { name: /relative to a creator/i })).toBeVisible({
      timeout: 15_000,
    });
    await expect(page.getByText(/an estimate of how this cover compares/i)).toBeVisible();
  });
});
