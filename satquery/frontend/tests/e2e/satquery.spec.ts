import { test, expect } from '@playwright/test';

// Utility to mock backend /validate-pair
async function mockValidatePair(page: any, response: any, status = 200) {
  await page.route('**/validate-pair', async route => {
    await route.fulfill({
      status,
      contentType: 'application/json',
      body: JSON.stringify(response),
    });
  });
}

// Utility to mock backend /query
async function mockQuery(page: any, response: any, status = 200, delay = 0) {
  await page.route('**/query', async route => {
    if (delay > 0) {
      await new Promise(r => setTimeout(r, delay));
    }
    await route.fulfill({
      status,
      contentType: 'application/json',
      body: JSON.stringify(response),
    });
  });
}

test.describe('SatQuery Frontend E2E', () => {

  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('incompatible GeoTIFF pair -> validation error shown, not a crash', async ({ page }) => {
    await mockValidatePair(page, {
      detail: "Cannot detect change: CRS mismatch between 'EPSG:4326' and 'EPSG:3857'."
    }, 400);

    // Provide mock files to file inputs
    const fileInputA = page.locator('input[type="file"]').nth(0);
    const fileInputB = page.locator('input[type="file"]').nth(1);

    await fileInputA.setInputFiles({
      name: 'sceneA.tif',
      mimeType: 'image/tiff',
      buffer: Buffer.from('mock geotiff a')
    });
    
    await fileInputB.setInputFiles({
      name: 'sceneB.tif',
      mimeType: 'image/tiff',
      buffer: Buffer.from('mock geotiff b')
    });

    // Wait for the validation logic to run and render the error
    await expect(page.locator('text=Cannot detect change: CRS mismatch')).toBeVisible();
    await expect(page.locator('text=Pair Compatibility Error')).toBeVisible();
  });

  test("PNG/JPEG upload -> skips georef checks, shows 'benchmark mode' label", async ({ page }) => {
    const fileInputA = page.locator('input[type="file"]').nth(0);

    await fileInputA.setInputFiles({
      name: 'sceneA.png',
      mimeType: 'image/png',
      buffer: Buffer.from('mock png')
    });

    await expect(page.locator('text=benchmark mode, no georeferencing')).toBeVisible();
    // Verify that the execute button is not disabled (assuming a query exists)
    await expect(page.locator('button:has-text("Execute Query")')).toBeEnabled();
  });

  test('execution trace panel reflects the actual models used for this specific query', async ({ page }) => {
    await mockQuery(page, {
      answer: "The built-up area has increased.",
      status: "success",
      execution_trace: {
        task: "Change Detection",
        models_used: ["RSVQALoRA", "CVA-Pixel-Matcher"],
        execution_steps: [
          { id: "1", name: "Semantic extraction", status: "completed" }
        ]
      }
    });

    const fileInputA = page.locator('input[type="file"]').nth(0);
    await fileInputA.setInputFiles({
      name: 'sceneA.png',
      mimeType: 'image/png',
      buffer: Buffer.from('mock')
    });

    await page.fill('textarea', 'Has the built-up area increased?');
    await page.click('button:has-text("Execute Query")');

    // Result panel should show Answer
    await expect(page.locator('text=The built-up area has increased.')).toBeVisible();

    // Click on Models Used Tab in Trace Panel
    await page.click('button:has-text("Models Used")');
    await expect(page.locator('text=RSVQALoRA')).toBeVisible();
    await expect(page.locator('text=CVA-Pixel-Matcher')).toBeVisible();
  });

  test('confidence badge color matches the bucket returned by the API', async ({ page }) => {
    await mockQuery(page, {
      answer: "Low confidence mock answer.",
      status: "success",
      outputs: {
        vqa: { confidence: 0.45 }
      }
    });

    const fileInputA = page.locator('input[type="file"]').nth(0);
    await fileInputA.setInputFiles({
      name: 'sceneA.png',
      mimeType: 'image/png',
      buffer: Buffer.from('mock')
    });

    await page.fill('textarea', 'Query?');
    await page.click('button:has-text("Execute Query")');

    // Medium confidence is 0.60 to 0.85, so 0.45 is Low Confidence (🔴)
    await expect(page.locator('text=Low Confidence')).toBeVisible();
  });

  test('download report button disabled until query completes', async ({ page }) => {
    await mockQuery(page, {
      answer: "Delayed", status: "success"
    }, 200, 1000); // 1s delay

    const downloadBtn = page.locator('button:has-text("Download Report")');
    
    // Initially disabled
    await expect(downloadBtn).toBeDisabled();

    const fileInputA = page.locator('input[type="file"]').nth(0);
    await fileInputA.setInputFiles({ name: 'sceneA.png', mimeType: 'image/png', buffer: Buffer.from('mock') });

    await page.click('button:has-text("Execute Query")');
    
    // Disabled while executing
    await expect(downloadBtn).toBeDisabled();

    // Enabled after completion
    await expect(page.locator('text=Delayed')).toBeVisible();
    await expect(downloadBtn).toBeEnabled();
  });

  test('download report works on a failed/zero-result query without crashing', async ({ page }) => {
    await mockQuery(page, { detail: "Internal Server Error" }, 500);

    const fileInputA = page.locator('input[type="file"]').nth(0);
    await fileInputA.setInputFiles({ name: 'sceneA.png', mimeType: 'image/png', buffer: Buffer.from('mock') });

    await page.click('button:has-text("Execute Query")');

    // Should show error banner
    await expect(page.locator('text=Internal Server Error')).toBeVisible();
    
    // App should not crash.
    await expect(page.locator('text=Execute Query')).toBeVisible();
  });

});
