import type { Page } from "@playwright/test";

export const ANALYSIS_ID = "analysis_ab12cd34ef56";

export const COMPLETE_REPORT = {
  analysis_id: ANALYSIS_ID,
  status: "complete",
  video_metadata: {
    duration_seconds: 12,
    width: 1080,
    height: 1920,
    fps: 30,
    has_audio: true,
    aspect_ratio: 0.5625,
    is_vertical_video: true,
    is_square_video: false,
  },
  scores: {
    top_quartile_probability: 0.72,
    view_performance_tier: "medium_high",
    engagement_tier: "medium",
    shareability_tier: "low",
    overall_presentation_score: 71,
    visual_quality_score: 68,
    audio_quality_score: 74,
    motion_score: 66,
    framing_score: 70,
  },
  features: {
    brightness_mean_full: 118,
    camera_stability_score: 0.81,
    face_visible_ratio: 0.9,
  },
  explanation: {
    strong_signals: ["Framing is stronger than in most reference covers."],
    weak_signals: ["Opening audio energy is weaker than usual."],
    neutral_or_missing_signals: [],
    recommendations: ["Keep the subject centered in the first seconds."],
  },
  limitations: [
    "Exploratory similarity to creator-relative top performers — not a virality guarantee.",
  ],
};

export async function mockAnalyzeApi(
  page: Page,
  options: {
    failUpload?: boolean;
    failStatus?: boolean;
    malformedReport?: boolean;
  } = {},
) {
  let polls = 0;
  await page.route("**/api/analyze**", async (route) => {
    const request = route.request();
    const method = request.method();
    const url = request.url();

    if (method === "POST" && url.endsWith("/api/analyze")) {
      if (options.failUpload) {
        await route.fulfill({
          status: 500,
          contentType: "application/json",
          body: JSON.stringify({ detail: "Backend failed." }),
        });
        return;
      }
      await route.fulfill({
        status: 202,
        contentType: "application/json",
        body: JSON.stringify({ analysis_id: ANALYSIS_ID, status: "processing" }),
      });
      return;
    }

    if (url.includes("/status")) {
      if (options.failStatus) {
        await route.fulfill({
          status: 500,
          contentType: "application/json",
          body: JSON.stringify({ detail: "Status failed." }),
        });
        return;
      }
      polls += 1;
      const complete = polls >= 2;
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          analysis_id: ANALYSIS_ID,
          status: complete ? "complete" : "processing",
          steps: {
            upload: "complete",
            metadata: "complete",
            frame_sampling: complete ? "complete" : "running",
            visual_quality: complete ? "complete" : "pending",
            framing: complete ? "complete" : "pending",
            motion: complete ? "complete" : "pending",
            audio: complete ? "complete" : "pending",
            ocr: complete ? "complete" : "pending",
            prediction: complete ? "complete" : "pending",
            report: complete ? "complete" : "pending",
          },
        }),
      });
      return;
    }

    if (url.includes("/report")) {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(
          options.malformedReport ? { unexpected: true } : COMPLETE_REPORT,
        ),
      });
      return;
    }

    await route.fallback();
  });
}

export function tinyVideoFile() {
  return {
    name: "cover.mp4",
    mimeType: "video/mp4",
    buffer: Buffer.from("fake-mp4-bytes"),
  };
}
