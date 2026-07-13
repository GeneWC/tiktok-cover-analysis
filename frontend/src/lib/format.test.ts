import { describe, expect, it } from "vitest";
import {
  NOT_AVAILABLE,
  formatNumber,
  formatProbability,
  formatScore,
  formatStepLabel,
  formatTier,
  humanize,
} from "./format";
import { groupForFeature, labelForFeature } from "./featureCatalog";

describe("formatTier", () => {
  it("maps enum values to readable labels", () => {
    expect(formatTier("medium_high")).toBe("Medium-High");
    expect(formatTier("high")).toBe("High");
  });
  it("renders null as Not available", () => {
    expect(formatTier(null)).toBe(NOT_AVAILABLE);
  });
});

describe("formatProbability", () => {
  it("renders a 0..1 value as a percentage", () => {
    expect(formatProbability(0.42)).toBe("42%");
    expect(formatProbability(1)).toBe("100%");
  });
  it("renders null / NaN as Not available", () => {
    expect(formatProbability(null)).toBe(NOT_AVAILABLE);
    expect(formatProbability(Number.NaN)).toBe(NOT_AVAILABLE);
  });
});

describe("formatScore", () => {
  it("rounds and stringifies", () => {
    expect(formatScore(72.6)).toBe("73");
  });
  it("handles null", () => {
    expect(formatScore(null)).toBe(NOT_AVAILABLE);
  });
});

describe("formatNumber", () => {
  it("renders booleans as Yes/No", () => {
    expect(formatNumber(true)).toBe("Yes");
    expect(formatNumber(false)).toBe("No");
  });
  it("renders null as Not available", () => {
    expect(formatNumber(null)).toBe(NOT_AVAILABLE);
  });
  it("keeps integers integral and rounds floats", () => {
    expect(formatNumber(1080)).toBe("1080");
    expect(formatNumber(0.123456)).toBe("0.123");
  });
});

describe("formatStepLabel", () => {
  it("uses friendly labels for known steps", () => {
    expect(formatStepLabel("frame_sampling")).toBe("Sampling frames");
  });
  it("humanizes unknown steps", () => {
    expect(formatStepLabel("some_new_step")).toBe("Some New Step");
  });
});

describe("humanize", () => {
  it("title-cases snake_case", () => {
    expect(humanize("audio_dynamic_range")).toBe("Audio Dynamic Range");
  });
});

describe("featureCatalog grouping", () => {
  it("routes features to the right dimension", () => {
    expect(groupForFeature("sharpness_full")).toBe("Visual quality");
    expect(groupForFeature("face_visible_ratio")).toBe("Framing & subject");
    expect(groupForFeature("camera_stability_score")).toBe("Motion & stability");
    expect(groupForFeature("audio_clipping_ratio")).toBe("Audio");
    expect(groupForFeature("text_present_anywhere")).toBe("On-screen text");
    expect(groupForFeature("duration_seconds")).toBe("Video");
    expect(groupForFeature("mystery_feature")).toBe("Other");
  });
  it("uses label overrides then humanizes", () => {
    expect(labelForFeature("duration_seconds")).toBe("Duration");
    expect(labelForFeature("unknown_metric")).toBe("Unknown Metric");
  });
});
