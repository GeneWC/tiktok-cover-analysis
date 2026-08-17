// Display formatting helpers. Keep presentation honest: null -> "Not available",
// tiers rendered as readable labels, probabilities as percentages.

import type { PerformanceTier, PipelineStep, StepStatus } from "../types";

export const NOT_AVAILABLE = "Not available";

const TIER_LABELS: Record<PerformanceTier, string> = {
  low: "Low",
  medium: "Medium",
  medium_high: "Medium-High",
  high: "High",
};

export function formatTier(tier: PerformanceTier | null | undefined): string {
  if (!tier) return NOT_AVAILABLE;
  return TIER_LABELS[tier] ?? tier;
}

/** 0..1 probability -> "42%". */
export function formatProbability(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return NOT_AVAILABLE;
  return `${Math.round(value * 100)}%`;
}

/** 0..100 presentation score, rounded, or "Not available". */
export function formatScore(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return NOT_AVAILABLE;
  return `${Math.round(value)}`;
}

export function formatSeconds(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return NOT_AVAILABLE;
  return `${value.toFixed(1)}s`;
}

/** Generic number formatter used by the feature table. */
export function formatNumber(
  value: number | boolean | null | undefined
): string {
  if (value == null) return NOT_AVAILABLE;
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (Number.isNaN(value)) return NOT_AVAILABLE;
  if (Number.isInteger(value)) return String(value);
  const abs = Math.abs(value);
  if (abs !== 0 && (abs < 0.001 || abs >= 100000)) {
    return value.toExponential(2);
  }
  return value.toFixed(abs < 1 ? 3 : 2);
}

// Friendly labels for the 10 pipeline steps (PRD 17.3).
const STEP_LABELS: Record<PipelineStep, string> = {
  upload: "Uploading",
  metadata: "Reading the file",
  frame_sampling: "Looking at frames",
  visual_quality: "Checking the picture",
  framing: "Checking the framing",
  motion: "Checking motion",
  audio: "Checking the sound",
  ocr: "Looking for on-screen text",
  prediction: "Scoring against the creator baseline",
  report: "Writing the report",
};

export function formatStepLabel(step: string): string {
  return (STEP_LABELS as Record<string, string>)[step] ?? humanize(step);
}

const STATUS_LABELS: Record<StepStatus, string> = {
  pending: "Pending",
  running: "Running",
  complete: "Complete",
  failed: "Failed",
  skipped: "Skipped",
};

export function formatStatusLabel(status: StepStatus): string {
  return STATUS_LABELS[status] ?? status;
}

/** snake_case / kebab-case -> "Title Case". */
export function humanize(key: string): string {
  return key
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}
