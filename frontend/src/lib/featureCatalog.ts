// Groups the raw feature dict into the PRD 16.4 sections with human labels.
// Grouping is prefix-driven so it stays robust if the extractor adds features;
// explicit label overrides make the common keys read nicely.

import { humanize } from "./format";

export const FEATURE_GROUP_ORDER = [
  "Video",
  "Visual quality",
  "Framing & subject",
  "Motion & stability",
  "Audio",
  "On-screen text",
  "Other",
] as const;

export type FeatureGroup = (typeof FEATURE_GROUP_ORDER)[number];

const VIDEO_KEYS = new Set([
  "duration_seconds",
  "fps",
  "width",
  "height",
  "aspect_ratio",
  "resolution_area",
  "bitrate",
  "has_audio",
  "is_vertical_video",
  "is_square_video",
]);

const LABEL_OVERRIDES: Record<string, string> = {
  duration_seconds: "Duration",
  fps: "Frame rate",
  aspect_ratio: "Aspect ratio",
  resolution_area: "Resolution (pixels)",
  bitrate: "Bitrate",
  has_audio: "Has audio track",
  is_vertical_video: "Vertical video",
  is_square_video: "Square video",
  sharpness_full: "Sharpness (overall)",
  contrast_full: "Contrast (overall)",
  colorfulness_full: "Colorfulness (overall)",
  brightness_mean_full: "Brightness (overall)",
  blur_full: "Blur (overall)",
  person_visible_ratio: "Performer visible ratio",
  face_visible_ratio: "Face visible ratio",
  hand_visible_ratio: "Hands visible ratio",
  upper_body_visible_ratio: "Upper body visible ratio",
  subject_centering_score: "Subject centering",
  subject_size_ratio: "Subject size in frame",
  face_size_ratio: "Face size in frame",
  motion_consistency: "Motion steadiness",
  camera_stability_score: "Camera stability",
  motion_energy_full: "Motion energy (overall)",
  audio_dynamic_range: "Audio dynamic range",
  audio_clipping_ratio: "Audio clipping",
  audio_silence_ratio: "Silence ratio",
  audio_rms_mean: "Loudness (RMS mean)",
  audio_peak_level: "Peak level",
  audio_spectral_centroid_mean: "Spectral centroid",
  text_present_anywhere: "On-screen text present",
  text_area_ratio_full: "Text area ratio (overall)",
  first_text_timestamp: "First text appears at",
  ocr_failed: "Text detection failed",
};

export function groupForFeature(key: string): FeatureGroup {
  if (VIDEO_KEYS.has(key)) return "Video";
  if (
    key.startsWith("brightness") ||
    key.startsWith("contrast") ||
    key.startsWith("sharpness") ||
    key.startsWith("blur") ||
    key.startsWith("colorfulness")
  ) {
    return "Visual quality";
  }
  if (
    key.startsWith("person_visible") ||
    key.startsWith("face_visible") ||
    key.startsWith("hand_visible") ||
    key.startsWith("upper_body") ||
    key.startsWith("subject_") ||
    key.startsWith("face_size") ||
    key === "hand_detection_failed"
  ) {
    return "Framing & subject";
  }
  if (
    key.startsWith("motion_") ||
    key.startsWith("hand_motion") ||
    key === "camera_stability_score"
  ) {
    return "Motion & stability";
  }
  if (key.startsWith("audio_")) return "Audio";
  if (
    key.startsWith("text_") ||
    key.startsWith("first_text") ||
    key.startsWith("average_text") ||
    key === "ocr_failed"
  ) {
    return "On-screen text";
  }
  return "Other";
}

export function labelForFeature(key: string): string {
  return LABEL_OVERRIDES[key] ?? humanize(key);
}
