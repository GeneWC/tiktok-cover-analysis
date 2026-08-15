// Types mirroring the backend Pydantic schema (backend/schemas/analysis.py).
// Keep these in sync with the API contract.

export type PerformanceTier = "low" | "medium" | "medium_high" | "high";

export type StepStatus =
  | "pending"
  | "running"
  | "complete"
  | "failed"
  | "skipped";

export type AnalysisStatus = "processing" | "complete" | "failed";

// Canonical ordered pipeline steps (backend/services/analysis_store.py).
export const PIPELINE_STEPS = [
  "upload",
  "metadata",
  "frame_sampling",
  "visual_quality",
  "framing",
  "motion",
  "audio",
  "ocr",
  "prediction",
  "report",
] as const;

export type PipelineStep = (typeof PIPELINE_STEPS)[number];

export interface AnalyzeResponse {
  analysis_id: string;
  status: AnalysisStatus;
}

export interface StatusResponse {
  analysis_id: string;
  status: AnalysisStatus;
  steps: Record<string, StepStatus>;
}

export interface VideoMetadata {
  duration_seconds: number;
  width: number;
  height: number;
  fps: number | null;
  has_audio: boolean;
  aspect_ratio: number | null;
  is_vertical_video: boolean;
  is_square_video: boolean;
}

export interface ReportScores {
  top_quartile_probability: number | null;
  view_performance_tier: PerformanceTier | null;
  engagement_tier: PerformanceTier | null;
  shareability_tier: PerformanceTier | null;
  overall_presentation_score: number | null;
  visual_quality_score: number | null;
  audio_quality_score: number | null;
  motion_score: number | null;
  framing_score: number | null;
}

export interface ReportExplanation {
  strong_signals: string[];
  weak_signals: string[];
  neutral_or_missing_signals: string[];
  recommendations: string[];
}

export type FeatureValue = number | boolean | null;

export interface ReportResponse {
  analysis_id: string;
  status: AnalysisStatus;
  video_metadata: VideoMetadata | null;
  scores: ReportScores;
  features: Record<string, FeatureValue>;
  explanation: ReportExplanation;
  limitations: string[];
}

// --- Channel diagnostics (D-018) ---

export const CHANNEL_STEPS = [
  "upload",
  "features",
  "diagnose",
  "report",
] as const;

export type ChannelStep = (typeof CHANNEL_STEPS)[number];

export interface ChannelAnalyzeResponse {
  channel_id: string;
  status: AnalysisStatus;
  n_videos: number;
}

export interface ChannelStatusResponse {
  channel_id: string;
  status: AnalysisStatus;
  steps: Record<string, StepStatus>;
  n_videos: number;
  n_features_done: number;
  error: string | null;
}

export interface ChannelFeatureDelta {
  feature: string;
  hit_mean: number;
  miss_mean: number;
  delta: number;
}

export interface ChannelVideoRank {
  video_id: string;
  filename: string | null;
  views: number | null;
  presentation_score: number;
  residual_l2: number;
  label: number | null;
}

export interface ChannelReportResponse {
  channel_id: string;
  status: AnalysisStatus;
  mode: "diagnostics";
  n_videos: number;
  n_labeled: number;
  n_hits: number | null;
  positive_rate: number | null;
  top_feature_deltas: ChannelFeatureDelta[];
  video_ranks: ChannelVideoRank[];
  recommendations: string[];
  limitations: string[];
  message: string | null;
}
