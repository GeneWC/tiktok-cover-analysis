// Typed client for the CoverSignal FastAPI backend.
// Base URL comes from VITE_API_BASE_URL, defaulting to the local dev backend.

import type {
  AnalyzeResponse,
  ChannelAnalyzeResponse,
  ChannelReportResponse,
  ChannelStatusResponse,
  ReportResponse,
  StatusResponse,
} from "../types";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000"
).replace(/\/$/, "");

/** Error carrying the HTTP status so callers can branch on 404 vs 4xx/5xx. */
export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/** Pull FastAPI's `{detail}` message out of an error response when present. */
async function extractDetail(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (body && typeof body.detail === "string") return body.detail;
    if (Array.isArray(body?.detail) && body.detail[0]?.msg) {
      return body.detail[0].msg as string;
    }
  } catch {
    // fall through to the generic message
  }
  return `Request failed (${response.status}).`;
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init);
  } catch {
    throw new ApiError(
      "Could not reach the analysis server. Is the backend running on " +
        `${API_BASE_URL}?`,
      0
    );
  }

  if (!response.ok) {
    throw new ApiError(await extractDetail(response), response.status);
  }
  return (await response.json()) as T;
}

export interface AnalyzeInput {
  file: File;
  instrument?: string;
  hashtags?: string;
}

/** POST /api/analyze — upload a video and register an analysis job. */
export async function analyze({
  file,
  instrument,
  hashtags,
}: AnalyzeInput): Promise<AnalyzeResponse> {
  const form = new FormData();
  form.append("video_file", file);
  if (instrument?.trim()) form.append("instrument", instrument.trim());
  if (hashtags?.trim()) form.append("hashtags", hashtags.trim());

  return requestJson<AnalyzeResponse>("/api/analyze", {
    method: "POST",
    body: form,
  });
}

/** GET /api/analyze/{id}/status — per-step progress for polling. */
export function getStatus(analysisId: string): Promise<StatusResponse> {
  return requestJson<StatusResponse>(
    `/api/analyze/${encodeURIComponent(analysisId)}/status`
  );
}

/** GET /api/analyze/{id}/report — the full (or pending) report. */
export function getReport(analysisId: string): Promise<ReportResponse> {
  return requestJson<ReportResponse>(
    `/api/analyze/${encodeURIComponent(analysisId)}/report`
  );
}

export interface ChannelDiagnoseInput {
  files: File[];
  /** Views aligned to `files` (same length), or omit for presentation-only. */
  views?: Array<number | null>;
}

/** POST /api/channel/diagnose — multi-video within-creator diagnostics. */
export async function diagnoseChannel({
  files,
  views,
}: ChannelDiagnoseInput): Promise<ChannelAnalyzeResponse> {
  const form = new FormData();
  for (const file of files) {
    form.append("video_files", file);
  }
  if (views && views.length === files.length) {
    form.append(
      "metrics_json",
      JSON.stringify(views.map((v) => ({ views: v })))
    );
  }
  return requestJson<ChannelAnalyzeResponse>("/api/channel/diagnose", {
    method: "POST",
    body: form,
  });
}

export function getChannelStatus(
  channelId: string
): Promise<ChannelStatusResponse> {
  return requestJson<ChannelStatusResponse>(
    `/api/channel/${encodeURIComponent(channelId)}/status`
  );
}

export function getChannelReport(
  channelId: string
): Promise<ChannelReportResponse> {
  return requestJson<ChannelReportResponse>(
    `/api/channel/${encodeURIComponent(channelId)}/report`
  );
}

export { API_BASE_URL };
