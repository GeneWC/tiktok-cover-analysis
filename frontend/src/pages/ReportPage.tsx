import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { getReport, ApiError } from "../api/client";
import { useAnalysis } from "../context/AnalysisContext";
import type { ReportResponse } from "../types";
import VideoSummary from "../components/VideoSummary";
import PredictionCard from "../components/PredictionCard";
import ScoreCard from "../components/ScoreCard";
import FeatureBreakdown from "../components/FeatureBreakdown";
import RecommendationPanel from "../components/RecommendationPanel";
import LimitationsNote from "../components/LimitationsNote";
import { getDefinition } from "../lib/glossary";

export default function ReportPage() {
  const { id } = useParams<{ id: string }>();
  const { previewUrlFor } = useAnalysis();

  const [report, setReport] = useState<ReportResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    setLoading(true);
    getReport(id)
      .then((data) => {
        if (!cancelled) {
          setReport(data);
          setError(null);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(
            err instanceof ApiError
              ? err.message
              : "Failed to load the report."
          );
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [id]);

  const previewUrl = id ? previewUrlFor(id) : null;

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-24 text-slate-500">
        <span className="h-8 w-8 animate-spin rounded-full border-2 border-indigo-500 border-t-transparent" />
        <p className="mt-4 text-sm">Loading report…</p>
      </div>
    );
  }

  if (error || !report) {
    return (
      <div className="mx-auto max-w-lg rounded-2xl bg-white p-8 text-center shadow-sm ring-1 ring-slate-200">
        <h1 className="text-xl font-semibold text-slate-900">
          Couldn't load report
        </h1>
        <p className="mt-2 text-sm text-slate-500">
          {error ?? "This report is not available."}
        </p>
        <Link
          to="/"
          className="mt-6 inline-block rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-700"
        >
          Analyze another video
        </Link>
      </div>
    );
  }

  const { scores } = report;
  const failed = report.status === "failed";

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Analysis report</h1>
          <p className="mt-1 text-sm text-slate-500">Analysis ID: {report.analysis_id}</p>
        </div>
        <Link
          to="/"
          className="rounded-lg bg-white px-4 py-2 text-sm font-semibold text-indigo-600 ring-1 ring-indigo-200 hover:bg-indigo-50"
        >
          Analyze another video
        </Link>
      </div>

      {failed && (
        <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          This video could not be fully analyzed, so predictions are
          unavailable. Any details below are partial.
        </div>
      )}

      <VideoSummary metadata={report.video_metadata} previewUrl={previewUrl} />

      {!failed && <PredictionCard scores={scores} />}

      <section>
        <h2 className="mb-3 text-lg font-semibold text-slate-900">
          Presentation scores
        </h2>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <ScoreCard
            label="Overall presentation"
            score={scores.overall_presentation_score}
            hint={getDefinition("overall presentation") ?? undefined}
            emphasize
          />
          <ScoreCard
            label="Visual quality"
            score={scores.visual_quality_score}
            hint={getDefinition("visual quality") ?? undefined}
          />
          <ScoreCard
            label="Audio quality"
            score={scores.audio_quality_score}
            hint={getDefinition("audio quality") ?? undefined}
          />
          <ScoreCard
            label="Motion"
            score={scores.motion_score}
            hint={getDefinition("motion") ?? undefined}
          />
          <ScoreCard
            label="Framing"
            score={scores.framing_score}
            hint={getDefinition("framing") ?? undefined}
          />
        </div>
        <p className="mt-2 text-xs text-slate-400">
          Scores are percentile ranks against comparable covers (higher is
          better). A dimension shows "Not available" when its inputs are missing
          — e.g. audio quality for a video with no audio track.
        </p>
      </section>

      <RecommendationPanel explanation={report.explanation} />

      <FeatureBreakdown features={report.features} />

      <LimitationsNote limitations={report.limitations} />
    </div>
  );
}
