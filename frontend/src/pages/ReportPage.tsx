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
              : "Could not load the report."
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
      <div className="muted flex flex-col items-center justify-center py-24">
        <span className="h-8 w-8 animate-spin rounded-full border-2 border-gold border-t-transparent" />
        <p className="mt-4 text-sm">Loading report…</p>
      </div>
    );
  }

  if (error || !report) {
    return (
      <div className="panel mx-auto max-w-lg rounded-sm p-8 text-center">
        <h1 className="font-display text-2xl text-ivory">
          Could not load report
        </h1>
        <p className="muted mt-2 text-sm">
          {error ?? "This report is not available."}
        </p>
        <Link to="/" className="btn-primary mt-6 rounded-sm px-4 py-2 text-sm">
          Analyze another video
        </Link>
      </div>
    );
  }

  const { scores } = report;
  const failed = report.status === "failed";

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="font-display text-3xl font-medium text-ivory">
            Your Cover
          </h1>
        </div>
        <Link to="/" className="btn-ghost btn-inline rounded-sm px-4 py-2 text-sm">
          Analyze another video
        </Link>
      </div>

      {failed && (
        <div className="alert rounded-sm px-4 py-3 text-sm">
          This video could not be fully analyzed. Predictions are unavailable.
          Details below may be partial.
        </div>
      )}

      {!failed && <RecommendationPanel explanation={report.explanation} />}

      <section>
        <h2 className="mb-4 font-display text-2xl text-ivory">Scores</h2>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <ScoreCard
            label="Overall"
            score={scores.overall_presentation_score}
            hint={getDefinition("overall presentation") ?? undefined}
            emphasize
          />
          <ScoreCard
            label="Visual"
            score={scores.visual_quality_score}
            hint={getDefinition("visual quality") ?? undefined}
          />
          <ScoreCard
            label="Audio"
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
        <p className="muted mt-3 text-xs">
          Higher is better. “Not available” means that part could not be
          measured.
        </p>
      </section>

      {!failed && <PredictionCard scores={scores} />}

      <VideoSummary metadata={report.video_metadata} previewUrl={previewUrl} />

      <FeatureBreakdown features={report.features} />

      <LimitationsNote limitations={report.limitations} />
    </div>
  );
}
