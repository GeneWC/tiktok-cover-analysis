import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ApiError, getChannelReport } from "../api/client";
import type { ChannelReportResponse } from "../types";
import LimitationsNote from "../components/LimitationsNote";
import { labelForFeature } from "../lib/featureCatalog";

export default function ChannelReportPage() {
  const { id } = useParams<{ id: string }>();
  const [report, setReport] = useState<ChannelReportResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    getChannelReport(id)
      .then(setReport)
      .catch((err) =>
        setError(
          err instanceof ApiError ? err.message : "Could not load the report."
        )
      );
  }, [id]);

  if (error) {
    return (
      <div className="mx-auto max-w-2xl">
        <p className="alert rounded-sm px-4 py-3">{error}</p>
        <div className="mt-3">
          <Link to="/channel" className="btn-ghost btn-inline rounded-sm px-4 py-2 text-sm">
            Try again
          </Link>
        </div>
      </div>
    );
  }

  if (!report) {
    return (
      <div className="muted flex flex-col items-center justify-center py-24">
        <span className="h-8 w-8 animate-spin rounded-full border-2 border-gold border-t-transparent" />
        <p className="mt-4 text-sm">Loading report…</p>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-3xl space-y-8">
      <div>
        <p className="mb-3 text-sm">
          <Link to="/channel" className="link">
            New comparison
          </Link>
          <span className="mx-2 text-gold/40">·</span>
          <Link to="/" className="nav-link">
            One video
          </Link>
        </p>
        <h1 className="font-display text-3xl font-medium text-ivory">
          Your Videos
        </h1>
        <p className="muted mt-3">
          Compared only to each other. {report.n_videos} videos
          {report.n_hits != null
            ? ` · ${report.n_hits} in the top quarter of this batch`
            : " · no view counts this time"}
          .
        </p>
      </div>

      {report.recommendations.length > 0 && (
        <section>
          <h2 className="font-display text-2xl text-ivory">
            Patterns to notice
          </h2>
          <ul className="mt-4 space-y-2 text-sm text-ivory">
            {report.recommendations.map((tip) => (
              <li key={tip} className="flex gap-2">
                <span className="text-gold">•</span>
                <span>{tip}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {report.top_feature_deltas.length > 0 && (
        <section className="panel rounded-sm p-5">
          <h2 className="font-display text-2xl text-ivory">
            What differs in stronger uploads
          </h2>
          <p className="muted mt-2 text-sm">
            Average difference between your top videos and the rest.
          </p>
          <ul className="mt-4 divide-y divide-gold/15">
            {report.top_feature_deltas.map((d) => (
              <li
                key={d.feature}
                className="flex items-center justify-between gap-3 py-2 text-sm"
              >
                <span className="text-ivory">{labelForFeature(d.feature)}</span>
                <span
                  className={
                    d.delta >= 0 ? "font-medium text-gold" : "font-medium text-ember"
                  }
                >
                  {d.delta >= 0 ? "Higher" : "Lower"}
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section className="panel rounded-sm p-5">
        <h2 className="font-display text-2xl text-ivory">Ranked videos</h2>
        <p className="muted mt-2 text-sm">
          Simple rank from brightness, sharpness, and framing in this batch.
        </p>
        <ul className="mt-4 divide-y divide-gold/15">
          {report.video_ranks.map((row, index) => (
            <li
              key={row.video_id}
              className="flex items-center justify-between gap-3 py-2 text-sm"
            >
              <div className="min-w-0">
                <p className="truncate font-medium text-ivory">
                  #{index + 1} {row.filename || row.video_id}
                </p>
                <p className="muted text-xs">
                  {row.views != null ? `${row.views.toLocaleString()} views · ` : ""}
                  {row.label === 1
                    ? "top quarter of this batch"
                    : row.label === 0
                      ? "below the top quarter"
                      : "no view count"}
                </p>
              </div>
              <span className="shrink-0 text-gold">
                {row.presentation_score.toFixed(1)}
              </span>
            </li>
          ))}
        </ul>
      </section>

      <LimitationsNote limitations={report.limitations} />
    </div>
  );
}
