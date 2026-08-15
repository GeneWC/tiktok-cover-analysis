import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ApiError, getChannelReport } from "../api/client";
import type { ChannelReportResponse } from "../types";
import LimitationsNote from "../components/LimitationsNote";

function formatFeature(name: string): string {
  return name.replaceAll("_", " ");
}

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
          err instanceof ApiError ? err.message : "Failed to load channel report."
        )
      );
  }, [id]);

  if (error) {
    return (
      <div className="max-w-2xl mx-auto text-rose-700">
        {error}
        <div className="mt-3">
          <Link to="/channel" className="underline">
            Try again
          </Link>
        </div>
      </div>
    );
  }

  if (!report) {
    return <p className="text-slate-500">Loading channel report…</p>;
  }

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div>
        <p className="text-sm text-indigo-600 mb-1">
          <Link to="/channel" className="hover:underline">
            New channel batch
          </Link>
          <span className="mx-2 text-slate-300">·</span>
          <Link to="/" className="hover:underline text-slate-500">
            Single-video analysis
          </Link>
        </p>
        <h1 className="text-3xl font-bold text-slate-900">
          Channel diagnostics
        </h1>
        <p className="mt-2 text-slate-600">
          Within-batch comparison only. {report.n_videos} videos
          {report.n_hits != null
            ? ` · ${report.n_hits} top-quartile in this batch`
            : " · no hit/miss labels (add views next time)"}
          .
        </p>
      </div>

      {report.recommendations.length > 0 && (
        <section className="rounded-2xl bg-white p-5 shadow-sm ring-1 ring-slate-200">
          <h2 className="text-lg font-semibold text-slate-900">
            Patterns to notice
          </h2>
          <ul className="mt-3 space-y-2 text-sm text-slate-700">
            {report.recommendations.map((tip) => (
              <li key={tip} className="flex gap-2">
                <span className="text-indigo-400">•</span>
                <span>{tip}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {report.top_feature_deltas.length > 0 && (
        <section className="rounded-2xl bg-white p-5 shadow-sm ring-1 ring-slate-200">
          <h2 className="text-lg font-semibold text-slate-900">
            What differs in your stronger uploads
          </h2>
          <p className="mt-1 text-sm text-slate-500">
            Mean feature level for batch top-quartile vs the rest (your data
            only).
          </p>
          <ul className="mt-4 divide-y divide-slate-100">
            {report.top_feature_deltas.map((d) => (
              <li
                key={d.feature}
                className="flex items-center justify-between gap-3 py-2 text-sm"
              >
                <span className="text-slate-700">
                  {formatFeature(d.feature)}
                </span>
                <span
                  className={
                    d.delta >= 0
                      ? "font-medium text-emerald-700"
                      : "font-medium text-rose-700"
                  }
                >
                  {d.delta >= 0 ? "+" : ""}
                  {d.delta.toFixed(3)}
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section className="rounded-2xl bg-white p-5 shadow-sm ring-1 ring-slate-200">
        <h2 className="text-lg font-semibold text-slate-900">
          Videos ranked by presentation proxy
        </h2>
        <p className="mt-1 text-sm text-slate-500">
          Simple within-batch average of brightness / sharpness / framing cues —
          not a model score.
        </p>
        <ul className="mt-4 divide-y divide-slate-100">
          {report.video_ranks.map((row, index) => (
            <li
              key={row.video_id}
              className="flex items-center justify-between gap-3 py-2 text-sm"
            >
              <div className="min-w-0">
                <p className="truncate font-medium text-slate-800">
                  #{index + 1} {row.filename || row.video_id}
                </p>
                <p className="text-xs text-slate-500">
                  {row.views != null ? `${row.views.toLocaleString()} views · ` : ""}
                  {row.label === 1
                    ? "batch top quartile"
                    : row.label === 0
                      ? "below batch top quartile"
                      : "unlabeled"}
                </p>
              </div>
              <span className="shrink-0 text-slate-600">
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
