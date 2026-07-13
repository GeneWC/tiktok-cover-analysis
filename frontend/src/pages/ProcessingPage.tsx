import { useEffect, useMemo } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import ProcessingStatus from "../components/ProcessingStatus";
import { useStatusPolling } from "../hooks/useStatusPolling";
import { useAnalysis } from "../context/AnalysisContext";
import { PIPELINE_STEPS } from "../types";

export default function ProcessingPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { previewUrlFor } = useAnalysis();
  const { data, error } = useStatusPolling(id);

  const previewUrl = id ? previewUrlFor(id) : null;

  useEffect(() => {
    if (data?.status === "complete" && id) {
      navigate(`/report/${id}`, { replace: true });
    }
  }, [data?.status, id, navigate]);

  const progress = useMemo(() => {
    if (!data) return 0;
    const done = PIPELINE_STEPS.filter((s) => {
      const st = data.steps[s];
      return st === "complete" || st === "skipped" || st === "failed";
    }).length;
    return Math.round((done / PIPELINE_STEPS.length) * 100);
  }, [data]);

  const failed = data?.status === "failed";

  return (
    <div className="max-w-3xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-slate-900">
          {failed ? "Analysis could not be completed" : "Analyzing your video…"}
        </h1>
        <p className="mt-1 text-sm text-slate-500">Analysis ID: {id}</p>
      </div>

      <div className="grid gap-6 md:grid-cols-[220px_1fr]">
        <div>
          <div className="aspect-[9/16] w-full overflow-hidden rounded-xl bg-slate-900 ring-1 ring-slate-200">
            {previewUrl ? (
              <video
                src={previewUrl}
                className="h-full w-full object-contain"
                muted
                playsInline
                controls
              />
            ) : (
              <div className="flex h-full items-center justify-center px-4 text-center text-xs text-slate-400">
                Preview unavailable
                <br />
                (reloaded page)
              </div>
            )}
          </div>
        </div>

        <div className="rounded-2xl bg-white shadow-sm ring-1 ring-slate-200 p-5">
          {!failed && (
            <div className="mb-4">
              <div className="flex items-center justify-between text-sm text-slate-600 mb-1">
                <span>Progress</span>
                <span>{progress}%</span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
                <div
                  className="h-full rounded-full bg-indigo-600 transition-all duration-500"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}

          {failed ? (
            <div className="rounded-lg bg-red-50 border border-red-200 px-4 py-3 text-sm text-red-700">
              <p className="font-medium">
                We couldn't finish analyzing this video.
              </p>
              <p className="mt-1">
                It may be corrupted, too short, or missing usable content. You
                can try a different file.
              </p>
            </div>
          ) : (
            <ProcessingStatus steps={data?.steps ?? {}} />
          )}

          {error && !failed && (
            <p className="mt-3 text-xs text-amber-600">
              Reconnecting to the analysis server… ({error})
            </p>
          )}

          {failed && (
            <Link
              to="/"
              className="mt-4 inline-block rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-700"
            >
              Upload another video
            </Link>
          )}
        </div>
      </div>
    </div>
  );
}
