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
    <div className="mx-auto max-w-3xl">
      <div className="mb-8">
        <h1 className="font-display text-3xl font-medium text-ivory">
          {failed ? "Could not finish this video" : "Analyzing…"}
        </h1>
        {!failed && (
          <p className="muted mt-2 max-w-prose text-sm">
            Progress follows finished analysis steps, not a stopwatch. Keep this
            link — a refresh can resume the job, but the local preview is gone.
          </p>
        )}
      </div>

      <div className="grid gap-6 md:grid-cols-[220px_1fr]">
        <div>
          <div className="aspect-[9/16] w-full overflow-hidden rounded-sm bg-dusk ring-1 ring-gold/25">
            {previewUrl ? (
              <video
                src={previewUrl}
                className="h-full w-full object-contain"
                muted
                playsInline
                controls
              />
            ) : (
              <div className="muted flex h-full items-center justify-center px-4 text-center text-xs">
                Preview unavailable
                <br />
                (page was reloaded)
              </div>
            )}
          </div>
        </div>

        <div className="panel rounded-sm p-5">
          {!failed && (
            <div className="mb-4">
              <div className="muted mb-1 flex items-center justify-between text-sm">
                <span>Progress</span>
                <span>{progress}%</span>
              </div>
              <div className="bar-track h-2 w-full overflow-hidden rounded-sm">
                <div
                  className="bar-fill h-full transition-all duration-500"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}

          {failed ? (
            <div className="alert rounded-sm px-4 py-3 text-sm">
              <p className="font-medium">This video could not be analyzed.</p>
              <p className="mt-1">
                It may be damaged, too short, or missing usable content. Try
                another file.
              </p>
            </div>
          ) : (
            <ProcessingStatus steps={data?.steps ?? {}} />
          )}

          {error && !failed && (
            <p className="mt-3 text-xs text-gold">
              Reconnecting… ({error})
            </p>
          )}

          {failed && (
            <Link to="/" className="btn-primary btn-inline mt-4 rounded-sm px-4 py-2 text-sm">
              Upload another video
            </Link>
          )}
        </div>
      </div>
    </div>
  );
}
