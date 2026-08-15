import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { ApiError, getChannelStatus } from "../api/client";
import { CHANNEL_STEPS, type StepStatus } from "../types";
import { formatStatusLabel } from "../lib/format";

const STEP_LABELS: Record<string, string> = {
  upload: "Upload",
  features: "Read each video",
  diagnose: "Compare the batch",
  report: "Build report",
};

export default function ChannelProcessingPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [steps, setSteps] = useState<Record<string, StepStatus> | null>(null);
  const [nDone, setNDone] = useState(0);
  const [nVideos, setNVideos] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    let timer: number | undefined;

    const tick = async () => {
      try {
        const status = await getChannelStatus(id);
        if (cancelled) return;
        setSteps(status.steps);
        setNDone(status.n_features_done);
        setNVideos(status.n_videos);
        if (status.status === "complete") {
          navigate(`/channel/report/${id}`, { replace: true });
          return;
        }
        if (status.status === "failed") {
          setError(status.error || "Comparison failed.");
          return;
        }
        timer = window.setTimeout(tick, 1500);
      } catch (err) {
        if (cancelled) return;
        setError(
          err instanceof ApiError
            ? err.message
            : "Could not check progress."
        );
      }
    };

    tick();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [id, navigate]);

  return (
    <div className="mx-auto max-w-xl">
      <h1 className="font-display text-3xl font-medium text-ivory">
        Comparing your videos
      </h1>
      <p className="muted mt-3 max-w-prose">
        Reading each video, then comparing them to each other. This can take a
        while.
      </p>

      {nVideos > 0 && (
        <p className="muted mt-4 text-sm">
          Videos read: {nDone} / {nVideos}
        </p>
      )}

      <ol className="mt-6 space-y-2">
        {CHANNEL_STEPS.map((step) => {
          const state = steps?.[step] ?? "pending";
          return (
            <li
              key={step}
              className="panel-quiet flex items-center justify-between rounded-sm px-3 py-2 text-sm"
            >
              <span>{STEP_LABELS[step] ?? step}</span>
              <span className="muted">{formatStatusLabel(state)}</span>
            </li>
          );
        })}
      </ol>

      {error && (
        <div className="alert mt-6 rounded-sm px-4 py-3 text-sm">
          {error}
          <div className="mt-2">
            <Link to="/channel" className="btn-ghost btn-inline mt-2 rounded-sm px-4 py-2 text-sm">
              Back to upload
            </Link>
          </div>
        </div>
      )}
    </div>
  );
}
