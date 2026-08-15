import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { ApiError, getChannelStatus } from "../api/client";
import { CHANNEL_STEPS, type StepStatus } from "../types";

const STEP_LABELS: Record<string, string> = {
  upload: "Upload",
  features: "Extract features",
  diagnose: "Within-batch diagnostics",
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
          setError(status.error || "Channel diagnostics failed.");
          return;
        }
        timer = window.setTimeout(tick, 1500);
      } catch (err) {
        if (cancelled) return;
        setError(
          err instanceof ApiError
            ? err.message
            : "Could not poll channel status."
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
    <div className="max-w-xl mx-auto">
      <h1 className="text-2xl font-bold text-slate-900">
        Running channel diagnostics
      </h1>
      <p className="mt-2 text-sm text-slate-600">
        Extracting presentation features for each video, then comparing within
        your batch. This can take a while on CPU.
      </p>

      {nVideos > 0 && (
        <p className="mt-4 text-sm text-slate-500">
          Features: {nDone} / {nVideos}
        </p>
      )}

      <ol className="mt-6 space-y-2">
        {CHANNEL_STEPS.map((step) => {
          const state = steps?.[step] ?? "pending";
          return (
            <li
              key={step}
              className="flex items-center justify-between rounded-lg bg-white px-3 py-2 ring-1 ring-slate-200 text-sm"
            >
              <span>{STEP_LABELS[step] ?? step}</span>
              <span className="text-slate-500 capitalize">{state}</span>
            </li>
          );
        })}
      </ol>

      {error && (
        <div className="mt-6 rounded-xl bg-rose-50 px-4 py-3 text-sm text-rose-700">
          {error}
          <div className="mt-2">
            <Link to="/channel" className="font-medium underline">
              Back to channel upload
            </Link>
          </div>
        </div>
      )}
    </div>
  );
}
