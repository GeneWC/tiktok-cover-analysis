import { PIPELINE_STEPS, type StepStatus } from "../types";
import { formatStepLabel } from "../lib/format";

interface ProcessingStatusProps {
  steps: Record<string, StepStatus>;
}

const STATUS_STYLES: Record<
  StepStatus,
  { dot: string; text: string; label: string }
> = {
  complete: { dot: "bg-emerald-500", text: "text-slate-900", label: "Done" },
  running: { dot: "bg-indigo-500 animate-pulse", text: "text-slate-900", label: "Running" },
  failed: { dot: "bg-red-500", text: "text-red-700", label: "Failed" },
  skipped: { dot: "bg-slate-300", text: "text-slate-500", label: "Skipped" },
  pending: { dot: "bg-slate-200", text: "text-slate-400", label: "Pending" },
};

function StepIcon({ status }: { status: StepStatus }) {
  if (status === "complete") {
    return (
      <span className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-emerald-500 text-white text-[11px]">
        ✓
      </span>
    );
  }
  if (status === "failed") {
    return (
      <span className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-red-500 text-white text-[11px]">
        ✕
      </span>
    );
  }
  if (status === "skipped") {
    return (
      <span className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-slate-300 text-white text-[11px]">
        –
      </span>
    );
  }
  if (status === "running") {
    return (
      <span className="inline-flex h-5 w-5 items-center justify-center rounded-full border-2 border-indigo-500 border-t-transparent animate-spin" />
    );
  }
  return (
    <span className="inline-flex h-5 w-5 items-center justify-center rounded-full border-2 border-slate-200" />
  );
}

export default function ProcessingStatus({ steps }: ProcessingStatusProps) {
  return (
    <ol className="space-y-1">
      {PIPELINE_STEPS.map((step) => {
        const status = steps[step] ?? "pending";
        const style = STATUS_STYLES[status];
        return (
          <li
            key={step}
            className="flex items-center justify-between gap-3 rounded-lg px-3 py-2 hover:bg-slate-50"
          >
            <div className="flex items-center gap-3">
              <StepIcon status={status} />
              <span className={`text-sm ${style.text}`}>
                {formatStepLabel(step)}
              </span>
            </div>
            <span className="text-xs font-medium text-slate-400">
              {style.label}
            </span>
          </li>
        );
      })}
    </ol>
  );
}
