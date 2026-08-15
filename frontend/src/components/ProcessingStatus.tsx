import { PIPELINE_STEPS, type StepStatus } from "../types";
import { formatStepLabel } from "../lib/format";

interface ProcessingStatusProps {
  steps: Record<string, StepStatus>;
}

const STATUS_STYLES: Record<
  StepStatus,
  { dot: string; text: string; label: string }
> = {
  complete: { dot: "bg-gold", text: "text-ivory", label: "Done" },
  running: { dot: "bg-ember animate-pulse", text: "text-ivory", label: "Running" },
  failed: { dot: "bg-brick", text: "text-gold", label: "Failed" },
  skipped: { dot: "bg-ridge", text: "text-dust", label: "Skipped" },
  pending: { dot: "bg-ridge", text: "text-dust", label: "Pending" },
};

function StepIcon({ status }: { status: StepStatus }) {
  if (status === "complete") {
    return (
      <span className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-gold text-[11px] text-dusk">
        ✓
      </span>
    );
  }
  if (status === "failed") {
    return (
      <span className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-brick text-[11px] text-ivory">
        ✕
      </span>
    );
  }
  if (status === "skipped") {
    return (
      <span className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-ridge text-[11px] text-ivory">
        –
      </span>
    );
  }
  if (status === "running") {
    return (
      <span className="inline-flex h-5 w-5 animate-spin items-center justify-center rounded-full border-2 border-gold border-t-transparent" />
    );
  }
  return (
    <span className="inline-flex h-5 w-5 items-center justify-center rounded-full border-2 border-gold/30" />
  );
}

export default function ProcessingStatus({ steps }: ProcessingStatusProps) {
  return (
    <ol className="space-y-1" aria-live="polite" aria-busy={PIPELINE_STEPS.some((s) => (steps[s] ?? "pending") === "running")}>
      {PIPELINE_STEPS.map((step) => {
        const status = steps[step] ?? "pending";
        const style = STATUS_STYLES[status];
        return (
          <li
            key={step}
            className="flex items-center justify-between gap-3 rounded-sm px-3 py-2 hover:bg-ridge/40"
          >
            <div className="flex items-center gap-3">
              <StepIcon status={status} />
              <span className={`text-sm ${style.text}`}>
                {formatStepLabel(step)}
              </span>
            </div>
            <span className="muted text-xs font-medium">{style.label}</span>
          </li>
        );
      })}
    </ol>
  );
}
