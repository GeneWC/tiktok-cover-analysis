import { formatScore, NOT_AVAILABLE } from "../lib/format";
import InfoTip from "./InfoTip";

interface ScoreCardProps {
  label: string;
  score: number | null;
  emphasize?: boolean;
  hint?: string;
}

function barColor(score: number): string {
  if (score >= 66) return "bg-emerald-500";
  if (score >= 33) return "bg-amber-500";
  return "bg-red-500";
}

export default function ScoreCard({
  label,
  score,
  emphasize = false,
  hint,
}: ScoreCardProps) {
  const available = score != null && !Number.isNaN(score);
  const pct = available ? Math.max(0, Math.min(100, score)) : 0;

  return (
    <div
      className={[
        "rounded-xl p-4 ring-1",
        emphasize
          ? "bg-indigo-600 text-white ring-indigo-600"
          : "bg-white ring-slate-200",
      ].join(" ")}
    >
      <div className="flex items-baseline justify-between">
        <span
          className={
            emphasize
              ? "flex items-center text-sm text-indigo-100"
              : "flex items-center text-sm text-slate-600"
          }
        >
          {label}
          {hint && <InfoTip text={hint} label={label} />}
        </span>
        <span
          className={
            emphasize
              ? "text-2xl font-bold"
              : "text-2xl font-bold text-slate-900"
          }
        >
          {available ? formatScore(score) : ""}
          {available && (
            <span
              className={
                emphasize
                  ? "text-sm font-normal text-indigo-100"
                  : "text-sm font-normal text-slate-400"
              }
            >
              /100
            </span>
          )}
        </span>
      </div>

      {available ? (
        <div
          className={[
            "mt-2 h-2 w-full overflow-hidden rounded-full",
            emphasize ? "bg-indigo-400/40" : "bg-slate-100",
          ].join(" ")}
        >
          <div
            className={[
              "h-full rounded-full",
              emphasize ? "bg-white" : barColor(pct),
            ].join(" ")}
            style={{ width: `${pct}%` }}
          />
        </div>
      ) : (
        <p
          className={
            emphasize
              ? "mt-2 text-sm text-indigo-100"
              : "mt-2 text-sm text-slate-400"
          }
        >
          {NOT_AVAILABLE}
        </p>
      )}
    </div>
  );
}
