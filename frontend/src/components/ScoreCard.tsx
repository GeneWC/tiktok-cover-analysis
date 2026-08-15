import { formatScore, NOT_AVAILABLE } from "../lib/format";

interface ScoreCardProps {
  label: string;
  score: number | null;
  emphasize?: boolean;
  hint?: string;
}

function barTone(score: number): string {
  if (score >= 66) return "bg-gold";
  if (score >= 33) return "bg-ember";
  return "bg-brick";
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
        "rounded-sm p-4",
        emphasize ? "bg-gold text-dusk" : "panel-quiet",
      ].join(" ")}
    >
      <div className="flex items-baseline justify-between gap-3">
        <span
          className={
            emphasize ? "text-sm text-dusk/80" : "muted text-sm"
          }
        >
          {label}
        </span>
        <span
          className={
            emphasize
              ? "font-display text-2xl font-medium"
              : "font-display text-2xl font-medium text-ivory"
          }
        >
          {available ? formatScore(score) : NOT_AVAILABLE}
          {available && (
            <span
              className={
                emphasize
                  ? "text-sm font-normal text-dusk/70"
                  : "muted text-sm font-normal"
              }
            >
              /100
            </span>
          )}
        </span>
      </div>
      {hint && (
        <p className={emphasize ? "mt-1 text-xs text-dusk/70" : "muted mt-1 text-xs"}>
          {hint}
        </p>
      )}

      {available && (
        <div
          className={[
            "mt-2 h-2 w-full overflow-hidden rounded-sm",
            emphasize ? "bg-dusk/20" : "bar-track",
          ].join(" ")}
        >
          <div
            className={[
              "h-full rounded-sm",
              emphasize ? "bg-dusk" : barTone(pct),
            ].join(" ")}
            style={{ width: `${pct}%` }}
          />
        </div>
      )}
    </div>
  );
}
