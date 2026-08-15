import type { ReportScores } from "../types";
import { formatProbability, NOT_AVAILABLE } from "../lib/format";
import { getDefinition } from "../lib/glossary";
import TierBadge from "./TierBadge";
import InfoTip from "./InfoTip";

interface PredictionCardProps {
  scores: ReportScores;
}

export default function PredictionCard({ scores }: PredictionCardProps) {
  const prob = scores.top_quartile_probability;
  const pct = prob != null ? Math.round(prob * 100) : null;

  return (
    <details className="panel-quiet rounded-sm">
      <summary className="cursor-pointer px-4 py-3 font-display text-xl text-ivory">
        Compared to other covers
      </summary>
      <div className="border-t border-gold/15 px-4 py-4">
        <p className="muted max-w-prose text-sm">
          A rough match only. Not a view forecast.
        </p>

        <div className="mt-4 grid gap-4 sm:grid-cols-2">
          <div>
            <p className="flex flex-wrap items-center gap-2 text-sm text-gold">
              <span className="flex items-center">
                Similarity to stronger covers
                {getDefinition("top quartile") && (
                  <InfoTip
                    text={getDefinition("top quartile")!}
                    label="similarity"
                  />
                )}
              </span>
            </p>
            <p className="mt-2 font-display text-2xl text-ivory">
              {formatProbability(prob)}
            </p>
            {pct != null && (
              <div className="bar-track mt-2 h-2 w-full overflow-hidden rounded-sm">
                <div className="bar-fill h-full" style={{ width: `${pct}%` }} />
              </div>
            )}
          </div>

          <div className="space-y-3">
            <TierRow
              label="View pattern"
              tier={scores.view_performance_tier}
              hint={getDefinition("view performance")}
            />
            <TierRow
              label="Like/comment pattern"
              tier={scores.engagement_tier}
              hint={getDefinition("engagement")}
            />
            <TierRow
              label="Share pattern"
              tier={scores.shareability_tier}
              hint={getDefinition("shareability")}
            />
          </div>
        </div>
      </div>
    </details>
  );
}

function TierRow({
  label,
  tier,
  hint,
}: {
  label: string;
  tier: ReportScores["view_performance_tier"];
  hint?: string | null;
}) {
  return (
    <div className="flex items-center justify-between rounded-sm bg-dusk/40 px-3 py-2">
      <span className="muted flex items-center text-sm">
        {label}
        {hint && <InfoTip text={hint} label={label} />}
      </span>
      {tier ? (
        <TierBadge tier={tier} />
      ) : (
        <span className="muted text-xs">{NOT_AVAILABLE}</span>
      )}
    </div>
  );
}
