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
    <section className="rounded-2xl bg-white shadow-sm ring-1 ring-slate-200 p-5">
      <h2 className="text-lg font-semibold text-slate-900">
        Predicted performance
      </h2>
      <p className="mt-1 text-sm text-slate-500">
        Model-based estimates from comparable instrumental covers. Shown as a
        probability and tiers, never exact view counts.
      </p>

      <div className="mt-4 grid gap-4 sm:grid-cols-2">
        <div className="rounded-xl bg-indigo-50 ring-1 ring-indigo-100 p-4">
          <p className="flex items-center text-sm text-indigo-700">
            Chance of landing in the top 25% of comparable covers
            {getDefinition("top quartile") && (
              <InfoTip
                text={getDefinition("top quartile")!}
                label="top-quartile probability"
              />
            )}
          </p>
          <p className="mt-1 text-3xl font-bold text-indigo-900">
            {formatProbability(prob)}
          </p>
          {pct != null && (
            <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-indigo-100">
              <div
                className="h-full rounded-full bg-indigo-600"
                style={{ width: `${pct}%` }}
              />
            </div>
          )}
        </div>

        <div className="space-y-3">
          <TierRow
            label="View performance"
            tier={scores.view_performance_tier}
            hint={getDefinition("view performance")}
          />
          <TierRow
            label="Engagement"
            tier={scores.engagement_tier}
            hint={getDefinition("engagement")}
          />
          <TierRow
            label="Shareability"
            tier={scores.shareability_tier}
            hint={getDefinition("shareability")}
          />
        </div>
      </div>
    </section>
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
    <div className="flex items-center justify-between rounded-lg bg-slate-50 px-3 py-2">
      <span className="flex items-center text-sm text-slate-600">
        {label}
        {hint && <InfoTip text={hint} label={label} />}
      </span>
      {tier ? (
        <TierBadge tier={tier} />
      ) : (
        <span className="text-xs text-slate-400">{NOT_AVAILABLE}</span>
      )}
    </div>
  );
}
