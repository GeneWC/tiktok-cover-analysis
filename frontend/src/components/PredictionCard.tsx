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
        Exploratory similarity scores
      </h2>
      <p className="mt-1 text-sm text-slate-500">
        Weak, cross-creator pattern matches — not a forecast of views or
        virality. Shown as a similarity probability and tiers only.
      </p>

      <div className="mt-4 grid gap-4 sm:grid-cols-2">
        <div className="rounded-xl bg-indigo-50 ring-1 ring-indigo-100 p-4">
          <p className="flex items-center gap-2 text-sm text-indigo-700">
            <span className="flex items-center">
              Similarity to top-quartile covers (exploratory)
              {getDefinition("top quartile") && (
                <InfoTip
                  text={getDefinition("top quartile")!}
                  label="top-quartile probability"
                />
              )}
            </span>
            <span className="rounded bg-amber-50 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-amber-800 ring-1 ring-amber-200">
              Low confidence
            </span>
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
            exploratory
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
            exploratory
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
  exploratory = false,
}: {
  label: string;
  tier: ReportScores["view_performance_tier"];
  hint?: string | null;
  exploratory?: boolean;
}) {
  return (
    <div className="flex items-center justify-between rounded-lg bg-slate-50 px-3 py-2">
      <span className="flex items-center gap-2 text-sm text-slate-600">
        <span className="flex items-center">
          {label}
          {hint && <InfoTip text={hint} label={label} />}
        </span>
        {exploratory && (
          <span className="rounded bg-amber-50 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-amber-800 ring-1 ring-amber-200">
            Low confidence
          </span>
        )}
      </span>
      {tier ? (
        <TierBadge tier={tier} />
      ) : (
        <span className="text-xs text-slate-400">{NOT_AVAILABLE}</span>
      )}
    </div>
  );
}
