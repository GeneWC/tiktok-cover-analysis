import type { ReportScores } from "../types";
import { formatProbability, NOT_AVAILABLE } from "../lib/format";
import { getDefinition } from "../lib/glossary";
import TierBadge from "./TierBadge";
import InfoTip from "./InfoTip";
import RelativeBand from "./RelativeBand";

interface PredictionCardProps {
  scores: ReportScores;
}

export default function PredictionCard({ scores }: PredictionCardProps) {
  const prob = scores.top_quartile_probability;

  return (
    <section className="panel rounded-sm p-5" aria-labelledby="relative-result">
      <h2 id="relative-result" className="font-display text-2xl text-ivory">
        Relative to a creator’s own videos
      </h2>
      <p className="muted mt-2 max-w-prose text-sm">
        An estimate of how this cover compares with stronger videos from the
        same kind of creator. Not a view forecast, and not a guarantee.
      </p>

      <div className="mt-5 grid gap-6 md:grid-cols-[1.2fr_1fr]">
        <div>
          <p className="flex flex-wrap items-center gap-2 text-sm text-gold">
            <span className="flex items-center">
              Resemblance to stronger covers
              {getDefinition("top quartile") && (
                <InfoTip
                  text={getDefinition("top quartile")!}
                  label="resemblance"
                />
              )}
            </span>
          </p>
          <p className="mt-2 font-display text-4xl text-ivory">
            {formatProbability(prob)}
          </p>
          <div className="mt-4">
            <RelativeBand probability={prob} />
          </div>
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
