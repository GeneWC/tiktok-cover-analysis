import type { PerformanceTier } from "../types";
import { formatTier, NOT_AVAILABLE } from "../lib/format";

const TIER_STYLES: Record<PerformanceTier, string> = {
  low: "bg-ridge text-dust",
  medium: "bg-wine text-ivory",
  medium_high: "bg-ember/25 text-gold",
  high: "bg-gold text-dusk",
};

export default function TierBadge({
  tier,
}: {
  tier: PerformanceTier | null | undefined;
}) {
  const style = tier ? TIER_STYLES[tier] : "bg-ridge text-dust";
  return (
    <span
      className={`inline-flex items-center rounded-sm px-2.5 py-0.5 text-xs font-semibold ${style}`}
    >
      {tier ? formatTier(tier) : NOT_AVAILABLE}
    </span>
  );
}
