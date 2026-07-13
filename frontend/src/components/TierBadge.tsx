import type { PerformanceTier } from "../types";
import { formatTier, NOT_AVAILABLE } from "../lib/format";

const TIER_STYLES: Record<PerformanceTier, string> = {
  low: "bg-slate-100 text-slate-600 ring-slate-200",
  medium: "bg-amber-50 text-amber-700 ring-amber-200",
  medium_high: "bg-sky-50 text-sky-700 ring-sky-200",
  high: "bg-emerald-50 text-emerald-700 ring-emerald-200",
};

export default function TierBadge({
  tier,
}: {
  tier: PerformanceTier | null | undefined;
}) {
  const style = tier
    ? TIER_STYLES[tier]
    : "bg-slate-50 text-slate-400 ring-slate-200";
  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold ring-1 ring-inset ${style}`}
    >
      {tier ? formatTier(tier) : NOT_AVAILABLE}
    </span>
  );
}
