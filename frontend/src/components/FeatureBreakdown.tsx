import { useMemo, useState } from "react";
import type { FeatureValue } from "../types";
import {
  FEATURE_GROUP_ORDER,
  type FeatureGroup,
  groupForFeature,
  labelForFeature,
} from "../lib/featureCatalog";
import { formatNumber } from "../lib/format";
import { getDefinition } from "../lib/glossary";
import InfoTip from "./InfoTip";

interface FeatureBreakdownProps {
  features: Record<string, FeatureValue>;
}

type Grouped = Record<FeatureGroup, Array<{ key: string; value: FeatureValue }>>;

export default function FeatureBreakdown({ features }: FeatureBreakdownProps) {
  const [expanded, setExpanded] = useState(false);

  const grouped = useMemo<Grouped>(() => {
    const acc = Object.fromEntries(
      FEATURE_GROUP_ORDER.map((g) => [g, [] as Array<{ key: string; value: FeatureValue }>])
    ) as Grouped;
    for (const [key, value] of Object.entries(features)) {
      acc[groupForFeature(key)].push({ key, value });
    }
    for (const group of FEATURE_GROUP_ORDER) {
      acc[group].sort((a, b) =>
        labelForFeature(a.key).localeCompare(labelForFeature(b.key))
      );
    }
    return acc;
  }, [features]);

  const hasFeatures = Object.keys(features).length > 0;

  if (!hasFeatures) {
    return (
      <section>
        <h2 className="font-display text-2xl text-ivory">Details</h2>
        <p className="muted mt-2 text-sm">No measurements for this video.</p>
      </section>
    );
  }

  return (
    <section>
      <div className="flex items-center justify-between">
        <h2 className="font-display text-2xl text-ivory">Details</h2>
        <button
          type="button"
          className="link text-sm"
          onClick={() => setExpanded((v) => !v)}
        >
          {expanded ? "Collapse all" : "Expand all"}
        </button>
      </div>
      <p className="muted mt-2 text-sm">
        Raw measurements from the video, grouped by type.
      </p>

      <div className="mt-4 space-y-3">
        {FEATURE_GROUP_ORDER.map((group) => {
          const rows = grouped[group];
          if (rows.length === 0) return null;
          return (
            <details
              key={group}
              open={expanded}
              className="panel-quiet group rounded-sm"
            >
              <summary className="flex cursor-pointer items-center justify-between px-4 py-3 text-sm font-semibold text-ivory marker:content-['']">
                <span>{group}</span>
                <span className="muted text-xs font-normal">
                  {rows.length} {rows.length === 1 ? "item" : "items"}
                </span>
              </summary>
              <div className="border-t border-gold/15 px-4 py-2">
                <dl className="grid gap-x-6 gap-y-1 sm:grid-cols-2">
                  {rows.map(({ key, value }) => {
                    const label = labelForFeature(key);
                    const definition = getDefinition(key);
                    return (
                      <div
                        key={key}
                        className="flex items-center justify-between gap-3 border-b border-gold/10 py-1.5"
                      >
                        <dt className="muted flex items-center text-sm" title={key}>
                          {label}
                          {definition && (
                            <InfoTip text={definition} label={label} />
                          )}
                        </dt>
                        <dd className="text-sm font-medium tabular-nums text-ivory">
                          {formatNumber(value)}
                        </dd>
                      </div>
                    );
                  })}
                </dl>
              </div>
            </details>
          );
        })}
      </div>
    </section>
  );
}
