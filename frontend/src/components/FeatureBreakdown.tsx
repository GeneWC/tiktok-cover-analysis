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
      <section className="rounded-2xl bg-white shadow-sm ring-1 ring-slate-200 p-5">
        <h2 className="text-lg font-semibold text-slate-900">
          Feature breakdown
        </h2>
        <p className="mt-2 text-sm text-slate-500">
          No feature values are available for this video.
        </p>
      </section>
    );
  }

  return (
    <section className="rounded-2xl bg-white shadow-sm ring-1 ring-slate-200 p-5">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-slate-900">
          Feature breakdown
        </h2>
        <button
          type="button"
          className="text-sm font-medium text-indigo-600 hover:text-indigo-700"
          onClick={() => setExpanded((v) => !v)}
        >
          {expanded ? "Collapse all" : "Expand all"}
        </button>
      </div>
      <p className="mt-1 text-sm text-slate-500">
        Raw measurements extracted from your video, grouped by dimension.
      </p>

      <div className="mt-4 space-y-3">
        {FEATURE_GROUP_ORDER.map((group) => {
          const rows = grouped[group];
          if (rows.length === 0) return null;
          return (
            <details
              key={group}
              open={expanded}
              className="group rounded-xl border border-slate-200"
            >
              <summary className="flex cursor-pointer items-center justify-between px-4 py-3 text-sm font-semibold text-slate-800 marker:content-['']">
                <span>{group}</span>
                <span className="text-xs font-normal text-slate-400">
                  {rows.length} {rows.length === 1 ? "feature" : "features"}
                </span>
              </summary>
              <div className="border-t border-slate-100 px-4 py-2">
                <dl className="grid gap-x-6 gap-y-1 sm:grid-cols-2">
                  {rows.map(({ key, value }) => {
                    const label = labelForFeature(key);
                    const definition = getDefinition(key);
                    return (
                      <div
                        key={key}
                        className="flex items-center justify-between gap-3 border-b border-slate-50 py-1.5"
                      >
                        <dt className="flex items-center text-sm text-slate-600" title={key}>
                          {label}
                          {definition && (
                            <InfoTip text={definition} label={label} />
                          )}
                        </dt>
                        <dd className="text-sm font-medium tabular-nums text-slate-900">
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
