import type { ReportExplanation } from "../types";
import { getInlineDefinition, getSignalTerm } from "../lib/glossary";
import InfoTip from "./InfoTip";

interface RecommendationPanelProps {
  explanation: ReportExplanation;
}

/** Render a signal, adding an info tooltip to its leading term when known. */
function SignalText({ signal }: { signal: string }) {
  const parsed = getSignalTerm(signal);
  if (!parsed) return <>{signal}</>;
  return (
    <>
      <span className="font-medium text-slate-800">{parsed.term}</span>
      <InfoTip text={parsed.definition} label={parsed.term} />
      {parsed.rest}
    </>
  );
}

/** Render a recommendation, adding a tooltip when it contains known jargon. */
function RecommendationText({ text }: { text: string }) {
  const inline = getInlineDefinition(text);
  return (
    <>
      {text}
      {inline && <InfoTip text={inline.definition} label={inline.phrase} />}
    </>
  );
}

function SignalList({
  title,
  items,
  tone,
  emptyText,
}: {
  title: string;
  items: string[];
  tone: "strong" | "weak" | "neutral";
  emptyText: string;
}) {
  const styles = {
    strong: { badge: "bg-emerald-500", ring: "ring-emerald-100" },
    weak: { badge: "bg-amber-500", ring: "ring-amber-100" },
    neutral: { badge: "bg-slate-400", ring: "ring-slate-100" },
  }[tone];

  return (
    <div className={`rounded-xl bg-white p-4 ring-1 ${styles.ring}`}>
      <h3 className="flex items-center gap-2 text-sm font-semibold text-slate-800">
        <span className={`inline-block h-2.5 w-2.5 rounded-full ${styles.badge}`} />
        {title}
      </h3>
      {items.length > 0 ? (
        <ul className="mt-2 space-y-1.5 text-sm text-slate-700">
          {items.map((item, i) => (
            <li key={i} className="flex gap-2">
              <span className="text-slate-300">•</span>
              <span>
                <SignalText signal={item} />
              </span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="mt-2 text-sm text-slate-400">{emptyText}</p>
      )}
    </div>
  );
}

export default function RecommendationPanel({
  explanation,
}: RecommendationPanelProps) {
  return (
    <section className="rounded-2xl bg-slate-100/60 ring-1 ring-slate-200 p-5">
      <h2 className="text-lg font-semibold text-slate-900">
        Signals & recommendations
      </h2>
      <p className="mt-1 text-sm text-slate-500">
        How this video compares to similar covers on interpretable production
        signals. These are correlations, not guarantees.
      </p>

      <div className="mt-4 grid gap-3 md:grid-cols-3">
        <SignalList
          title="Strong signals"
          tone="strong"
          items={explanation.strong_signals}
          emptyText="No standout strengths detected."
        />
        <SignalList
          title="Weak signals"
          tone="weak"
          items={explanation.weak_signals}
          emptyText="No notable weaknesses detected."
        />
        <SignalList
          title="Neutral / not evaluated"
          tone="neutral"
          items={explanation.neutral_or_missing_signals}
          emptyText="Nothing to note."
        />
      </div>

      {explanation.recommendations.length > 0 && (
        <div className="mt-4 rounded-xl bg-white p-4 ring-1 ring-indigo-100">
          <h3 className="text-sm font-semibold text-indigo-800">
            Suggested improvements
          </h3>
          <ol className="mt-2 space-y-1.5 text-sm text-slate-700">
            {explanation.recommendations.map((rec, i) => (
              <li key={i} className="flex gap-2">
                <span className="font-semibold text-indigo-500">{i + 1}.</span>
                <span>
                  <RecommendationText text={rec} />
                </span>
              </li>
            ))}
          </ol>
        </div>
      )}
    </section>
  );
}
