import type { ReportExplanation } from "../types";
import { getInlineDefinition, getSignalTerm } from "../lib/glossary";
import InfoTip from "./InfoTip";

interface RecommendationPanelProps {
  explanation: ReportExplanation;
}

function SignalText({ signal }: { signal: string }) {
  const parsed = getSignalTerm(signal);
  if (!parsed) return <>{signal}</>;
  return (
    <>
      <span className="font-medium text-ivory">{parsed.term}</span>
      <InfoTip text={parsed.definition} label={parsed.term} />
      {parsed.rest}
    </>
  );
}

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
  const badge = {
    strong: "bg-gold",
    weak: "bg-ember",
    neutral: "bg-brick",
  }[tone];

  return (
    <div>
      <h3 className="flex items-center gap-2 text-sm font-semibold text-ivory">
        <span className={`inline-block h-2.5 w-2.5 rounded-full ${badge}`} />
        {title}
      </h3>
      {items.length > 0 ? (
        <ul className="mt-2 space-y-1.5 text-sm text-ivory">
          {items.map((item, i) => (
            <li key={i} className="flex gap-2">
              <span className="text-gold/50">•</span>
              <span>
                <SignalText signal={item} />
              </span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted mt-2 text-sm">{emptyText}</p>
      )}
    </div>
  );
}

export default function RecommendationPanel({
  explanation,
}: RecommendationPanelProps) {
  return (
    <section>
      <h2 className="font-display text-2xl text-ivory">What stands out</h2>
      <p className="muted mt-2 max-w-prose text-sm">
        What this video does well, and where it is weaker.
      </p>

      <div className="mt-5 grid gap-6 md:grid-cols-3">
        <SignalList
          title="Strong"
          tone="strong"
          items={explanation.strong_signals}
          emptyText="Nothing stood out as strong."
        />
        <SignalList
          title="Weak"
          tone="weak"
          items={explanation.weak_signals}
          emptyText="Nothing stood out as weak."
        />
        <SignalList
          title="Other"
          tone="neutral"
          items={explanation.neutral_or_missing_signals}
          emptyText="Nothing else to note."
        />
      </div>

      {explanation.recommendations.length > 0 && (
        <div className="panel mt-6 rounded-sm p-4">
          <h3 className="text-sm font-semibold text-gold">Try this</h3>
          <ol className="mt-2 space-y-1.5 text-sm text-ivory">
            {explanation.recommendations.map((rec, i) => (
              <li key={i} className="flex gap-2">
                <span className="font-semibold text-gold">{i + 1}.</span>
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
