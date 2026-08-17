import { formatProbability } from "../lib/format";

type Band = "stronger" | "typical" | "weaker";

function bandFor(probability: number | null | undefined): Band | null {
  if (probability == null || Number.isNaN(probability)) return null;
  if (probability >= 0.6) return "stronger";
  if (probability <= 0.4) return "weaker";
  return "typical";
}

const BAND_COPY: Record<Band, string> = {
  stronger: "Stronger than the typical reference cover",
  typical: "Close to the typical reference cover",
  weaker: "Weaker than the typical reference cover",
};

export default function RelativeBand({
  probability,
}: {
  probability: number | null | undefined;
}) {
  const band = bandFor(probability);
  const pct = probability == null ? null : Math.round(probability * 100);

  return (
    <div>
      <p className="sr-only">
        Estimated resemblance to stronger covers:
        {probability == null
          ? " not available"
          : ` ${formatProbability(probability)}, ${band ? BAND_COPY[band] : ""}`}
      </p>
      <div className="bar-track relative h-3 w-full overflow-hidden rounded-sm">
        <div
          className="absolute inset-y-0 left-0 bg-gold/25"
          style={{ width: "40%" }}
          aria-hidden="true"
        />
        <div
          className="absolute inset-y-0 left-[40%] bg-ember/35"
          style={{ width: "20%" }}
          aria-hidden="true"
        />
        <div
          className="absolute inset-y-0 left-[60%] right-0 bg-gold/40"
          aria-hidden="true"
        />
        {pct != null && (
          <div
            className="absolute top-1/2 h-4 w-0.5 -translate-y-1/2 bg-ivory"
            style={{ left: `${pct}%` }}
            aria-hidden="true"
          />
        )}
      </div>
      <div className="muted mt-2 flex justify-between text-[11px] uppercase tracking-wide">
        <span>Weaker</span>
        <span>Typical</span>
        <span>Stronger</span>
      </div>
      {band && (
        <p className="mt-2 text-sm text-ivory">{BAND_COPY[band]}.</p>
      )}
    </div>
  );
}
