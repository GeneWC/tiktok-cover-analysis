interface LimitationsNoteProps {
  limitations: string[];
}

const DEFAULT_DISCLAIMER =
  "This is an exploratory analysis for creative feedback only. Predictions are based on patterns in comparable instrumental covers and are not a guarantee of real-world performance.";

export default function LimitationsNote({ limitations }: LimitationsNoteProps) {
  const items = limitations.length > 0 ? limitations : [DEFAULT_DISCLAIMER];

  return (
    <section className="rounded-2xl border border-amber-200 bg-amber-50 p-5">
      <div className="flex items-start gap-3">
        <svg
          className="mt-0.5 h-5 w-5 flex-shrink-0 text-amber-500"
          viewBox="0 0 24 24"
          fill="currentColor"
          aria-hidden="true"
        >
          <path
            fillRule="evenodd"
            d="M9.401 3.003c1.155-2 4.043-2 5.197 0l7.355 12.748c1.154 2-.29 4.5-2.599 4.5H4.645c-2.309 0-3.752-2.5-2.598-4.5L9.4 3.003ZM12 8.25a.75.75 0 0 1 .75.75v3.75a.75.75 0 0 1-1.5 0V9a.75.75 0 0 1 .75-.75Zm0 8.25a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5Z"
            clipRule="evenodd"
          />
        </svg>
        <div>
          <h2 className="text-sm font-semibold text-amber-900">
            Limitations & disclaimers
          </h2>
          <ul className="mt-2 space-y-1.5 text-sm text-amber-800">
            {items.map((item, i) => (
              <li key={i} className="flex gap-2">
                <span className="text-amber-400">•</span>
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
