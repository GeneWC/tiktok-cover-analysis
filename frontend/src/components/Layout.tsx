import { Link } from "react-router-dom";
import type { ReactNode } from "react";

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-full flex flex-col">
      <header className="border-b border-slate-200 bg-white/80 backdrop-blur sticky top-0 z-10">
        <div className="mx-auto max-w-5xl px-4 py-3 flex items-center gap-3">
          <Link to="/" className="flex items-center gap-2 group">
            <span className="inline-flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-600 text-white font-bold">
              C
            </span>
            <span className="font-semibold text-slate-900 group-hover:text-indigo-700">
              CoverSignal
            </span>
          </Link>
          <span className="text-sm text-slate-400 hidden sm:inline">
            Instrumental cover video analyzer
          </span>
        </div>
      </header>

      <main className="flex-1 mx-auto w-full max-w-5xl px-4 py-8">
        {children}
      </main>

      <footer className="border-t border-slate-200 bg-white">
        <div className="mx-auto max-w-5xl px-4 py-4 text-xs text-slate-500">
          Exploratory analysis for creative feedback only — not a guarantee of
          performance. Predictions are shown as tiers and probabilities, never
          exact view counts.
        </div>
      </footer>
    </div>
  );
}
