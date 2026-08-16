import { Link, NavLink, useLocation } from "react-router-dom";
import type { ReactNode } from "react";
import FlameMark from "./FlameMark";

export default function Layout({ children }: { children: ReactNode }) {
  const { pathname } = useLocation();
  const oneVideo =
    pathname === "/" ||
    pathname.startsWith("/processing") ||
    pathname.startsWith("/report");
  const several = pathname.startsWith("/channel");

  return (
    <div className="app-shell">
      <a href="#main" className="skip-link">
        Skip to content
      </a>
      <header className="site-header">
        <div className="mx-auto flex max-w-5xl items-center gap-3 px-4 py-2">
          <Link to="/" className="group flex min-h-11 items-center gap-2.5">
            <FlameMark className="h-8 w-8 text-gold" />
            <span className="font-display text-lg font-medium text-ivory group-hover:text-gold">
              Zukover
            </span>
          </Link>
          <nav className="ml-auto flex items-center gap-1 text-sm">
            <NavLink
              to="/"
              end
              aria-current={oneVideo ? "page" : undefined}
              className={`nav-link inline-flex min-h-11 items-center px-3 ${oneVideo ? "nav-link-active" : ""}`}
            >
              One video
            </NavLink>
            <NavLink
              to="/channel"
              aria-current={several ? "page" : undefined}
              className={`nav-link inline-flex min-h-11 items-center px-3 ${several ? "nav-link-active" : ""}`}
            >
              Several videos
            </NavLink>
          </nav>
        </div>
      </header>

      <main id="main" className="mx-auto w-full max-w-5xl flex-1 px-4 py-10">
        {children}
      </main>

      <footer className="site-footer mt-auto">
        <div className="mx-auto max-w-5xl px-4 py-5 text-center text-sm">
          <p className="muted">Feedback only. Not a guarantee of views.</p>
          <p className="mt-2 text-ivory">
            made by{" "}
            <a
              href="https://www.tiktok.com/@suibianmusic"
              className="link"
              target="_blank"
              rel="noreferrer"
            >
              @suibianmusic
            </a>
            {" · Formerly CoverSignal"}
          </p>
        </div>
      </footer>
    </div>
  );
}
