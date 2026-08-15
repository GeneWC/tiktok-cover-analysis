import { useId, useState } from "react";

interface InfoTipProps {
  text: string;
  label?: string;
}

export default function InfoTip({ text, label }: InfoTipProps) {
  const [open, setOpen] = useState(false);
  const id = useId();

  return (
    <span className="group/tip relative inline-flex align-middle">
      <button
        type="button"
        aria-label={label ? `About ${label}` : "More info"}
        aria-expanded={open}
        aria-controls={id}
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setOpen((v) => !v);
        }}
        onBlur={() => setOpen(false)}
        className="inline-flex h-11 w-11 -my-3 -mx-1 items-center justify-center text-ivory transition hover:text-gold focus:outline-none focus-visible:ring-2 focus-visible:ring-gold"
      >
        <span
          aria-hidden="true"
          className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-ridge text-[10px] font-bold leading-none"
        >
          ?
        </span>
      </button>
      <span
        id={id}
        role="tooltip"
        className={[
          "pointer-events-none absolute bottom-full left-1/2 z-20 mb-2 w-60 -translate-x-1/2 rounded-sm bg-dusk px-3 py-2 text-left text-xs font-normal leading-snug text-ivory ring-1 ring-gold/30 transition-opacity duration-150",
          open
            ? "opacity-100"
            : "opacity-0 group-hover/tip:opacity-100 group-focus-within/tip:opacity-100",
        ].join(" ")}
      >
        {text}
        <span className="absolute left-1/2 top-full -mt-1 h-2 w-2 -translate-x-1/2 rotate-45 bg-dusk" />
      </span>
    </span>
  );
}
