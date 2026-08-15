export default function FlameMark({ className = "h-8 w-8" }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 32 32"
      fill="none"
      aria-hidden="true"
    >
      <circle
        cx="16"
        cy="16"
        r="14.25"
        stroke="currentColor"
        strokeWidth="1.2"
      />
      <path
        fill="currentColor"
        d="M16 6.8c1.15 3.2-.45 5.25-1.7 6.5 2.8-.35 5.55 1.35 5.55 4.85 0 3.2-2.4 5.75-5.5 5.75s-5.5-2.55-5.5-5.75c0-2.4 1.15-4.2 2.35-5.65C12.55 10.6 14.2 8.85 16 6.8Z"
      />
    </svg>
  );
}
