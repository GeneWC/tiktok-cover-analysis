interface LimitationsNoteProps {
  limitations: string[];
}

const DEFAULT_NOTE =
  "This is feedback on how the video is presented. It is not a promise of views.";

export default function LimitationsNote({ limitations }: LimitationsNoteProps) {
  const items = limitations.length > 0 ? limitations : [DEFAULT_NOTE];

  return (
    <section className="panel-quiet rounded-sm p-5">
      <h2 className="font-display text-lg text-ivory">Notes</h2>
      <ul className="muted mt-2 space-y-1.5 text-sm">
        {items.map((item, i) => (
          <li key={i} className="flex gap-2">
            <span className="text-gold/60">•</span>
            <span>{item}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
