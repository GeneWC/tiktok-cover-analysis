import type { VideoMetadata } from "../types";
import { formatSeconds, NOT_AVAILABLE } from "../lib/format";

interface VideoSummaryProps {
  metadata: VideoMetadata | null;
  previewUrl: string | null;
}

function orientation(meta: VideoMetadata): string {
  if (meta.is_vertical_video) return "Vertical";
  if (meta.is_square_video) return "Square";
  return "Horizontal";
}

export default function VideoSummary({
  metadata,
  previewUrl,
}: VideoSummaryProps) {
  return (
    <section className="panel rounded-sm p-5">
      <div className="grid gap-5 sm:grid-cols-[160px_1fr]">
        <div className="aspect-[9/16] w-full overflow-hidden rounded-sm bg-dusk">
          {previewUrl ? (
            <video
              src={previewUrl}
              className="h-full w-full object-contain"
              muted
              playsInline
              controls
            />
          ) : (
            <div className="muted flex h-full items-center justify-center px-3 text-center text-xs">
              Preview unavailable
            </div>
          )}
        </div>

        <div>
          <h2 className="font-display text-2xl text-ivory">The video</h2>
          {metadata ? (
            <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
              <Item label="Duration" value={formatSeconds(metadata.duration_seconds)} />
              <Item
                label="Resolution"
                value={`${metadata.width}×${metadata.height}`}
              />
              <Item
                label="Frame Rate"
                value={metadata.fps != null ? `${metadata.fps.toFixed(0)} fps` : NOT_AVAILABLE}
              />
              <Item label="Orientation" value={orientation(metadata)} />
              <Item
                label="Aspect Ratio"
                value={
                  metadata.aspect_ratio != null
                    ? metadata.aspect_ratio.toFixed(3)
                    : NOT_AVAILABLE
                }
              />
              <Item
                label="Audio"
                value={metadata.has_audio ? "Yes" : "None"}
              />
            </dl>
          ) : (
            <p className="muted mt-3 text-sm">{NOT_AVAILABLE}</p>
          )}
        </div>
      </div>
    </section>
  );
}

function Item({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="muted">{label}</dt>
      <dd className="font-medium text-ivory">{value}</dd>
    </div>
  );
}
