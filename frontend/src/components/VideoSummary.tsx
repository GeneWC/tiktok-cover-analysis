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
    <section className="rounded-2xl bg-white shadow-sm ring-1 ring-slate-200 p-5">
      <div className="grid gap-5 sm:grid-cols-[160px_1fr]">
        <div className="aspect-[9/16] w-full overflow-hidden rounded-xl bg-slate-900">
          {previewUrl ? (
            <video
              src={previewUrl}
              className="h-full w-full object-contain"
              muted
              playsInline
              controls
            />
          ) : (
            <div className="flex h-full items-center justify-center px-3 text-center text-xs text-slate-400">
              Preview unavailable
            </div>
          )}
        </div>

        <div>
          <h2 className="text-lg font-semibold text-slate-900">
            Video summary
          </h2>
          {metadata ? (
            <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
              <Item label="Duration" value={formatSeconds(metadata.duration_seconds)} />
              <Item
                label="Resolution"
                value={`${metadata.width}×${metadata.height}`}
              />
              <Item
                label="Frame rate"
                value={metadata.fps != null ? `${metadata.fps.toFixed(0)} fps` : NOT_AVAILABLE}
              />
              <Item label="Orientation" value={orientation(metadata)} />
              <Item
                label="Aspect ratio"
                value={
                  metadata.aspect_ratio != null
                    ? metadata.aspect_ratio.toFixed(3)
                    : NOT_AVAILABLE
                }
              />
              <Item
                label="Audio track"
                value={metadata.has_audio ? "Present" : "None"}
              />
            </dl>
          ) : (
            <p className="mt-3 text-sm text-slate-500">{NOT_AVAILABLE}</p>
          )}
        </div>
      </div>
    </section>
  );
}

function Item({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-slate-500">{label}</dt>
      <dd className="font-medium text-slate-900">{value}</dd>
    </div>
  );
}
