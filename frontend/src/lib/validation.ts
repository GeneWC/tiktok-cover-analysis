// Client-side pre-checks that mirror the backend upload validation
// (backend/core/config.py + video_validation.py) for fast feedback. The
// backend remains the source of truth and re-validates every upload.

export const SUPPORTED_EXTENSIONS = [".mp4", ".mov", ".m4v"] as const;
export const MAX_FILE_SIZE_MB = 200;
export const MAX_DURATION_SECONDS = 120;
export const MIN_DURATION_SECONDS = 1;

function extensionOf(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot === -1 ? "" : name.slice(dot).toLowerCase();
}

/** Synchronous checks (extension + size). Returns an error string or null. */
export function validateFileBasics(file: File): string | null {
  const ext = extensionOf(file.name);
  if (!SUPPORTED_EXTENSIONS.includes(ext as (typeof SUPPORTED_EXTENSIONS)[number])) {
    return `Unsupported file type "${ext || "unknown"}". Supported: ${SUPPORTED_EXTENSIONS.join(
      ", "
    )}.`;
  }
  const sizeMb = file.size / (1024 * 1024);
  if (sizeMb > MAX_FILE_SIZE_MB) {
    return `File is too large (${sizeMb.toFixed(
      0
    )} MB). Maximum is ${MAX_FILE_SIZE_MB} MB.`;
  }
  return null;
}

/** Read the video duration in the browser to validate the length bound. */
export function readVideoDuration(objectUrl: string): Promise<number> {
  return new Promise((resolve, reject) => {
    const el = document.createElement("video");
    el.preload = "metadata";
    el.onloadedmetadata = () => resolve(el.duration);
    el.onerror = () => reject(new Error("Could not read video metadata."));
    el.src = objectUrl;
  });
}

/** Validate duration bounds. Returns an error string or null. */
export function validateDuration(seconds: number): string | null {
  if (!Number.isFinite(seconds)) return null; // let the backend decide
  if (seconds > MAX_DURATION_SECONDS) {
    return `Video is ${seconds.toFixed(
      0
    )}s long. Maximum is ${MAX_DURATION_SECONDS}s.`;
  }
  if (seconds < MIN_DURATION_SECONDS) {
    return `Video is too short (min ${MIN_DURATION_SECONDS}s).`;
  }
  return null;
}
