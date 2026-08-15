import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import MultiVideoUploader from "../components/MultiVideoUploader";
import { ApiError, diagnoseChannel } from "../api/client";
import { validateFileBasics } from "../lib/validation";

const MIN_FILES = 5;

export default function ChannelUploadPage() {
  const navigate = useNavigate();
  const [files, setFiles] = useState<File[]>([]);
  const [viewsText, setViewsText] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const handleFiles = (next: File[]) => {
    setError(null);
    for (const file of next) {
      const basic = validateFileBasics(file);
      if (basic) {
        setError(basic);
        return;
      }
    }
    setFiles(next);
  };

  const viewsAligned = useMemo(() => {
    return files.map((f) => {
      const raw = (viewsText[f.name] ?? "").trim();
      if (!raw) return null;
      const n = Number(raw);
      return Number.isFinite(n) && n >= 0 ? Math.floor(n) : null;
    });
  }, [files, viewsText]);

  const allViewsFilled = viewsAligned.every((v) => v != null);
  const anyViewFilled = viewsAligned.some((v) => v != null);

  const handleSubmit = async () => {
    if (submitting) return;
    if (files.length < MIN_FILES) {
      setError(`Add at least ${MIN_FILES} videos.`);
      return;
    }
    if (anyViewFilled && !allViewsFilled) {
      setError("Enter views for every video, or leave all blank.");
      return;
    }

    setSubmitting(true);
    setError(null);
    try {
      const { channel_id } = await diagnoseChannel({
        files,
        views: allViewsFilled ? (viewsAligned as number[]) : undefined,
      });
      navigate(`/channel/processing/${channel_id}`);
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : "Could not upload the videos."
      );
      setSubmitting(false);
    }
  };

  return (
    <div className="mx-auto max-w-2xl">
      <div className="mb-10 text-center">
        <h1 className="font-display text-4xl font-medium text-ivory">
          Compare Your Videos
        </h1>
        <p className="muted mx-auto mt-3 max-w-md">
          Upload {MIN_FILES} or more of your covers. See what the stronger ones
          share.
        </p>
      </div>

      <div className="panel rounded-sm p-6">
        <MultiVideoUploader
          files={files}
          onFilesSelected={handleFiles}
          disabled={submitting}
          minFiles={MIN_FILES}
        />

        {files.length > 0 && (
          <div className="mt-6">
            <h2 className="font-display text-lg text-ivory">
              View counts (optional)
            </h2>
            <p className="muted mt-1 text-sm">
              Fill in every video, or leave them all blank.
            </p>
            <div className="mt-3 space-y-2">
              {files.map((file) => (
                <label
                  key={file.name}
                  className="flex items-center gap-3 text-sm"
                >
                  <span className="flex-1 truncate text-ivory">
                    {file.name}
                  </span>
                  <input
                    type="number"
                    min={0}
                    inputMode="numeric"
                    placeholder="views"
                    disabled={submitting}
                    value={viewsText[file.name] ?? ""}
                    onChange={(e) =>
                      setViewsText((prev) => ({
                        ...prev,
                        [file.name]: e.target.value,
                      }))
                    }
                    className="field min-h-11 w-32 rounded-sm px-2"
                  />
                </label>
              ))}
            </div>
          </div>
        )}

        {error && (
          <p className="alert mt-4 rounded-sm px-4 py-3 text-sm" role="alert">
            {error}
          </p>
        )}

        <button
          type="button"
          disabled={submitting || files.length < MIN_FILES}
          onClick={handleSubmit}
          className="btn-primary mt-6 rounded-sm px-4 py-3"
        >
          {submitting ? "Uploading…" : "Compare videos"}
        </button>
      </div>
    </div>
  );
}
