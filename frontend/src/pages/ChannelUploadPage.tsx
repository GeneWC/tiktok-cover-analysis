import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
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
      setError(`Add at least ${MIN_FILES} videos from the same creator.`);
      return;
    }
    if (anyViewFilled && !allViewsFilled) {
      setError(
        "Enter views for every video, or leave all blank for presentation-only ranks."
      );
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
          : "Something went wrong uploading your channel videos."
      );
      setSubmitting(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto">
      <div className="text-center mb-8">
        <p className="text-sm font-medium text-indigo-600 mb-2">
          <Link to="/" className="hover:underline">
            Single video
          </Link>
          <span className="mx-2 text-slate-300">/</span>
          Channel diagnostics
        </p>
        <h1 className="text-3xl font-bold text-slate-900">
          Diagnose your channel batch
        </h1>
        <p className="mt-2 text-slate-600">
          Upload {MIN_FILES}+ of your own covers and optional view counts. We
          show what differs in <em>your</em> stronger uploads — not a cross-creator
          virality forecast.
        </p>
      </div>

      <div className="rounded-2xl bg-white shadow-sm ring-1 ring-slate-200 p-6">
        <MultiVideoUploader
          files={files}
          onFilesSelected={handleFiles}
          disabled={submitting}
          minFiles={MIN_FILES}
        />

        {files.length > 0 && (
          <div className="mt-6">
            <h2 className="text-sm font-semibold text-slate-800">
              Optional views (for hit/miss deltas)
            </h2>
            <p className="mt-1 text-xs text-slate-500">
              Provide views for all videos to compute within-batch top-quartile
              labels. Leave blank for presentation ranks only.
            </p>
            <div className="mt-3 space-y-2">
              {files.map((file) => (
                <label
                  key={file.name}
                  className="flex items-center gap-3 text-sm"
                >
                  <span className="flex-1 truncate text-slate-700">
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
                    className="w-28 rounded-lg border border-slate-200 px-2 py-1.5 text-slate-900"
                  />
                </label>
              ))}
            </div>
          </div>
        )}

        {error && (
          <p className="mt-4 text-sm text-rose-600" role="alert">
            {error}
          </p>
        )}

        <button
          type="button"
          disabled={submitting || files.length < MIN_FILES}
          onClick={handleSubmit}
          className="mt-6 w-full rounded-xl bg-indigo-600 px-4 py-3 text-sm font-semibold text-white hover:bg-indigo-700 disabled:opacity-50"
        >
          {submitting ? "Uploading…" : "Run channel diagnostics"}
        </button>
      </div>
    </div>
  );
}
