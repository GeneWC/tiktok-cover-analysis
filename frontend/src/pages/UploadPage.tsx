import { useState } from "react";
import { useNavigate } from "react-router-dom";
import VideoUploader from "../components/VideoUploader";
import { analyze, ApiError } from "../api/client";
import { useAnalysis } from "../context/AnalysisContext";
import {
  readVideoDuration,
  validateDuration,
  validateFileBasics,
} from "../lib/validation";

export default function UploadPage() {
  const navigate = useNavigate();
  const { setUploadedVideo } = useAnalysis();

  const [file, setFile] = useState<File | null>(null);
  const [instrument, setInstrument] = useState("");
  const [hashtags, setHashtags] = useState("");
  const [showOptional, setShowOptional] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const handleSelect = (selected: File) => {
    setError(validateFileBasics(selected));
    setFile(selected);
  };

  const handleSubmit = async () => {
    if (!file || submitting) return;

    const basicError = validateFileBasics(file);
    if (basicError) {
      setError(basicError);
      return;
    }

    setSubmitting(true);
    setError(null);

    // Best-effort duration pre-check (non-fatal if the browser can't read it).
    try {
      const objectUrl = URL.createObjectURL(file);
      const duration = await readVideoDuration(objectUrl);
      URL.revokeObjectURL(objectUrl);
      const durationError = validateDuration(duration);
      if (durationError) {
        setError(durationError);
        setSubmitting(false);
        return;
      }
    } catch {
      // ignore — backend re-validates duration
    }

    try {
      const { analysis_id } = await analyze({ file, instrument, hashtags });
      setUploadedVideo(analysis_id, file);
      navigate(`/processing/${analysis_id}`);
    } catch (err) {
      const message =
        err instanceof ApiError
          ? err.message
          : "Something went wrong uploading your video.";
      setError(message);
      setSubmitting(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-slate-900">
          Analyze your cover video
        </h1>
        <p className="mt-2 text-slate-600">
          Upload an instrumental cover clip and get transparent, feature-based
          feedback on how it presents — plus exploratory similarity tiers.
        </p>
      </div>

      <div className="rounded-2xl bg-white shadow-sm ring-1 ring-slate-200 p-6">
        <VideoUploader
          selectedFile={file}
          onFileSelected={handleSelect}
          disabled={submitting}
        />

        <div className="mt-4">
          <button
            type="button"
            className="text-sm font-medium text-indigo-600 hover:text-indigo-700"
            onClick={() => setShowOptional((v) => !v)}
          >
            {showOptional ? "Hide" : "Add"} optional context (instrument,
            hashtags)
          </button>

          {showOptional && (
            <div className="mt-3 grid gap-3 sm:grid-cols-2">
              <label className="text-sm">
                <span className="block text-slate-600 mb-1">Instrument</span>
                <input
                  type="text"
                  value={instrument}
                  onChange={(e) => setInstrument(e.target.value)}
                  placeholder="e.g. violin"
                  className="w-full rounded-lg border border-slate-300 px-3 py-2 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 outline-none"
                />
              </label>
              <label className="text-sm">
                <span className="block text-slate-600 mb-1">Hashtags</span>
                <input
                  type="text"
                  value={hashtags}
                  onChange={(e) => setHashtags(e.target.value)}
                  placeholder="#violin #cover"
                  className="w-full rounded-lg border border-slate-300 px-3 py-2 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 outline-none"
                />
              </label>
            </div>
          )}
        </div>

        {error && (
          <div
            role="alert"
            className="mt-4 rounded-lg bg-red-50 border border-red-200 px-4 py-3 text-sm text-red-700"
          >
            {error}
          </div>
        )}

        <button
          type="button"
          onClick={handleSubmit}
          disabled={!file || submitting}
          className="mt-6 w-full rounded-xl bg-indigo-600 px-4 py-3 font-semibold text-white transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {submitting ? "Uploading…" : "Analyze video"}
        </button>

        <p className="mt-3 text-center text-xs text-slate-400">
          Hashtags and instrument are optional and not used by the current
          model.
        </p>
      </div>
    </div>
  );
}
