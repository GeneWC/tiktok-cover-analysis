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
      const { analysis_id } = await analyze({ file });
      setUploadedVideo(analysis_id, file);
      navigate(`/processing/${analysis_id}`);
    } catch (err) {
      const message =
        err instanceof ApiError
          ? err.message
          : "Could not upload the video.";
      setError(message);
      setSubmitting(false);
    }
  };

  return (
    <div className="mx-auto max-w-2xl">
      <div className="mb-10 text-center">
        <h1 className="font-display text-4xl font-medium text-ivory">
          Analyze Your Cover
        </h1>
        <p className="muted mx-auto mt-3 max-w-md">
          Upload a cover. Get a plain read on filming and sound.
        </p>
      </div>

      <div className="panel rounded-sm p-6">
        <VideoUploader
          selectedFile={file}
          onFileSelected={handleSelect}
          disabled={submitting}
        />

        {error && (
          <div role="alert" className="alert mt-4 rounded-sm px-4 py-3 text-sm">
            {error}
          </div>
        )}

        <button
          type="button"
          onClick={handleSubmit}
          disabled={!file || submitting}
          className="btn-primary mt-6 rounded-sm px-4 py-3"
        >
          {submitting ? "Uploading…" : "Analyze video"}
        </button>
      </div>
    </div>
  );
}
