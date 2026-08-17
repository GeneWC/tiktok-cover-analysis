import { useRef, useState, type DragEvent } from "react";
import {
  MAX_DURATION_SECONDS,
  MAX_FILE_SIZE_MB,
  SUPPORTED_EXTENSIONS,
} from "../lib/validation";

interface VideoUploaderProps {
  selectedFile: File | null;
  onFileSelected: (file: File) => void;
  disabled?: boolean;
}

function formatSize(bytes: number): string {
  const mb = bytes / (1024 * 1024);
  return mb >= 1 ? `${mb.toFixed(1)} MB` : `${(bytes / 1024).toFixed(0)} KB`;
}

export default function VideoUploader({
  selectedFile,
  onFileSelected,
  disabled = false,
}: VideoUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);

  const openPicker = () => inputRef.current?.click();

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragActive(false);
    if (disabled) return;
    const file = e.dataTransfer.files?.[0];
    if (file) onFileSelected(file);
  };

  return (
    <div>
      <div
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-disabled={disabled}
        aria-describedby="upload-hints"
        aria-label={
          selectedFile
            ? `Selected ${selectedFile.name}. Click to choose a different file`
            : "Upload a short cover video"
        }
        onClick={openPicker}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") openPicker();
        }}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
        className={[
          "dropzone flex flex-col items-center justify-center gap-3 rounded-sm px-6 py-12 text-center",
          disabled ? "cursor-not-allowed opacity-60" : "cursor-pointer",
          dragActive ? "dropzone-active" : "",
        ].join(" ")}
      >
        <svg
          className="h-10 w-10 text-gold"
          fill="none"
          viewBox="0 0 24 24"
          strokeWidth={1.5}
          stroke="currentColor"
          aria-hidden="true"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5"
          />
        </svg>

        {selectedFile ? (
          <div>
            <p className="break-all font-medium text-ivory">
              {selectedFile.name}
            </p>
            <p id="upload-hints" className="muted text-sm">
              {formatSize(selectedFile.size)} · click to choose a different file
            </p>
          </div>
        ) : (
          <div>
            <p className="font-medium text-ivory">
              Drop your cover here, or click to browse
            </p>
            <p id="upload-hints" className="muted text-sm">
              {SUPPORTED_EXTENSIONS.join(", ")} · up to {MAX_FILE_SIZE_MB} MB ·
              1–{MAX_DURATION_SECONDS}s. Compared with other videos by the same
              kind of creator.
            </p>
          </div>
        )}

        <input
          ref={inputRef}
          type="file"
          accept="video/mp4,video/quicktime,.mp4,.mov,.m4v"
          className="hidden"
          tabIndex={-1}
          aria-label="Upload a short cover video"
          disabled={disabled}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onFileSelected(file);
            e.target.value = "";
          }}
        />
      </div>
    </div>
  );
}
