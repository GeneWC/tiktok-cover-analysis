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
        tabIndex={0}
        aria-label="Upload a video"
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
          "flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed px-6 py-12 text-center transition",
          disabled ? "opacity-60 cursor-not-allowed" : "cursor-pointer",
          dragActive
            ? "border-indigo-500 bg-indigo-50"
            : "border-slate-300 bg-slate-50 hover:border-indigo-400 hover:bg-indigo-50/40",
        ].join(" ")}
      >
        <svg
          className="h-10 w-10 text-indigo-500"
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
            <p className="font-medium text-slate-900 break-all">
              {selectedFile.name}
            </p>
            <p className="text-sm text-slate-500">
              {formatSize(selectedFile.size)} · click to choose a different file
            </p>
          </div>
        ) : (
          <div>
            <p className="font-medium text-slate-900">
              Drop your cover video here, or click to browse
            </p>
            <p className="text-sm text-slate-500">
              {SUPPORTED_EXTENSIONS.join(", ")} · up to {MAX_FILE_SIZE_MB} MB ·
              max {MAX_DURATION_SECONDS}s
            </p>
          </div>
        )}

        <input
          ref={inputRef}
          type="file"
          accept="video/mp4,video/quicktime,.mp4,.mov,.m4v"
          className="hidden"
          disabled={disabled}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onFileSelected(file);
            // reset so selecting the same file again re-triggers onChange
            e.target.value = "";
          }}
        />
      </div>
    </div>
  );
}
