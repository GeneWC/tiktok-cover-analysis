import { useRef, useState, type DragEvent } from "react";
import {
  MAX_DURATION_SECONDS,
  MAX_FILE_SIZE_MB,
  SUPPORTED_EXTENSIONS,
} from "../lib/validation";

interface MultiVideoUploaderProps {
  files: File[];
  onFilesSelected: (files: File[]) => void;
  disabled?: boolean;
  minFiles?: number;
}

function formatSize(bytes: number): string {
  const mb = bytes / (1024 * 1024);
  return mb >= 1 ? `${mb.toFixed(1)} MB` : `${(bytes / 1024).toFixed(0)} KB`;
}

export default function MultiVideoUploader({
  files,
  onFilesSelected,
  disabled = false,
  minFiles = 5,
}: MultiVideoUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);

  const mergeFiles = (incoming: FileList | File[]) => {
    const list = Array.from(incoming);
    const byName = new Map<string, File>();
    for (const f of files) byName.set(f.name, f);
    for (const f of list) byName.set(f.name, f);
    onFilesSelected(Array.from(byName.values()));
  };

  return (
    <div>
      <div
        role="button"
        tabIndex={0}
        aria-label={
          files.length
            ? `${files.length} videos selected. Click to add more`
            : "Upload channel videos"
        }
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
        }}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={(e: DragEvent<HTMLDivElement>) => {
          e.preventDefault();
          setDragActive(false);
          if (!disabled && e.dataTransfer.files?.length) {
            mergeFiles(e.dataTransfer.files);
          }
        }}
        className={[
          "dropzone flex flex-col items-center justify-center gap-3 rounded-sm px-6 py-10 text-center",
          disabled ? "cursor-not-allowed opacity-60" : "cursor-pointer",
          dragActive ? "dropzone-active" : "",
        ].join(" ")}
      >
        <p className="font-medium text-ivory">
          Drop {minFiles}+ videos, or click to browse
        </p>
        <p className="muted text-sm">
          {SUPPORTED_EXTENSIONS.join(", ")} · up to {MAX_FILE_SIZE_MB} MB each ·
          max {MAX_DURATION_SECONDS}s
        </p>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept="video/mp4,video/quicktime,.mp4,.mov,.m4v"
          className="hidden"
          tabIndex={-1}
          disabled={disabled}
          onChange={(e) => {
            if (e.target.files?.length) mergeFiles(e.target.files);
            e.target.value = "";
          }}
        />
      </div>

      {files.length > 0 && (
        <ul className="panel-quiet mt-4 divide-y divide-gold/15 rounded-sm">
          {files.map((file) => (
            <li
              key={file.name}
              className="flex items-center justify-between gap-3 px-3 py-2 text-sm"
            >
              <span className="truncate text-ivory">{file.name}</span>
              <span className="flex shrink-0 items-center gap-2">
                <span className="muted">{formatSize(file.size)}</span>
                <button
                  type="button"
                  disabled={disabled}
                  aria-label={`Remove ${file.name}`}
                  onClick={() =>
                    onFilesSelected(files.filter((f) => f.name !== file.name))
                  }
                  className="inline-flex min-h-11 min-w-11 items-center justify-center text-gold hover:text-ivory"
                >
                  Remove
                </button>
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
