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
        aria-label="Upload channel videos"
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
          "flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed px-6 py-10 text-center transition",
          disabled ? "opacity-60 cursor-not-allowed" : "cursor-pointer",
          dragActive
            ? "border-indigo-500 bg-indigo-50"
            : "border-slate-300 bg-slate-50 hover:border-indigo-400 hover:bg-indigo-50/40",
        ].join(" ")}
      >
        <p className="font-medium text-slate-900">
          Drop {minFiles}+ videos from the same creator, or click to browse
        </p>
        <p className="text-sm text-slate-500">
          {SUPPORTED_EXTENSIONS.join(", ")} · up to {MAX_FILE_SIZE_MB} MB each ·
          max {MAX_DURATION_SECONDS}s · multi-select
        </p>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept="video/mp4,video/quicktime,.mp4,.mov,.m4v"
          className="hidden"
          disabled={disabled}
          onChange={(e) => {
            if (e.target.files?.length) mergeFiles(e.target.files);
            e.target.value = "";
          }}
        />
      </div>

      {files.length > 0 && (
        <ul className="mt-4 divide-y divide-slate-100 rounded-xl ring-1 ring-slate-200 bg-white">
          {files.map((file) => (
            <li
              key={file.name}
              className="flex items-center justify-between gap-3 px-3 py-2 text-sm"
            >
              <span className="truncate text-slate-800">{file.name}</span>
              <span className="shrink-0 text-slate-400">
                {formatSize(file.size)}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
