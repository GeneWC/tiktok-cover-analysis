// Holds the just-uploaded video (File + object URL) so the Processing and Report
// pages can show a live preview without re-fetching bytes. On a hard reload this
// is empty; pages fall back to metadata/report from the API and skip the preview.

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

interface UploadedVideo {
  analysisId: string;
  file: File;
  objectUrl: string;
}

interface AnalysisContextValue {
  video: UploadedVideo | null;
  /** Store the uploaded file for a given analysis id, creating an object URL. */
  setUploadedVideo: (analysisId: string, file: File) => void;
  /** Release the object URL and clear state. */
  clear: () => void;
  /** Return the preview URL only if it matches the given analysis id. */
  previewUrlFor: (analysisId: string) => string | null;
}

const AnalysisContext = createContext<AnalysisContextValue | null>(null);

export function AnalysisProvider({ children }: { children: ReactNode }) {
  const [video, setVideo] = useState<UploadedVideo | null>(null);
  const urlRef = useRef<string | null>(null);

  const revoke = useCallback(() => {
    if (urlRef.current) {
      URL.revokeObjectURL(urlRef.current);
      urlRef.current = null;
    }
  }, []);

  const setUploadedVideo = useCallback(
    (analysisId: string, file: File) => {
      revoke();
      const objectUrl = URL.createObjectURL(file);
      urlRef.current = objectUrl;
      setVideo({ analysisId, file, objectUrl });
    },
    [revoke]
  );

  const clear = useCallback(() => {
    revoke();
    setVideo(null);
  }, [revoke]);

  const previewUrlFor = useCallback(
    (analysisId: string) =>
      video && video.analysisId === analysisId ? video.objectUrl : null,
    [video]
  );

  const value = useMemo(
    () => ({ video, setUploadedVideo, clear, previewUrlFor }),
    [video, setUploadedVideo, clear, previewUrlFor]
  );

  return (
    <AnalysisContext.Provider value={value}>
      {children}
    </AnalysisContext.Provider>
  );
}

export function useAnalysis(): AnalysisContextValue {
  const ctx = useContext(AnalysisContext);
  if (!ctx) {
    throw new Error("useAnalysis must be used within an AnalysisProvider");
  }
  return ctx;
}
