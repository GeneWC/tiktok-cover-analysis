// Polls GET /status on an interval until the job reaches a terminal state
// (complete/failed) or the network errors out. Cleans up on unmount.

import { useEffect, useRef, useState } from "react";
import { getStatus } from "../api/client";
import type { StatusResponse } from "../types";

const POLL_INTERVAL_MS = 1500;

interface PollingState {
  data: StatusResponse | null;
  error: string | null;
}

export function useStatusPolling(analysisId: string | undefined): PollingState {
  const [data, setData] = useState<StatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!analysisId) return;

    let cancelled = false;

    const poll = async () => {
      try {
        const status = await getStatus(analysisId);
        if (cancelled) return;
        setData(status);
        setError(null);
        if (status.status === "processing") {
          timerRef.current = setTimeout(poll, POLL_INTERVAL_MS);
        }
      } catch (err) {
        if (cancelled) return;
        setError(
          err instanceof Error ? err.message : "Failed to fetch status."
        );
        // keep trying — the backend may still be warming up
        timerRef.current = setTimeout(poll, POLL_INTERVAL_MS);
      }
    };

    poll();

    return () => {
      cancelled = true;
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [analysisId]);

  return { data, error };
}
