"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { API_BASE } from "@/lib/apiBase";

const POLL_INTERVAL_MS = 2000;

export interface CaptureStatusData {
  capture_id: string;
  status: string;
  validation_passed: boolean;
  side: string;
  mrz_parsed: Record<string, unknown> | null;
  ocr_fields: Record<string, unknown> | null;
  mrz_check_digits_valid: boolean | null;
  fields?: Record<string, unknown> | null;
  raw_ocr?: Array<Record<string, unknown>> | null;
  ocr_confidence?: number | null;
}

export interface UseCaptureStatusReturn {
  data: CaptureStatusData | null;
  isPolling: boolean;
  error: string | null;
  startPolling: (captureId: string, token: string) => void;
  stopPolling: () => void;
}

export function useCaptureStatus(): UseCaptureStatusReturn {
  const [data, setData] = useState<CaptureStatusData | null>(null);
  const [isPolling, setIsPolling] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setIsPolling(false);
  }, []);

  const startPolling = useCallback(
    (captureId: string, token: string) => {
      stopPolling();
      setIsPolling(true);
      setError(null);

      const poll = async () => {
        try {
          const res = await fetch(
            `${API_BASE}/api/capture/status/${captureId}`,
            {
              headers: { Authorization: `Bearer ${token}` },
            }
          );
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const statusJson: CaptureStatusData = await res.json();

          // If completed, also fetch structured fields
          if (statusJson.status === "completed") {
            try {
              const fieldsRes = await fetch(
                `${API_BASE}/api/capture/${captureId}/fields`,
                {
                  headers: { Authorization: `Bearer ${token}` },
                }
              );
              if (fieldsRes.ok) {
                const fieldsJson = await fieldsRes.json();
                statusJson.fields = fieldsJson.fields ?? null;
                statusJson.raw_ocr = fieldsJson.raw_ocr ?? null;
                statusJson.ocr_confidence = fieldsJson.ocr_confidence ?? null;
              }
            } catch {
              // Non-fatal: fields endpoint may not be ready yet
            }
          }

          setData(statusJson);

          // Stop polling when processing is done
          if (statusJson.status === "completed" || statusJson.status === "failed") {
            stopPolling();
          }
        } catch (err) {
          setError(err instanceof Error ? err.message : "Polling failed");
        }
      };

      // Immediate first poll
      poll();
      intervalRef.current = setInterval(poll, POLL_INTERVAL_MS);
    },
    [stopPolling]
  );

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  return { data, isPolling, error, startPolling, stopPolling };
}
