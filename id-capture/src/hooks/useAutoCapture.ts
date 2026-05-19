"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const HOLD_DURATION_MS = 1500;

export interface UseAutoCaptureReturn {
  isHolding: boolean;
  holdProgress: number; // 0..1
  captured: boolean;
  reset: () => void;
}

/**
 * Triggers auto-capture when `readyToCapture` stays true for 1.5 seconds.
 * Calls `onCapture` when the timer completes.
 */
export function useAutoCapture(
  readyToCapture: boolean,
  onCapture: () => void
): UseAutoCaptureReturn {
  const [isHolding, setIsHolding] = useState(false);
  const [holdProgress, setHoldProgress] = useState(0);
  const [captured, setCaptured] = useState(false);

  const holdStartRef = useRef<number | null>(null);
  const rafRef = useRef<number | null>(null);

  const reset = useCallback(() => {
    setCaptured(false);
    setIsHolding(false);
    setHoldProgress(0);
    holdStartRef.current = null;
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
  }, []);

  useEffect(() => {
    if (captured) return;

    if (!readyToCapture) {
      // Reset hold
      holdStartRef.current = null;
      requestAnimationFrame(() => { setIsHolding(false); setHoldProgress(0); });
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      return;
    }

    // Start hold timer
    if (!holdStartRef.current) {
      holdStartRef.current = Date.now();
      setIsHolding(true);
    }

    const tick = () => {
      if (!holdStartRef.current) return;

      const elapsed = Date.now() - holdStartRef.current;
      const progress = Math.min(1, elapsed / HOLD_DURATION_MS);
      setHoldProgress(progress);

      if (progress >= 1) {
        setCaptured(true);
        setIsHolding(false);
        onCapture();
        return;
      }

      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [readyToCapture, captured, onCapture]);

  return { isHolding, holdProgress, captured, reset };
}
