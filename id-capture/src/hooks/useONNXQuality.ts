"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { QualityResult, runQualityCheck } from "@/lib/onnxLoader";
import { grabFrameResized } from "@/lib/frameEncoder";

export interface UseONNXQualityReturn {
  quality: QualityResult | null;
  isModelLoaded: boolean;
  runCheck: (video: HTMLVideoElement) => Promise<QualityResult>;
}

export function useONNXQuality(): UseONNXQualityReturn {
  const [quality, setQuality] = useState<QualityResult | null>(null);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    // Create an offscreen canvas for frame resizing
    canvasRef.current = document.createElement("canvas");

    // Pre-load the model
    import("@/lib/onnxLoader").then(({ loadQualityModel }) => {
      loadQualityModel()
        .then(() => setIsModelLoaded(true))
        .catch(() => {
          // Heuristic fallback is fine
          setIsModelLoaded(true);
        });
    });
  }, []);

  const runCheck = useCallback(
    async (video: HTMLVideoElement): Promise<QualityResult> => {
      if (!canvasRef.current) {
        canvasRef.current = document.createElement("canvas");
      }

      const imageData = grabFrameResized(video, canvasRef.current, 224, 224);
      const result = await runQualityCheck(imageData);
      setQuality(result);
      return result;
    },
    []
  );

  return { quality, isModelLoaded, runCheck };
}
