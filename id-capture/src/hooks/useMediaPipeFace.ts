"use client";

import { useRef, useState, useCallback, useEffect } from "react";
import { FaceLandmarker, FilesetResolver, type NormalizedLandmark } from "@mediapipe/tasks-vision";

interface UseMediaPipeFaceReturn {
  landmarks: NormalizedLandmark[][] | null;
  faceDetected: boolean;
  isReady: boolean;
  error: string | null;
  processFrame: (video: HTMLVideoElement) => void;
}

let faceLandmarker: FaceLandmarker | null = null;
let initPromise: Promise<void> | null = null;

async function ensureLandmarker(): Promise<FaceLandmarker> {
  if (faceLandmarker) return faceLandmarker;

  if (!initPromise) {
    initPromise = (async () => {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
      );
      faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
          delegate: "GPU",
        },
        outputFaceBlendshapes: false,
        runningMode: "VIDEO",
        numFaces: 1,
      });
    })();
  }

  await initPromise;
  return faceLandmarker!;
}

export function useMediaPipeFace(): UseMediaPipeFaceReturn {
  const [isReady, setIsReady] = useState(false);
  const [faceDetected, setFaceDetected] = useState(false);
  const [landmarks, setLandmarks] = useState<NormalizedLandmark[][] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const lastTimestampRef = useRef(0);

  useEffect(() => {
    ensureLandmarker()
      .then(() => {
        setIsReady(true);
      })
      .catch((e) => {
        setError(e instanceof Error ? e.message : "MediaPipe failed to load");
      });

    return () => {
      // Keep the singleton loaded across unmounts
    };
  }, []);

  const processFrame = useCallback((video: HTMLVideoElement) => {
    if (!faceLandmarker) return;

    const now = performance.now();
    const timestamp = Math.floor(now);
    // Skip duplicate timestamps
    if (timestamp <= lastTimestampRef.current) {
      lastTimestampRef.current = timestamp + 1;
    } else {
      lastTimestampRef.current = timestamp;
    }

    try {
      const results = faceLandmarker.detectForVideo(video, lastTimestampRef.current);
      if (results.faceLandmarks && results.faceLandmarks.length > 0) {
        setLandmarks(results.faceLandmarks);
        setFaceDetected(true);
      } else {
        setFaceDetected(false);
      }
    } catch {
      setFaceDetected(false);
    }
  }, []);

  return { landmarks, faceDetected, isReady, error, processFrame };
}

/** Minimum confidence a landmark must have to be considered valid. */
export const MIN_LANDMARK_CONFIDENCE = 0.5;

/** Validate that landmark array has all 468 points with sufficient confidence. */
export function validateLandmarks(lm: NormalizedLandmark[][]): boolean {
  if (!lm || lm.length === 0) return false;
  const points = lm[0];
  if (points.length !== 468) return false;
  return points.every((p) => p.visibility == null || p.visibility >= MIN_LANDMARK_CONFIDENCE);
}

/** Serialize 468 landmarks to a plain array for JSON storage. */
export function serializeLandmarks(lm: NormalizedLandmark[][]): number[][] {
  return lm[0].map((p) => [p.x, p.y, p.z]);
}
