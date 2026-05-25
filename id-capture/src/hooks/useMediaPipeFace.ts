"use client";

import { useRef, useState, useCallback, useEffect } from "react";
import {
  FaceLandmarker,
  FilesetResolver,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";

interface UseMediaPipeFaceReturn {
  landmarks: NormalizedLandmark[][] | null;
  faceDetected: boolean;
  isReady: boolean;
  error: string | null;
  processFrame: (video: HTMLVideoElement, canvas: HTMLCanvasElement) => boolean;
}

let faceLandmarker: FaceLandmarker | null = null;
let initPromise: Promise<FaceLandmarker> | null = null;

async function ensureLandmarker(): Promise<FaceLandmarker> {
  if (faceLandmarker) return faceLandmarker;

  if (!initPromise) {
    initPromise = (async (): Promise<FaceLandmarker> => {
      console.log("[MediaPipe] Loading WASM files...");
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
      );
      console.log("[MediaPipe] WASM loaded, creating FaceLandmarker...");
      faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
          delegate: "GPU",
        },
        outputFaceBlendshapes: false,
        outputFacialTransformationMatrixes: false,
        runningMode: "IMAGE",
        numFaces: 1,
      });
      console.log("[MediaPipe] FaceLandmarker ready");
      return faceLandmarker;
    })();
  }

  return initPromise;
}

export function useMediaPipeFace(): UseMediaPipeFaceReturn {
  const [isReady, setIsReady] = useState(false);
  const [faceDetected, setFaceDetected] = useState(false);
  const [landmarks, setLandmarks] = useState<NormalizedLandmark[][] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const loadAttempted = useRef(false);
  const frameCount = useRef(0);
  const detectCount = useRef(0);
  const missCount = useRef(0);

  useEffect(() => {
    if (loadAttempted.current) return;
    loadAttempted.current = true;

    ensureLandmarker()
      .then(() => setIsReady(true))
      .catch((e) => {
        console.error("[MediaPipe] Load failed:", e);
        setError(e instanceof Error ? e.message : "MediaPipe failed to load");
      });
  }, []);

  const processFrame = useCallback(
    (video: HTMLVideoElement, canvas: HTMLCanvasElement): boolean => {
      if (!faceLandmarker) return false;

      frameCount.current++;
      if (canvas.width === 0 || canvas.height === 0) return false;

      try {
        const results = faceLandmarker.detect(canvas);
        if (results.faceLandmarks && results.faceLandmarks.length > 0) {
          detectCount.current++;
          setLandmarks(results.faceLandmarks);
          setFaceDetected(true);
          // Log every 60 frames
          if (detectCount.current % 60 === 1) {
            console.log(
              `[MediaPipe] Face detected (frame ${frameCount.current}, ` +
              `hits=${detectCount.current}, misses=${missCount.current})`
            );
          }
          return true;
        } else {
          missCount.current++;
          if (frameCount.current === 1 || frameCount.current % 120 === 0) {
            console.log(
              `[MediaPipe] No face (frame ${frameCount.current}, ` +
              `canvas=${canvas.width}x${canvas.height}, ` +
              `hits=${detectCount.current}, misses=${missCount.current})`
            );
          }
          setFaceDetected(false);
          return false;
        }
      } catch (e) {
        console.error("[MediaPipe] detect error:", e);
        setFaceDetected(false);
        return false;
      }
    },
    []
  );

  return { landmarks, faceDetected, isReady, error, processFrame };
}

export const MIN_LANDMARK_CONFIDENCE = 0.5;

export function validateLandmarks(lm: NormalizedLandmark[][]): boolean {
  if (!lm || lm.length === 0) return false;
  const points = lm[0];
  if (points.length !== 468) return false;
  return points.every(
    (p) => p.visibility == null || p.visibility >= MIN_LANDMARK_CONFIDENCE
  );
}

export function serializeLandmarks(lm: NormalizedLandmark[][]): number[][] {
  return lm[0].map((p) => [p.x, p.y, p.z]);
}
