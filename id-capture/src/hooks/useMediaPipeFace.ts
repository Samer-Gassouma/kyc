"use client";

import { useRef, useState, useEffect, useCallback } from "react";
import {
  FaceLandmarker,
  FilesetResolver,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";

let _landmarker: FaceLandmarker | null = null;
let _loading: Promise<FaceLandmarker> | null = null;

async function loadLandmarker(): Promise<FaceLandmarker> {
  if (_landmarker) return _landmarker;
  if (!_loading) {
    _loading = (async () => {
      console.log("[MP] Loading WASM...");
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
      );
      console.log("[MP] Creating FaceLandmarker (IMAGE mode)...");
      _landmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
          delegate: "GPU",
        },
        runningMode: "IMAGE",
        numFaces: 1,
      });
      console.log("[MP] FaceLandmarker ready");
      return _landmarker;
    })();
  }
  return _loading;
}

export function useMediaPipeFace() {
  const [ready, setReady] = useState(false);
  const [landmarks, setLandmarks] = useState<NormalizedLandmark[][] | null>(null);
  const [detected, setDetected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const started = useRef(false);
  const frameIdx = useRef(0);

  // Load the model once
  useEffect(() => {
    if (started.current) return;
    started.current = true;
    loadLandmarker()
      .then(() => setReady(true))
      .catch((e) => {
        console.error("[MP] Load error:", e);
        setError(String(e));
      });
  }, []);

  const detect = useCallback(
    (video: HTMLVideoElement, outputCanvas: HTMLCanvasElement): boolean => {
      const lm = _landmarker;
      if (!lm) return false;

      const vw = video.videoWidth;
      const vh = video.videoHeight;
      if (!vw || !vh) return false;

      // Draw the current video frame to the canvas
      outputCanvas.width = vw;
      outputCanvas.height = vh;
      const ctx = outputCanvas.getContext("2d");
      if (!ctx) return false;
      ctx.drawImage(video, 0, 0, vw, vh);

      // Run detection on the canvas
      try {
        const res = lm.detect(outputCanvas);
        frameIdx.current++;
        if (res.faceLandmarks?.length) {
          setLandmarks(res.faceLandmarks);
          setDetected(true);
          if (frameIdx.current === 1 || frameIdx.current % 60 === 0) {
            console.log(`[MP] frame=${frameIdx.current} FACE DETECTED pts=${res.faceLandmarks[0].length}`);
          }
          return true;
        } else {
          setDetected(false);
          if (frameIdx.current === 1 || frameIdx.current % 60 === 0) {
            console.log(`[MP] frame=${frameIdx.current} no face (canvas=${vw}x${vh})`);
          }
          return false;
        }
      } catch (e) {
        console.error("[MP] detect crashed:", e);
        return false;
      }
    },
    []
  );

  return {
    landmarks,
    faceDetected: detected,
    isReady: ready,
    error,
    detect,
  };
}

export type { NormalizedLandmark };
export { DrawingUtils } from "@mediapipe/tasks-vision";
