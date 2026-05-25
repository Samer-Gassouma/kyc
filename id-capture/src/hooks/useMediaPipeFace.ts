"use client";

import { useRef, useState, useEffect, useCallback } from "react";
import {
  FaceLandmarker,
  FilesetResolver,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";

export interface HeadPose {
  yaw: number;    // degrees, positive = turning right, negative = turning left
  pitch: number;  // degrees, positive = looking up, negative = looking down
  roll: number;   // degrees, positive = tilting right
}

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
      console.log("[MP] Creating FaceLandmarker with pose matrix...");
      _landmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
          delegate: "GPU",
        },
        runningMode: "IMAGE",
        numFaces: 1,
        outputFacialTransformationMatrixes: true,
      });
      console.log("[MP] FaceLandmarker ready");
      return _landmarker;
    })();
  }
  return _loading;
}

/** Extract yaw/pitch/roll (degrees) from a 4x4 column-major transformation matrix */
function matrixToEuler(data: number[]): HeadPose {
  // MediaPipe Matrix has .data: number[] in column-major order
  // Upper-left 3x3 is the rotation. Column-major in data[]:
  // data[0] data[4] data[8]  |  data[1] data[5] data[9]  |  data[2] data[6] data[10]
  const r00 = data[0], r01 = data[4], r02 = data[8];
  const r10 = data[1], r11 = data[5], r12 = data[9];
  const r20 = data[2], r21 = data[6], r22 = data[10];

  // Decompose rotation matrix → Euler angles (radians)
  // Using ZYX convention (yaw=Y, pitch=X, roll=Z)
  const sy = Math.sqrt(r00 * r00 + r10 * r10);
  const singular = sy < 1e-6;

  let yaw: number, pitch: number, roll: number;
  if (!singular) {
    yaw = Math.atan2(-r10, r00);     // Y axis
    pitch = Math.atan2(r20, r22);     // X axis
    roll = Math.asin(-r21);           // Clamp for numerical stability...
    roll = Math.asin(Math.max(-1, Math.min(1, -r21)));  // Z axis
  } else {
    yaw = Math.atan2(r02, r22);
    pitch = 0;
    roll = Math.asin(Math.max(-1, Math.min(1, -r21)));
  }

  return {
    yaw: (yaw * 180) / Math.PI,
    pitch: (pitch * 180) / Math.PI,
    roll: (roll * 180) / Math.PI,
  };
}

export function useMediaPipeFace() {
  const [ready, setReady] = useState(false);
  const [landmarks, setLandmarks] = useState<NormalizedLandmark[][] | null>(null);
  const [detected, setDetected] = useState(false);
  const [headPose, setHeadPose] = useState<HeadPose | null>(null);
  const [error, setError] = useState<string | null>(null);
  const started = useRef(false);
  const frameIdx = useRef(0);

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

      outputCanvas.width = vw;
      outputCanvas.height = vh;
      const ctx = outputCanvas.getContext("2d");
      if (!ctx) return false;
      ctx.drawImage(video, 0, 0, vw, vh);

      try {
        const res = lm.detect(outputCanvas);
        frameIdx.current++;

        if (res.faceLandmarks?.length) {
          setLandmarks(res.faceLandmarks);
          setDetected(true);

          // Extract head pose from transformation matrix
          if (res.facialTransformationMatrixes?.length) {
            const pose = matrixToEuler(res.facialTransformationMatrixes[0].data);
            setHeadPose(pose);
          }

          if (frameIdx.current === 1 || frameIdx.current % 60 === 0) {
            console.log(`[MP] frame=${frameIdx.current} FACE pts=${res.faceLandmarks[0].length}`);
          }
          return true;
        } else {
          setDetected(false);
          setHeadPose(null);
          if (frameIdx.current === 1 || frameIdx.current % 120 === 0) {
            console.log(`[MP] frame=${frameIdx.current} no face (${vw}x${vh})`);
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

  return { landmarks, faceDetected: detected, headPose, isReady: ready, error, detect };
}

export type { NormalizedLandmark };
export { DrawingUtils } from "@mediapipe/tasks-vision";
