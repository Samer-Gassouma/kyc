"use client";

import { useRef, useState, useEffect, useCallback } from "react";
import {
  FaceLandmarker,
  FilesetResolver,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";

export interface HeadPose {
  yaw: number;    // degrees, negative = turning LEFT, positive = turning RIGHT
  pitch: number;  // degrees, negative = looking UP, positive = looking DOWN
  roll: number;
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

/** Decompose 4x4 column-major rotation matrix → Euler angles (degrees).
 *  Front camera is mirrored, so we negate yaw and pitch so the signs match
 *  the user's physical movement:
 *    yaw < 0  = turning head LEFT
 *    yaw > 0  = turning head RIGHT
 *    pitch < 0 = looking UP
 *    pitch > 0 = looking DOWN
 */
function matrixToEuler(data: number[]): HeadPose {
  const r00 = data[0], r01 = data[4], r02 = data[8];
  const r10 = data[1], r11 = data[5], r12 = data[9];
  const r20 = data[2], r21 = data[6], r22 = data[10];

  const sy = Math.sqrt(r00 * r00 + r10 * r10);

  let rawYaw: number, rawPitch: number, roll: number;
  if (sy > 1e-6) {
    rawYaw = Math.atan2(-r10, r00);
    rawPitch = Math.atan2(r20, r22);
    roll = Math.asin(Math.max(-1, Math.min(1, -r21)));
  } else {
    rawYaw = Math.atan2(r02, r22);
    rawPitch = 0;
    roll = Math.asin(Math.max(-1, Math.min(1, -r21)));
  }

  return {
    // Negate for front camera mirror:
    //   screen-left = user's right → yaw < 0 is "left" from user's view
    yaw: -(rawYaw * 180) / Math.PI,
    //   screen-up = user's down → pitch < 0 is "up" from user's view
    pitch: -(rawPitch * 180) / Math.PI,
    roll: (roll * 180) / Math.PI,
  };
}

/** Connection = [startIdx, endIdx] pairs for the face mesh wireframe. */
export type Tessellation = [number, number][];

let _tessellation: Tessellation | null = null;

function getTessellation(): Tessellation {
  if (_tessellation) return _tessellation;
  if (_landmarker && (FaceLandmarker as any).FACE_LANDMARKS_TESSELATION) {
    const raw = (FaceLandmarker as any).FACE_LANDMARKS_TESSELATION as { start: number; end: number }[];
    _tessellation = raw.map((c) => [c.start, c.end] as [number, number]);
  }
  // Fallback: key contour edges (eyes, nose, lips, oval) if static accessor unavailable
  if (!_tessellation || _tessellation.length === 0) {
    _tessellation = _fallbackTessellation();
  }
  return _tessellation;
}

function _fallbackTessellation(): Tessellation {
  const c: Tessellation = [];
  const ring = (idx: number[]) => { for (let i = 0; i < idx.length; i++) c.push([idx[i], idx[(i + 1) % idx.length]]); };
  ring([33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7]);
  ring([362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382]);
  ring([46,53,52,65,55,70,63,105,66,107]);
  ring([276,283,282,295,285,300,293,334,296,336]);
  ring([61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185]);
  ring([78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95]);
  ring([10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]);
  // Vertical connectors
  for (const [a,b] of [[10,151],[151,9],[9,8],[8,168],[168,6],[6,197],[197,195],[195,5],[5,4],[4,1],[1,19],[19,94],[94,2],[2,200],[200,199],[199,175],[175,152]]) c.push([a,b]);
  // Eye-brow connectors
  for (const [a,b] of [[33,46],[133,53],[173,52],[157,65],[158,55],[159,70],[160,63],[161,105],[246,107],[362,276],[263,283],[249,282],[390,295],[373,285],[374,300],[380,293],[381,334],[382,296],[398,336]]) c.push([a,b]);
  // Nose-to-eye connectors
  for (const [a,b] of [[6,33],[6,362],[168,133],[168,263],[197,157],[197,390],[195,158],[195,373]]) c.push([a,b]);
  // Cheek grid
  for (let i = 0; i < 16; i++) {
    const top = [234,127,162,21,54,103,67,109,10,338,297,332,284,251,389,356,454][i];
    const bot = [93,132,58,172,136,150,149,176,148,152,377,400,378,379,365,397,288][i] || 152;
    if (top && bot) c.push([top, bot]);
  }
  return c;
}

export function getFaceTessellation(): Tessellation {
  return getTessellation();
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

          if (res.facialTransformationMatrixes?.length) {
            const pose = matrixToEuler(res.facialTransformationMatrixes[0].data);
            setHeadPose(pose);
            // Live debug — check in browser console as window.__headPose
            (window as any).__headPose = pose;
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
