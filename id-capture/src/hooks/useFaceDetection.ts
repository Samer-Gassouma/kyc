"use client";

import { useEffect, useState, useRef } from "react";

const MODEL_URL = "/models/face-api";

let _faceapi: any = null;
let _loaded = false;
let _loading: Promise<void> | null = null;

async function getFaceApi(): Promise<any> {
  if (_faceapi) return _faceapi;
  _faceapi = await import("@vladmandic/face-api");
  return _faceapi;
}

async function ensureModels(): Promise<void> {
  if (_loaded) return;
  if (!_loading) {
    _loading = (async () => {
      const fa = await getFaceApi();
      console.log("[face-api] Loading tiny face detector only...");
      await fa.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
      _loaded = true;
      console.log("[face-api] Ready (bbox-only)");
    })();
  }
  return _loading;
}

export interface FaceBox {
  x: number; y: number; width: number; height: number;
}

export type PoseState = "center" | "left" | "right" | "up" | "none";

function detectPose(box: FaceBox, frameW: number, frameH: number): PoseState {
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  const nx = (cx - frameW / 2) / (frameW / 2);
  const ny = (cy - frameH / 2) / (frameH / 2);

  if (nx > 0.22) return "left";
  if (nx < -0.22) return "right";
  if (ny < -0.18) return "up";
  if (Math.abs(nx) < 0.12 && Math.abs(ny) < 0.12) return "center";
  return "none";
}

export function useFaceDetection() {
  const [isReady, setIsReady] = useState(false);
  const started = useRef(false);

  useEffect(() => {
    if (started.current) return;
    started.current = true;
    ensureModels().then(() => setIsReady(true)).catch(e => console.error("[face-api] fail:", e));
  }, []);

  async function detect(video: HTMLVideoElement): Promise<{ box: FaceBox; pose: PoseState } | null> {
    const fa = _faceapi;
    if (!fa || !_loaded) return null;
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh) return null;

    try {
      const result = await fa.detectSingleFace(
        video,
        new fa.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 })
      );
      // NO .withFaceLandmarks() — bounding box only, much faster

      if (!result) return null;

      const raw = result.box; // { x, y, width, height }
      const box: FaceBox = { x: raw.x, y: raw.y, width: raw.width, height: raw.height };
      const pose = detectPose(box, vw, vh);
      (window as any).__pose = { pose, box };

      return { box, pose };
    } catch {
      return null;
    }
  }

  return { isReady, detect };
}
