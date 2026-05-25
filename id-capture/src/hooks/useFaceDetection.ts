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
      const faceapi = await getFaceApi();
      console.log("[face-api] Loading models...");
      await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
      await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
      await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
      _loaded = true;
      console.log("[face-api] Models ready");
    })();
  }
  return _loading;
}

function computePose(landmarks: any, boxW: number, boxH: number) {
  const nose = landmarks.getNose();
  const leftEye = landmarks.getLeftEye();
  const rightEye = landmarks.getRightEye();
  const jaw = landmarks.getJawOutline();

  const noseTip = nose[3];
  const eyeCenterX = (leftEye[0].x + rightEye[3].x) / 2;
  const eyeCenterY = (leftEye[0].y + rightEye[3].y) / 2;
  const chinTip = jaw[8];
  const leftCheek = jaw[0];
  const rightCheek = jaw[16];

  const faceW = Math.max(rightCheek.x - leftCheek.x, 1);
  const noseOffX = (noseTip.x - eyeCenterX) / faceW;

  const faceH = Math.max(chinTip.y - eyeCenterY, 1);
  const noseOffY = (noseTip.y - eyeCenterY) / faceH;

  return {
    yaw: -noseOffX * 100,
    pitch: noseOffY * 100,
    roll: 0,
  };
}

export interface FaceResult {
  box: { x: number; y: number; width: number; height: number };
  landmarks: any;
  pose: { yaw: number; pitch: number; roll: number };
}

export function useFaceDetection() {
  const [isReady, setIsReady] = useState(false);
  const started = useRef(false);

  useEffect(() => {
    if (started.current) return;
    started.current = true;
    ensureModels().then(() => setIsReady(true)).catch((e) => {
      console.error("[face-api] Load failed:", e);
    });
  }, []);

  async function detect(video: HTMLVideoElement): Promise<FaceResult | null> {
    if (!_loaded) return null;
    const faceapi = _faceapi;
    if (!faceapi) return null;

    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh) return null;

    const result = await faceapi.detectSingleFace(
      video,
      new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 })
    ).withFaceLandmarks();

    if (!result) return null;

    const box = result.detection.box;
    const pose = computePose(result.landmarks, box.width, box.height);
    (window as any).__headPose = pose;

    return { box, landmarks: result.landmarks, pose };
  }

  return { isReady, detect };
}
