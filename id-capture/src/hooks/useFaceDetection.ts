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
      console.log("[face-api] Loading models...");
      await fa.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
      await fa.nets.faceLandmark68TinyNet.loadFromUri(MODEL_URL);
      _loaded = true;
      console.log("[face-api] Ready (bbox + 68-tiny landmarks)");
    })();
  }
  return _loading;
}

export interface FaceBox {
  x: number; y: number; width: number; height: number;
}

export type PoseState = "center" | "left" | "right" | "up" | "none";

/** Detect head pose from nose tip position RELATIVE to the face bounding box.
 *
 *  face-api processes the RAW (un-mirrored) camera frame.
 *  In the raw image: person's LEFT  = right side of image
 *                    person's RIGHT = left side of image
 *
 *  When turning head LEFT:  nose appears on the RIGHT side of the face box
 *  When turning head RIGHT: nose appears on the LEFT side of the face box
 *  When looking UP:         nose appears HIGHER in the face box
 */
function detectPose(box: FaceBox, noseTip: { x: number; y: number }): PoseState {
  const boxCx = box.x + box.width / 2;
  const boxCy = box.y + box.height / 2;
  const bw = Math.max(box.width, 1);
  const bh = Math.max(box.height, 1);

  const noseOffX = (noseTip.x - boxCx) / bw;
  const noseOffY = (noseTip.y - boxCy) / bh;

  // Nose on right side of face box (in raw image) = turned LEFT
  if (noseOffX > 0.05) return "left";
  // Nose on left side of face box = turned RIGHT
  if (noseOffX < -0.045) return "right";
  // Nose above box center = looking UP
  if (noseOffY < -0.04) return "up";
  // Nose roughly centered
  if (Math.abs(noseOffX) < 0.035 && Math.abs(noseOffY) < 0.035) return "center";
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

  async function detect(video: HTMLVideoElement): Promise<{
    box: FaceBox; pose: PoseState; noseTip: { x: number; y: number };
  } | null> {
    const fa = _faceapi;
    if (!fa || !_loaded) return null;
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh) return null;

    try {
      const result = await fa.detectSingleFace(
        video,
        new fa.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 })
      ).withFaceLandmarks();  // uses faceLandmark68TinyNet

      if (!result) {
        (window as any).__pose = { pose: "none", reason: "no face detected" };
        return null;
      }

      const raw = result.box;
      const box: FaceBox = { x: raw.x, y: raw.y, width: raw.width, height: raw.height };
      const nose = result.landmarks.getNose();
      const noseTip = { x: nose[3].x, y: nose[3].y };
      const boxCx = box.x + box.width / 2;
      const boxCy = box.y + box.height / 2;
      const noseOffX = (noseTip.x - boxCx) / Math.max(box.width, 1);
      const noseOffY = (noseTip.y - boxCy) / Math.max(box.height, 1);
      const pose = detectPose(box, noseTip);

      // Heavy debug logging
      if (Math.random() < 0.02) {
        console.log(
          `[Pose] box=(${box.x.toFixed(0)},${box.y.toFixed(0)} ${box.width.toFixed(0)}x${box.height.toFixed(0)}) ` +
          `nose=(${noseTip.x.toFixed(0)},${noseTip.y.toFixed(0)}) ` +
          `offX=${noseOffX.toFixed(3)} offY=${noseOffY.toFixed(3)} → ${pose}`
        );
      }

      (window as any).__pose = { pose, box, noseTip, noseOffX: noseOffX.toFixed(3), noseOffY: noseOffY.toFixed(3) };

      return { box, pose, noseTip };
    } catch {
      return null;
    }
  }

  return { isReady, detect };
}
