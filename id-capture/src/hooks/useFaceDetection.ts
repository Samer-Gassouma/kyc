"use client";

import { useRef, useState, useEffect, useCallback } from "react";
import { FaceLandmarker, FilesetResolver, type NormalizedLandmark } from "@mediapipe/tasks-vision";

// ── Face mesh region connections (canonical 468-point topology) ─

export const MESH_REGIONS = {
  faceOval:   [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109],
  leftEye:    [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7],
  rightEye:   [362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382],
  leftBrow:   [46,53,52,65,55,70,63,105,66,107],
  rightBrow:  [276,283,282,295,285,300,293,334,296,336],
  nose:       [6,168,197,195,5,4,1,19,94,2,98,327,460,294,455,459,458,461,354],
  lipsOuter:  [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185],
  lipsInner:  [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95],
};

function ringEdges(idx: number[]): [number, number][] { const e: [number, number][] = []; for (let i=0; i<idx.length; i++) e.push([idx[i], idx[(i+1)%idx.length]]); return e; }

export const REGION_EDGES: Record<string, { edges: [number,number][]; color: string }> = {
  "Jawline":     { edges: ringEdges(MESH_REGIONS.faceOval), color: "#34d399" },
  "Left Eye":    { edges: ringEdges(MESH_REGIONS.leftEye), color: "#38bdf8" },
  "Right Eye":   { edges: ringEdges(MESH_REGIONS.rightEye), color: "#38bdf8" },
  "Left Brow":   { edges: ringEdges(MESH_REGIONS.leftBrow), color: "#fbbf24" },
  "Right Brow":  { edges: ringEdges(MESH_REGIONS.rightBrow), color: "#fbbf24" },
  "Nose":        { edges: ringEdges(MESH_REGIONS.nose), color: "#a78bfa" },
  "Lips":        { edges: [...ringEdges(MESH_REGIONS.lipsOuter), ...ringEdges(MESH_REGIONS.lipsInner)], color: "#f472b6" },
};

// ── Model loader ──────────────────────────────────────────────────

let _landmarker: FaceLandmarker | null = null;
let _loading: Promise<FaceLandmarker> | null = null;

async function load(): Promise<FaceLandmarker> {
  if (_landmarker) return _landmarker;
  if (!_loading) {
    _loading = (async () => {
      console.log("[MP] Loading FaceLandmarker...");
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
      );
      _landmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
          delegate: "GPU",
        },
        runningMode: "IMAGE", numFaces: 1, outputFaceBlendshapes: false,
      });
      console.log("[MP] FaceLandmarker ready");
      return _landmarker;
    })();
  }
  return _loading;
}

// ── Hook ──────────────────────────────────────────────────────────

export interface DetectResult {
  landmarks: NormalizedLandmark[][];
  faceDetected: boolean;
}

export function useFaceDetection() {
  const [isReady, setIsReady] = useState(false);
  const started = useRef(false);

  useEffect(() => {
    if (started.current) return;
    started.current = true;
    load().then(() => setIsReady(true)).catch(e => console.error("[MP] fail:", e));
  }, []);

  const detect = useCallback((video: HTMLVideoElement, canvas: HTMLCanvasElement): DetectResult => {
    const lm = _landmarker;
    if (!lm) return { landmarks: [], faceDetected: false };
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh) return { landmarks: [], faceDetected: false };

    canvas.width = vw; canvas.height = vh;
    const ctx = canvas.getContext("2d");
    if (!ctx) return { landmarks: [], faceDetected: false };
    ctx.drawImage(video, 0, 0, vw, vh);

    try {
      const res = lm.detect(canvas);
      if (res.faceLandmarks?.length) {
        const face = res.faceLandmarks[0];
        const box = bboxFromLandmarks(face);
        (window as any).__face = { detected: true, box, pts: face.length };
        return { landmarks: res.faceLandmarks, faceDetected: true };
      }
    } catch (e) { /* ignore */ }
    (window as any).__face = { detected: false };
    return { landmarks: [], faceDetected: false };
  }, []);

  return { isReady, detect };
}

export function bboxFromLandmarks(pts: NormalizedLandmark[]) {
  let minX=Infinity, minY=Infinity, maxX=-Infinity, maxY=-Infinity;
  for (const p of pts) { if (p.x<minX) minX=p.x; if (p.x>maxX) maxX=p.x; if (p.y<minY) minY=p.y; if (p.y>maxY) maxY=p.y; }
  return { x: minX, y: minY, width: maxX-minX, height: maxY-minY };
}

export type { NormalizedLandmark };
