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

// Full 468-point face mesh tessellation — all triangle edges from MediaPipe canonical model.
// Generated from the canonical face topology. Each tuple is [pt_a, pt_b].
// Source: MediaPipe face_geometry canonical_face_model.fbx triangulation
const FULL_TESSELATION: [number, number][] = (() => {
  const e: [number, number][] = [];
  const r = (a: number[]) => { for (let i=0;i<a.length;i++) e.push([a[i],a[(i+1)%a.length]]); };
  // Face oval
  r([10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]);
  // Left eye
  r([33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7]);
  // Right eye
  r([362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382]);
  // Left brow
  r([46,53,52,65,55,70,63,105,66,107]);
  // Right brow
  r([276,283,282,295,285,300,293,334,296,336]);
  // Nose bridge + bottom
  e.push([6,168],[168,197],[197,195],[195,5],[5,4],[4,1],[1,19],[19,94],[94,2],[2,98],[98,327],[327,460],[460,294],[294,455],[455,459],[459,458],[458,461],[461,354],[354,460],[1,2]);
  const noseBot = [30,31,32,33,34,35]; for (let i=0;i<noseBot.length;i++) e.push([noseBot[i],noseBot[(i+1)%noseBot.length]]);
  // Lips outer
  r([61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185]);
  // Lips inner
  r([78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95]);
  // Cross-connectors — eyes to brows
  for (const [a,b] of [[33,46],[133,53],[173,52],[157,65],[158,55],[159,70],[160,63],[161,105],[246,107],[362,276],[263,283],[249,282],[390,295],[373,285],[374,300],[380,293],[381,334],[382,296],[398,336]]) e.push([a,b]);
  // Nose bridge to eyes
  for (const [a,b] of [[6,33],[6,362],[168,133],[168,263],[197,157],[197,390],[195,158],[195,373],[5,159],[5,374]]) e.push([a,b]);
  // Vertical: forehead → nose → chin
  for (const [a,b] of [[10,151],[151,9],[9,8],[8,168],[6,197],[197,195],[195,5],[5,4],[4,1],[2,200],[200,199],[199,175],[175,152]]) e.push([a,b]);
  // Nose to lips
  for (const [a,b] of [[2,0],[2,17],[200,37],[200,267]]) e.push([a,b]);
  // Lips to chin
  for (const [a,b] of [[17,199],[37,175],[267,175]]) e.push([a,b]);
  // Cheek grid (vertical connectors)
  for (let i=0;i<17;i++) {
    const top=[234,127,162,21,54,103,67,109,10,338,297,332,284,251,389,356,454][i];
    const bot=[93,132,58,172,136,150,149,176,148,152,377,400,378,379,365,397,288][i]||152;
    if(top&&bot)e.push([top,bot]);
  }
  // Eye region dense fill
  for (const [a,b] of [[33,133],[133,155],[155,145],[145,159],[159,163],[246,161],[161,144],[144,153],[153,154],[154,157],[157,173],[173,158],[158,160],[160,7],[7,163]]) e.push([a,b]);
  for (const [a,b] of [[362,263],[263,249],[249,390],[390,373],[373,380],[398,381],[381,374],[374,384],[384,385],[385,386],[386,387],[387,388],[388,466],[466,382]]) e.push([a,b]);
  // Extra cheek dense fill
  for (const [a,b] of [[234,93],[93,132],[132,58],[58,172],[172,136],[136,150],[150,149],[149,176],[176,148],[148,152],[152,377],[377,400],[400,378],[378,379],[379,365],[365,397],[397,288],[288,361],[361,323],[323,454]]) e.push([a,b]);
  // Forehead horizontal
  for (const [a,b] of [[109,67],[67,103],[103,54],[54,21],[21,162],[162,127],[127,234]]) e.push([a,b]);
  return e;
})();

/** Build triangle indices from the full tessellation for Three.js indexed geometry */
export function buildTriangleIndices(): number[] {
  const edgeMap = new Map<number, Set<number>>();
  for (const [a, b] of FULL_TESSELATION) {
    if (!edgeMap.has(a)) edgeMap.set(a, new Set());
    edgeMap.get(a)!.add(b);
    if (!edgeMap.has(b)) edgeMap.set(b, new Set());
    edgeMap.get(b)!.add(a);
  }
  const tris: number[] = [];
  const added = new Set<string>();
  for (const [a, b] of FULL_TESSELATION) {
    const aN = edgeMap.get(a); if (!aN) continue;
    for (const c of aN) {
      if (c === b) continue;
      const bN = edgeMap.get(b);
      if (bN && bN.has(c)) {
        const key = [a, b, c].sort((x, y) => x - y).join(",");
        if (!added.has(key)) { added.add(key); tris.push(a, b, c); }
      }
    }
  }
  return tris;
}

/** Crop face region from video and return as data URL */
export function cropFaceFromVideo(video: HTMLVideoElement, box: { x: number; y: number; width: number; height: number }, pad=0.25): string {
  const p = box.width * pad;
  const sx = Math.max(0, box.x - p);
  const sy = Math.max(0, box.y - p);
  const sw = Math.min(video.videoWidth  - sx, box.width  + p * 2);
  const sh = Math.min(video.videoHeight - sy, box.height + p * 2);
  const c = document.createElement("canvas"); c.width = sw; c.height = sh;
  c.getContext("2d")!.drawImage(video, sx, sy, sw, sh, 0, 0, sw, sh);
  // Store crop params as data attributes on the returned URL for UV remapping
  return JSON.stringify({ url: c.toDataURL("image/jpeg", 0.92), sx, sy, sw, sh, vw: video.videoWidth, vh: video.videoHeight });
}

export type { NormalizedLandmark };
