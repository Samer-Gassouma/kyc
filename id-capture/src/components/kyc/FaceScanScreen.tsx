"use client";

import { useRef, useState, useEffect, useCallback } from "react";
import { useKYCStore } from "@/store/kycStore";
import { useFaceDetection, REGION_EDGES } from "@/hooks/useFaceDetection";
import { canvasToJpegBlob } from "@/lib/frameEncoder";

export default function FaceScanScreen() {
  const { submitFace, setStep, faceMatchPassed } = useKYCStore();
  const { isReady, detect } = useFaceDetection();

  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animRef = useRef(0);
  const stableCount = useRef(0);
  const detectedCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const [msg, setMsg] = useState("Loading...");
  const [countdown, setCountdown] = useState(0);
  const [progress, setProgress] = useState(0);
  const [submitting, setSubmitting] = useState(false);
  const [retries, setRetries] = useState(0);
  const [error, setError] = useState<string | null>(null);

  // Start camera
  useEffect(() => {
    let running = true;
    (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
          audio: false,
        });
        if (!running) { stream.getTracks().forEach((t) => t.stop()); return; }
        streamRef.current = stream;
        const v = videoRef.current;
        if (v) { v.srcObject = stream; await v.play(); }
        setMsg("Position your face in the frame");
      } catch (e) {
        setMsg("Camera access denied");
        setError("Camera access denied");
      }
    })();
    return () => {
      running = false;
      streamRef.current?.getTracks().forEach((t) => t.stop());
      cancelAnimationFrame(animRef.current);
    };
  }, []);

  // Face detection loop
  useEffect(() => {
    if (!isReady) return;
    let running = true;
    if (!detectedCanvasRef.current) detectedCanvasRef.current = document.createElement("canvas");

    const loop = () => {
      if (!running) return;
      const v = videoRef.current;
      const dc = detectedCanvasRef.current;
      if (!v || v.videoWidth === 0 || !dc) {
        animRef.current = requestAnimationFrame(loop);
        return;
      }

      const result = detect(v, dc);
      if (!running) return;

      if (result.faceDetected && result.landmarks[0]) {
        const pts = result.landmarks[0];
        drawFaceMesh(overlayRef.current!, pts, v.videoWidth, v.videoHeight);

        const box = faceBbox(pts, v.videoWidth, v.videoHeight);
        const goodSize = box.width / v.videoWidth > 0.25;

        if (goodSize) {
          stableCount.current++;
          const remaining = Math.max(0, Math.ceil((50 - stableCount.current) / 30));
          setCountdown(remaining);
          setProgress(Math.round((stableCount.current / 50) * 100));

          if (stableCount.current >= 50) {
            running = false;
            doCapture(v);
            return;
          }
          setMsg(remaining > 0 ? `Hold still... ${remaining}` : "Scanning...");
        } else {
          stableCount.current = Math.max(0, stableCount.current - 1);
          setCountdown(0);
          setProgress(0);
          setMsg("Move closer — face too small");
        }
      } else {
        stableCount.current = Math.max(0, stableCount.current - 3);
        setCountdown(0);
        setProgress(0);
        setMsg("No face detected");
        const ov = overlayRef.current;
        if (ov) {
          const c = ov.getContext("2d");
          if (c) c.clearRect(0, 0, ov.width, ov.height);
        }
      }
      animRef.current = requestAnimationFrame(loop);
    };
    animRef.current = requestAnimationFrame(loop);
    return () => { running = false; };
  }, [isReady, detect]);

  const doCapture = useCallback(async (video: HTMLVideoElement) => {
    setSubmitting(true);
    setMsg("Capturing...");

    const c = document.createElement("canvas");
    c.width = video.videoWidth;
    c.height = video.videoHeight;
    c.getContext("2d")!.drawImage(video, 0, 0);
    const blob = await canvasToJpegBlob(c, 0.85);

    const match = await submitFace(blob, 1.0);
    setSubmitting(false);

    if (match) {
      setStep("EMAIL_INPUT");
    } else if (retries < 2) {
      setRetries((r) => r + 1);
      stableCount.current = 0;
      setProgress(0);
      setCountdown(0);
      setError(`Face didn't match document photo. ${2 - retries} retries left.`);
      setMsg("Try again — position your face clearly");
    } else {
      setError("Face verification failed after 3 attempts. Document photo may not match.");
    }
  }, [submitFace, setStep, retries]);

  return (
    <div className="space-y-4 w-full">
      <h2 className="text-xl font-bold text-white text-center">Face Scan</h2>
      <p className="text-zinc-400 text-sm text-center">
        We&apos;ll compare your face with your document photo
      </p>

      <div className="relative w-full overflow-hidden rounded-2xl bg-black" style={{ maxWidth: 400, margin: "0 auto" }}>
        <div className="relative">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="h-full w-full object-cover"
            style={{ aspectRatio: "3/4", transform: "scaleX(-1)" }}
          />
          <canvas
            ref={overlayRef}
            className="pointer-events-none absolute inset-0 h-full w-full"
            style={{ transform: "scaleX(-1)" }}
          />
          {countdown > 0 && countdown <= 3 && (
            <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/30">
              <span className="text-7xl font-bold text-white animate-pulse">{countdown}</span>
            </div>
          )}
        </div>
      </div>

      <div className="flex flex-col items-center gap-2 text-center">
        <p className="text-sm text-zinc-400">{msg}</p>
        {progress > 0 && (
          <div className="h-1 w-40 rounded-full bg-zinc-700">
            <div
              className="h-full rounded-full bg-sky-400 transition-all"
              style={{ width: `${progress}%` }}
            />
          </div>
        )}
        {submitting && (
          <div className="h-5 w-5 animate-spin rounded-full border-2 border-blue-500 border-t-transparent mt-2" />
        )}
        {error && (
          <p className="text-red-400 text-sm">{error}</p>
        )}
      </div>
    </div>
  );
}

// ── Helpers ──────────────────────────────────────────────────────────

function faceBbox(pts: any[], vw: number, vh: number) {
  let x = Infinity, y = Infinity, X = -Infinity, Y = -Infinity;
  for (const p of pts) {
    if (p.x < x) x = p.x; if (p.x > X) X = p.x;
    if (p.y < y) y = p.y; if (p.y > Y) Y = p.y;
  }
  return {
    x: x * vw,
    y: y * vh,
    width: (X - x) * vw,
    height: (Y - y) * vh,
  };
}

function drawFaceMesh(
  canvas: HTMLCanvasElement,
  pts: any[],
  vw: number,
  vh: number
) {
  canvas.width = vw;
  canvas.height = vh;
  const ctx = canvas.getContext("2d");
  if (!ctx || pts.length < 400) return;
  ctx.clearRect(0, 0, vw, vh);
  for (const [, r] of Object.entries(REGION_EDGES)) {
    ctx.strokeStyle = r.color + "99";
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    for (const [a, b] of r.edges) {
      if (a >= pts.length || b >= pts.length) continue;
      ctx.moveTo(pts[a].x * vw, pts[a].y * vh);
      ctx.lineTo(pts[b].x * vw, pts[b].y * vh);
    }
    ctx.stroke();
  }
  // Small dots at landmarks
  ctx.fillStyle = "rgba(255,255,255,0.5)";
  for (let i = 0; i < pts.length; i += 3) {
    ctx.beginPath();
    ctx.arc(pts[i].x * vw, pts[i].y * vh, 1.1, 0, Math.PI * 2);
    ctx.fill();
  }
}
