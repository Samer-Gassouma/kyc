"use client";

import { useRef, useState, useEffect, useCallback } from "react";
import { useKYCStore } from "@/store/kycStore";
import { useFaceDetection } from "@/hooks/useFaceDetection";
import { API_BASE } from "@/lib/apiBase";

export default function LivenessScreen() {
  const { sessionId, setLivenessScore, setStep } = useKYCStore();
  const { isReady, detect } = useFaceDetection();

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animRef = useRef(0);
  const positionHistory = useRef<{ x: number; y: number }[]>([]);
  const eyeClosed = useRef(false);

  const [cameraReady, setCameraReady] = useState(false);
  const [progress, setProgress] = useState(0);
  const [blinkDetected, setBlinkDetected] = useState(false);
  const [message, setMessage] = useState("Loading...");

  // Start camera
  useEffect(() => {
    let running = true;
    (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 640 } },
          audio: false,
        });
        if (!running) { stream.getTracks().forEach((t) => t.stop()); return; }
        streamRef.current = stream;
        const v = videoRef.current;
        if (v) { v.srcObject = stream; await v.play(); }
        setCameraReady(true);
        setMessage("Move your head slowly to complete the circle");
      } catch {
        setMessage("Camera access denied");
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
    if (!cameraReady || !isReady) return;
    let running = true;

    const loop = () => {
      if (!running) return;
      const v = videoRef.current;
      const c = canvasRef.current;
      if (!v || v.videoWidth === 0 || !c) {
        animRef.current = requestAnimationFrame(loop);
        return;
      }

      const result = detect(v, c);
      if (!running) return;

      if (result.faceDetected && result.landmarks[0]) {
        const box = faceBbox(result.landmarks[0], v.videoWidth, v.videoHeight);
        const cx = box.x + box.width / 2;
        const cy = box.y + box.height / 2;

        positionHistory.current.push({ x: cx, y: cy });
        if (positionHistory.current.length > 90) positionHistory.current.shift();

        // Compute progress: how many of 12 angular sectors have been covered
        if (positionHistory.current.length > 5) {
          const center = {
            x: positionHistory.current.reduce((s, p) => s + p.x, 0) / positionHistory.current.length,
            y: positionHistory.current.reduce((s, p) => s + p.y, 0) / positionHistory.current.length,
          };

          const covered = new Set<number>();
          positionHistory.current.forEach((p) => {
            const angle = Math.atan2(p.y - center.y, p.x - center.x);
            const sector = Math.floor(((angle + Math.PI) / (Math.PI * 2)) * 12);
            covered.add(sector);
          });

          const p = Math.min(covered.size / 12, 1.0);
          setProgress(p);
        }

        // Blink detection
        if (result.blendshapes) {
          const blinkL = result.blendshapes.find(
            (b: any) => b.categoryName === "eyeBlinkLeft"
          )?.score ?? 0;
          const blinkR = result.blendshapes.find(
            (b: any) => b.categoryName === "eyeBlinkRight"
          )?.score ?? 0;

          if ((blinkL > 0.35 || blinkR > 0.35) && !eyeClosed.current) {
            eyeClosed.current = true;
          }
          if (blinkL < 0.1 && blinkR < 0.1 && eyeClosed.current) {
            setBlinkDetected(true);
            eyeClosed.current = false;
          }
        }

        // Update message
        if (progress < 1) {
          setMessage("Move your head slowly to complete the circle");
        } else if (!blinkDetected) {
          setMessage("Now blink naturally");
        } else {
          setMessage("✓ Liveness confirmed");
        }

        // Draw ring
        drawLivenessRing(c, cx, cy, Math.min(box.width, box.height) * 0.55, progress, blinkDetected);
      } else {
        setMessage("No face detected");
        const ctx = c.getContext("2d");
        if (ctx) ctx.clearRect(0, 0, c.width, c.height);
      }

      animRef.current = requestAnimationFrame(loop);
    };

    animRef.current = requestAnimationFrame(loop);
    return () => { running = false; };
  }, [cameraReady, isReady, progress, blinkDetected, detect]);

  // Complete when both conditions met
  useEffect(() => {
    if (progress >= 1.0 && blinkDetected) {
      const timeout = setTimeout(async () => {
        setLivenessScore(1.0);
        try {
          const fd = new FormData();
          fd.append("liveness_passed", "true");
          fd.append("liveness_score", "1.0");
          await fetch(`${API_BASE}/api/kyc/session/${sessionId}/liveness`, {
            method: "PATCH",
            body: fd,
          });
        } catch {}
        setStep("PHONE_INPUT");
      }, 800);
      return () => clearTimeout(timeout);
    }
  }, [progress, blinkDetected, sessionId, setLivenessScore, setStep]);

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold text-white text-center">Liveness Check</h2>
      <p className="text-zinc-400 text-sm text-center">
        Move your head in a circle, then blink to prove you&apos;re real
      </p>

      <div className="relative w-full aspect-square rounded-xl overflow-hidden bg-black">
        <video
          ref={videoRef}
          className="w-full h-full object-cover"
          style={{ transform: "scaleX(-1)" }}
          autoPlay
          muted
          playsInline
        />
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full"
          style={{ transform: "scaleX(-1)" }}
        />

        {!cameraReady && (
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <div className="w-48 h-48 rounded-full border-4 border-dashed border-zinc-600 animate-spin flex items-center justify-center"
                 style={{ animationDuration: "3s" }}>
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#888" strokeWidth="1.5">
                <circle cx="12" cy="8" r="4" />
                <path d="M4 20c0-4 3.6-7 8-7s8 3 8 7" />
              </svg>
            </div>
          </div>
        )}

        {cameraReady && (
          <div className="absolute bottom-4 left-0 right-0 flex justify-center">
            <span className="bg-black/70 text-white px-4 py-2 rounded-full text-sm">
              {message}
            </span>
          </div>
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
  return { x: x * vw, y: y * vh, width: (X - x) * vw, height: (Y - y) * vh };
}

function drawLivenessRing(
  canvas: HTMLCanvasElement,
  cx: number,
  cy: number,
  radius: number,
  progress: number,
  blinkDone: boolean
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  canvas.width = canvas.clientWidth;
  canvas.height = canvas.clientHeight;

  const TICKS = 60;
  const filled = Math.floor(progress * TICKS);

  for (let i = 0; i < TICKS; i++) {
    const angle = (i / TICKS) * Math.PI * 2 - Math.PI / 2;
    const inner = radius - 10;
    const outer = radius + 2;
    const x1 = cx + Math.cos(angle) * inner;
    const y1 = cy + Math.sin(angle) * inner;
    const x2 = cx + Math.cos(angle) * outer;
    const y2 = cy + Math.sin(angle) * outer;

    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.lineWidth = 3;
    ctx.strokeStyle = i < filled ? "#22c55e" : "rgba(255,255,255,0.25)";
    ctx.lineCap = "round";
    ctx.stroke();
  }

  // Circular clip mask for face
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, radius - 16, 0, Math.PI * 2);
  ctx.clip();

  // Draw video frame inside circle
  ctx.globalCompositeOperation = "source-over";

  ctx.restore();

  // Blink indicator dot
  if (blinkDone && progress >= 1) {
    ctx.beginPath();
    ctx.arc(cx, cy - radius - 20, 6, 0, Math.PI * 2);
    ctx.fillStyle = "#22c55e";
    ctx.fill();
  }
}
