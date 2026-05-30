"use client";

import { useRef, useState, useEffect } from "react";
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
  const [instruction, setInstruction] = useState("Loading...");

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
        setInstruction("Center your face in the circle");
      } catch {
        setInstruction("Camera access denied");
      }
    })();
    return () => {
      running = false;
      streamRef.current?.getTracks().forEach((t) => t.stop());
      cancelAnimationFrame(animRef.current);
    };
  }, []);

  // ── Frame loop ──────────────────────────────────────────────────────
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

      const canvasW = v.videoWidth;
      const canvasH = v.videoHeight;
      c.width  = canvasW;
      c.height = canvasH;

      const ctx = c.getContext("2d");
      if (!ctx) { animRef.current = requestAnimationFrame(loop); return; }

      const result = detect(v, c);
      if (!running) return;

      ctx.clearRect(0, 0, canvasW, canvasH);

      // ── Fixed ring geometry — never moves ──────────────────────────
      const ringCenterX = canvasW / 2;
      const ringCenterY = canvasH / 2;
      const ringRadius  = Math.min(canvasW, canvasH) * 0.38;

      let instructionText = "Center your face in the circle";
      let currentProgress = progress;
      let currentBlink = blinkDetected;

      if (result.faceDetected && result.landmarks[0]) {
        const box = faceBbox(result.landmarks[0], canvasW, canvasH);

        // Face center normalized to canvas (0..1)
        const faceCx = (box.x + box.width  / 2) / canvasW;
        const faceCy = (box.y + box.height / 2) / canvasH;

        // Check if face is roughly centered
        const faceOffsetX = Math.abs(faceCx - 0.5);
        const faceOffsetY = Math.abs(faceCy - 0.5);
        const faceInPosition = faceOffsetX < 0.22 && faceOffsetY < 0.22;

        if (faceInPosition) {
          // Accumulate position history (normalized coords)
          positionHistory.current.push({ x: faceCx, y: faceCy });
          if (positionHistory.current.length > 90) positionHistory.current.shift();

          // Compute angular coverage from average center
          if (positionHistory.current.length > 10) {
            const avgX = positionHistory.current.reduce((s, p) => s + p.x, 0) / positionHistory.current.length;
            const avgY = positionHistory.current.reduce((s, p) => s + p.y, 0) / positionHistory.current.length;

            const covered = new Set<number>();
            for (const p of positionHistory.current) {
              const dx = p.x - avgX;
              const dy = p.y - avgY;
              const dist = Math.sqrt(dx * dx + dy * dy);
              if (dist < 0.015) continue; // ignore micro-movements
              const angle = Math.atan2(dy, dx);
              const sector = Math.floor(((angle + Math.PI) / (Math.PI * 2)) * 12);
              covered.add(sector);
            }
            currentProgress = Math.min(covered.size / 12, 1.0);
            setProgress(currentProgress);
          }

          instructionText = currentProgress >= 1
            ? currentBlink ? "✓ Liveness confirmed" : "Now blink naturally"
            : "Move your head slowly to complete the circle";

        } else {
          // Don't accumulate — face not centered
          instructionText = "Center your face in the circle";
        }

        // ── Blink detection ──────────────────────────────────────────
        if (result.blendshapes && faceInPosition) {
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
            currentBlink = true;
            setBlinkDetected(true);
            eyeClosed.current = false;
            if (currentProgress >= 1) {
              instructionText = "✓ Liveness confirmed";
            }
          }
        }
      } else {
        instructionText = "No face detected";
        setProgress(0);
        positionHistory.current = [];
      }

      setInstruction(instructionText);

      // ── Draw everything ────────────────────────────────────────────

      // 1. Static guide circle
      ctx.save();
      ctx.beginPath();
      ctx.arc(ringCenterX, ringCenterY, ringRadius - 16, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 6]);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();

      // 2. Tick ring (always fixed at center)
      drawLivenessRing(ctx, ringCenterX, ringCenterY, ringRadius, currentProgress);

      // 3. Blink indicator dot
      if (currentBlink && currentProgress >= 1) {
        ctx.beginPath();
        ctx.arc(ringCenterX, ringCenterY - ringRadius - 20, 8, 0, Math.PI * 2);
        ctx.fillStyle = "#22c55e";
        ctx.fill();
        ctx.beginPath();
        ctx.arc(ringCenterX, ringCenterY - ringRadius - 20, 8, 0, Math.PI * 2);
        ctx.strokeStyle = "#22c55e";
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      animRef.current = requestAnimationFrame(loop);
    };

    animRef.current = requestAnimationFrame(loop);
    return () => { running = false; };
  }, [cameraReady, isReady, progress, blinkDetected, detect]);

  // ── Complete when both conditions met ───────────────────────────────
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
      }, 1000);
      return () => clearTimeout(timeout);
    }
  }, [progress, blinkDetected, sessionId, setLivenessScore, setStep]);

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold text-white text-center">Liveness Check</h2>
      <p className="text-zinc-400 text-sm text-center">
        Center your face in the circle, then move your head around and blink
      </p>

      <div className="relative w-full aspect-square rounded-2xl overflow-hidden bg-black">
        {/* Video — mirrored */}
        <video
          ref={videoRef}
          className="absolute inset-0 w-full h-full object-cover"
          style={{ transform: "scaleX(-1)" }}
          autoPlay
          muted
          playsInline
        />

        {/* Canvas overlay — matches video mirroring */}
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full"
          style={{ transform: "scaleX(-1)" }}
        />

        {/* Dark gradient vignette over edges */}
        <div className="absolute inset-0 pointer-events-none rounded-2xl"
          style={{
            boxShadow: "inset 0 0 80px 40px rgba(0,0,0,0.6)",
          }}
        />

        {!cameraReady && (
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <div className="w-16 h-16 rounded-full border-4 border-dashed border-zinc-500 animate-spin"
                 style={{ animationDuration: "3s" }} />
          </div>
        )}

        {cameraReady && (
          <div className="absolute bottom-6 left-0 right-0 flex justify-center z-10">
            <span className={`px-5 py-2.5 rounded-full text-sm font-medium backdrop-blur-md transition-colors duration-300 ${
              instruction.includes("✓")
                ? "bg-green-500/80 text-white"
                : instruction.includes("Center")
                  ? "bg-white/10 text-white/80"
                  : "bg-black/60 text-white"
            }`}>
              {instruction}
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
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  radius: number,
  progress: number
) {
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
    ctx.strokeStyle = i < filled ? "#22c55e" : "rgba(255,255,255,0.20)";
    ctx.lineCap = "round";
    ctx.stroke();
  }
}
