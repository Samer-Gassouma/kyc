"use client";

import { useRef, useState, useEffect } from "react";
import { useKYCStore } from "@/store/kycStore";
import { useFaceDetection } from "@/hooks/useFaceDetection";
import { API_BASE } from "@/lib/apiBase";

// ── Step definitions ──────────────────────────────────────────────────

const LIVENESS_STEPS = [
  { id: "left",  label: "Turn your head LEFT  →", ringEnd: 0.50 },
  { id: "right", label: "← Turn your head RIGHT", ringEnd: 1.00 },
] as const;

const HOLD_REQUIRED = 30; // frames at ~30fps = ~1 second

// ── Direction detection — bounding box only ───────────────────────────

function detectDirection(
  box: { x: number; y: number; width: number; height: number },
  canvasW: number,
  canvasH: number
): "center" | "left" | "right" | "up" | "none" {
  const faceCx = box.x + box.width / 2;
  const faceCy = box.y + box.height / 2;
  const nx = faceCx / canvasW;
  const ny = faceCy / canvasH;

  // Front camera is mirrored:
  // user turns LEFT  → face appears on RIGHT  side of frame → nx > 0.55
  // user turns RIGHT → face appears on LEFT   side of frame → nx < 0.45
  // user looks UP    → face moves UP in frame               → ny < 0.38
  if (nx > 0.58) return "left";
  if (nx < 0.42) return "right";
  if (ny < 0.38) return "up";
  if (Math.abs(nx - 0.5) < 0.12 && Math.abs(ny - 0.5) < 0.14) return "center";
  return "none";
}

// ── Component ─────────────────────────────────────────────────────────

export default function LivenessScreen() {
  const { sessionId, setLivenessScore, setStep } = useKYCStore();
  const { isReady, detect } = useFaceDetection();

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animRef = useRef(0);
  const holdFrames = useRef(0);
  const completedRef = useRef(false);

  const [cameraReady, setCameraReady] = useState(false);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [ringProgress, setRingProgress] = useState(0);
  const [faceDetected, setFaceDetected] = useState(false);

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
      } catch {
        // camera denied
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
    if (completedRef.current) return;
    let running = true;

    const loop = () => {
      if (!running || completedRef.current) return;
      const v = videoRef.current;
      const c = canvasRef.current;
      if (!v || v.videoWidth === 0 || !c) {
        animRef.current = requestAnimationFrame(loop);
        return;
      }

      const canvasW = v.videoWidth;
      const canvasH = v.videoHeight;
      c.width = canvasW;
      c.height = canvasH;

      const ctx = c.getContext("2d");
      if (!ctx) { animRef.current = requestAnimationFrame(loop); return; }

      const result = detect(v, c);
      if (!running || completedRef.current) return;

      ctx.clearRect(0, 0, canvasW, canvasH);

      // ── Fixed ring geometry ──────────────────────────────────────────
      const ringCenterX = canvasW / 2;
      const ringCenterY = canvasH / 2;
      const ringRadius = Math.min(canvasW, canvasH) * 0.38;

      const hasFace = !!(result.faceDetected && result.landmarks[0]);
      setFaceDetected(hasFace);

      if (hasFace) {
        const box = faceBbox(result.landmarks[0], canvasW, canvasH);
        const currentStep = LIVENESS_STEPS[currentStepIndex];

        // ── Direction step ────────────────────────────────────────────
        const direction = detectDirection(box, canvasW, canvasH);
        const stepStart = currentStepIndex * 0.5;

        if (direction === currentStep.id) {
          holdFrames.current += 1;

          // Animate ring smoothly during hold
          const holdFraction = Math.min(holdFrames.current / HOLD_REQUIRED, 1.0);
          const prog = stepStart + holdFraction * 0.5;
          setRingProgress(prog);

          if (holdFrames.current >= HOLD_REQUIRED) {
            // Step complete — advance to next
            holdFrames.current = 0;
            const nextIdx = currentStepIndex + 1;
            setRingProgress(currentStep.ringEnd);
            setCurrentStepIndex(nextIdx);
            if (nextIdx >= LIVENESS_STEPS.length) {
              completedRef.current = true;
              handleComplete();
              running = false;
              return;
            }
          }
        } else {
          // Wrong direction — decay hold progress slowly
          holdFrames.current = Math.max(0, holdFrames.current - 2);
          const holdFraction = holdFrames.current / HOLD_REQUIRED;
          setRingProgress(stepStart + holdFraction * 0.5);
        }
      }

      // ── Draw everything ──────────────────────────────────────────────

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

      // 2. Tick ring with quarter coloring
      drawLivenessRing(ctx, ringCenterX, ringCenterY, ringRadius, ringProgress, currentStepIndex);

      // 3. Blink complete indicator
      if (ringProgress >= 1.0) {
        ctx.beginPath();
        ctx.arc(ringCenterX, ringCenterY - ringRadius - 20, 8, 0, Math.PI * 2);
        ctx.fillStyle = "#22c55e";
        ctx.fill();
      }

      animRef.current = requestAnimationFrame(loop);
    };

    animRef.current = requestAnimationFrame(loop);
    return () => { running = false; };
  }, [cameraReady, isReady, currentStepIndex, detect]);

  // ── Submit liveness to backend ──────────────────────────────────────
  async function handleComplete() {
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
    setTimeout(() => setStep("PHONE_INPUT"), 1200);
  }

  // ── Instruction text ────────────────────────────────────────────────
  const instructionLabel = faceDetected
    ? (currentStepIndex < LIVENESS_STEPS.length
        ? LIVENESS_STEPS[currentStepIndex].label
        : "✓ Liveness confirmed")
    : "Position your face in the circle";

  const isComplete = ringProgress >= 1.0;

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold text-white text-center">Liveness Check</h2>
      <p className="text-zinc-400 text-sm text-center">
        Follow the prompts to verify you&apos;re a real person
      </p>

      {/* Progress dots */}
      <div className="flex justify-center gap-2">
        {LIVENESS_STEPS.map((step, i) => (
          <div
            key={step.id}
            className={`h-1.5 rounded-full transition-all duration-300 ${
              i < currentStepIndex
                ? "w-8 bg-green-500"
                : i === currentStepIndex
                  ? "w-8 bg-cyan-400"
                  : "w-4 bg-zinc-700"
            }`}
          />
        ))}
      </div>

      {/* Camera view */}
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

        {/* Canvas overlay — mirrors video */}
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full"
          style={{ transform: "scaleX(-1)" }}
        />

        {/* Vignette */}
        <div
          className="absolute inset-0 pointer-events-none rounded-2xl"
          style={{ boxShadow: "inset 0 0 80px 40px rgba(0,0,0,0.6)" }}
        />

        {/* Loading spinner */}
        {!cameraReady && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div
              className="w-16 h-16 rounded-full border-4 border-dashed border-zinc-500 animate-spin"
              style={{ animationDuration: "3s" }}
            />
          </div>
        )}

        {/* Instruction pill */}
        {cameraReady && (
          <div className="absolute bottom-6 left-0 right-0 flex justify-center z-10">
            <span
              className={`px-5 py-2.5 rounded-full text-sm font-medium backdrop-blur-md transition-all duration-300 ${
                isComplete
                  ? "bg-green-500/80 text-white scale-105"
                  : faceDetected
                    ? "bg-black/60 text-white"
                    : "bg-white/10 text-white/80"
              }`}
            >
              {instructionLabel}
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
  progress: number,
  activeQuarter: number
) {
  const TICKS = 60;
  const filled = Math.floor(progress * TICKS);
  const quarterTicks = 30; // 60 / 2 steps

  for (let i = 0; i < TICKS; i++) {
    const angle = (i / TICKS) * Math.PI * 2 - Math.PI / 2;
    const inner = radius - 10;
    const outer = radius + 2;
    const x1 = cx + Math.cos(angle) * inner;
    const y1 = cy + Math.sin(angle) * inner;
    const x2 = cx + Math.cos(angle) * outer;
    const y2 = cy + Math.sin(angle) * outer;

    let color: string;
    const activeQuarterStart = activeQuarter * quarterTicks;

    if (i < filled - (activeQuarter >= 4 ? 0 : quarterTicks)) {
      // Completed quarters — solid green
      color = "#22c55e";
    } else if (i < filled && activeQuarter < 4) {
      // Current quarter being filled — cyan
      color = "#38bdf8";
    } else {
      // Unfilled — dim white
      color = "rgba(255, 255, 255, 0.18)";
    }

    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.lineWidth = 3;
    ctx.strokeStyle = color;
    ctx.lineCap = "round";
    ctx.stroke();
  }
}
