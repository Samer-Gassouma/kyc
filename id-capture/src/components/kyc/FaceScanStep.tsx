"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { canvasToJpegBlob, grabFrame } from "@/lib/frameEncoder";
import { API_BASE } from "@/lib/apiBase";
import { useMediaPipeFace, serializeLandmarks } from "@/hooks/useMediaPipeFace";
import { checkLiveness, prepareLivenessInput } from "@/lib/silentFaceLiveness";
import clsx from "clsx";
import { CheckCircle, Loader2, XCircle, Camera } from "lucide-react";

interface FaceScanStepProps {
  token: string;
  userId: string;
  onComplete: (result: {
    passed: boolean;
    confidence: number;
    user_id: string;
  }) => void;
}

type ScanState =
  | "idle"
  | "preparing"
  | "scanning"
  | "capturing"
  | "verifying"
  | "passed"
  | "failed";

export default function FaceScanStep({
  token,
  userId,
  onComplete,
}: FaceScanStepProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animRef = useRef<number | null>(null);
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;

  const [scanState, setScanState] = useState<ScanState>("idle");
  const [camError, setCamError] = useState<string | null>(null);
  const [statusText, setStatusText] = useState("");
  const [confidence, setConfidence] = useState(0);

  const { landmarks, faceDetected, isReady, processFrame } =
    useMediaPipeFace();

  const cleanup = useCallback(() => {
    if (animRef.current) cancelAnimationFrame(animRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }, []);

  // Start camera
  const startCamera = useCallback(async () => {
    try {
      setScanState("preparing");
      setStatusText("Loading face detection...");

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
        audio: false,
      });
      streamRef.current = stream;

      const video = document.createElement("video");
      video.setAttribute("playsinline", "");
      video.srcObject = stream;
      videoRef.current = video;
      await video.play();
    } catch (err) {
      setCamError(
        err instanceof Error ? err.message : "Camera access denied"
      );
      setScanState("failed");
    }
  }, []);

  // Initialize camera on mount
  useEffect(() => {
    startCamera();
    return cleanup;
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Transition to scanning when models + camera are ready
  useEffect(() => {
    if (isReady && videoRef.current && scanState === "preparing") {
      setScanState("scanning");
      setStatusText("Position your face in the frame");
    }
  }, [isReady, scanState]);

  // Face positioning quality check using 468 landmarks
  function faceIsWellPositioned(): boolean {
    if (!landmarks || landmarks.length === 0) return false;
    const pts = landmarks[0];
    if (pts.length < 468) return false;

    let minX = Infinity,
      minY = Infinity,
      maxX = -Infinity,
      maxY = -Infinity;
    for (const p of pts) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }

    const faceW = maxX - minX;
    const faceH = maxY - minY;
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;

    // Face must occupy at least 8% of frame area
    const faceAreaRatio = faceW * faceH;
    if (faceAreaRatio < 0.08) return false;

    // Face center within ~25% of frame center
    const distFromCenter = Math.sqrt(
      (centerX - 0.5) ** 2 + (centerY - 0.5) ** 2
    );
    if (distFromCenter > 0.25) return false;

    // Rough head-pose: nose tip (1) should be between eye centers
    const nose = pts[1];
    const leftEyeInner = pts[133];
    const rightEyeInner = pts[362];
    const eyeMidX = (leftEyeInner.x + rightEyeInner.x) / 2;
    const eyeMidY = (leftEyeInner.y + rightEyeInner.y) / 2;
    const noseOffsetX = Math.abs(nose.x - eyeMidX);
    const noseOffsetY = Math.abs(nose.y - eyeMidY);
    if (noseOffsetX > 0.06 || noseOffsetY > 0.12) return false;

    return true;
  }

  // Draw landmark dots and face oval on overlay canvas
  function drawOverlay() {
    const canvas = overlayRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (landmarks && landmarks.length > 0) {
      const pts = landmarks[0];

      // Draw landmark dots
      ctx.fillStyle = "rgba(59, 130, 246, 0.55)";
      for (const p of pts) {
        const x = p.x * canvas.width;
        const y = p.y * canvas.height;
        ctx.beginPath();
        ctx.arc(x, y, 1.8, 0, Math.PI * 2);
        ctx.fill();
      }

      // Draw face oval guideline
      const chin = pts[152];
      const forehead = pts[10];
      const leftCheek = pts[234];
      const rightCheek = pts[454];
      const cx = ((leftCheek.x + rightCheek.x) / 2) * canvas.width;
      const cy = ((forehead.y + chin.y) / 2) * canvas.height;
      const rx =
        (Math.abs(rightCheek.x - leftCheek.x) / 2) * canvas.width * 1.3;
      const ry =
        (Math.abs(chin.y - forehead.y) / 2) * canvas.height * 1.3;

      const wellPositioned = faceIsWellPositioned();
      ctx.strokeStyle = wellPositioned
        ? "rgba(34, 197, 94, 0.5)"
        : "rgba(255, 255, 255, 0.2)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  // Frame loop — MediaPipe processing + overlay rendering
  useEffect(() => {
    if (scanState !== "scanning" && scanState !== "capturing") return;

    const canvas = document.createElement("canvas");

    const loop = () => {
      const video = videoRef.current;
      if (!video || video.readyState < 2) {
        animRef.current = requestAnimationFrame(loop);
        return;
      }

      grabFrame(video, canvas);
      processFrame(video);
      drawOverlay();

      animRef.current = requestAnimationFrame(loop);
    };

    animRef.current = requestAnimationFrame(loop);
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scanState, landmarks]);

  // Auto-capture when face is well-positioned for ~1s
  const captureTriggeredRef = useRef(false);
  const stableCountRef = useRef(0);

  useEffect(() => {
    if (scanState !== "scanning" || captureTriggeredRef.current) return;

    if (faceIsWellPositioned()) {
      stableCountRef.current++;
      const remaining = Math.max(1, Math.ceil((30 - stableCountRef.current) / 10));
      setStatusText(`Hold still... ${remaining}`);
      if (stableCountRef.current >= 30) {
        captureTriggeredRef.current = true;
        handleCapture();
      }
    } else {
      stableCountRef.current = 0;
      setStatusText(
        faceDetected
          ? "Center your face in the oval"
          : "Position your face in the frame"
      );
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [landmarks, scanState]);

  async function handleCapture() {
    setScanState("capturing");
    setStatusText("Checking liveness...");

    const video = videoRef.current;
    if (!video || !landmarks || landmarks.length === 0) {
      resetCapture();
      return;
    }

    // Compute face bbox from landmarks
    const pts = landmarks[0];
    let minX = Infinity,
      minY = Infinity,
      maxX = -Infinity,
      maxY = -Infinity;
    for (const p of pts) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
    const bbox = {
      x: minX * video.videoWidth,
      y: minY * video.videoHeight,
      width: (maxX - minX) * video.videoWidth,
      height: (maxY - minY) * video.videoHeight,
    };

    // Capture burst of 3 frames, run liveness on each
    let bestLiveness = 0;
    let bestBlob: Blob | null = null;

    for (let i = 0; i < 3; i++) {
      await new Promise((r) => setTimeout(r, 150));

      const input = prepareLivenessInput(video, bbox);
      if (!input) continue;

      const score = await checkLiveness(input);
      if (score > bestLiveness) {
        bestLiveness = score;
        const frameCanvas = document.createElement("canvas");
        grabFrame(video, frameCanvas);
        bestBlob = await canvasToJpegBlob(frameCanvas, 0.85);
      }
    }

    if (!bestBlob || bestLiveness < 0.5) {
      setScanState("failed");
      setCamError(
        bestLiveness < 0.5
          ? "Spoof detected — use a real face, not a photo or screen"
          : "Liveness check failed"
      );
      return;
    }

    // Send best frame to server for verification
    setScanState("verifying");
    setStatusText("Verifying identity...");

    try {
      const formData = new FormData();
      formData.append("image", bestBlob, "face.jpg");
      formData.append("user_id", userId);

      const res = await fetch(`${API_BASE}/api/face/verify`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(
          (err as { detail?: string }).detail || `HTTP ${res.status}`
        );
      }

      const result = await res.json();
      setConfidence(result.confidence);

      if (result.matched) {
        setScanState("passed");
        setStatusText("");
        onCompleteRef.current({
          passed: true,
          confidence: result.confidence,
          user_id: result.user_id,
        });
      } else {
        setScanState("failed");
        setCamError(
          `Face doesn't match (${(result.confidence * 100).toFixed(0)}% — need ${(result.threshold_used * 100).toFixed(0)}%)`
        );
      }
    } catch (err) {
      setScanState("failed");
      setCamError(
        err instanceof Error ? err.message : "Verification failed"
      );
    }
  }

  function resetCapture() {
    captureTriggeredRef.current = false;
    stableCountRef.current = 0;
    setScanState("scanning");
    setStatusText("Position your face in the frame");
  }

  function handleRetry() {
    captureTriggeredRef.current = false;
    stableCountRef.current = 0;
    setCamError(null);
    setConfidence(0);
    cleanup();
    startCamera();
  }

  const showVideo =
    scanState === "preparing" ||
    scanState === "scanning" ||
    scanState === "capturing";

  return (
    <div className="flex flex-col items-center gap-4">
      {camError && (
        <div className="flex flex-col items-center gap-3 p-4 text-center">
          <p className="text-sm text-red-400">{camError}</p>
          <button
            onClick={handleRetry}
            className="rounded-full bg-blue-600 px-4 py-2 text-sm text-white hover:bg-blue-500"
          >
            Retry
          </button>
        </div>
      )}

      {/* Camera view */}
      <div
        className="relative w-full overflow-hidden rounded-2xl bg-black"
        style={{ maxWidth: 400 }}
      >
        {showVideo && (
          <div className="relative">
            <video
              ref={(el) => {
                if (el && videoRef.current !== el) {
                  videoRef.current = el;
                }
              }}
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
          </div>
        )}

        {/* Verifying spinner placeholder */}
        {scanState === "verifying" && (
          <div
            className="flex items-center justify-center bg-black"
            style={{ aspectRatio: "3/4" }}
          >
            <div className="flex flex-col items-center gap-3">
              <Loader2 className="h-10 w-10 animate-spin text-blue-400" />
              <p className="text-sm text-zinc-300">{statusText}</p>
            </div>
          </div>
        )}

        {/* Passed state */}
        {scanState === "passed" && (
          <div
            className="flex items-center justify-center bg-green-950/50"
            style={{ aspectRatio: "3/4" }}
          >
            <div className="flex flex-col items-center gap-3">
              <CheckCircle className="h-16 w-16 text-green-400" />
              <p className="text-sm font-medium text-green-400">
                Identity verified ({(confidence * 100).toFixed(0)}%)
              </p>
            </div>
          </div>
        )}

        {/* Idle / preparing */}
        {(scanState === "idle" || scanState === "preparing") && (
          <div
            className="flex items-center justify-center bg-black"
            style={{ aspectRatio: "3/4" }}
          >
            <div className="flex flex-col items-center gap-3">
              <Loader2 className="h-10 w-10 animate-spin text-blue-400" />
              <p className="text-sm text-zinc-300">{statusText}</p>
            </div>
          </div>
        )}
      </div>

      {/* Status text */}
      <div className="flex flex-col items-center gap-2 text-center">
        {scanState === "scanning" && (
          <div className="flex items-center gap-2 text-sm text-zinc-400">
            <Camera className="h-4 w-4" />
            {statusText}
          </div>
        )}

        {scanState === "capturing" && (
          <div className="flex items-center gap-2 text-sm text-blue-400">
            <Loader2 className="h-4 w-4 animate-spin" />
            {statusText}
          </div>
        )}

        {scanState === "verifying" && (
          <div className="flex items-center gap-2 text-sm text-blue-400">
            <Loader2 className="h-4 w-4 animate-spin" />
            {statusText}
          </div>
        )}

        {scanState === "passed" && (
          <div className="flex items-center gap-2 rounded-full bg-green-600 px-5 py-2 text-sm font-medium text-white">
            <CheckCircle className="h-4 w-4" />
            Face verified
          </div>
        )}

        {scanState === "failed" && !camError && (
          <div className="flex flex-col items-center gap-3">
            <div className="flex items-center gap-2 rounded-full bg-red-600 px-5 py-2 text-sm font-medium text-white">
              <XCircle className="h-4 w-4" />
              Verification failed
            </div>
            <button
              onClick={handleRetry}
              className="rounded-full bg-blue-600 px-6 py-2.5 text-sm font-medium text-white hover:bg-blue-500"
            >
              Try Again
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
