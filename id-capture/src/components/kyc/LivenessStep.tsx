"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { canvasToJpegBlob, grabFrame } from "@/lib/frameEncoder";
import { API_BASE, getWsUrl } from "@/lib/apiBase";
import clsx from "clsx";
import {
  CheckCircle,
  Loader2,
  XCircle,
  Shield,
  Eye,
  EyeOff,
  ArrowLeft,
  ArrowRight,
  Smile,
} from "lucide-react";

interface LivenessStepProps {
  token: string;
  sessionId: string;
  frontCaptureId?: string;
  onComplete: (passed: boolean) => void;
}

interface LivenessResponse {
  passed: boolean;
  failed: boolean;
  instruction: string;
  face_detected?: boolean;
  spoof_detected?: boolean;
  selfie_ready?: boolean;
  liveness_score?: number;
  progress?: number;
  selfie_url?: string | null;
  face_bbox?: number[] | null;
  face_landmarks?: {x: number; y: number}[] | null;
}

type LivenessState = "connecting" | "calibrating" | "running" | "passed" | "failed";


function FaceTrackingOverlay({
  landmarks,
  bbox,
  canvasRef,
  videoRef,
  visible,
}: {
  landmarks: { x: number; y: number }[] | null;
  bbox: number[] | null;
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  visible: boolean;
}) {
  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || !visible) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw face bounding box
    if (bbox && bbox.length === 4) {
      const [x, y, w, h] = bbox;
      const sx = canvas.width / video.videoWidth;
      const sy = canvas.height / video.videoHeight;
      const rx = (video.videoWidth - (x + w)) * sx;  // mirror for selfie view
      const ry = y * sy;
      const rw = w * sx;
      const rh = h * sy;

      ctx.strokeStyle = "rgba(74, 222, 128, 0.6)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.roundRect(rx, ry, rw, rh, 12);
      ctx.stroke();
    }

    // Draw landmarks
    if (landmarks) {
      const sx = canvas.width / video.videoWidth;
      const sy = canvas.height / video.videoHeight;
      ctx.fillStyle = "rgba(147, 197, 253, 0.8)";
      for (const lm of landmarks) {
        const lx = (video.videoWidth - lm.x) * sx;  // mirror
        const ly = lm.y * sy;
        ctx.beginPath();
        ctx.arc(lx, ly, 2, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }, [landmarks, bbox, visible, canvasRef, videoRef]);

  if (!visible) return null;
  return null;
}

export default function LivenessStep({
  token,
  sessionId,
  frontCaptureId,
  onComplete,
}: LivenessStepProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animRef = useRef<number | null>(null);
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;

  const [livenessState, setLivenessState] =
    useState<LivenessState>("connecting");
  const [instruction, setInstruction] = useState("انظر إلى الكاميرا");
  const [faceDetected, setFaceDetected] = useState(false);
  const [finalizing, setFinalizing] = useState(false);
  const [camError, setCamError] = useState<string | null>(null);
  const [selfieUrl, setSelfieUrl] = useState<string | null>(null);
  const [faceLandmarks, setFaceLandmarks] = useState<{x: number; y: number}[] | null>(null);
  const [progress, setProgress] = useState(0);
  const [progressNeeded, setProgressNeeded] = useState(2);
  const [faceBBox, setFaceBBox] = useState<number[] | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);

  // ── Single unified effect: camera → websocket → frames → cleanup ─
  useEffect(() => {
    if (!sessionId) return;

    let stopped = false;
    const canvas = document.createElement("canvas");
    canvasRef.current = canvas;

    const cleanup = () => {
      stopped = true;
      wsRef.current?.close();
      wsRef.current = null;
      if (animRef.current) cancelAnimationFrame(animRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    };

    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "user",
            width: { ideal: 640 },
            height: { ideal: 480 },
          },
          audio: false,
        });
        if (stopped) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;

        // Need a video element for the stream
        let video = videoRef.current;
        if (!video) {
          video = document.createElement("video");
          video.setAttribute("playsinline", "");
          videoRef.current = video;
        }
        video.srcObject = stream;
        await video.play();

        if (stopped) return;

        // ── Now open WebSocket ──
        const wsUrl = getWsUrl(`/ws/liveness/${sessionId}`);
        const ws = new WebSocket(wsUrl);
        ws.binaryType = "arraybuffer";
        wsRef.current = ws;

        ws.onopen = () => {
          if (stopped) {
            ws.close();
            return;
          }
          setLivenessState("running");
          setInstruction("انظر إلى الكاميرا");
          startFrameLoop(video!, canvas, ws);
        };

        ws.onmessage = (event) => {
          if (stopped) return;
          try {
            const data: LivenessResponse = JSON.parse(event.data);
            setFaceDetected(data.face_detected ?? true);
            setInstruction(data.instruction);
            if (data.face_landmarks) setFaceLandmarks(data.face_landmarks);
            setProgress(data.progress ?? 0);
            if (data.selfie_url) setSelfieUrl(data.selfie_url);
            setProgressNeeded(100);
            if (data.face_bbox) setFaceBBox(data.face_bbox);

            setLivenessState("running");
            if (data.passed) {
              setLivenessState("passed");
              cleanup();
              if (frontCaptureId) {
                setFinalizing(true);
                const fd = new FormData();
                fd.append("front_capture_id", frontCaptureId);
                fetch(`${API_BASE}/api/kyc/finalize/${sessionId}`, {
                  method: "POST",
                  headers: { Authorization: `Bearer ${token}` },
                  body: fd,
                })
                  .then((r) => r.json())
                  .then((res) => onCompleteRef.current(res.kyc_passed))
                  .catch(() => onCompleteRef.current(false))
                  .finally(() => setFinalizing(false));
              } else {
                onCompleteRef.current(true);
              }
              return;
            }
            if (data.failed) {
              setLivenessState("failed");
              cleanup();
            }
          } catch {
            /* ignore */
          }
        };

        ws.onerror = () => setInstruction("خطأ في الاتصال");
        ws.onclose = () => {
          if (!stopped && livenessState === "running")
            setInstruction("انقطع الاتصال");
        };
      } catch (err) {
        if (!stopped) {
          setCamError(
            err instanceof Error ? err.message : "Camera access denied",
          );
        }
      }
    };

    startCamera();
    return cleanup;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  // ── Frame loop ──────────────────────────────────────────────
  const startFrameLoop = (
    video: HTMLVideoElement,
    canvas: HTMLCanvasElement,
    ws: WebSocket,
  ) => {
    let lastSend = 0;
    const INTERVAL = 100;

    const loop = () => {
      if (ws.readyState !== WebSocket.OPEN) return;
      const now = Date.now();
      if (now - lastSend >= INTERVAL && video.readyState >= 2) {
        lastSend = now;
        grabFrame(video, canvas);
        canvasToJpegBlob(canvas, 0.7).then((blob) => {
          blob.arrayBuffer().then((buf) => {
            if (ws.readyState === WebSocket.OPEN) ws.send(buf);
          });
        });
      }
      animRef.current = requestAnimationFrame(loop);
    };

    animRef.current = requestAnimationFrame(loop);
  };

  // ── Retry handler ───────────────────────────────────────────
  const handleRetry = useCallback(async () => {
    await fetch(`${API_BASE}/api/liveness/reset/${sessionId}`, {
      method: "POST",
    });
    window.location.reload();
  }, [sessionId]);

  // ── Gesture icon ────────────────────────────────────────────
  const getGestureIcon = () => {
    const txt = instruction;
    if (txt.includes("أغمض") || txt.includes("blink") || txt.includes("Blink"))
      return <EyeOff className="h-8 w-8 animate-pulse" />;
    if (txt.includes("يسار") || txt.includes("left") || txt.includes("Left"))
      return <ArrowLeft className="h-8 w-8 animate-bounce" />;
    if (txt.includes("يمين") || txt.includes("right") || txt.includes("Right"))
      return <ArrowRight className="h-8 w-8 animate-bounce" />;
    if (txt.includes("ابتسم") || txt.includes("smile") || txt.includes("Smile"))
      return <Smile className="h-8 w-8 animate-pulse" />;
    return <Eye className="h-8 w-8" />;
  };

  const getBorderColor = () => {
    switch (livenessState) {
      case "passed":
        return "border-green-400 bg-green-500/10";
      case "failed":
      case "spoof":
        return "border-red-400 bg-red-500/10";
      case "running":
        return faceDetected ? "border-yellow-400" : "border-white/30";
      default:
        return "border-white/20";
    }
  };

  return (
    <div className="flex flex-col items-center gap-6">
      {camError && (
        <div className="flex flex-col items-center gap-3 p-8 text-center">
          <p className="text-sm text-red-400">{camError}</p>
          <button
            onClick={handleRetry}
            className="rounded-full bg-blue-600 px-4 py-2 text-sm text-white"
          >
            Retry
          </button>
        </div>
      )}

      <div
        className="relative w-full overflow-hidden rounded-2xl bg-black"
        style={{ maxWidth: 400 }}
      >
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="h-full w-full object-cover"
          style={{ aspectRatio: "3/4", transform: "scaleX(-1)" }}
        />
        {/* Face tracking canvas overlay */}
        <canvas
          ref={overlayRef}
          className="pointer-events-none absolute inset-0 h-full w-full"
          style={{ transform: "scaleX(-1)" }}
        />
        <FaceTrackingOverlay
          landmarks={faceLandmarks}
          bbox={faceBBox}
          canvasRef={overlayRef}
          videoRef={videoRef}
          visible={livenessState === "running" || livenessState === "calibrating"}
        />
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <div
            className={clsx(
              "flex h-56 w-56 items-center justify-center rounded-full border-[3px] transition-colors duration-500",
              getBorderColor(),
            )}
          >
            {livenessState === "passed" && selfieUrl && (
            <div className="mb-4 overflow-hidden rounded-xl border-2 border-green-400">
              <img
                src={`${API_BASE}${selfieUrl}`}
                alt="Selfie"
                className="h-48 w-full object-cover"
              />
            </div>
          )}
          {livenessState === "passed" && (
              <CheckCircle className="h-16 w-16 animate-scaleIn text-green-400" />
            )}
            {(livenessState === "failed" || livenessState === "spoof") && (
              <XCircle className="h-16 w-16 animate-scaleIn text-red-400" />
            )}
          </div>
        </div>
      </div>

      <div className="flex flex-col items-center gap-3 text-center">
        {livenessState === "running" && (
          <>
            <div className="flex items-center gap-2 text-lg font-medium text-white">
              {getGestureIcon()}
              <span>{instruction}</span>
            </div>
            {livenessState === "running" && progressNeeded > 1 && (
              <div className="h-1.5 w-48 overflow-hidden rounded-full bg-zinc-700">
                <div
                  className="h-full rounded-full bg-green-400 transition-all duration-200"
                  style={{ width: `${Math.min(100, (progress / progressNeeded) * 100)}%` }}
                />
              </div>
            )}
            {!faceDetected && (
              <p className="text-xs text-yellow-500">
                وجهك غير ظاهر — انظر إلى الكاميرا
              </p>
            )}
          </>
        )}

        {livenessState === "spoof" && (
          <>
            <div className="flex items-center gap-2 rounded-full bg-red-600 px-5 py-2.5 text-sm font-medium text-white">
              <Shield className="h-5 w-5" />
              <span>تم اكتشاف محاولة احتيال</span>
            </div>
            <p className="text-sm text-zinc-400">{instruction}</p>
          </>
        )}

        {livenessState === "passed" && selfieUrl && (
            <div className="mb-4 overflow-hidden rounded-xl border-2 border-green-400">
              <img
                src={`${API_BASE}${selfieUrl}`}
                alt="Selfie"
                className="h-48 w-full object-cover"
              />
            </div>
          )}
          {livenessState === "passed" && (
          <>
            <div className="flex items-center gap-2 rounded-full bg-green-600 px-5 py-2.5 text-sm font-medium text-white">
              <CheckCircle className="h-5 w-5" />
              <span>تم التحقق — جاري التأكيد...</span>
            </div>
            {finalizing && (
              <div className="flex items-center gap-2 text-sm text-zinc-400">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span>مطابقة الوجه...</span>
              </div>
            )}
          </>
        )}

        {(livenessState === "failed" || livenessState === "spoof") && (
          <button
            onClick={handleRetry}
            className="rounded-full bg-blue-600 px-6 py-2.5 text-sm font-medium text-white hover:bg-blue-500"
          >
            إعادة المحاولة
          </button>
        )}

        {livenessState === "calibrating" && (
          <div className="flex items-center gap-2 text-sm text-blue-400">
            <Loader2 className="h-4 w-4 animate-spin" /><span>جاري المعايرة...</span>
          </div>
        )}

        {livenessState === "connecting" && (
          <div className="flex items-center gap-2 text-sm text-zinc-400">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span>جاري الاتصال...</span>
          </div>
        )}
      </div>
    </div>
  );
}
