"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useCamera } from "@/hooks/useCamera";
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
  spoof_score?: number;
  selfie_ready?: boolean;
}

type LivenessState = "connecting" | "running" | "passed" | "failed" | "spoof";

export default function LivenessStep({
  token,
  sessionId,
  frontCaptureId,
  onComplete,
}: LivenessStepProps) {
  const {
    videoRef,
    isReady,
    error: camError,
    start,
    stop,
  } = useCamera({
    facingMode: "user",
    width: 640,
    height: 480,
  });

  const [livenessState, setLivenessState] =
    useState<LivenessState>("connecting");
  const [instruction, setInstruction] = useState("انظر إلى الكاميرا");
  const [faceDetected, setFaceDetected] = useState(false);
  const [finalizing, setFinalizing] = useState(false);
  const [spoofScore, setSpoofScore] = useState<number>(1.0);
  const [reconnectKey, setReconnectKey] = useState(0);
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;

  const wsRef = useRef<WebSocket | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const loopRef = useRef<number | null>(null);

  // Start camera
  useEffect(() => {
    canvasRef.current = document.createElement("canvas");
    start();
    return () => {
      stop();
      wsRef.current?.close();
      if (loopRef.current) cancelAnimationFrame(loopRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Connect WebSocket when camera ready
  useEffect(() => {
    if (!isReady || !sessionId) return;

    const wsUrl = getWsUrl(`/ws/liveness/${sessionId}`);
    const ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onopen = () => {
      setLivenessState("running");
      setInstruction("انظر إلى الكاميرا");
    };

    ws.onmessage = (event) => {
      try {
        const data: LivenessResponse = JSON.parse(event.data);

        setFaceDetected(data.face_detected ?? true);
        setInstruction(data.instruction);
        if (data.spoof_score !== undefined) setSpoofScore(data.spoof_score);

        if (data.spoof_detected) {
          setLivenessState("spoof");
          ws.close();
          if (loopRef.current) cancelAnimationFrame(loopRef.current);
          return;
        }

        if (data.passed) {
          setLivenessState("passed");
          ws.close();
          if (loopRef.current) cancelAnimationFrame(loopRef.current);

          setFinalizing(true);
          const formData = new FormData();
          if (frontCaptureId)
            formData.append("front_capture_id", frontCaptureId);
          fetch(`${API_BASE}/api/kyc/finalize/${sessionId}`, {
            method: "POST",
            headers: { Authorization: `Bearer ${token}` },
            body: formData,
          })
            .then((r) => r.json())
            .then((result) => {
              setFinalizing(false);
              onCompleteRef.current(result.kyc_passed);
            })
            .catch(() => {
              setFinalizing(false);
              onCompleteRef.current(false);
            });
        }

        if (data.failed) {
          setLivenessState("failed");
          ws.close();
          if (loopRef.current) cancelAnimationFrame(loopRef.current);
        }
      } catch {
        // ignore parse errors
      }
    };

    ws.onerror = () => setInstruction("خطأ في الاتصال — إعادة المحاولة...");
    ws.onclose = () => {
      if (livenessState === "running") setInstruction("انقطع الاتصال");
    };

    return () => {
      ws.close();
    };
  }, [isReady, sessionId, token, reconnectKey]);

  // Frame sending loop — 10fps
  useEffect(() => {
    if (!isReady || livenessState !== "running") return;

    let lastSend = 0;
    const INTERVAL = 100;

    const loop = () => {
      const now = Date.now();
      if (
        now - lastSend >= INTERVAL &&
        wsRef.current?.readyState === WebSocket.OPEN &&
        videoRef.current &&
        canvasRef.current
      ) {
        lastSend = now;
        grabFrame(videoRef.current, canvasRef.current);
        canvasToJpegBlob(canvasRef.current, 0.7).then((blob) => {
          blob.arrayBuffer().then((buf) => {
            if (wsRef.current?.readyState === WebSocket.OPEN) {
              wsRef.current.send(buf);
            }
          });
        });
      }
      loopRef.current = requestAnimationFrame(loop);
    };

    loopRef.current = requestAnimationFrame(loop);
    return () => {
      if (loopRef.current) cancelAnimationFrame(loopRef.current);
    };
  }, [isReady, livenessState, videoRef]);

  // ── Gesture icon ────────────────────────────────────────────
  const getGestureIcon = () => {
    const txt = instruction.toLowerCase();
    if (txt.includes("اغمض") || txt.includes("blink"))
      return <EyeOff className="h-8 w-8 animate-pulse" />;
    if (txt.includes("يسار") || txt.includes("left"))
      return <ArrowLeft className="h-8 w-8 animate-bounce" />;
    if (txt.includes("يمين") || txt.includes("right"))
      return <ArrowRight className="h-8 w-8 animate-bounce" />;
    if (txt.includes("ابتسم") || txt.includes("smile"))
      return <Smile className="h-8 w-8 animate-pulse" />;
    return <Eye className="h-8 w-8" />;
  };

  // ── Border color ────────────────────────────────────────────
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
      {/* Camera error */}
      {camError && (
        <div className="flex flex-col items-center gap-3 p-8 text-center">
          <p className="text-sm text-red-400">{camError}</p>
          <button
            onClick={start}
            className="rounded-full bg-blue-600 px-4 py-2 text-sm text-white"
          >
            Retry Camera
          </button>
        </div>
      )}

      {/* Video + face oval */}
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

        {/* Face oval guide */}
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <div
            className={clsx(
              "flex h-56 w-56 items-center justify-center rounded-full border-[3px] transition-colors duration-500",
              getBorderColor(),
            )}
          >
            {livenessState === "passed" && (
              <CheckCircle className="h-16 w-16 animate-scaleIn text-green-400" />
            )}
            {(livenessState === "failed" || livenessState === "spoof") && (
              <XCircle className="h-16 w-16 animate-scaleIn text-red-400" />
            )}
          </div>
        </div>
      </div>

      {/* Status area */}
      <div className="flex flex-col items-center gap-3 text-center">
        {/* Running: show instruction + gesture icon */}
        {livenessState === "running" && (
          <>
            <div className="flex items-center gap-2 text-lg font-medium text-white">
              {getGestureIcon()}
              <span>{instruction}</span>
            </div>
            {!faceDetected && (
              <p className="text-xs text-yellow-500">
                وجهك غير ظاهر — انظر إلى الكاميرا
              </p>
            )}
          </>
        )}

        {/* Spoof detected */}
        {livenessState === "spoof" && (
          <>
            <div className="flex items-center gap-2 rounded-full bg-red-600 px-5 py-2.5 text-sm font-medium text-white">
              <Shield className="h-5 w-5" />
              <span>تم اكتشاف محاولة احتيال</span>
            </div>
            <p className="text-sm text-zinc-400">{instruction}</p>
          </>
        )}

        {/* Passed */}
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

        {/* Failed */}
        {(livenessState === "failed" || livenessState === "spoof") && (
          <button
            onClick={async () => {
              await fetch(`${API_BASE}/api/liveness/reset/${sessionId}`, {
                method: "POST",
              });
              setReconnectKey((k) => k + 1);
              setLivenessState("running");
              setInstruction("انظر إلى الكاميرا");
              start();
            }}
            className="rounded-full bg-blue-600 px-6 py-2.5 text-sm font-medium text-white hover:bg-blue-500"
          >
            إعادة المحاولة
          </button>
        )}

        {/* Connecting */}
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
