"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useCamera } from "@/hooks/useCamera";
import { canvasToJpegBlob, grabFrame } from "@/lib/frameEncoder";
import { API_BASE, getWsUrl } from "@/lib/apiBase";
import clsx from "clsx";
import { CheckCircle, Loader2, XCircle, Shield } from "lucide-react";

interface LivenessStepProps {
  token: string;
  sessionId: string;
  frontCaptureId?: string;
  onComplete: (passed: boolean) => void;
}

interface LivenessFrame {
  face_detected: boolean;
  instruction: string;
  passed: boolean;
  failed: boolean;
  progress?: number;
  selfie_ready?: boolean;
}

export default function LivenessStep({ token, sessionId, frontCaptureId, onComplete }: LivenessStepProps) {
  const { videoRef, isReady, error: camError, start, stop } = useCamera({
    facingMode: "user",
    width: 1280,
    height: 720,
  });

  const [state, setState] = useState("collecting");
  const [progress, setProgress] = useState(0);
  const [passed, setPassed] = useState(false);
  const [failed, setFailed] = useState(false);
  const [faceDetected, setFaceDetected] = useState(false);
  const [instruction, setInstruction] = useState("Look at the camera");
  const [finalizing, setFinalizing] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const loopRef = useRef<number | null>(null);

  // Start camera on mount
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

  // Connect WebSocket when camera is ready
  useEffect(() => {
    if (!isReady || !sessionId) return;

    const wsUrl = getWsUrl(`/ws/liveness/${sessionId}`);
    const ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      setInstruction("Look at the camera");
    };

    ws.onmessage = (event) => {
      try {
        const data: LivenessFrame = JSON.parse(event.data);
        setFaceDetected(data.face_detected);
        setInstruction(data.instruction);
        setProgress(data.progress ?? 0);

        if (data.passed) {
          setPassed(true);
          ws.close();
          if (loopRef.current) cancelAnimationFrame(loopRef.current);
          // Call finalize endpoint
          setFinalizing(true);
          const formData = new FormData();
          if (frontCaptureId) {
            formData.append("front_capture_id", frontCaptureId);
          }
          fetch(`${API_BASE}/api/kyc/finalize/${sessionId}`, {
            method: "POST",
            headers: {
              Authorization: `Bearer ${token}`,
            },
            body: formData,
          })
            .then((res) => res.json())
            .then((result) => {
              setFinalizing(false);
              onComplete(result.kyc_passed);
            })
            .catch(() => {
              setFinalizing(false);
              onComplete(false);
            });
        }

        if (data.failed) {
          setFailed(true);
          ws.close();
          if (loopRef.current) cancelAnimationFrame(loopRef.current);
        }
      } catch {
        // ignore
      }
    };

    ws.onerror = () => {
      setInstruction("Connection error — retrying...");
    };

    ws.onclose = () => {
      if (!passed && !failed) {
        setInstruction("Connection closed");
      }
    };

    wsRef.current = ws;

    return () => {
      ws.close();
    };
  }, [isReady, sessionId, token, onComplete, passed, failed]);

  // Frame sending loop — 10fps
  useEffect(() => {
    if (!isReady || passed || failed) return;

    let lastSend = 0;
    const INTERVAL = 100; // 10fps

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
        canvasToJpegBlob(canvasRef.current, 0.75).then((blob) => {
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
  }, [isReady, passed, failed, videoRef]);

  const getBorderColor = () => {
    if (passed) return "border-green-400 bg-green-500/20";
    if (failed) return "border-red-400 bg-red-500/20";
    if (!faceDetected) return "border-white/40";
    return "border-yellow-400";
  };

  const showProgress = state === "collecting" && !passed && !failed;

  return (
    <div className="flex flex-col items-center gap-4">
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

      {/* Video with face oval overlay */}
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
          style={{ aspectRatio: "3 / 4", transform: "scaleX(-1)" }}
        />

        {/* Face oval guide */}
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <div
            className={clsx(
              "h-56 w-56 rounded-full border-[3px] transition-colors duration-300",
              getBorderColor()
            )}
          >
            {passed && (
              <div className="flex h-full items-center justify-center">
                <CheckCircle className="h-16 w-16 text-green-400 animate-scaleIn" />
              </div>
            )}
            {failed && (
              <div className="flex h-full items-center justify-center">
                <XCircle className="h-16 w-16 text-red-400 animate-scaleIn" />
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Challenge prompt */}
      <div className="flex flex-col items-center gap-2">
        {showProgress && (
          <div className="flex w-full max-w-xs flex-col items-center gap-2">
            <div className="flex items-center gap-2 text-sm font-medium text-white">
              <Shield className="h-5 w-5 text-blue-400" />
              <span>{instruction}</span>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-zinc-700">
              <div
                className="h-full rounded-full bg-blue-500 transition-all duration-300"
                style={{ width: `${Math.min(100, progress)}%` }}
              />
            </div>
            <span className="text-xs text-zinc-400">
              {Math.min(100, progress)}%
            </span>
          </div>
        )}

        {passed && (
          <div className="flex items-center gap-2 rounded-full bg-green-600 px-5 py-2.5 text-sm font-medium text-white">
            <CheckCircle className="h-5 w-5" />
            <span>Liveness passed — finalizing...</span>
          </div>
        )}

        {failed && (
          <div className="flex items-center gap-2 rounded-full bg-red-600 px-5 py-2.5 text-sm font-medium text-white">
            <XCircle className="h-5 w-5" />
            <span>Liveness failed — please retry</span>
          </div>
        )}

        {finalizing && (
          <div className="flex items-center gap-2 text-sm text-zinc-500">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span>Running face match...</span>
          </div>
        )}

        <p className="text-center text-sm text-zinc-500 dark:text-zinc-400">
          {instruction}
        </p>

        {!faceDetected && !passed && !failed && isReady && (
          <p className="text-xs text-yellow-500">
            No face detected — look at the camera
          </p>
        )}
      </div>
    </div>
  );
}
