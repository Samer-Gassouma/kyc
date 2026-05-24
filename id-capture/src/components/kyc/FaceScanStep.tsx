"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { canvasToJpegBlob, grabFrame } from "@/lib/frameEncoder";
import { getWsUrl } from "@/lib/apiBase";
import clsx from "clsx";
import { CheckCircle, Loader2, XCircle, Camera } from "lucide-react";

interface FaceScanStepProps {
  token: string;
  sessionId: string;
  onComplete: (passed: boolean) => void;
}

interface ScanResponse {
  passed: boolean;
  failed: boolean;
  face_detected: boolean;
  liveness_score: number;
  progress: number;
  face_bbox: number[] | null;
}

type ScanState = "connecting" | "running" | "passed" | "failed";

export default function FaceScanStep({
  token,
  sessionId,
  onComplete,
}: FaceScanStepProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animRef = useRef<number | null>(null);
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;

  const [scanState, setScanState] = useState<ScanState>("connecting");
  const [faceDetected, setFaceDetected] = useState(false);
  const [progress, setProgress] = useState(0);
  const [camError, setCamError] = useState<string | null>(null);

  const cleanup = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    if (animRef.current) cancelAnimationFrame(animRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }, []);

  // Start camera + WebSocket
  useEffect(() => {
    if (!sessionId) return;

    let stopped = false;
    const canvas = document.createElement("canvas");

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

        let video = videoRef.current;
        if (!video) {
          video = document.createElement("video");
          video.setAttribute("playsinline", "");
          videoRef.current = video;
        }
        video.srcObject = stream;
        await video.play();
        if (stopped) return;

        // Connect WebSocket
        const wsUrl = getWsUrl(`/ws/face-scan/${sessionId}`);
        const ws = new WebSocket(wsUrl);
        ws.binaryType = "arraybuffer";
        wsRef.current = ws;

        ws.onopen = () => {
          if (stopped) {
            ws.close();
            return;
          }
          setScanState("running");

          // Frame loop ~10fps
          let lastSend = 0;
          const loop = () => {
            if (ws.readyState !== WebSocket.OPEN) return;
            const now = Date.now();
            if (now - lastSend >= 100) {
              lastSend = now;
              grabFrame(video!, canvas);
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

        ws.onmessage = (event) => {
          if (stopped) return;
          try {
            const data: ScanResponse = JSON.parse(event.data);
            setFaceDetected(data.face_detected);
            setProgress(data.progress ?? 0);

            if (data.passed) {
              setScanState("passed");
              cleanup();
              onCompleteRef.current(true);
            } else if (data.failed) {
              setScanState("failed");
              cleanup();
            }
          } catch {
            /* ignore */
          }
        };

        ws.onclose = () => {
          if (!stopped && scanState === "connecting") {
            setCamError("Connection failed");
            setScanState("failed");
          }
        };

        ws.onerror = () => {
          if (!stopped) setCamError("Connection error");
        };
      } catch (err) {
        if (!stopped) {
          setCamError(
            err instanceof Error ? err.message : "Camera access denied"
          );
          setScanState("failed");
        }
      }
    };

    startCamera();
    return () => {
      stopped = true;
      cleanup();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  const handleRetry = () => {
    window.location.reload();
  };

  return (
    <div className="flex flex-col items-center gap-4">
      {camError && (
        <div className="flex flex-col items-center gap-3 p-4 text-center">
          <p className="text-sm text-red-400">{camError}</p>
          <button
            onClick={handleRetry}
            className="rounded-full bg-blue-600 px-4 py-2 text-sm text-white"
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
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="h-full w-full object-cover"
          style={{ aspectRatio: "3/4", transform: "scaleX(-1)" }}
        />

        {/* Face guide overlay */}
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <div
            className={clsx(
              "flex h-56 w-56 items-center justify-center rounded-full border-[3px] transition-colors duration-500",
              scanState === "passed" && "border-green-400 bg-green-500/10",
              scanState === "failed" && "border-red-400 bg-red-500/10",
              scanState === "running" && faceDetected && "border-blue-400",
              scanState === "running" && !faceDetected && "border-white/20",
              scanState === "connecting" && "border-white/10"
            )}
          >
            {scanState === "passed" && (
              <CheckCircle className="h-16 w-16 text-green-400" />
            )}
            {scanState === "failed" && (
              <XCircle className="h-16 w-16 text-red-400" />
            )}
          </div>
        </div>
      </div>

      {/* Status + progress */}
      <div className="flex flex-col items-center gap-2 text-center">
        {scanState === "connecting" && (
          <div className="flex items-center gap-2 text-sm text-zinc-400">
            <Loader2 className="h-4 w-4 animate-spin" />
            Starting camera...
          </div>
        )}

        {scanState === "running" && (
          <>
            <div className="flex items-center gap-2 text-sm text-white">
              <Camera className="h-4 w-4" />
              {!faceDetected
                ? "Position your face in the circle"
                : progress < 100
                ? "Hold still..."
                : "Verifying..."}
            </div>
            <div className="h-1.5 w-48 overflow-hidden rounded-full bg-zinc-700">
              <div
                className="h-full rounded-full bg-blue-400 transition-all duration-200"
                style={{ width: `${progress}%` }}
              />
            </div>
          </>
        )}

        {scanState === "passed" && (
          <div className="flex items-center gap-2 rounded-full bg-green-600 px-5 py-2 text-sm font-medium text-white">
            <CheckCircle className="h-4 w-4" />
            Face verified
          </div>
        )}

        {scanState === "failed" && (
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
