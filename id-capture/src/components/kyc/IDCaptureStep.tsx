"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ImagePlus } from "lucide-react";
import { useCamera } from "@/hooks/useCamera";
import { useONNXQuality } from "@/hooks/useONNXQuality";
import { useIDWebSocket } from "@/hooks/useIDWebSocket";
import { useAutoCapture } from "@/hooks/useAutoCapture";
import { canvasToJpegBlob, grabFrame } from "@/lib/frameEncoder";
import { API_BASE } from "@/lib/apiBase";
import CameraOverlay, { BorderState } from "./CameraOverlay";
import QualityIndicator from "./QualityIndicator";
import CaptureReview from "./CaptureReview";

interface IDCaptureStepProps {
  side: "front" | "back";
  token: string;
  onCaptureComplete: (captureId: string, blob: Blob) => void;
}

type Phase = "camera" | "preview" | "review";

export default function IDCaptureStep({
  side,
  token,
  onCaptureComplete,
}: IDCaptureStepProps) {
  const { videoRef, isReady, error: camError, start, stop } = useCamera({
    facingMode: "environment",
  });
  const { quality, runCheck } = useONNXQuality();
  const {
    detection,
    isConnected,
    connect,
    disconnect,
    sendFrame,
  } = useIDWebSocket();

  const [phase, setPhase] = useState<Phase>("camera");
  const [capturedImageUrl, setCapturedImageUrl] = useState<string>("");
  const [cropImageUrl, setCropImageUrl] = useState<string>("");
  const [capturedBlob, setCapturedBlob] = useState<Blob | null>(null);
  const [reviewStatus, setReviewStatus] = useState<
    "preview" | "validating" | "success" | "failed"
  >("preview");
  const [rejectionReason, setRejectionReason] = useState<string | null>(null);
  const [videoSize, setVideoSize] = useState<{ w: number; h: number }>({ w: 0, h: 0 });

  const loopRef = useRef<number | null>(null);
  const fullResCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Start camera + WebSocket on mount
  useEffect(() => {
    start();
    connect();
    fullResCanvasRef.current = document.createElement("canvas");

    return () => {
      stop();
      disconnect();
      if (loopRef.current) cancelAnimationFrame(loopRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Main processing loop — send frames to YOLO via WS
  useEffect(() => {
    if (!isReady || phase !== "camera") return;

    const loop = async () => {
      const video = videoRef.current;
      if (!video) {
        loopRef.current = requestAnimationFrame(loop);
        return;
      }

      if (isConnected) {
        await sendFrame(video);
      }

      loopRef.current = requestAnimationFrame(loop);
    };

    loopRef.current = requestAnimationFrame(loop);

    return () => {
      if (loopRef.current) cancelAnimationFrame(loopRef.current);
    };
  }, [isReady, phase, isConnected, videoRef, runCheck, sendFrame]);

  // Pause camera when leaving camera phase
  const pauseCamera = useCallback(() => {
    stop();
    disconnect();
    if (loopRef.current) cancelAnimationFrame(loopRef.current);
  }, [stop, disconnect]);

  // ── Capture: grab frame, show preview ─────────────────────────
  const doCapture = useCallback(
    async (blob: Blob) => {
      const url = URL.createObjectURL(blob);
      setCapturedBlob(blob);
      setCapturedImageUrl(url);
      setCropImageUrl("");
      setRejectionReason(null);
      setReviewStatus("preview");
      setPhase("preview");
      pauseCamera();
    },
    [pauseCamera]
  );

  // ── Manual capture (button press) ─────────────────────────────
  const handleManualCapture = useCallback(async () => {
    const video = videoRef.current;
    const canvas = fullResCanvasRef.current;
    if (!video || !canvas) return;

    grabFrame(video, canvas);
    const blob = await canvasToJpegBlob(canvas, 0.95);
    await doCapture(blob);
  }, [videoRef, doCapture]);

  // ── Auto-capture (hold still 1.5s) ────────────────────────────
  const readyToCapture = detection?.ready_to_capture ?? false;
  const { isHolding, holdProgress, reset: resetAutoCapture } = useAutoCapture(
    readyToCapture && phase === "camera",
    handleManualCapture
  );

  // ── User clicked "Use This" → validate ────────────────────────
  const handleProceed = useCallback(async () => {
    if (!capturedBlob) return;

    setPhase("review");
    setReviewStatus("validating");

    try {
      const formData = new FormData();
      formData.append("file", capturedBlob, `${side}_capture.jpg`);
      formData.append("side", side);

      const res = await fetch(`${API_BASE}/api/capture/validate`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const result = await res.json();

      if (result.validation_passed) {
        setReviewStatus("success");
        if (result.crop_base64) {
          setCropImageUrl(`data:image/jpeg;base64,${result.crop_base64}`);
        }
        setTimeout(() => onCaptureComplete(result.capture_id, capturedBlob), 1200);
      } else {
        setReviewStatus("failed");
        setRejectionReason(result.rejection_reason);
        if (result.crop_base64) {
          setCropImageUrl(`data:image/jpeg;base64,${result.crop_base64}`);
        }
      }
    } catch (err) {
      setReviewStatus("failed");
      setRejectionReason(
        err instanceof Error ? err.message : "Validation request failed"
      );
    }
  }, [capturedBlob, side, token, onCaptureComplete]);

  // ── Gallery upload → preview, then validate ───────────────────
  const submitGallery = useCallback(
    async (file: File) => {
      const url = URL.createObjectURL(file);
      setCapturedBlob(file);
      setCapturedImageUrl(url);
      setCropImageUrl("");
      setRejectionReason(null);
      setReviewStatus("preview");
      setPhase("preview");
      pauseCamera();
    },
    [pauseCamera]
  );

  const handleGallerySelect = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      await submitGallery(file);
      if (fileInputRef.current) fileInputRef.current.value = "";
    },
    [submitGallery]
  );

  // ── Retry: back to camera ─────────────────────────────────────
  const handleRetry = useCallback(() => {
    setPhase("camera");
    setCapturedImageUrl("");
    setCropImageUrl("");
    setCapturedBlob(null);
    setReviewStatus("preview");
    setRejectionReason(null);
    resetAutoCapture();
    start();
    connect();
  }, [start, connect, resetAutoCapture]);

  // ── Border state from detection result ────────────────────────
  const getBorderState = (): BorderState => {
    if (!detection || !detection.detected) return "neutral";
    if (detection.ready_to_capture) return "success";
    if (detection.issues.length > 2 || detection.confidence < 0.6) return "error";
    return "warning";
  };

  const sideLabel = side === "front" ? "Front of ID" : "Back of ID";

  // ── Preview / Review phase ────────────────────────────────────
  if (phase === "preview" || phase === "review") {
    return (
      <CaptureReview
        imageUrl={capturedImageUrl}
        cropImageUrl={cropImageUrl}
        status={reviewStatus}
        rejectionReason={rejectionReason}
        onRetry={handleRetry}
        onProceed={handleProceed}
      />
    );
  }

  // ── Camera phase ──────────────────────────────────────────────
  return (
    <div className="flex w-full flex-col items-center">
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

      {/* Video + overlay */}
      <div
        className="relative w-full overflow-hidden bg-black"
        style={{ maxWidth: 480, aspectRatio: "3 / 4" }}
      >
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="h-full w-full object-cover"
          onLoadedMetadata={(e) => {
            const v = e.currentTarget;
            if (v.videoWidth > 0) setVideoSize({ w: v.videoWidth, h: v.videoHeight });
          }}
        />

        <CameraOverlay
          borderState={getBorderState()}
          holdProgress={holdProgress}
          label={sideLabel}
          bbox={detection?.bbox}
          videoSize={videoSize}
        />

        {detection?.detected && (
          <div className="absolute right-3 top-3 z-10 flex items-center gap-1.5 rounded-full bg-black/60 px-2.5 py-1 text-[11px] font-medium text-white backdrop-blur-sm">
            <span
              className="inline-block h-2 w-2 rounded-full"
              style={{
                backgroundColor:
                  detection.confidence > 0.7
                    ? "#22c55e"
                    : detection.confidence > 0.5
                    ? "#facc15"
                    : "#ef4444",
              }}
            />
            {(detection.confidence * 100).toFixed(0)}% detected
          </div>
        )}
      </div>

      {/* Quality issues */}
      <div className="mt-2.5 w-full max-w-[480px]">
        <QualityIndicator
          issues={detection?.issues ?? []}
          localQuality={quality}
        />
      </div>

      <p className="mt-3 text-center text-sm text-zinc-600 dark:text-zinc-400">
        Place your ID within the frame and take a picture.
      </p>

      {/* Action buttons */}
      <div className="mt-5 flex items-center gap-6">
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="flex h-12 w-12 items-center justify-center rounded-full bg-zinc-200 text-zinc-600 transition-colors hover:bg-zinc-300 dark:bg-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-600"
          aria-label="Upload from gallery"
        >
          <ImagePlus className="h-5 w-5" />
        </button>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleGallerySelect}
        />

        <button
          type="button"
          onClick={handleManualCapture}
          disabled={!isReady}
          className="group relative flex h-[68px] w-[68px] items-center justify-center rounded-full border-[3px] border-blue-500 transition-all active:scale-95 disabled:opacity-40"
          aria-label="Capture photo"
        >
          <span className="block h-[54px] w-[54px] rounded-full bg-blue-500 transition-colors group-hover:bg-blue-400 group-active:bg-blue-600" />

          {holdProgress > 0 && holdProgress < 1 && (
            <svg
              className="absolute -inset-1 h-[76px] w-[76px]"
              viewBox="0 0 76 76"
            >
              <circle
                cx="38"
                cy="38"
                r="35"
                fill="none"
                stroke="#22c55e"
                strokeWidth="3"
                strokeLinecap="round"
                strokeDasharray={`${holdProgress * 219.91} 219.91`}
                transform="rotate(-90 38 38)"
              />
            </svg>
          )}
        </button>

        <div className="h-12 w-12" />
      </div>

      {!isConnected && isReady && (
        <p className="mt-2 text-xs text-yellow-500">
          Connecting to detection server...
        </p>
      )}
    </div>
  );
}
