"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { useCardDetection } from "@/hooks/useCardDetection";
import { API_BASE } from "@/lib/apiBase";
import { Camera, AlertTriangle } from "lucide-react";

interface Props {
  side: "front" | "back";
  token: string;
  onCaptureComplete: (captureId: string, blob: Blob) => void;
}

export default function IDCaptureStep({ side, token, onCaptureComplete }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animRef = useRef(0);
  const detectCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const [phase, setPhase] = useState<"camera" | "preview">("camera");
  const [cardDetected, setCardDetected] = useState(false);
  const [capturedURL, setCapturedURL] = useState("");
  const [capturedBlob, setCapturedBlob] = useState<Blob | null>(null);
  const [qualityWarn, setQualityWarn] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { detect, drawHighlight } = useCardDetection();

  const stop = useCallback(() => {
    cancelAnimationFrame(animRef.current);
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
  }, []);

  const startCam = useCallback(async () => {
    setError(null); setPhase("camera"); setCardDetected(false);
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1920 }, height: { ideal: 1080 } },
        audio: false,
      });
      streamRef.current = s;
      const v = videoRef.current; if (!v) throw new Error("no video");
      v.srcObject = s; await v.play();
    } catch (e) { setError(e instanceof Error ? e.message : "Camera error"); }
  }, []);

  useEffect(() => { startCam(); return stop; }, []); // eslint-disable-line

  useEffect(() => {
    if (phase !== "camera") return;
    if (!detectCanvasRef.current) detectCanvasRef.current = document.createElement("canvas");
    let running = true;

    const loop = () => {
      if (!running) return;
      const v = videoRef.current;
      const dc = detectCanvasRef.current;
      const ov = overlayRef.current;
      if (!v || v.videoWidth === 0 || !dc || !ov) {
        animRef.current = requestAnimationFrame(loop); return;
      }
      const result = detect(v, dc);
      setCardDetected(result.detected);
      ov.width = v.videoWidth; ov.height = v.videoHeight;
      const ctx = ov.getContext("2d");
      if (ctx) { ctx.clearRect(0, 0, ov.width, ov.height); if (result.detected) drawHighlight(ov, result); }
      animRef.current = requestAnimationFrame(loop);
    };
    animRef.current = requestAnimationFrame(loop);
    return () => { running = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase]);

  function handleCapture() {
    const v = videoRef.current; if (!v) return;
    const c = document.createElement("canvas");
    c.width = v.videoWidth; c.height = v.videoHeight;
    const ctx = c.getContext("2d")!;
    ctx.drawImage(v, 0, 0);
    const img = ctx.getImageData(0, 0, c.width, c.height).data;
    let bright = 0;
    for (let i = 0; i < img.length; i += 4) bright += (img[i] + img[i + 1] + img[i + 2]) / 3;
    bright /= (img.length / 4);
    if (bright < 50) setQualityWarn("Too dark — try better lighting");
    else if (bright > 220) setQualityWarn("Overexposed — move away from light");
    else setQualityWarn(null);
    c.toBlob(blob => {
      if (!blob) return;
      setCapturedBlob(blob);
      setCapturedURL(URL.createObjectURL(blob));
      setPhase("preview");
    }, "image/jpeg", 0.92);
  }

  function handleRetake() {
    if (capturedURL) URL.revokeObjectURL(capturedURL);
    setCapturedURL(""); setCapturedBlob(null); setQualityWarn(null);
    setPhase("camera");
  }

  async function handleProceed() {
    if (!capturedBlob) return;
    setSubmitting(true);
    try {
      const fd = new FormData();
      fd.append("file", capturedBlob, `${side}.jpg`);
      fd.append("side", side);
      const res = await fetch(`${API_BASE}/api/capture/validate`, {
        method: "POST", headers: { Authorization: `Bearer ${token}` }, body: fd,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (!data.validation_passed && data.rejection_reason) {
        setError(data.rejection_reason); handleRetake(); setSubmitting(false); return;
      }
      onCaptureComplete(data.capture_id, capturedBlob);
    } catch (e) { setError(e instanceof Error ? e.message : "Upload failed"); setSubmitting(false); }
  }

  const label = side === "front" ? "Front of ID" : "Back of ID";

  return (
    <div className="flex flex-col items-center gap-4 w-full">
      {error && <div className="text-sm text-red-400 text-center">{error}</div>}
      <div className="relative w-full overflow-hidden rounded-2xl bg-black" style={{ maxWidth: 400 }}>
        {phase === "camera" && (
          <div className="relative">
            <video ref={videoRef} autoPlay playsInline muted className="h-full w-full object-cover" style={{ aspectRatio: "3/4" }} />
            <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full" />
          </div>
        )}
        {phase === "preview" && capturedURL && (
          <div className="relative">
            <img src={capturedURL} className="w-full object-cover" style={{ aspectRatio: "3/4" }} alt={label} />
            <div className="absolute top-3 left-3 bg-black/60 px-3 py-1 rounded-full text-xs text-white">{label}</div>
          </div>
        )}
      </div>
      {phase === "camera" && (
        <>
          <p className="text-sm text-zinc-400">{cardDetected ? "Card detected — ready to capture" : "Position your ID card flat in the frame"}</p>
          <button onClick={handleCapture} disabled={!cardDetected}
            className={`w-full py-4 rounded-2xl text-white font-semibold text-lg transition-all ${cardDetected ? "bg-green-500 shadow-lg shadow-green-500/30 active:scale-95" : "bg-zinc-700 text-zinc-500 cursor-not-allowed"}`}>
            <Camera className="mr-2 inline h-5 w-5" />{cardDetected ? "Capture Card" : "Waiting for card..."}
          </button>
        </>
      )}
      {phase === "preview" && (
        <div className="flex flex-col gap-3 w-full">
          <div className="flex gap-3">
            <button onClick={handleRetake} className="flex-1 py-3 rounded-xl border border-zinc-600 text-zinc-300 text-sm">↩ Retake</button>
            <button onClick={handleProceed} disabled={submitting} className="flex-1 py-3 rounded-xl bg-blue-600 text-white font-semibold text-sm">{submitting ? "Uploading..." : "Use this photo →"}</button>
          </div>
          {qualityWarn && <p className="text-yellow-400 text-xs text-center flex items-center justify-center gap-1"><AlertTriangle className="h-3 w-3" />{qualityWarn}</p>}
        </div>
      )}
    </div>
  );
}
