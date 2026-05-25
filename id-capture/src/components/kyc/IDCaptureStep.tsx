"use client";

import { useEffect, useRef, useState, useCallback } from "react";
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

  const [phase, setPhase] = useState<"camera" | "preview">("camera");
  const [capturedURL, setCapturedURL] = useState("");
  const [capturedBlob, setCapturedBlob] = useState<Blob | null>(null);
  const [qualityWarn, setQualityWarn] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const stop = useCallback(() => {
    cancelAnimationFrame(animRef.current);
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
  }, []);

  const startCam = useCallback(async () => {
    setError(null); setPhase("camera");
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

  // Draw static card guide frame
  useEffect(() => {
    if (phase !== "camera") return;
    let running = true;
    const loop = () => {
      if (!running) return;
      const v = videoRef.current;
      const ov = overlayRef.current;
      if (!v || v.videoWidth === 0 || !ov) { animRef.current = requestAnimationFrame(loop); return; }

      ov.width = v.videoWidth; ov.height = v.videoHeight;
      const ctx = ov.getContext("2d");
      if (ctx) {
        ctx.clearRect(0, 0, ov.width, ov.height);
        drawGuide(ctx, ov.width, ov.height);
      }
      animRef.current = requestAnimationFrame(loop);
    };
    animRef.current = requestAnimationFrame(loop);
    return () => { running = false; };
  }, [phase]);

  function drawGuide(ctx: CanvasRenderingContext2D, w: number, h: number) {
    // Card guide: centered rectangle with ~1.58:1 aspect ratio (ID card)
    const margin = 0.08;
    const gw = w * (1 - margin * 2);
    const gh = gw / 1.58;
    const gx = (w - gw) / 2;
    const gy = (h - gh) / 2;

    // Dim outside the guide
    ctx.fillStyle = "rgba(0,0,0,0.45)";
    ctx.fillRect(0, 0, w, gy);
    ctx.fillRect(0, gy + gh, w, h - gy - gh);
    ctx.fillRect(0, gy, gx, gh);
    ctx.fillRect(gx + gw, gy, w - gx - gw, gh);

    // Guide border — dashed white
    ctx.strokeStyle = "rgba(255,255,255,0.6)";
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 4]);
    ctx.strokeRect(gx, gy, gw, gh);
    ctx.setLineDash([]);

    // Corner brackets
    const bs = Math.min(28, gw * 0.1);
    ctx.strokeStyle = "rgba(255,255,255,0.8)";
    ctx.lineWidth = 3;
    const corners = [[gx,gy],[gx+gw,gy],[gx+gw,gy+gh],[gx,gy+gh]];
    corners.forEach(([cx, cy]) => {
      const bx = cx === gx ? 1 : -1;
      const by = cy === gy ? 1 : -1;
      ctx.beginPath();
      ctx.moveTo(cx, cy + by * bs); ctx.lineTo(cx, cy); ctx.lineTo(cx + bx * bs, cy);
      ctx.stroke();
    });
  }

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
    if (bright < 60) setQualityWarn("Too dark — try better lighting");
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
          <p className="text-sm text-zinc-400">Align your card within the guide frame</p>
          <button onClick={handleCapture}
            className="w-full py-4 rounded-2xl bg-green-500 text-white font-semibold text-lg shadow-lg shadow-green-500/30 active:scale-95 transition-all">
            <Camera className="mr-2 inline h-5 w-5" />Capture {side === "front" ? "Front" : "Back"}
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
