"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { canvasToJpegBlob, grabFrame } from "@/lib/frameEncoder";
import { API_BASE } from "@/lib/apiBase";
import { useFaceDetection } from "@/hooks/useFaceDetection";
import { CheckCircle, Loader2, XCircle, Camera } from "lucide-react";

interface FaceScanStepProps {
  token: string;
  userId: string;
  onComplete: (result: { passed: boolean; confidence: number; user_id: string }) => void;
}

type ScanState = "idle" | "preparing" | "scanning" | "verifying" | "passed" | "failed";

export default function FaceScanStep({ token, userId, onComplete }: FaceScanStepProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animRef = useRef<number>(0);
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;
  const stableRef = useRef(0);

  const [scanState, setScanState] = useState<ScanState>("idle");
  const [camError, setCamError] = useState<string | null>(null);
  const [statusText, setStatusText] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [progress, setProgress] = useState(0);
  const [glow, setGlow] = useState<"none" | "yellow" | "green">("none");

  const { isReady, detect } = useFaceDetection();

  const cleanup = useCallback(() => {
    if (animRef.current) cancelAnimationFrame(animRef.current);
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
  }, []);

  const startCamera = useCallback(async () => {
    try {
      setScanState("preparing"); setStatusText("Loading face detection...");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } }, audio: false,
      });
      streamRef.current = stream;
      const v = videoRef.current; if (!v) throw new Error("no video");
      v.srcObject = stream; await v.play();
    } catch (err) {
      setCamError(err instanceof Error ? err.message : "Camera"); setScanState("failed");
    }
  }, []);

  useEffect(() => { startCamera(); return cleanup; }, []); // eslint-disable-line

  useEffect(() => {
    if (isReady && videoRef.current && videoRef.current.videoWidth > 0 && scanState === "preparing") {
      setScanState("scanning"); setStatusText("Center your face in the oval");
    }
  }, [isReady, scanState]);

  // Frame loop
  useEffect(() => {
    if (scanState !== "scanning") return;
    let running = true, busy = false;

    async function loop() {
      if (!running) return;
      const video = videoRef.current;
      if (!video || video.videoWidth === 0 || busy) { animRef.current = requestAnimationFrame(loop); return; }
      busy = true;
      let res;
      try { res = await detect(video); } catch { res = null; }
      busy = false;
      if (!running) return;

      if (res) {
        const centered = res.pose === "center";
        drawOval(video, res.box, centered ? "green" : "yellow", 0);
        setGlow(centered ? "green" : "yellow");
        if (centered) {
          stableRef.current = Math.min(25, stableRef.current + 1);
          setProgress(Math.round((stableRef.current / 25) * 100));
          setStatusText("Hold still...");
          if (stableRef.current >= 25) {
            running = false;
            await doCapture(video);
            return;
          }
        } else {
          stableRef.current = Math.max(0, stableRef.current - 2);
          setProgress(Math.round((stableRef.current / 25) * 100));
          setStatusText("Center your face in the oval");
        }
      } else {
        drawOval(video, null, "none", 0);
        setGlow("none"); stableRef.current = 0; setProgress(0);
        setStatusText("Position your face");
      }
      animRef.current = requestAnimationFrame(loop);
    }
    animRef.current = requestAnimationFrame(loop);
    return () => { running = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scanState]);

  async function doCapture(video: HTMLVideoElement) {
    setScanState("verifying"); setStatusText("Verifying...");
    const c = document.createElement("canvas");
    grabFrame(video, c);
    const blob = await canvasToJpegBlob(c, 0.85);
    try {
      const fd = new FormData(); fd.append("image", blob, "face.jpg"); fd.append("user_id", userId);
      const res = await fetch(`${API_BASE}/api/face/verify`, { method: "POST", headers: { Authorization: `Bearer ${token}` }, body: fd });
      if (!res.ok) throw new Error((await res.json().catch(()=>({}))).detail || `HTTP ${res.status}`);
      const data = await res.json();
      setConfidence(data.confidence);
      if (data.matched) { setScanState("passed"); onCompleteRef.current({ passed: true, confidence: data.confidence, user_id: data.user_id }); }
      else { setScanState("failed"); setCamError(`No match (${(data.confidence*100).toFixed(0)}%)`); }
    } catch (err) { setScanState("failed"); setCamError(err instanceof Error ? err.message : "Verification failed"); }
  }

  function drawOval(video: HTMLVideoElement, box: any, g: string, _p: number) {
    const canvas = overlayRef.current;
    if (!canvas || !video) return;
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d"); if (!ctx) return;
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    const ox = w / 2, oy = h / 2.1, orx = w * 0.21, ory = h * 0.29;
    const gc = g === "green" ? "rgba(34,197,94,0.5)" : g === "yellow" ? "rgba(234,179,8,0.5)" : "rgba(255,255,255,0.12)";
    ctx.save();
    ctx.shadowColor = g === "green" ? "#22c55e" : g === "yellow" ? "#eab308" : "#ffffff";
    ctx.shadowBlur = g === "green" ? 22 : g === "yellow" ? 15 : 6;
    ctx.strokeStyle = gc; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.ellipse(ox, oy, orx, ory, 0, 0, Math.PI * 2); ctx.stroke();
    ctx.restore();
    if (box) { ctx.strokeStyle = "rgba(255,255,255,0.12)"; ctx.lineWidth = 1; ctx.strokeRect(box.x, box.y, box.width, box.height); }
  }

  function handleRetry() { stableRef.current = 0; setProgress(0); setCamError(null); cleanup(); startCamera(); }

  return (
    <div className="flex flex-col items-center gap-4">
      {camError && <div className="flex flex-col items-center gap-3 p-4 text-center"><p className="text-sm text-red-400">{camError}</p><button onClick={handleRetry} className="rounded-full bg-blue-600 px-4 py-2 text-sm text-white">Retry</button></div>}
      <div className="relative w-full overflow-hidden rounded-2xl bg-black" style={{ maxWidth: 400 }}>
        {(scanState === "preparing" || scanState === "scanning") && (
          <div className="relative">
            <video ref={videoRef} autoPlay playsInline muted className="h-full w-full object-cover" style={{ aspectRatio: "3/4", transform: "scaleX(-1)" }} />
            <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full" style={{ transform: "scaleX(-1)" }} />
          </div>
        )}
        {scanState === "verifying" && <div className="flex items-center justify-center bg-black" style={{ aspectRatio: "3/4" }}><Loader2 className="h-10 w-10 animate-spin text-blue-400" /></div>}
        {scanState === "passed" && <div className="flex items-center justify-center bg-green-950/50" style={{ aspectRatio: "3/4" }}><CheckCircle className="h-16 w-16 text-green-400" /></div>}
      </div>
      <div className="flex flex-col items-center gap-2 text-center">
        {scanState === "scanning" && <><div className="flex items-center gap-2 text-sm text-zinc-400"><Camera className="h-4 w-4" />{statusText}</div>{progress > 0 && <div className="h-1 w-40 rounded-full bg-zinc-700"><div className="h-full rounded-full bg-sky-400 transition-all" style={{ width: `${progress}%` }} /></div>}</>}
        {scanState === "passed" && <div className="flex items-center gap-2 rounded-full bg-green-600 px-5 py-2 text-sm font-medium text-white"><CheckCircle className="h-4 w-4" />Verified ({(confidence*100).toFixed(0)}%)</div>}
        {scanState === "failed" && !camError && <div className="flex flex-col items-center gap-3"><div className="flex items-center gap-2 rounded-full bg-red-600 px-5 py-2 text-sm text-white"><XCircle className="h-4 w-4" />Failed</div><button onClick={handleRetry} className="rounded-full bg-blue-600 px-6 py-2.5 text-sm text-white">Try Again</button></div>}
      </div>
    </div>
  );
}
