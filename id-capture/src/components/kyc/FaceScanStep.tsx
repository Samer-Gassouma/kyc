"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { canvasToJpegBlob, grabFrame } from "@/lib/frameEncoder";
import { API_BASE } from "@/lib/apiBase";
import { useFaceDetection } from "@/hooks/useFaceDetection";
import clsx from "clsx";
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

  const [scanState, setScanState] = useState<ScanState>("idle");
  const [camError, setCamError] = useState<string | null>(null);
  const [statusText, setStatusText] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [livePts, setLivePts] = useState<number[][]>([]);
  const [progress, setProgress] = useState(0);

  const { isReady, detect } = useFaceDetection();
  const stableRef = useRef(0);
  const captureRef = useRef(false);

  const cleanup = useCallback(() => {
    if (animRef.current) cancelAnimationFrame(animRef.current);
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
  }, []);

  const startCamera = useCallback(async () => {
    try {
      setScanState("preparing");
      setStatusText("Loading face detection...");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } }, audio: false,
      });
      streamRef.current = stream;
      const v = videoRef.current;
      if (!v) throw new Error("no video");
      v.srcObject = stream;
      await v.play();
    } catch (err) {
      setCamError(err instanceof Error ? err.message : "Camera access denied");
      setScanState("failed");
    }
  }, []);

  useEffect(() => { startCamera(); return cleanup; }, []); // eslint-disable-line

  useEffect(() => {
    if (isReady && videoRef.current && videoRef.current.videoWidth > 0 && scanState === "preparing") {
      setScanState("scanning");
      setStatusText("Position your face in the frame");
    }
  }, [isReady, scanState]);

  // Frame loop
  useEffect(() => {
    if (scanState !== "scanning") return;
    let running = true;

    async function loop() {
      if (!running) return;
      const video = videoRef.current;
      if (!video || video.videoWidth === 0) {
        animRef.current = requestAnimationFrame(loop);
        return;
      }

      const result = await detect(video);
      if (!running) return;

      if (result && !captureRef.current) {
        setLivePts(result.landmarks.positions.map((p: any) => [p.x / video!.videoWidth, p.y / video!.videoHeight]));
        drawOverlay(video);

        // Check if centered
        const { yaw, pitch } = result.pose;
        const centered = Math.abs(yaw) < 12 && Math.abs(pitch) < 15;

        if (centered) {
          stableRef.current = Math.min(30, stableRef.current + 1);
          setProgress(Math.round((stableRef.current / 30) * 100));
          setStatusText("Hold still...");
          if (stableRef.current >= 30) {
            captureRef.current = true;
            await handleCapture(video, result.landmarks.positions);
            return;
          }
        } else {
          stableRef.current = Math.max(0, stableRef.current - 2);
          setProgress(Math.round((stableRef.current / 30) * 100));
          setStatusText("Center your face in the frame");
        }
      } else if (!result) {
        stableRef.current = 0;
        setProgress(0);
        setStatusText("Position your face in the frame");
      }

      animRef.current = requestAnimationFrame(loop);
    }
    animRef.current = requestAnimationFrame(loop);
    return () => { running = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scanState]);

  const JAW = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16];
  const LEFT_BROW = [17,18,19,20,21]; const RIGHT_BROW = [22,23,24,25,26];
  const NOSE_BRIDGE = [27,28,29,30]; const NOSE_BOTTOM = [31,32,33,34,35];
  const LEFT_EYE = [36,37,38,39,40,41]; const RIGHT_EYE = [42,43,44,45,46,47];
  const MOUTH_OUTER = [48,49,50,51,52,53,54,55,56,57,58,59];
  const MOUTH_INNER = [60,61,62,63,64,65,66,67];

  function drawOverlay(video: HTMLVideoElement) {
    const canvas = overlayRef.current;
    if (!canvas || !video) return;
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (livePts.length < 60) return;
    const w = canvas.width, h = canvas.height;

    const line = (ix: number[], c: string, lw: number) => {
      ctx.strokeStyle = c; ctx.lineWidth = lw; ctx.beginPath();
      ctx.moveTo(livePts[ix[0]][0]*w, livePts[ix[0]][1]*h);
      for (let i = 1; i < ix.length; i++) ctx.lineTo(livePts[ix[i]][0]*w, livePts[ix[i]][1]*h);
      if (ix.length > 2) ctx.closePath(); ctx.stroke();
    };
    const poly = (ix: number[], fc: string, sc: string, lw: number) => {
      ctx.fillStyle = fc; ctx.strokeStyle = sc; ctx.lineWidth = lw; ctx.beginPath();
      ctx.moveTo(livePts[ix[0]][0]*w, livePts[ix[0]][1]*h);
      for (let i = 1; i < ix.length; i++) ctx.lineTo(livePts[ix[i]][0]*w, livePts[ix[i]][1]*h);
      ctx.closePath(); ctx.fill(); ctx.stroke();
    };

    line(JAW, "rgba(255,255,255,0.2)", 2);
    line(LEFT_BROW, "rgba(251,191,36,0.6)", 2.5);
    line(RIGHT_BROW, "rgba(251,191,36,0.6)", 2.5);
    poly(LEFT_EYE, "rgba(255,255,255,0.08)", "rgba(96,165,250,0.6)", 2);
    poly(RIGHT_EYE, "rgba(255,255,255,0.08)", "rgba(96,165,250,0.6)", 2);
    line(NOSE_BRIDGE, "rgba(168,85,247,0.4)", 1.5);
    poly(NOSE_BOTTOM, "rgba(168,85,247,0.08)", "rgba(168,85,247,0.5)", 1.5);
    poly(MOUTH_OUTER, "rgba(239,68,68,0.08)", "rgba(239,68,68,0.5)", 2);
    line(MOUTH_INNER, "rgba(239,68,68,0.35)", 1);

    for (let i = 0; i < livePts.length; i++) {
      const r = LEFT_EYE.includes(i) || RIGHT_EYE.includes(i) ? 2.2 : MOUTH_OUTER.includes(i) || MOUTH_INNER.includes(i) ? 2 : NOSE_BRIDGE.includes(i) || NOSE_BOTTOM.includes(i) ? 1.8 : 1.1;
      const col = LEFT_EYE.includes(i) || RIGHT_EYE.includes(i) ? "rgba(96,165,250,0.75)" : MOUTH_OUTER.includes(i) || MOUTH_INNER.includes(i) ? "rgba(239,68,68,0.65)" : NOSE_BRIDGE.includes(i) || NOSE_BOTTOM.includes(i) ? "rgba(168,85,247,0.65)" : "rgba(255,255,255,0.35)";
      ctx.fillStyle = col; ctx.beginPath(); ctx.arc(livePts[i][0]*w, livePts[i][1]*h, r, 0, Math.PI*2); ctx.fill();
    }
  }

  async function handleCapture(video: HTMLVideoElement, pts: any[]) {
    setScanState("verifying");
    setStatusText("Verifying identity...");

    const c = document.createElement("canvas");
    grabFrame(video, c);
    const blob = await canvasToJpegBlob(c, 0.85);

    try {
      const fd = new FormData();
      fd.append("image", blob, "face.jpg");
      fd.append("user_id", userId);
      const res = await fetch(`${API_BASE}/api/face/verify`, { method: "POST", headers: { Authorization: `Bearer ${token}` }, body: fd });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
      const data = await res.json();
      setConfidence(data.confidence);
      if (data.matched) {
        setScanState("passed");
        onCompleteRef.current({ passed: true, confidence: data.confidence, user_id: data.user_id });
      } else {
        setScanState("failed");
        setCamError(`Face doesn't match (${(data.confidence * 100).toFixed(0)}% — need ${(data.threshold_used * 100).toFixed(0)}%)`);
      }
    } catch (err) {
      setScanState("failed");
      setCamError(err instanceof Error ? err.message : "Verification failed");
    }
  }

  function handleRetry() {
    captureRef.current = false;
    stableRef.current = 0;
    setProgress(0);
    setCamError(null);
    setConfidence(0);
    cleanup();
    startCamera();
  }

  const showVideo = scanState === "preparing" || scanState === "scanning";

  return (
    <div className="flex flex-col items-center gap-4">
      {camError && (
        <div className="flex flex-col items-center gap-3 p-4 text-center">
          <p className="text-sm text-red-400">{camError}</p>
          <button onClick={handleRetry} className="rounded-full bg-blue-600 px-4 py-2 text-sm text-white">Retry</button>
        </div>
      )}
      <div className="relative w-full overflow-hidden rounded-2xl bg-black" style={{ maxWidth: 400 }}>
        {showVideo && (
          <div className="relative">
            <video ref={videoRef} autoPlay playsInline muted className="h-full w-full object-cover" style={{ aspectRatio: "3/4", transform: "scaleX(-1)" }} />
            <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full" style={{ transform: "scaleX(-1)" }} />
          </div>
        )}
        {scanState === "verifying" && (
          <div className="flex items-center justify-center bg-black" style={{ aspectRatio: "3/4" }}>
            <Loader2 className="h-10 w-10 animate-spin text-blue-400" />
          </div>
        )}
        {scanState === "passed" && (
          <div className="flex items-center justify-center bg-green-950/50" style={{ aspectRatio: "3/4" }}>
            <CheckCircle className="h-16 w-16 text-green-400" />
          </div>
        )}
      </div>
      <div className="flex flex-col items-center gap-2 text-center">
        {scanState === "scanning" && (
          <>
            <div className="flex items-center gap-2 text-sm text-zinc-400"><Camera className="h-4 w-4" />{statusText}</div>
            {progress > 0 && <div className="h-1.5 w-48 rounded-full bg-zinc-700"><div className="h-full rounded-full bg-sky-400 transition-all" style={{ width: `${progress}%` }} /></div>}
          </>
        )}
        {scanState === "passed" && (
          <div className="flex items-center gap-2 rounded-full bg-green-600 px-5 py-2 text-sm font-medium text-white"><CheckCircle className="h-4 w-4" />Face verified ({(confidence*100).toFixed(0)}%)</div>
        )}
        {scanState === "failed" && !camError && (
          <div className="flex flex-col items-center gap-3">
            <div className="flex items-center gap-2 rounded-full bg-red-600 px-5 py-2 text-sm font-medium text-white"><XCircle className="h-4 w-4" />Verification failed</div>
            <button onClick={handleRetry} className="rounded-full bg-blue-600 px-6 py-2.5 text-sm font-medium text-white">Try Again</button>
          </div>
        )}
      </div>
    </div>
  );
}
