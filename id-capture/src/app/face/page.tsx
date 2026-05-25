"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { API_BASE } from "@/lib/apiBase";
import { canvasToJpegBlob } from "@/lib/frameEncoder";
import { useMediaPipeFace } from "@/hooks/useMediaPipeFace";
import { checkLiveness, prepareLivenessInput } from "@/lib/silentFaceLiveness";
import Link from "next/link";
import { ArrowLeft, Camera, Loader2, CheckCircle, XCircle, UserPlus, Fingerprint } from "lucide-react";

type Mode = "enroll" | "verify";

export default function FacePage() {
  const [mode, setMode] = useState<Mode>("enroll");
  const [token, setToken] = useState("");
  const [phase, setPhase] = useState<"idle" | "active" | "verifying" | "done" | "error">("idle");
  const [statusMsg, setStatusMsg] = useState("");
  const [errMsg, setErrMsg] = useState<string | null>(null);
  const [userId, setUserId] = useState("");
  const [result, setResult] = useState<{ matched?: boolean; confidence?: number; threshold_used?: number; user_id?: string } | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animRef = useRef<number>(0);
  const detectCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const { landmarks, faceDetected, isReady, detect } = useMediaPipeFace();

  // JWT
  useEffect(() => {
    fetch(`${API_BASE}/api/auth/token`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: `face_${Date.now()}` }) })
      .then(r => r.json()).then(d => setToken(d.access_token)).catch(() => setToken("dev_token"));
  }, []);

  // ── Start / Stop camera ────────────────────────────────────────

  const stop = useCallback(() => {
    cancelAnimationFrame(animRef.current);
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
  }, []);

  const start = useCallback(async () => {
    setErrMsg(null);
    setResult(null);
    setPhase("active");
    setStatusMsg("Loading...");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      streamRef.current = stream;
      const v = videoRef.current;
      if (!v) throw new Error("no video element");
      v.srcObject = stream;
      await v.play();
      setStatusMsg(mode === "enroll" ? "Position your face" : "Position your face");
    } catch (e) {
      setErrMsg(e instanceof Error ? e.message : "Camera error");
      setPhase("error");
    }
  }, [mode]);

  useEffect(() => stop, []); // eslint-disable-line

  // ── Frame loop ─────────────────────────────────────────────────

  const stableRef = useRef(0);     // 0–30 progress
  const captureRef = useRef(false);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (phase !== "active") {
      stableRef.current = 0;
      captureRef.current = false;
      setProgress(0);
      return;
    }
    if (!detectCanvasRef.current) detectCanvasRef.current = document.createElement("canvas");

    const loop = () => {
      const video = videoRef.current;
      const dCanvas = detectCanvasRef.current;
      if (!video || !dCanvas || video.videoWidth === 0) {
        animRef.current = requestAnimationFrame(loop);
        return;
      }

      detect(video, dCanvas);
      drawOverlay(video);

      if (!captureRef.current) {
        if (faceIsWellPositioned()) {
          stableRef.current = Math.min(30, stableRef.current + 1);
        } else {
          stableRef.current = Math.max(0, stableRef.current - 2);
        }
        const pct = Math.round((stableRef.current / 30) * 100);
        setProgress(pct);

        if (stableRef.current >= 30) {
          captureRef.current = true;
          setStatusMsg("Processing...");
          handleAction();
          return;
        } else if (stableRef.current > 15) {
          setStatusMsg("Hold still...");
        } else if (landmarks) {
          setStatusMsg("Center your face in the frame");
        } else {
          setStatusMsg(mode === "enroll" ? "Position your face" : "Position your face");
        }
      }

      animRef.current = requestAnimationFrame(loop);
    };

    animRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, landmarks]);

  // ── Face position check ────────────────────────────────────────

  function faceIsWellPositioned(): boolean {
    if (!landmarks || !landmarks[0] || landmarks[0].length < 468) return false;
    const pts = landmarks[0];
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of pts) {
      if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
    }
    const area = (maxX - minX) * (maxY - minY);
    const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
    return area > 0.06 && Math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2) < 0.3;
  }

  // ── Face mesh connections (MediaPipe topology) ─────────────────

  // Indices for key facial feature contours
  const FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109];
  const LEFT_EYE = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246];
  const RIGHT_EYE = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398];
  const LEFT_EYEBROW = [46,53,52,65,55,70,63,105,66,107];
  const RIGHT_EYEBROW = [276,283,282,295,285,300,293,334,296,336];
  const NOSE_BRIDGE = [6,168,197,195,5,4,1,19,94,2];
  const NOSE_TIP = [1,2,98,327,460,459,458,461,354,455,460];
  const LIPS_OUTER = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185];
  const LIPS_INNER = [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95];

  function drawMeshLine(ctx: CanvasRenderingContext2D, pts: any[], indices: number[], w: number, h: number, color: string, width: number) {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    for (let i = 0; i < indices.length; i++) {
      const p = pts[indices[i]];
      if (!p) continue;
      const x = p.x * w, y = p.y * h;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.stroke();
  }

  function drawOverlay(video: HTMLVideoElement) {
    const canvas = overlayRef.current;
    if (!canvas || !video) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!landmarks || !landmarks[0]) return;

    const pts = landmarks[0];
    const w = canvas.width, h = canvas.height;
    const positioned = faceIsWellPositioned();
    const alpha = positioned ? "0.8" : "0.3";

    // Face oval
    drawMeshLine(ctx, pts, FACE_OVAL, w, h, `rgba(255,255,255,${alpha})`, 2);

    // Eyes
    drawMeshLine(ctx, pts, LEFT_EYE, w, h, `rgba(96,165,250,${alpha})`, 1.5);
    drawMeshLine(ctx, pts, RIGHT_EYE, w, h, `rgba(96,165,250,${alpha})`, 1.5);

    // Eyebrows
    drawMeshLine(ctx, pts, LEFT_EYEBROW, w, h, `rgba(250,204,21,${alpha})`, 2);
    drawMeshLine(ctx, pts, RIGHT_EYEBROW, w, h, `rgba(250,204,21,${alpha})`, 2);

    // Nose
    drawMeshLine(ctx, pts, NOSE_BRIDGE, w, h, `rgba(168,85,247,${alpha})`, 1.5);
    drawMeshLine(ctx, pts, NOSE_TIP, w, h, `rgba(168,85,247,${alpha})`, 1.5);

    // Lips
    drawMeshLine(ctx, pts, LIPS_OUTER, w, h, `rgba(239,68,68,${alpha})`, 1.5);
    drawMeshLine(ctx, pts, LIPS_INNER, w, h, `rgba(239,68,68,${alpha})`, 1);

    // Iris centers
    if (pts[468] && pts[473]) {
      [[468,4],[473,4]].forEach(([idx,r]) => {
        const p = pts[idx as number];
        ctx.fillStyle = `rgba(255,255,255,${alpha})`;
        ctx.beginPath();
        ctx.arc(p.x * w, p.y * h, (r as number), 0, Math.PI * 2);
        ctx.fill();
      });
    }
  }

  // ── Enroll or Verify ───────────────────────────────────────────

  async function handleAction() {
    setPhase("verifying");
    setStatusMsg("Checking liveness...");

    const video = videoRef.current;
    if (!video || !landmarks || !landmarks[0]) { setPhase("error"); setErrMsg("Lost face"); return; }

    const pts = landmarks[0];
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of pts) {
      if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
    }
    const bbox = { x: minX * video.videoWidth, y: minY * video.videoHeight, width: (maxX - minX) * video.videoWidth, height: (maxY - minY) * video.videoHeight };

    let bestLive = 0, bestBlob: Blob | null = null;
    for (let i = 0; i < 3; i++) {
      await new Promise(r => setTimeout(r, 150));
      const input = prepareLivenessInput(video, bbox);
      if (!input) continue;
      const score = await checkLiveness(input);
      if (score > bestLive) {
        bestLive = score;
        const fc = document.createElement("canvas");
        fc.width = video.videoWidth; fc.height = video.videoHeight;
        fc.getContext("2d")!.drawImage(video, 0, 0);
        bestBlob = await canvasToJpegBlob(fc, 0.85);
      }
    }

    if (!bestBlob || bestLive < 0.5) {
      setPhase("error");
      setErrMsg(bestLive < 0.5 ? "Spoof detected — use a real face" : "Liveness failed");
      return;
    }

    setStatusMsg(mode === "enroll" ? "Generating embedding..." : "Verifying...");
    try {
      if (mode === "enroll") {
        const fd = new FormData();
        fd.append("image", bestBlob, "face.jpg");
        fd.append("liveness_score", String(bestLive));
        const res = await fetch(`${API_BASE}/api/face/enroll`, { method: "POST", headers: { Authorization: `Bearer ${token}` }, body: fd });
        if (!res.ok) throw new Error((await res.json().catch(()=>({}))).detail || `HTTP ${res.status}`);
        const data = await res.json();
        setUserId(data.user_id);
        setResult(data);
        setPhase("done");
      } else {
        if (!userId) { setPhase("error"); setErrMsg("Enroll first or paste a user ID above"); return; }
        const fd = new FormData();
        fd.append("image", bestBlob, "face.jpg");
        fd.append("user_id", userId);
        const res = await fetch(`${API_BASE}/api/face/verify`, { method: "POST", headers: { Authorization: `Bearer ${token}` }, body: fd });
        if (!res.ok) throw new Error((await res.json().catch(()=>({}))).detail || `HTTP ${res.status}`);
        setResult(await res.json());
        setPhase("done");
      }
    } catch (e) {
      setPhase("error");
      setErrMsg(e instanceof Error ? e.message : "Request failed");
    }
  }

  function retry() {
    stop();
    setPhase("idle");
    setErrMsg(null);
    setResult(null);
  }

  return (
    <div className="flex min-h-screen flex-col bg-zinc-950 text-zinc-100">
      <header className="flex items-center gap-3 border-b border-zinc-800 px-4 py-3">
        <Link href="/" className="rounded-full p-1.5 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"><ArrowLeft className="h-5 w-5" /></Link>
        <Fingerprint className="h-5 w-5 text-blue-400" />
        <h1 className="text-base font-semibold">Face Pipeline Test</h1>
      </header>

      <main className="mx-auto flex w-full max-w-md flex-1 flex-col items-center p-4">
        {/* Tabs */}
        <div className="mb-4 flex w-full rounded-lg bg-zinc-900 p-1">
          <button onClick={() => { setMode("enroll"); retry(); }}
            className={`flex-1 rounded-md px-4 py-2 text-sm font-medium ${mode === "enroll" ? "bg-blue-600 text-white" : "text-zinc-400"}`}>
            <UserPlus className="mr-2 inline h-4 w-4" />Enroll
          </button>
          <button onClick={() => { setMode("verify"); retry(); }}
            className={`flex-1 rounded-md px-4 py-2 text-sm font-medium ${mode === "verify" ? "bg-blue-600 text-white" : "text-zinc-400"}`}>
            <Fingerprint className="mr-2 inline h-4 w-4" />Verify
          </button>
        </div>

        {/* Idle */}
        {phase === "idle" && (
          <div className="flex flex-1 flex-col items-center justify-center gap-4">
            <p className="text-sm text-zinc-400">{mode === "enroll" ? "Capture a face to enroll a new identity" : "Verify against an enrolled identity"}</p>
            <button onClick={start} className="flex items-center gap-2 rounded-xl bg-blue-600 px-6 py-3 text-sm font-medium text-white hover:bg-blue-500">
              <Camera className="h-5 w-5" />Start Camera
            </button>
            {mode === "verify" && (
              <input type="text" value={userId} onChange={e => setUserId(e.target.value)}
                placeholder="Paste user_id from enrollment..."
                className="mt-2 w-full rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-200" />
            )}
          </div>
        )}

        {/* Camera */}
        <div className="relative w-full overflow-hidden rounded-2xl bg-black" style={{ maxWidth: 400 }}>
          {phase === "active" && (
            <div className="relative">
              <video ref={videoRef} autoPlay playsInline muted className="h-full w-full object-cover" style={{ aspectRatio: "3/4", transform: "scaleX(-1)" }} />
              <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full" style={{ transform: "scaleX(-1)" }} />
            </div>
          )}
          {phase === "verifying" && (
            <div className="flex items-center justify-center bg-black" style={{ aspectRatio: "3/4" }}>
              <Loader2 className="h-10 w-10 animate-spin text-blue-400" />
            </div>
          )}
          {phase === "done" && (
            <div className={`flex items-center justify-center ${mode === "enroll" || result?.matched ? "bg-green-950/50" : "bg-red-950/50"}`} style={{ aspectRatio: "3/4" }}>
              {(mode === "enroll" || result?.matched) ? <CheckCircle className="h-16 w-16 text-green-400" /> : <XCircle className="h-16 w-16 text-red-400" />}
            </div>
          )}
        </div>

        {/* Status / Error */}
        <div className="mt-4 text-center">
          {phase === "active" && (
            <div className="flex flex-col items-center gap-2">
              <p className="text-sm text-zinc-400"><Camera className="mr-1 inline h-4 w-4" />{isReady ? statusMsg : "Loading face detection..."}</p>
              {progress > 0 && (
                <div className="h-1.5 w-48 overflow-hidden rounded-full bg-zinc-700">
                  <div
                    className="h-full rounded-full bg-blue-400 transition-all duration-200"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              )}
            </div>
          )}
          {phase === "verifying" && <p className="text-sm text-blue-400"><Loader2 className="mr-1 inline h-4 w-4 animate-spin" />{statusMsg}</p>}
          {phase === "error" && (
            <div className="flex flex-col items-center gap-3">
              <p className="text-sm text-red-400"><XCircle className="mr-1 inline h-4 w-4" />{errMsg}</p>
              <button onClick={retry} className="rounded-full bg-blue-600 px-6 py-2.5 text-sm font-medium text-white">Try Again</button>
            </div>
          )}
        </div>

        {/* Results */}
        {phase === "done" && result && (
          <div className="mt-4 w-full space-y-3">
            {mode === "enroll" && (
              <div className="rounded-xl bg-zinc-900 p-4">
                <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">Enrolled</h2>
                <code className="block break-all rounded bg-zinc-800 p-2 text-xs text-green-400">{result.user_id}</code>
                <button onClick={() => { setMode("verify"); setUserId(result.user_id || ""); setPhase("idle"); }}
                  className="mt-3 w-full rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white">Switch to Verify</button>
              </div>
            )}
            {mode === "verify" && (
              <div className={`rounded-xl p-4 ${result.matched ? "bg-green-500/10 ring-1 ring-green-500/30" : "bg-red-500/10 ring-1 ring-red-500/30"}`}>
                <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">Result</h2>
                <p className={`text-lg font-bold ${result.matched ? "text-green-400" : "text-red-400"}`}>
                  {result.matched ? "MATCH" : "NO MATCH"} — {((result.confidence ?? 0) * 100).toFixed(1)}%
                </p>
                <p className="text-xs text-zinc-500">threshold: {((result.threshold_used ?? 0) * 100).toFixed(0)}%</p>
                <button onClick={retry} className="mt-3 w-full rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white">Test Again</button>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
