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

  useEffect(() => {
    if (phase !== "active") return;
    if (!detectCanvasRef.current) detectCanvasRef.current = document.createElement("canvas");

    let stable = 0;
    let captured = false;

    const loop = () => {
      const video = videoRef.current;
      const dCanvas = detectCanvasRef.current;
      if (!video || !dCanvas || video.videoWidth === 0) {
        animRef.current = requestAnimationFrame(loop);
        return;
      }

      // Detect + draw frame in one call
      detect(video, dCanvas);

      // Draw landmarks overlay
      drawOverlay(video);

      // Auto-capture when face is well-positioned
      if (!captured && faceIsWellPositioned()) {
        stable++;
        const remaining = Math.max(1, Math.ceil((30 - stable) / 10));
        setStatusMsg(`Hold still... ${remaining}`);
        if (stable >= 30) {
          captured = true;
          handleAction();
          return; // loop stops, handleAction takes over
        }
      } else if (!captured) {
        stable = 0;
        if (landmarks) {
          setStatusMsg("Center your face in the oval");
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

  // ── Overlay ────────────────────────────────────────────────────

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
    ctx.fillStyle = "rgba(59, 130, 246, 0.55)";
    for (const p of pts) {
      ctx.beginPath();
      ctx.arc(p.x * canvas.width, p.y * canvas.height, 1.6, 0, Math.PI * 2);
      ctx.fill();
    }
    const chin = pts[152], forehead = pts[10], left = pts[234], right = pts[454];
    const cx = ((left.x + right.x) / 2) * canvas.width;
    const cy = ((forehead.y + chin.y) / 2) * canvas.height;
    const rx = Math.abs(right.x - left.x) / 2 * canvas.width * 1.3;
    const ry = Math.abs(chin.y - forehead.y) / 2 * canvas.height * 1.3;
    ctx.strokeStyle = faceIsWellPositioned() ? "rgba(34,197,94,0.5)" : "rgba(255,255,255,0.2)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
    ctx.stroke();
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
          {phase === "active" && <p className="text-sm text-zinc-400"><Camera className="mr-1 inline h-4 w-4" />{isReady ? statusMsg : "Loading face detection..."}</p>}
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
