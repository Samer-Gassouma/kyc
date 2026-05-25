"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { API_BASE } from "@/lib/apiBase";
import { canvasToJpegBlob } from "@/lib/frameEncoder";
import { useFaceDetection, type PoseState } from "@/hooks/useFaceDetection";
import Link from "next/link";
import { ArrowLeft, Camera, Loader2, CheckCircle, XCircle, UserPlus, Fingerprint, ArrowUp, ArrowLeftCircle, ArrowRightCircle, Focus } from "lucide-react";

type Mode = "enroll" | "verify";
type Angle = "center" | "left" | "right" | "up";

const ANGLES: Angle[] = ["center", "left", "right", "up"];
const LABEL: Record<Angle, string> = {
  center: "Center your face in the oval",
  left: "Turn your head LEFT",
  right: "Turn your head RIGHT",
  up: "Tilt your head UP",
};
const ICON: Record<Angle, typeof Focus> = {
  center: Focus, left: ArrowLeftCircle, right: ArrowRightCircle, up: ArrowUp,
};

const POSE_MATCH: Record<Angle, (p: PoseState) => boolean> = {
  center: (p) => p === "center",
  left:   (p) => p === "left",
  right:  (p) => p === "right",
  up:     (p) => p === "up",
};

// ── Page ──────────────────────────────────────────────────────────

export default function FacePage() {
  const [mode, setMode] = useState<Mode>("enroll");
  const [token, setToken] = useState("");
  const [phase, setPhase] = useState<"idle" | "active" | "capturing" | "verifying" | "done" | "error">("idle");
  const [statusMsg, setStatusMsg] = useState("");
  const [errMsg, setErrMsg] = useState<string | null>(null);
  const [userId, setUserId] = useState("");
  const [result, setResult] = useState<any>(null);
  const [currentAngle, setCurrentAngle] = useState<Angle>("center");
  const [angleProgress, setAngleProgress] = useState(0);
  const [completedAngles, setCompletedAngles] = useState<Set<Angle>>(new Set());
  const [guideGlow, setGuideGlow] = useState<"none" | "yellow" | "green">("none");

  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animRef = useRef<number>(0);
  const stableRef = useRef(0);
  const capturedFrames = useRef<Map<Angle, Blob>>(new Map());
  const drawingRef = useRef({ box: null as any, pose: "none" as PoseState, progress: 0, glow: "none" as string, angle: "center" as string });

  const { isReady, detect } = useFaceDetection();

  useEffect(() => {
    fetch(`${API_BASE}/api/auth/token`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: `face_${Date.now()}` }) })
      .then(r => r.json()).then(d => setToken(d.access_token)).catch(() => setToken("dev_token"));
  }, []);

  const stop = useCallback(() => {
    cancelAnimationFrame(animRef.current);
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
  }, []);

  const start = useCallback(async () => {
    setErrMsg(null); setResult(null); setPhase("active");
    setCurrentAngle("center"); setAngleProgress(0);
    setCompletedAngles(new Set()); setGuideGlow("none");
    capturedFrames.current.clear(); stableRef.current = 0;
    setStatusMsg(LABEL.center);
    drawingRef.current = { box: null, pose: "none", progress: 0, glow: "none", angle: "center" };
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } }, audio: false,
      });
      streamRef.current = stream;
      const v = videoRef.current; if (!v) throw new Error("no video");
      v.srcObject = stream; await v.play();
    } catch (e) {
      setErrMsg(e instanceof Error ? e.message : "Camera"); setPhase("error");
    }
  }, []);

  useEffect(() => stop, []); // eslint-disable-line

  // ── Frame loop ─────────────────────────────────────────────────

  useEffect(() => {
    if (phase !== "active") return;
    let running = true, busy = false;

    async function loop() {
      if (!running) return;
      const video = videoRef.current;
      if (!video || video.videoWidth === 0 || busy) {
        animRef.current = requestAnimationFrame(loop);
        return;
      }
      busy = true;
      let res;
      try { res = await detect(video); } catch { res = null; }
      busy = false;
      if (!running) return;

      const d = drawingRef.current;
      const target = currentAngle;

      if (res) {
        d.box = res.box;
        d.pose = res.pose;
        d.angle = target;

        const matches = POSE_MATCH[target](res.pose);

        if (matches && !capturedFrames.current.has(target)) {
          stableRef.current = Math.min(30, stableRef.current + 1);
        } else {
          stableRef.current = Math.max(0, stableRef.current - 2);
        }

        const pct = Math.round((stableRef.current / 30) * 100);
        setAngleProgress(pct);
        d.progress = pct;
        d.glow = res.pose === "none" ? "none" : matches ? "green" : "yellow";
        setGuideGlow(d.glow as any);

        if (matches) {
          setStatusMsg(`${LABEL[target]} — hold still`);
        } else {
          // Show live debug: nose position relative to box center
          const dbg = (window as any).__pose;
          const sx = dbg?.noseOffX != null ? ` noseOffX=${dbg.noseOffX}` : "";
          const sy = dbg?.noseOffY != null ? ` noseOffY=${dbg.noseOffY}` : "";
          setStatusMsg(`${LABEL[target]}${sx}${sy}`);
        }

        if (stableRef.current >= 30 && !capturedFrames.current.has(target)) {
          busy = true;
          await captureAngle(target, video);
          busy = false;
          stableRef.current = 0;
          drawingRef.current.progress = 0;
          setAngleProgress(0);
        }
      } else {
        d.box = null; d.pose = "none"; d.progress = 0; d.glow = "none";
        setGuideGlow("none"); setAngleProgress(0);
        setStatusMsg("Position your face in the oval");
        stableRef.current = 0;
      }

      drawOval(video, d.box, d.glow, d.progress);
      animRef.current = requestAnimationFrame(loop);
    }
    animRef.current = requestAnimationFrame(loop);
    return () => { running = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, currentAngle]);

  // ── Capture ────────────────────────────────────────────────────

  async function captureAngle(angle: Angle, video: HTMLVideoElement) {
    setPhase("capturing");
    const c = document.createElement("canvas");
    c.width = video.videoWidth; c.height = video.videoHeight;
    c.getContext("2d")!.drawImage(video, 0, 0);
    const blob = await canvasToJpegBlob(c, 0.85);
    capturedFrames.current.set(angle, blob);

    const done = new Set(completedAngles); done.add(angle);
    setCompletedAngles(done);

    const idx = ANGLES.indexOf(angle);
    if (idx < ANGLES.length - 1) {
      setCurrentAngle(ANGLES[idx + 1]);
      setStatusMsg(LABEL[ANGLES[idx + 1]]);
      setAngleProgress(0);
      drawingRef.current.progress = 0;
      setPhase("active");
    } else {
      setPhase("verifying"); setStatusMsg("Processing...");
      await verify();
    }
  }

  async function verify() {
    const best = capturedFrames.current.get("center") || capturedFrames.current.get(ANGLES[0])!;
    try {
      if (mode === "enroll") {
        const fd = new FormData(); fd.append("image", best, "face.jpg"); fd.append("liveness_score", "0.95");
        const res = await fetch(`${API_BASE}/api/face/enroll`, { method: "POST", headers: { Authorization: `Bearer ${token}` }, body: fd });
        if (!res.ok) throw new Error((await res.json().catch(()=>({}))).detail || `HTTP ${res.status}`);
        setUserId((await res.json()).user_id); setResult(await res.json());
      } else {
        if (!userId) { setPhase("error"); setErrMsg("Enroll first or paste a user ID"); return; }
        const fd = new FormData(); fd.append("image", best, "face.jpg"); fd.append("user_id", userId);
        const res = await fetch(`${API_BASE}/api/face/verify`, { method: "POST", headers: { Authorization: `Bearer ${token}` }, body: fd });
        if (!res.ok) throw new Error((await res.json().catch(()=>({}))).detail || `HTTP ${res.status}`);
        setResult(await res.json());
      }
      setPhase("done");
    } catch (e) { setPhase("error"); setErrMsg(e instanceof Error ? e.message : "Request failed"); }
  }

  function retry() { stop(); setPhase("idle"); setErrMsg(null); setResult(null); }

  // ── Oval guide drawing ─────────────────────────────────────────

  function drawOval(video: HTMLVideoElement, box: any, glow: string, progress: number) {
    const canvas = overlayRef.current;
    if (!canvas || !video) return;
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Oval params
    const ox = w / 2, oy = h / 2.1;
    const orx = w * 0.21, ory = h * 0.29;

    // Glow color
    const glowColor = glow === "green" ? "rgba(34,197,94,0.5)" : glow === "yellow" ? "rgba(234,179,8,0.5)" : "rgba(255,255,255,0.15)";

    // Outer glow
    ctx.save();
    ctx.shadowColor = glow === "green" ? "#22c55e" : glow === "yellow" ? "#eab308" : "#ffffff";
    ctx.shadowBlur = glow === "green" ? 25 : glow === "yellow" ? 18 : 8;
    ctx.strokeStyle = glowColor;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.ellipse(ox, oy, orx, ory, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();

    // Progress arc
    if (progress > 0 && progress < 100) {
      const startAngle = -Math.PI / 2;
      const endAngle = startAngle + (progress / 100) * Math.PI * 2;
      ctx.strokeStyle = glow === "green" ? "rgba(34,197,94,0.9)" : "rgba(56,189,248,0.8)";
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.ellipse(ox, oy, orx + 3, ory + 3, 0, startAngle, endAngle);
      ctx.stroke();
    }

    // Face bounding box (subtle)
    if (box) {
      ctx.strokeStyle = glow === "green" ? "rgba(34,197,94,0.3)" : "rgba(255,255,255,0.15)";
      ctx.lineWidth = 1;
      ctx.strokeRect(box.x, box.y, box.width, box.height);
    }
  }

  const Icon = ICON[currentAngle];

  return (
    <div className="flex min-h-screen flex-col bg-zinc-950 text-zinc-100">
      <header className="flex items-center gap-3 border-b border-zinc-800 px-4 py-3">
        <Link href="/" className="rounded-full p-1.5 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"><ArrowLeft className="h-5 w-5" /></Link>
        <Fingerprint className="h-5 w-5 text-blue-400" /><h1 className="text-base font-semibold">Face Pipeline Test</h1>
      </header>
      <main className="mx-auto flex w-full max-w-md flex-1 flex-col items-center p-4">
        <div className="mb-4 flex w-full rounded-lg bg-zinc-900 p-1">
          <button onClick={() => { setMode("enroll"); retry(); }} className={`flex-1 rounded-md px-4 py-2 text-sm font-medium ${mode === "enroll" ? "bg-blue-600 text-white" : "text-zinc-400"}`}>
            <UserPlus className="mr-2 inline h-4 w-4" />Enroll</button>
          <button onClick={() => { setMode("verify"); retry(); }} className={`flex-1 rounded-md px-4 py-2 text-sm font-medium ${mode === "verify" ? "bg-blue-600 text-white" : "text-zinc-400"}`}>
            <Fingerprint className="mr-2 inline h-4 w-4" />Verify</button>
        </div>

        {phase === "idle" && (
          <div className="flex flex-1 flex-col items-center justify-center gap-4">
            <p className="text-sm text-zinc-400">{mode === "enroll" ? "Capture your face from multiple angles" : "Verify against enrolled identity"}</p>
            <button onClick={start} className="flex items-center gap-2 rounded-xl bg-blue-600 px-6 py-3 text-sm font-medium text-white hover:bg-blue-500"><Camera className="h-5 w-5" />Start Camera</button>
            {mode === "verify" && <input type="text" value={userId} onChange={e => setUserId(e.target.value)} placeholder="Paste user_id from enrollment..." className="mt-2 w-full rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-200" />}
          </div>
        )}

        <div className="relative w-full overflow-hidden rounded-2xl bg-black" style={{ maxWidth: 400 }}>
          {(phase === "active" || phase === "capturing") && (
            <div className="relative">
              <video ref={videoRef} autoPlay playsInline muted className="h-full w-full object-cover" style={{ aspectRatio: "3/4", transform: "scaleX(-1)" }} />
              <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full" style={{ transform: "scaleX(-1)" }} />
            </div>
          )}
          {phase === "verifying" && <div className="flex flex-col items-center justify-center bg-black gap-3" style={{ aspectRatio: "3/4" }}><Loader2 className="h-10 w-10 animate-spin text-blue-400" /><p className="text-sm text-zinc-300">{statusMsg}</p></div>}
          {phase === "done" && <div className={`flex items-center justify-center ${mode === "enroll" || result?.matched ? "bg-green-950/50" : "bg-red-950/50"}`} style={{ aspectRatio: "3/4" }}>{(mode === "enroll" || result?.matched) ? <CheckCircle className="h-16 w-16 text-green-400" /> : <XCircle className="h-16 w-16 text-red-400" />}</div>}
        </div>

        <div className="mt-4 flex flex-col items-center gap-2 text-center w-full">
          {(phase === "active" || phase === "capturing") && (
            <div className="flex items-center gap-2 mb-2">
              {ANGLES.map((a, i) => (
                <div key={a} className="flex items-center gap-2">
                  <div className={`flex h-7 w-7 items-center justify-center rounded-full text-xs font-bold ${completedAngles.has(a) ? "bg-green-500 text-white" : a === currentAngle ? "bg-blue-600 text-white ring-2 ring-blue-400" : "bg-zinc-700 text-zinc-400"}`}>
                    {completedAngles.has(a) ? <CheckCircle className="h-3.5 w-3.5" /> : i + 1}
                  </div>
                  {i < 3 && <div className="h-0.5 w-5 bg-zinc-700" />}
                </div>
              ))}
            </div>
          )}
          {phase === "active" && <div className="flex items-center gap-2 text-sm text-zinc-400"><Icon className="h-4 w-4 text-sky-400" />{isReady ? statusMsg : "Loading..."}</div>}
          {phase === "error" && <div className="flex flex-col items-center gap-3"><p className="text-sm text-red-400"><XCircle className="mr-1 inline h-4 w-4" />{errMsg}</p><button onClick={retry} className="rounded-full bg-blue-600 px-6 py-2.5 text-sm text-white">Try Again</button></div>}
        </div>

        {phase === "done" && result && (
          <div className="mt-4 w-full space-y-3">
            {mode === "enroll" && (
              <div className="rounded-xl bg-zinc-900 p-4">
                <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">Enrolled</h2>
                <code className="block break-all rounded bg-zinc-800 p-2 text-xs text-green-400">{result.user_id}</code>
                <button onClick={() => { setMode("verify"); setUserId(result.user_id || ""); setPhase("idle"); }} className="mt-3 w-full rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white">Switch to Verify</button>
              </div>
            )}
            {mode === "verify" && (
              <div className={`rounded-xl p-4 ${result.matched ? "bg-green-500/10 ring-1 ring-green-500/30" : "bg-red-500/10 ring-1 ring-red-500/30"}`}>
                <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">Result</h2>
                <p className={`text-lg font-bold ${result.matched ? "text-green-400" : "text-red-400"}`}>{result.matched ? "MATCH" : "NO MATCH"} — {((result.confidence ?? 0) * 100).toFixed(1)}%</p>
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
