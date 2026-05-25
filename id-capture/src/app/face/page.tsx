"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { API_BASE } from "@/lib/apiBase";
import { canvasToJpegBlob } from "@/lib/frameEncoder";
import { useFaceDetection } from "@/hooks/useFaceDetection";
import Link from "next/link";
import { ArrowLeft, Camera, Loader2, CheckCircle, XCircle, UserPlus, Fingerprint, ArrowUp, ArrowLeftCircle, ArrowRightCircle, Focus } from "lucide-react";
import Face3DViewer from "@/components/kyc/Face3DViewer";

type Mode = "enroll" | "verify";
type Angle = "center" | "left" | "right" | "up";

const ANGLES: Angle[] = ["center", "left", "right", "up"];
const LABEL: Record<Angle, string> = {
  center: "Look straight ahead",
  left: "Turn your head to the LEFT",
  right: "Turn your head to the RIGHT",
  up: "Tilt your head UP",
};
const ICON: Record<Angle, typeof Focus> = {
  center: Focus, left: ArrowLeftCircle, right: ArrowRightCircle, up: ArrowUp,
};

export default function FacePage() {
  const [mode, setMode] = useState<Mode>("enroll");
  const [token, setToken] = useState("");
  const [phase, setPhase] = useState<"idle" | "active" | "capturing" | "verifying" | "done" | "error">("idle");
  const [statusMsg, setStatusMsg] = useState("");
  const [errMsg, setErrMsg] = useState<string | null>(null);
  const [userId, setUserId] = useState("");
  const [result, setResult] = useState<{ matched?: boolean; confidence?: number; threshold_used?: number; user_id?: string } | null>(null);
  const [currentAngle, setCurrentAngle] = useState<Angle>("center");
  const [angleProgress, setAngleProgress] = useState(0);
  const [completedAngles, setCompletedAngles] = useState<Set<Angle>>(new Set());
  const [livePoints, setLivePoints] = useState<number[][]>([]);
  const [liveColors, setLiveColors] = useState<string[]>([]);

  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animRef = useRef<number>(0);
  const stableRef = useRef(0);
  const capturedFrames = useRef<Map<Angle, Blob>>(new Map());
  const lastResult = useRef<{ pose: { yaw: number; pitch: number }; landmarks: any } | null>(null);

  const { isReady, detect } = useFaceDetection();

  // JWT
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
    setCompletedAngles(new Set()); setLivePoints([]);
    capturedFrames.current.clear(); stableRef.current = 0;
    setStatusMsg(LABEL.center);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } }, audio: false,
      });
      streamRef.current = stream;
      const v = videoRef.current;
      if (!v) throw new Error("no video");
      v.srcObject = stream;
      await v.play();
    } catch (e) {
      setErrMsg(e instanceof Error ? e.message : "Camera error");
      setPhase("error");
    }
  }, []);

  useEffect(() => stop, []); // eslint-disable-line

  // ── Frame loop ─────────────────────────────────────────────────

  useEffect(() => {
    if (phase !== "active") return;

    let running = true;
    let captureTriggered = false;

    async function loop() {
      if (!running) return;
      const video = videoRef.current;
      if (!video || video.videoWidth === 0) {
        animRef.current = requestAnimationFrame(loop);
        return;
      }

      const result = await detect(video);
      if (!running) return;

      if (result) {
        lastResult.current = result;
        const newPts = result.landmarks.positions.map((p: any) => [p.x / video.videoWidth, p.y / video.videoHeight]);
        setLivePoints(newPts);
        setLiveColors(sampleColors(video, newPts));
        drawOverlay(video);

        // Pose check
        const { yaw, pitch } = result.pose;
        const target = currentAngle;
        let matchesAngle = false;

        if (target === "left" && yaw < -18) matchesAngle = true;
        else if (target === "right" && yaw > 18) matchesAngle = true;
        else if (target === "up" && pitch < -12) matchesAngle = true;
        else if (target === "center" && Math.abs(yaw) < 10 && Math.abs(pitch) < 12) matchesAngle = true;

        if (matchesAngle && !captureTriggered) {
          stableRef.current = Math.min(30, stableRef.current + 1);
          setAngleProgress(Math.round((stableRef.current / 30) * 100));
          setStatusMsg(`${LABEL[target]} — hold still`);
          if (stableRef.current >= 30) {
            captureTriggered = true;
            await captureAngle(target, video);
            if (!running) return;
            captureTriggered = false;
          }
        } else if (!matchesAngle) {
          stableRef.current = Math.max(0, stableRef.current - 2);
          setAngleProgress(Math.round((stableRef.current / 30) * 100));
          if (stableRef.current < 5) setStatusMsg(LABEL[target]);
        }
      } else {
        stableRef.current = Math.max(0, stableRef.current - 3);
        setAngleProgress(Math.round((stableRef.current / 30) * 100));
        setStatusMsg("Position your face in the frame");
      }

      animRef.current = requestAnimationFrame(loop);
    }

    animRef.current = requestAnimationFrame(loop);
    return () => { running = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, currentAngle]);

  // ── Capture one angle ──────────────────────────────────────────

  async function captureAngle(angle: Angle, video: HTMLVideoElement) {
    setPhase("capturing");

    const c = document.createElement("canvas");
    c.width = video.videoWidth; c.height = video.videoHeight;
    c.getContext("2d")!.drawImage(video, 0, 0);
    const blob = await canvasToJpegBlob(c, 0.85);
    capturedFrames.current.set(angle, blob);

    const done = new Set(completedAngles);
    done.add(angle);
    setCompletedAngles(done);
    stableRef.current = 0;
    setAngleProgress(0);

    const idx = ANGLES.indexOf(angle);
    if (idx < ANGLES.length - 1) {
      setCurrentAngle(ANGLES[idx + 1]);
      setStatusMsg(LABEL[ANGLES[idx + 1]]);
      setPhase("active");
    } else {
      setPhase("verifying");
      setStatusMsg("Processing...");
      await verify();
    }
  }

  // ── Verify / Enroll ────────────────────────────────────────────

  async function verify() {
    const best = capturedFrames.current.get("center") || capturedFrames.current.get(ANGLES[0])!;

    try {
      if (mode === "enroll") {
        const fd = new FormData();
        fd.append("image", best, "face.jpg");
        fd.append("liveness_score", "0.95");
        if (livePoints.length > 0) fd.append("landmarks_3d", JSON.stringify(livePoints));

        const res = await fetch(`${API_BASE}/api/face/enroll`, { method: "POST", headers: { Authorization: `Bearer ${token}` }, body: fd });
        if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
        const data = await res.json();
        setUserId(data.user_id);
        setResult(data);
        setPhase("done");
      } else {
        if (!userId) { setPhase("error"); setErrMsg("Enroll first or paste a user ID"); return; }
        const fd = new FormData();
        fd.append("image", best, "face.jpg");
        fd.append("user_id", userId);
        const res = await fetch(`${API_BASE}/api/face/verify`, { method: "POST", headers: { Authorization: `Bearer ${token}` }, body: fd });
        if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
        setResult(await res.json());
        setPhase("done");
      }
    } catch (e) {
      setPhase("error");
      setErrMsg(e instanceof Error ? e.message : "Request failed");
    }
  }

  function retry() { stop(); setPhase("idle"); setErrMsg(null); setResult(null); }

  // ── Overlay — 68-point wireframe ───────────────────────────────

  // Feature group definitions for styled overlay
  const JAW = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16];
  const LEFT_BROW = [17,18,19,20,21];
  const RIGHT_BROW = [22,23,24,25,26];
  const NOSE_BRIDGE = [27,28,29,30];
  const NOSE_BOTTOM = [31,32,33,34,35];
  const LEFT_EYE = [36,37,38,39,40,41];
  const RIGHT_EYE = [42,43,44,45,46,47];
  const MOUTH_OUTER = [48,49,50,51,52,53,54,55,56,57,58,59];
  const MOUTH_INNER = [60,61,62,63,64,65,66,67];

  function drawPoly(ctx: CanvasRenderingContext2D, pts: number[][], indices: number[], w: number, h: number, fill: string, stroke: string, lw: number) {
    if (indices.some(i => i >= pts.length)) return;
    ctx.fillStyle = fill;
    ctx.strokeStyle = stroke;
    ctx.lineWidth = lw;
    ctx.beginPath();
    ctx.moveTo(pts[indices[0]][0] * w, pts[indices[0]][1] * h);
    for (let i = 1; i < indices.length; i++) ctx.lineTo(pts[indices[i]][0] * w, pts[indices[i]][1] * h);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }

  function drawLine(ctx: CanvasRenderingContext2D, pts: number[][], indices: number[], w: number, h: number, color: string, lw: number) {
    if (indices.some(i => i >= pts.length)) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = lw;
    ctx.beginPath();
    ctx.moveTo(pts[indices[0]][0] * w, pts[indices[0]][1] * h);
    for (let i = 1; i < indices.length; i++) ctx.lineTo(pts[indices[i]][0] * w, pts[indices[i]][1] * h);
    if (indices.length > 2) ctx.closePath();
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
    const pts = livePoints;
    if (pts.length < 60) return;
    const w = canvas.width, h = canvas.height;

    // Jaw — subtle white outline
    drawLine(ctx, pts, JAW, w, h, "rgba(255,255,255,0.25)", 2);

    // Eyebrows — thicker warm lines
    drawLine(ctx, pts, LEFT_BROW, w, h, "rgba(251,191,36,0.7)", 2.5);
    drawLine(ctx, pts, RIGHT_BROW, w, h, "rgba(251,191,36,0.7)", 2.5);

    // Eyes — filled with subtle white + blue outline
    drawPoly(ctx, pts, LEFT_EYE, w, h, "rgba(255,255,255,0.10)", "rgba(96,165,250,0.7)", 2);
    drawPoly(ctx, pts, RIGHT_EYE, w, h, "rgba(255,255,255,0.10)", "rgba(96,165,250,0.7)", 2);

    // Nose bridge — thin purple
    drawLine(ctx, pts, NOSE_BRIDGE, w, h, "rgba(168,85,247,0.5)", 1.5);
    // Nose bottom ring — filled
    drawPoly(ctx, pts, NOSE_BOTTOM, w, h, "rgba(168,85,247,0.10)", "rgba(168,85,247,0.6)", 1.5);

    // Mouth — filled pink + red outline
    drawPoly(ctx, pts, MOUTH_OUTER, w, h, "rgba(239,68,68,0.10)", "rgba(239,68,68,0.6)", 2);
    drawLine(ctx, pts, MOUTH_INNER, w, h, "rgba(239,68,68,0.4)", 1);

    // Landmark dots — larger on key features
    for (let i = 0; i < pts.length; i++) {
      const isEye = LEFT_EYE.includes(i) || RIGHT_EYE.includes(i);
      const isMouth = MOUTH_OUTER.includes(i) || MOUTH_INNER.includes(i);
      const isNose = NOSE_BRIDGE.includes(i) || NOSE_BOTTOM.includes(i);
      const r = isEye ? 2.2 : isMouth ? 2 : isNose ? 1.8 : 1.2;
      ctx.fillStyle = isEye ? "rgba(96,165,250,0.8)" : isMouth ? "rgba(239,68,68,0.7)" : isNose ? "rgba(168,85,247,0.7)" : "rgba(255,255,255,0.4)";
      ctx.beginPath();
      ctx.arc(pts[i][0] * w, pts[i][1] * h, r, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Sample colors from video for 3D face
  function sampleColors(video: HTMLVideoElement, pts: number[][]): string[] {
    if (!video || video.videoWidth === 0 || pts.length < 30) return [];
    const c = document.createElement("canvas");
    c.width = video.videoWidth; c.height = video.videoHeight;
    const ctx = c.getContext("2d");
    if (!ctx) return [];
    ctx.drawImage(video, 0, 0);
    const colors: string[] = [];
    for (const pt of pts) {
      const px = Math.round(pt[0] * video.videoWidth);
      const py = Math.round(pt[1] * video.videoHeight);
      try {
        const [r, g, b] = ctx.getImageData(Math.min(px, video.videoWidth-1), Math.min(py, video.videoHeight-1), 1, 1).data;
        colors.push(`rgb(${r},${g},${b})`);
      } catch { colors.push("#38bdf8"); }
    }
    return colors;
  }

  const Icon = ICON[currentAngle];

  return (
    <div className="flex min-h-screen flex-col bg-zinc-950 text-zinc-100">
      <header className="flex items-center gap-3 border-b border-zinc-800 px-4 py-3">
        <Link href="/" className="rounded-full p-1.5 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"><ArrowLeft className="h-5 w-5" /></Link>
        <Fingerprint className="h-5 w-5 text-blue-400" />
        <h1 className="text-base font-semibold">Face Pipeline Test</h1>
      </header>

      <main className="mx-auto flex w-full max-w-md flex-1 flex-col items-center p-4">
        <div className="mb-4 flex w-full rounded-lg bg-zinc-900 p-1">
          <button onClick={() => { setMode("enroll"); retry(); }} className={`flex-1 rounded-md px-4 py-2 text-sm font-medium ${mode === "enroll" ? "bg-blue-600 text-white" : "text-zinc-400"}`}>
            <UserPlus className="mr-2 inline h-4 w-4" />Enroll
          </button>
          <button onClick={() => { setMode("verify"); retry(); }} className={`flex-1 rounded-md px-4 py-2 text-sm font-medium ${mode === "verify" ? "bg-blue-600 text-white" : "text-zinc-400"}`}>
            <Fingerprint className="mr-2 inline h-4 w-4" />Verify
          </button>
        </div>

        {phase === "idle" && (
          <div className="flex flex-1 flex-col items-center justify-center gap-4">
            <p className="text-sm text-zinc-400">{mode === "enroll" ? "Capture your face from multiple angles" : "Verify against enrolled identity"}</p>
            <button onClick={start} className="flex items-center gap-2 rounded-xl bg-blue-600 px-6 py-3 text-sm font-medium text-white hover:bg-blue-500">
              <Camera className="h-5 w-5" />Start Camera
            </button>
            {mode === "verify" && (
              <input type="text" value={userId} onChange={e => setUserId(e.target.value)} placeholder="Paste user_id from enrollment..."
                className="mt-2 w-full rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-200" />
            )}
          </div>
        )}

        <div className="relative w-full overflow-hidden rounded-2xl bg-black" style={{ maxWidth: 400 }}>
          {(phase === "active" || phase === "capturing") && (
            <div className="relative">
              <video ref={videoRef} autoPlay playsInline muted className="h-full w-full object-cover" style={{ aspectRatio: "3/4", transform: "scaleX(-1)" }} />
              <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full" style={{ transform: "scaleX(-1)" }} />
            </div>
          )}
          {phase === "verifying" && (
            <div className="flex flex-col items-center justify-center bg-black gap-3" style={{ aspectRatio: "3/4" }}>
              <Loader2 className="h-10 w-10 animate-spin text-blue-400" />
              <p className="text-sm text-zinc-300">{statusMsg}</p>
            </div>
          )}
          {phase === "done" && (
            <div className={`flex items-center justify-center ${mode === "enroll" || result?.matched ? "bg-green-950/50" : "bg-red-950/50"}`} style={{ aspectRatio: "3/4" }}>
              {(mode === "enroll" || result?.matched) ? <CheckCircle className="h-16 w-16 text-green-400" /> : <XCircle className="h-16 w-16 text-red-400" />}
            </div>
          )}
        </div>

        <div className="mt-4 flex flex-col items-center gap-2 text-center w-full">
          {(phase === "active" || phase === "capturing") && (
            <div className="flex items-center gap-2 mb-2">
              {ANGLES.map((a, i) => (
                <div key={a} className="flex items-center gap-2">
                  <div className={`flex h-7 w-7 items-center justify-center rounded-full text-xs font-bold transition-all ${
                    completedAngles.has(a) ? "bg-green-500 text-white" : a === currentAngle ? "bg-blue-600 text-white ring-2 ring-blue-400" : "bg-zinc-700 text-zinc-400"
                  }`}>
                    {completedAngles.has(a) ? <CheckCircle className="h-3.5 w-3.5" /> : i + 1}
                  </div>
                  {i < 3 && <div className="h-0.5 w-5 bg-zinc-700" />}
                </div>
              ))}
            </div>
          )}
          {(phase === "active" || phase === "capturing") && angleProgress > 0 && (
            <div className="h-1.5 w-48 overflow-hidden rounded-full bg-zinc-700">
              <div className="h-full rounded-full bg-sky-400 transition-all duration-200" style={{ width: `${angleProgress}%` }} />
            </div>
          )}
          {phase === "active" && (
            <div className="flex items-center gap-2 text-sm text-zinc-400">
              <Icon className="h-4 w-4 text-sky-400" />{isReady ? statusMsg : "Loading face detection..."}
            </div>
          )}
          {phase === "error" && (
            <div className="flex flex-col items-center gap-3">
              <p className="text-sm text-red-400"><XCircle className="mr-1 inline h-4 w-4" />{errMsg}</p>
              <button onClick={retry} className="rounded-full bg-blue-600 px-6 py-2.5 text-sm font-medium text-white">Try Again</button>
            </div>
          )}
        </div>

        {(phase === "active" || phase === "done") && livePoints.length > 0 && (
          <div className="mt-4 w-full rounded-xl bg-zinc-900 p-4">
            <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">3D Face Scan</h2>
            <p className="mb-1 text-xs text-zinc-600">Drag to rotate — scroll to zoom</p>
            <Face3DViewer points={livePoints} colors={liveColors} width={368} height={400} />
          </div>
        )}

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
